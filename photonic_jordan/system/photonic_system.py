"""Top-level orchestration object for the photonic Jordan framework."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np

from ..dynamics import PassiveLODynamics
from ..hierarchy import InvariantEngine, JordanFiltration
from ..math import normalize_density, safe_matmul
from ..measurement import ObservableFactory
from ..schur import SchurWeylDecomposition
from ..spaces import LabeledTensorSpace, SymmetricGroupProjectors
from ..specs import ModelSpec
from ..state import (
    ArrayLike,
    MultiplicityLabel,
    Partition,
    PhotonicState,
    StateBuilder,
    StateFactory,
    UnitaryFactory,
)


class PhotonicSystem:
    """User-facing entry point for preparing states and evaluating invariants.

    This object owns all cached structures:
    - labeled tensor space and one-body generators,
    - Schur/multiplicity projectors,
    - global and scoped Jordan filtrations.
    - observable constructors for lifted one-body measurements.

    Scope semantics
    ---------------
    Exactly one of the following can be active in an invariant query:
    1) global (default),
    2) ``sector=lambda``,
    3) ``multiplicity=(lambda, a)``.
    """

    def __init__(
        self,
        m_ext: int,
        n_particles: int,
        particle_type: str = "boson",
        rng: Optional[np.random.Generator] = None,
    ):
        self.spec = ModelSpec(m_ext=m_ext, n_particles=n_particles, particle_type=particle_type)
        self.space = LabeledTensorSpace(self.spec)
        self.rng = np.random.default_rng() if rng is None else rng

        self._state_factory = StateFactory(self.space, rng=self.rng)
        self._dynamics = PassiveLODynamics(self.space)
        self._filtration = JordanFiltration(self.space)
        self._invariants = InvariantEngine(self._filtration, self._dynamics)
        self._built_order = -1

        try:
            self._projectors = SymmetricGroupProjectors(self.space)
        except NotImplementedError:
            self._projectors = None

        self._decomposition: Optional[SchurWeylDecomposition] = None
        self._scoped_filtrations: Dict[Tuple[Union[str, Partition], ...], JordanFiltration] = {}
        self._scoped_built_order: Dict[Tuple[Union[str, Partition], ...], int] = {}

        self.state = StateBuilder(self)
        self.unitary = UnitaryFactory(self)
        self.observable = ObservableFactory(self)

    @property
    def hilbert_dim(self) -> int:
        """Return external Hilbert dimension ``m_ext ** n_particles``."""
        return self.spec.hilbert_dim

    @property
    def dynamics(self) -> PassiveLODynamics:
        """Return passive linear-optics dynamics helper."""
        return self._dynamics

    @property
    def invariants_engine(self) -> InvariantEngine:
        """Return invariant diagnostic helper bound to global filtration."""
        return self._invariants

    @property
    def projectors(self) -> Optional[SymmetricGroupProjectors]:
        """Return symmetric-group projector helper (or ``None`` outside demo scope)."""
        return self._projectors

    @property
    def decomposition(self) -> SchurWeylDecomposition:
        """Return lazily built Schur-Weyl decomposition cache.

        Returns
        -------
        SchurWeylDecomposition
            Decomposition object exposing sector and multiplicity projectors.
        """
        if self._projectors is None:
            raise NotImplementedError("Schur decomposition is currently demo-supported for n=2,3 only.")
        if self._decomposition is None:
            self._decomposition = SchurWeylDecomposition(self.space, self._projectors)
        return self._decomposition

    def ensure_filtration(self, max_order: Optional[int] = None) -> None:
        """Build global Jordan filtration up to ``max_order``.

        Parameters
        ----------
        max_order:
            Maximal hierarchy order. Defaults to ``n_particles``.
        """
        if max_order is None:
            max_order = self.spec.n_particles
        if max_order > self._built_order:
            self._filtration.build(max_order=max_order)
            self._built_order = max_order

    @staticmethod
    def _resolve_multiplicity_arg(
        multiplicity: Optional[MultiplicityLabel],
        copy: Optional[MultiplicityLabel],
    ) -> Optional[MultiplicityLabel]:
        if multiplicity is not None and copy is not None:
            raise ValueError("Provide only one of multiplicity=... or copy=... (legacy alias).")
        return multiplicity if multiplicity is not None else copy

    @staticmethod
    def _validate_scope_inputs(sector: Optional[Partition], multiplicity: Optional[MultiplicityLabel]) -> None:
        if sector is not None and multiplicity is not None:
            raise ValueError("Provide at most one scope: global, sector=..., or multiplicity=....")

    @staticmethod
    def _scope_key(
        sector: Optional[Partition],
        multiplicity: Optional[MultiplicityLabel],
    ) -> Tuple[Union[str, Partition], ...]:
        if sector is not None:
            return ("sector", tuple(sector))
        if multiplicity is not None:
            lam, a = multiplicity
            return ("multiplicity", tuple(lam), int(a))
        return ("global",)

    def scope_projector(
        self,
        sector: Optional[Partition] = None,
        multiplicity: Optional[MultiplicityLabel] = None,
        copy: Optional[MultiplicityLabel] = None,
    ) -> Optional[np.ndarray]:
        """Return projector associated with active scope.

        Parameters
        ----------
        sector:
            Schur sector label ``lambda``.
        multiplicity:
            Multiplicity-local label ``(lambda, a)``.
        copy:
            Deprecated alias for ``multiplicity``.

        Returns
        -------
        np.ndarray or None
            Scope projector for local computations; ``None`` means global scope.
        """
        multiplicity = self._resolve_multiplicity_arg(multiplicity, copy)
        self._validate_scope_inputs(sector=sector, multiplicity=multiplicity)
        if sector is not None:
            return self.sector_projector(sector)
        if multiplicity is not None:
            lam, a = multiplicity
            return self.multiplicity_projector(lam, a)
        return None

    def project_density_to_scope(
        self,
        rho: np.ndarray,
        sector: Optional[Partition] = None,
        multiplicity: Optional[MultiplicityLabel] = None,
        copy: Optional[MultiplicityLabel] = None,
    ) -> np.ndarray:
        """Compress ``rho`` into active scope as ``Q rho Q``."""
        Q = self.scope_projector(sector=sector, multiplicity=multiplicity, copy=copy)
        if Q is None:
            return rho
        return safe_matmul(Q, rho, Q)

    def ensure_scope_filtration(
        self,
        max_order: int,
        sector: Optional[Partition] = None,
        multiplicity: Optional[MultiplicityLabel] = None,
        copy: Optional[MultiplicityLabel] = None,
    ) -> JordanFiltration:
        """Build or reuse filtration associated with selected scope.

        Parameters
        ----------
        max_order:
            Maximal hierarchy order.
        sector:
            Schur sector label.
        multiplicity:
            Multiplicity-local label ``(lambda, a)``.
        copy:
            Deprecated alias for ``multiplicity``.

        Returns
        -------
        JordanFiltration
            Global filtration or local filtration compressed by scope projector.

        Notes
        -----
        The scoped filtration is generated by projected one-body generators
        ``Q E_st Q`` and seed operator ``Q``.
        """
        multiplicity = self._resolve_multiplicity_arg(multiplicity, copy)
        self._validate_scope_inputs(sector=sector, multiplicity=multiplicity)
        key = self._scope_key(sector=sector, multiplicity=multiplicity)

        if key == ("global",):
            self.ensure_filtration(max_order=max_order)
            return self._filtration

        if key not in self._scoped_filtrations:
            Q = self.scope_projector(sector=sector, multiplicity=multiplicity)
            if Q is None:
                raise RuntimeError("Scoped filtration requires a non-global scope projector.")
            generators = [safe_matmul(Q, G, Q) for G in self.space.generators.values()]
            self._scoped_filtrations[key] = JordanFiltration(
                self.space,
                generator_list=generators,
                seed_operator=Q,
                support_projector=Q,
            )
            self._scoped_built_order[key] = -1

        if max_order > self._scoped_built_order[key]:
            self._scoped_filtrations[key].build(max_order=max_order)
            self._scoped_built_order[key] = max_order
        return self._scoped_filtrations[key]

    def available_partitions(self):
        """Return available Schur partitions in current demo scope."""
        if self._projectors is None:
            return []
        return self.decomposition.partitions()

    def sector_projector(self, lam: Partition) -> np.ndarray:
        """Return Schur sector projector ``Q_lambda``."""
        if self._projectors is None:
            raise NotImplementedError("Sector projectors are only implemented for n=2,3 in this demo.")
        return self.decomposition.sector_projector(lam)

    def multiplicity_projector(self, lam: Partition, a: int) -> np.ndarray:
        """Return multiplicity-local projector ``Q_{lambda,a}``."""
        if self._projectors is None:
            raise NotImplementedError("Multiplicity projectors are only implemented for n=2,3 in this demo.")
        return self.decomposition.multiplicity_projector(lam, a)

    # Backward-compatible alias.
    def copy_projector(self, lam: Partition, a: int) -> np.ndarray:
        """Alias of :meth:`multiplicity_projector` for backward compatibility."""
        return self.multiplicity_projector(lam, a)

    def density_state(self, rho: ArrayLike, label: Optional[str] = None) -> PhotonicState:
        """Build normalized :class:`PhotonicState` from tensor-basis matrix.

        Parameters
        ----------
        rho:
            Density matrix in tensor basis.
        label:
            Optional human-readable label.
        """
        arr = np.asarray(rho, dtype=complex)
        if arr.shape != (self.hilbert_dim, self.hilbert_dim):
            raise ValueError(f"Density matrix must have shape {(self.hilbert_dim, self.hilbert_dim)}.")
        return PhotonicState(system=self, data=normalize_density(arr), label=label)
