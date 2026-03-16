"""High-level photonic state object and invariant views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..math import normalize_density, safe_matmul

if TYPE_CHECKING:
    from ..measurement.observable import SingleParticleObservable
    from ..system.photonic_system import PhotonicSystem

ArrayLike = Union[np.ndarray, Sequence[Sequence[complex]]]
GramInput = Union[str, float, complex, np.ndarray, Sequence[Sequence[complex]]]
Partition = Tuple[int, ...]
MultiplicityLabel = Tuple[Partition, int]


@dataclass
class InvariantReport:
    """Container of cumulative/exact Jordan weights.

    Parameters
    ----------
    cumulative:
        Dictionary ``j -> I_{<=j}``.
    exact:
        Dictionary ``j -> I_j``.
    sector_weights:
        Optional sector traces ``Tr(Q_lambda rho)``.
    """

    cumulative: Dict[int, float]
    exact: Dict[int, float]
    sector_weights: Optional[Dict[Partition, float]] = None

    def summary(self, digits: int = 8) -> str:
        """Return multiline human-readable summary."""
        lines = ["InvariantReport"]
        lines.append("  cumulative weights:")
        for j in sorted(self.cumulative):
            lines.append(f"    j={j}: {self.cumulative[j]:.{digits}f}")

        lines.append("  exact-layer weights:")
        for j in sorted(self.exact):
            lines.append(f"    j={j}: {self.exact[j]:.{digits}f}")

        if self.sector_weights is not None:
            lines.append("  sector weights:")
            for lam, w in self.sector_weights.items():
                lines.append(f"    lambda={lam}: {w:.{digits}f}")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


class StateInvariantView:
    """Bound invariant API for a specific :class:`PhotonicState`.

    Notes
    -----
    Invariants are evaluated on tensor-space density operators projected to a
    chosen scope (global / sector / multiplicity), then measured against the
    corresponding Jordan filtration.
    """

    def __init__(self, state: "PhotonicState"):
        self.state = state

    def _density_in_scope(
        self,
        sector: Optional[Partition],
        multiplicity: Optional[MultiplicityLabel] = None,
        copy: Optional[MultiplicityLabel] = None,
    ) -> np.ndarray:
        rho_t = self.state._tensor_matrix()
        return self.state.system.project_density_to_scope(
            rho_t,
            sector=sector,
            multiplicity=multiplicity,
            copy=copy,
        )

    def I_exact(
        self,
        order: int,
        sector: Optional[Partition] = None,
        multiplicity: Optional[MultiplicityLabel] = None,
        copy: Optional[MultiplicityLabel] = None,
    ) -> float:
        """Return exact-layer invariant ``I_order``.

        Parameters
        ----------
        order:
            Jordan hierarchy order.
        sector:
            Optional Schur sector label.
        multiplicity:
            Optional multiplicity-local label ``(lambda, a)``.
        copy:
            Deprecated alias for ``multiplicity``.
        """
        filtration = self.state.system.ensure_scope_filtration(
            max_order=order,
            sector=sector,
            multiplicity=multiplicity,
            copy=copy,
        )
        return filtration.layer_weight(
            self._density_in_scope(sector=sector, multiplicity=multiplicity, copy=copy),
            order,
        )

    def I_cumulative(
        self,
        order: int,
        sector: Optional[Partition] = None,
        multiplicity: Optional[MultiplicityLabel] = None,
        copy: Optional[MultiplicityLabel] = None,
    ) -> float:
        """Return cumulative invariant ``I_{<=order}``."""
        filtration = self.state.system.ensure_scope_filtration(
            max_order=order,
            sector=sector,
            multiplicity=multiplicity,
            copy=copy,
        )
        return filtration.cumulative_weight(
            self._density_in_scope(sector=sector, multiplicity=multiplicity, copy=copy),
            order,
        )

    def report(
        self,
        max_order: Optional[int] = None,
        include_sectors: bool = True,
        sector: Optional[Partition] = None,
        multiplicity: Optional[MultiplicityLabel] = None,
        copy: Optional[MultiplicityLabel] = None,
    ) -> InvariantReport:
        """Return full invariant report up to ``max_order``."""
        if max_order is None:
            max_order = self.state.system.spec.n_particles

        filtration = self.state.system.ensure_scope_filtration(
            max_order=max_order,
            sector=sector,
            multiplicity=multiplicity,
            copy=copy,
        )
        rho_t = self._density_in_scope(sector=sector, multiplicity=multiplicity, copy=copy)

        cumulative = {j: filtration.cumulative_weight(rho_t, j) for j in range(max_order + 1)}
        exact = {j: filtration.layer_weight(rho_t, j) for j in range(max_order + 1)}

        sectors = (
            self.state.sector_weights()
            if include_sectors and sector is None and multiplicity is None and copy is None
            else None
        )
        return InvariantReport(cumulative=cumulative, exact=exact, sector_weights=sectors)


class StateMeasurementView:
    """Bound measurement API for a specific :class:`PhotonicState`."""

    def __init__(self, state: "PhotonicState"):
        self.state = state

    def expectation(
        self,
        observable: "SingleParticleObservable",
        sector: Optional[Partition] = None,
        multiplicity: Optional[MultiplicityLabel] = None,
        copy: Optional[MultiplicityLabel] = None,
        conditional: bool = False,
    ) -> float:
        """Return expectation value of an observable for this state."""
        return observable.expectation(
            self.state,
            sector=sector,
            multiplicity=multiplicity,
            copy=copy,
            conditional=conditional,
        )

    def variance(
        self,
        observable: "SingleParticleObservable",
        sector: Optional[Partition] = None,
        multiplicity: Optional[MultiplicityLabel] = None,
        copy: Optional[MultiplicityLabel] = None,
        conditional: bool = False,
    ) -> float:
        """Return variance of an observable for this state."""
        return observable.variance(
            self.state,
            sector=sector,
            multiplicity=multiplicity,
            copy=copy,
            conditional=conditional,
        )

    def distribution(
        self,
        observable: "SingleParticleObservable",
        sector: Optional[Partition] = None,
        multiplicity: Optional[MultiplicityLabel] = None,
        copy: Optional[MultiplicityLabel] = None,
        conditional: bool = False,
        tol: float = 1e-10,
    ):
        """Return grouped spectral distribution for this state."""
        return observable.distribution(
            self.state,
            sector=sector,
            multiplicity=multiplicity,
            copy=copy,
            conditional=conditional,
            tol=tol,
        )

    def sample(
        self,
        observable: "SingleParticleObservable",
        shots: int = 1,
        rng: Optional[np.random.Generator] = None,
        sector: Optional[Partition] = None,
        multiplicity: Optional[MultiplicityLabel] = None,
        copy: Optional[MultiplicityLabel] = None,
        conditional: bool = False,
    ):
        """Sample measurement outcomes for this state."""
        return observable.sample(
            self.state,
            shots=shots,
            rng=rng,
            sector=sector,
            multiplicity=multiplicity,
            copy=copy,
            conditional=conditional,
        )


class PhotonicState:
    """State object independent of matrix representation choice.

    Parameters
    ----------
    system:
        Parent :class:`~photonic_jordan.system.photonic_system.PhotonicSystem`.
    data:
        Tensor-basis operator matrix representing the state.
    label:
        Optional display label for reports/notebooks.
    _cache:
        Internal cache for alternate representations, keyed by representation name.

    Notes
    -----
    The state itself is representation-independent. Use :meth:`density_matrix`
    to obtain matrices in ``tensor`` or ``schur`` representation.
    Invariant and measurement helpers are exposed as ``state.invariant`` and
    ``state.measure``.
    """

    def __init__(
        self,
        system: "PhotonicSystem",
        data: np.ndarray,
        label: Optional[str] = None,
        _cache: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.system = system
        self.label = label
        self.invariant = StateInvariantView(self)
        self.measure = StateMeasurementView(self)

        tensor = np.asarray(data, dtype=complex)
        if _cache is None:
            self._cache: Dict[str, np.ndarray] = {"tensor": tensor}
        else:
            self._cache = {k: np.asarray(v, dtype=complex) for k, v in _cache.items()}
            self._cache["tensor"] = tensor

    def _tensor_matrix(self) -> np.ndarray:
        return self._cache["tensor"]

    def _schur_matrix(self) -> np.ndarray:
        if "schur" not in self._cache:
            self._cache["schur"] = self.system.decomposition.to_schur_operator(self._tensor_matrix())
        return self._cache["schur"]

    def density_matrix(self, rep: str = "tensor", copy: bool = True) -> np.ndarray:
        """Return density matrix in requested representation.

        Parameters
        ----------
        rep:
            ``tensor`` or ``schur``.
        copy:
            If ``True``, return a defensive copy.

        Returns
        -------
        np.ndarray
            Matrix representation of the same state.
        """
        rep_key = rep.strip().lower()
        if rep_key == "tensor":
            out = self._tensor_matrix()
        elif rep_key == "schur":
            out = self._schur_matrix()
        else:
            raise ValueError("rep must be one of {'tensor', 'schur'} in the current implementation.")
        return out.copy() if copy else out

    def copy(self) -> "PhotonicState":
        """Return deep-copied state and representation cache."""
        cloned_cache = {k: v.copy() for k, v in self._cache.items()}
        return PhotonicState(system=self.system, data=self._tensor_matrix().copy(), label=self.label, _cache=cloned_cache)

    def evolve(self, S: ArrayLike) -> "PhotonicState":
        """Evolve state by single-particle unitary ``S`` lifted to ``S^{\\otimes n}``."""
        S = self.system.unitary.from_matrix(S)
        rho_out_tensor = self.system.dynamics.evolve_density(self._tensor_matrix(), S)
        label = None if self.label is None else f"{self.label} -> evolved"
        return PhotonicState(system=self.system, data=rho_out_tensor, label=label, _cache={"tensor": rho_out_tensor})

    def evolve_haar(self, seed: Optional[int] = None) -> "PhotonicState":
        """Evolve with Haar-random single-particle unitary."""
        return self.evolve(self.system.unitary.haar(seed=seed))

    def project_jordan(
        self,
        order: int,
        kind: str = "exact",
        sector: Optional[Partition] = None,
        multiplicity: Optional[MultiplicityLabel] = None,
        copy: Optional[MultiplicityLabel] = None,
    ) -> "PhotonicState":
        """Project state onto Jordan exact/cumulative subspace.

        Parameters
        ----------
        order:
            Hierarchy order.
        kind:
            ``exact`` for layer projector, ``cumulative`` for cumulative projector.
        sector:
            Optional Schur sector scope.
        multiplicity:
            Optional multiplicity scope ``(lambda, a)``.
        copy:
            Deprecated alias for ``multiplicity``.
        """
        filtration = self.system.ensure_scope_filtration(
            max_order=order,
            sector=sector,
            multiplicity=multiplicity,
            copy=copy,
        )

        rho_t = self.system.project_density_to_scope(
            self._tensor_matrix(),
            sector=sector,
            multiplicity=multiplicity,
            copy=copy,
        )

        kind = kind.lower()
        if kind in {"exact", "layer", "delta"}:
            data = filtration.apply_projector_layer(rho_t, order)
            tag = f"jordan_exact_{order}"
        elif kind in {"cumulative", "cum", "leq", "<="}:
            data = filtration.apply_projector_cumulative(rho_t, order)
            tag = f"jordan_cumulative_{order}"
        else:
            raise ValueError("kind must be 'exact' or 'cumulative'.")

        if sector is not None:
            tag = f"{tag}_sector_{tuple(sector)}"
        elif multiplicity is not None or copy is not None:
            active = multiplicity if multiplicity is not None else copy
            tag = f"{tag}_multiplicity_{tuple(active[0])}_{active[1]}"

        label = None if self.label is None else f"{self.label} | {tag}"
        return PhotonicState(system=self.system, data=data, label=label, _cache={"tensor": data})

    def project_sector(self, lam: Partition) -> "PhotonicState":
        """Project state as ``Q_lambda rho Q_lambda``."""
        Q = self.system.sector_projector(lam)
        data = safe_matmul(Q, self._tensor_matrix(), Q)
        label = None if self.label is None else f"{self.label} | sector {lam}"
        return PhotonicState(system=self.system, data=data, label=label, _cache={"tensor": data})

    def project_multiplicity(self, lam: Partition, a: int) -> "PhotonicState":
        """Project to multiplicity-local block ``Q_{lambda,a}``.

        The basis inside multiplicity spaces is not mathematically unique.
        This package uses a deterministic convention (stable seeded commutant
        diagonalization), so labels ``a=0,1,...`` are reproducible across runs
        for fixed ``(m_ext, n_particles, lambda)``.
        """
        Qa = self.system.multiplicity_projector(lam, a)
        data = safe_matmul(Qa, self._tensor_matrix(), Qa)
        label = None if self.label is None else f"{self.label} | multiplicity ({lam}, {a})"
        return PhotonicState(system=self.system, data=data, label=label, _cache={"tensor": data})

    # Backward-compatible alias.
    def project_copy(self, lam: Partition, a: int) -> "PhotonicState":
        """Alias of :meth:`project_multiplicity` for backward compatibility."""
        return self.project_multiplicity(lam, a)

    def blocks(self) -> Dict[Partition, "PhotonicState"]:
        """Return all Schur sector blocks as dictionary ``lambda -> state``."""
        return {lam: self.project_sector(lam) for lam in self.system.available_partitions()}

    def block(self, lam: Partition) -> "PhotonicState":
        """Alias of :meth:`project_sector`."""
        return self.project_sector(lam)

    def multiplicity_block(self, lam: Partition, a: int) -> "PhotonicState":
        """Alias of :meth:`project_multiplicity`."""
        return self.project_multiplicity(lam, a)

    # Backward-compatible alias.
    def copy_block(self, lam: Partition, a: int) -> "PhotonicState":
        """Alias of :meth:`multiplicity_block` for backward compatibility."""
        return self.multiplicity_block(lam, a)

    def sector_weights(self) -> Optional[Dict[Partition, float]]:
        """Return sector population weights ``Tr(Q_lambda rho)``."""
        if self.system.projectors is None:
            return None
        rho_t = self._tensor_matrix()
        weights = {}
        for lam in self.system.available_partitions():
            Q = self.system.sector_projector(lam)
            weights[lam] = float(np.real(np.trace(safe_matmul(Q, rho_t))))
        return weights

    def analyze(
        self,
        max_order: Optional[int] = None,
        include_sectors: bool = True,
        sector: Optional[Partition] = None,
        multiplicity: Optional[MultiplicityLabel] = None,
        copy: Optional[MultiplicityLabel] = None,
    ) -> InvariantReport:
        """Compute invariant report for current state."""
        return self.invariant.report(
            max_order=max_order,
            include_sectors=include_sectors,
            sector=sector,
            multiplicity=multiplicity,
            copy=copy,
        )

    def trace(self) -> complex:
        """Return trace in tensor representation."""
        return np.trace(self._tensor_matrix())

    def purity(self) -> float:
        """Return purity ``Tr(rho^2)`` in tensor representation."""
        rho_t = self._tensor_matrix()
        return float(np.real(np.trace(rho_t @ rho_t)))

    def is_physical(self, tol: float = 1e-8) -> bool:
        """Check Hermiticity, unit trace, and positivity within tolerance."""
        rho_t = self._tensor_matrix()
        hermitian = np.allclose(rho_t, rho_t.conj().T, atol=tol)
        unit_trace = np.allclose(np.trace(rho_t), 1.0, atol=tol)
        evals = np.linalg.eigvalsh((rho_t + rho_t.conj().T) / 2.0)
        positive = np.min(evals) >= -tol
        return bool(hermitian and unit_trace and positive)

    @property
    def matrix(self) -> np.ndarray:
        """Return tensor-basis matrix (same physical state)."""
        return self._tensor_matrix()

    def __array__(self):
        return self._tensor_matrix()

    def __repr__(self) -> str:
        rho_t = self._tensor_matrix()
        tr = np.trace(rho_t)
        return (
            f"PhotonicState(shape={rho_t.shape}, "
            f"trace={tr.real:.6f}{tr.imag:+.2e}j, label={self.label!r})"
        )


def resolve_gram_input(gram: GramInput, n_particles: int) -> np.ndarray:
    """Resolve shorthand Gram-matrix specifications to explicit arrays.

    Parameters
    ----------
    gram:
        String shortcut, scalar overlap, or explicit matrix.
    n_particles:
        Number of particles (matrix dimension).

    Returns
    -------
    np.ndarray
        Hermitian matrix candidate to be validated by state constructors.
    """
    if isinstance(gram, str):
        key = gram.strip().lower()
        if key in {"indistinguishable", "identical", "fully_indistinguishable"}:
            return np.ones((n_particles, n_particles), dtype=complex)
        if key in {"distinguishable", "orthogonal", "fully_distinguishable"}:
            return np.eye(n_particles, dtype=complex)
        raise ValueError("Unknown gram string. Use 'indistinguishable' or 'distinguishable'.")

    if np.isscalar(gram):
        x = complex(gram)
        G = np.full((n_particles, n_particles), x, dtype=complex)
        np.fill_diagonal(G, 1.0)
        return G

    G = np.asarray(gram, dtype=complex)
    if G.shape != (n_particles, n_particles):
        raise ValueError(f"Gram matrix must have shape {(n_particles, n_particles)}.")
    return G


def gram_description(gram: GramInput) -> str:
    """Human-readable short description for notebook labels."""
    if isinstance(gram, str):
        return gram
    if np.isscalar(gram):
        return f"pairwise-overlap={complex(gram)}"
    return "custom-matrix"


def as_normalized_density(rho: ArrayLike) -> np.ndarray:
    """Convert user input into normalized density matrix array."""
    return normalize_density(np.asarray(rho, dtype=complex))
