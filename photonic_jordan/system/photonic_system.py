"""Top-level orchestration object for the photonic Jordan framework."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import warnings

import numpy as np

from ..dynamics import PassiveLODynamics
from ..hierarchy import InvariantEngine, JordanFiltration
from ..math import normalize_density, safe_matmul
from ..measurement import ObservableFactory
from ..schur import SchurWeylDecomposition
from ..spaces import BosonicFockSpace, LabeledTensorSpace, SymmetricGroupProjectors
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

    _GLOBAL_AUTO_CACHE: bool = True
    _GLOBAL_CACHE_DIR: Path = (
        Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "linear_optics_toolkit"
    )

    def __init__(
        self,
        m_ext: int,
        n_particles: int,
        particle_type: str = "boson",
        rng: Optional[np.random.Generator] = None,
        auto_cache: Optional[bool] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.spec = ModelSpec(m_ext=m_ext, n_particles=n_particles, particle_type=particle_type)
        self.space = LabeledTensorSpace(self.spec)
        self._fock_space: Optional[BosonicFockSpace] = (
            BosonicFockSpace(self.spec, tensor_space=self.space)
            if self.spec.particle_type == "boson"
            else None
        )
        self.rng = np.random.default_rng() if rng is None else rng

        self._state_factory = StateFactory(self.space, rng=self.rng)
        self._dynamics = PassiveLODynamics(self.space)
        self._filtration: Optional[JordanFiltration] = None
        self._invariants: Optional[InvariantEngine] = None
        self._built_order = -1
        self._fock_filtration: Optional[JordanFiltration] = None
        self._fock_built_order = -1
        self.auto_cache = self._GLOBAL_AUTO_CACHE if auto_cache is None else bool(auto_cache)
        if cache_dir is None:
            self.cache_dir = self._GLOBAL_CACHE_DIR
        else:
            self.cache_dir = Path(cache_dir).expanduser()
        self._fock_cache_dirty = False
        self._cache_io_in_progress = False

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

        if self.auto_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._try_auto_load_fock_cache()

    @property
    def hilbert_dim(self) -> int:
        """Return external Hilbert dimension ``m_ext ** n_particles``."""
        return self.spec.hilbert_dim

    @property
    def dynamics(self) -> PassiveLODynamics:
        """Return passive linear-optics dynamics helper."""
        return self._dynamics

    @property
    def fock_space(self) -> Optional[BosonicFockSpace]:
        """Return bosonic fixed-``n`` Fock backend when available."""
        return self._fock_space

    @classmethod
    def configure_global_cache(
        cls,
        enabled: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Configure default cache behavior for future ``PhotonicSystem`` instances."""
        cls._GLOBAL_AUTO_CACHE = bool(enabled)
        if cache_dir is not None:
            cls._GLOBAL_CACHE_DIR = Path(cache_dir).expanduser()

    def default_fock_cache_path(self) -> Path:
        """Return canonical Fock cache path for current model."""
        filename = (
            f"fock_cache_v1_{self.spec.particle_type}_"
            f"m{self.spec.m_ext}_n{self.spec.n_particles}.npz"
        )
        return self.cache_dir / filename

    @property
    def invariants_engine(self) -> InvariantEngine:
        """Return invariant diagnostic helper bound to global filtration."""
        if self._invariants is None:
            self._filtration = JordanFiltration(self.space)
            self._invariants = InvariantEngine(self._filtration, self._dynamics)
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
        if self._filtration is None:
            self._filtration = JordanFiltration(self.space)
            self._invariants = InvariantEngine(self._filtration, self._dynamics)
        if max_order > self._built_order:
            self._filtration.build(max_order=max_order)
            self._built_order = max_order

    def _mark_fock_cache_dirty(self) -> None:
        if self.auto_cache:
            self._fock_cache_dirty = True

    def _try_auto_load_fock_cache(self) -> None:
        if not self.auto_cache or self._fock_space is None:
            return
        path = self.default_fock_cache_path()
        if not path.exists():
            return
        try:
            self._cache_io_in_progress = True
            self.load_fock_cache(path, load_generators=True, load_hierarchy=True)
            self._fock_cache_dirty = False
        except Exception as exc:
            warnings.warn(
                f"Failed to load fock cache from '{path}': {exc}. Will rebuild cache lazily.",
                RuntimeWarning,
                stacklevel=2,
            )
        finally:
            self._cache_io_in_progress = False

    def _auto_save_fock_cache_if_needed(self, force: bool = False) -> None:
        if (
            not self.auto_cache
            or self._fock_space is None
            or self._cache_io_in_progress
            or (not force and not self._fock_cache_dirty)
        ):
            return
        path = self.default_fock_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._cache_io_in_progress = True
            self.save_fock_cache(path, max_order=None, include_generators=True)
            self._fock_cache_dirty = False
        finally:
            self._cache_io_in_progress = False

    def fock_generators(self) -> Dict[Tuple[int, int], np.ndarray]:
        """Return Fock one-body generators and mark auto-cache dirty if newly built."""
        if self._fock_space is None:
            raise NotImplementedError("Fock backend is only available for particle_type='boson'.")
        was_empty = self._fock_space._generators_cache is None
        gens = self._fock_space.generators
        if was_empty:
            self._mark_fock_cache_dirty()
            self._auto_save_fock_cache_if_needed()
        return gens

    def ensure_fock_filtration(self, max_order: Optional[int] = None) -> JordanFiltration:
        """Build bosonic symmetric-space filtration up to ``max_order``.

        Notes
        -----
        This backend is only available when ``particle_type='boson'``.
        """
        if self._fock_space is None:
            raise NotImplementedError("Fock backend is only available for particle_type='boson'.")
        if max_order is None:
            max_order = self.spec.n_particles

        gens = self.fock_generators()
        if self._fock_filtration is None:
            self._fock_filtration = JordanFiltration(
                self._fock_space,
                generator_list=list(gens.values()),
            )
        if max_order > self._fock_built_order:
            self._fock_filtration.build(max_order=max_order)
            self._fock_built_order = max_order
            self._mark_fock_cache_dirty()
            self._auto_save_fock_cache_if_needed()
        return self._fock_filtration

    def tensor_to_fock_operator(self, operator: np.ndarray) -> np.ndarray:
        """Project tensor-basis operator into symmetric Fock basis."""
        if self._fock_space is None:
            raise NotImplementedError("Fock backend is only available for particle_type='boson'.")
        op = np.asarray(operator, dtype=complex)
        if op.shape != (self.hilbert_dim, self.hilbert_dim):
            raise ValueError(f"Tensor operator must have shape {(self.hilbert_dim, self.hilbert_dim)}.")
        V = self._fock_space.isometry_to_tensor
        return safe_matmul(V.conj().T, op, V)

    def fock_to_tensor_operator(self, operator: np.ndarray) -> np.ndarray:
        """Embed Fock-basis operator into tensor basis using symmetric isometry."""
        if self._fock_space is None:
            raise NotImplementedError("Fock backend is only available for particle_type='boson'.")
        op = np.asarray(operator, dtype=complex)
        shape = (self._fock_space.dim, self._fock_space.dim)
        if op.shape != shape:
            raise ValueError(f"Fock operator must have shape {shape}.")
        V = self._fock_space.isometry_to_tensor
        return safe_matmul(V, op, V.conj().T)

    def total_unitary_fock_from_single_particle(self, S: np.ndarray) -> np.ndarray:
        """Return Fock-basis lifted unitary for single-particle ``S``."""
        if self._fock_space is None:
            raise NotImplementedError("Fock backend is only available for particle_type='boson'.")
        S = self.unitary.from_matrix(S)
        return self._fock_space.total_unitary_from_single_particle(S)

    def evolve_density_fock(self, rho_fock: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Evolve Fock-basis density with native bosonic lifted unitary."""
        if self._fock_space is None:
            raise NotImplementedError("Fock backend is only available for particle_type='boson'.")
        S = self.unitary.from_matrix(S)
        return self._fock_space.evolve_density(rho_fock, S)

    def save_fock_cache(
        self,
        path: Union[str, Path],
        max_order: Optional[int] = None,
        include_generators: bool = True,
    ) -> None:
        """Persist Fock generators and/or built Fock hierarchy to disk.

        Parameters
        ----------
        path:
            Output ``.npz`` file path.
        max_order:
            If provided, ensure Fock hierarchy is built up to this order before saving.
        include_generators:
            If ``True``, include cached Fock generators in the checkpoint.
        """
        if self._fock_space is None:
            raise NotImplementedError("Fock backend is only available for particle_type='boson'.")
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        old_io_flag = self._cache_io_in_progress
        self._cache_io_in_progress = True
        try:
            if max_order is not None:
                self.ensure_fock_filtration(max_order=max_order)

            meta = {
                "format": "photonic_jordan_fock_cache",
                "version": 1,
                "m_ext": int(self.spec.m_ext),
                "n_particles": int(self.spec.n_particles),
                "particle_type": self.spec.particle_type,
            }
            arrays: Dict[str, np.ndarray] = {"meta": np.array(json.dumps(meta))}

            if include_generators:
                gens = self.fock_generators()
                for (s, t), G in gens.items():
                    arrays[f"gen_{s}_{t}"] = G
                arrays["has_generators"] = np.array(1, dtype=np.int64)
            else:
                arrays["has_generators"] = np.array(0, dtype=np.int64)

            if self._fock_filtration is None or self._fock_built_order < 0:
                arrays["fock_built_order"] = np.array(-1, dtype=np.int64)
            else:
                arrays["fock_built_order"] = np.array(self._fock_built_order, dtype=np.int64)
                for j in range(self._fock_built_order + 1):
                    arrays[f"cum_{j}"] = self._fock_filtration.cumulative_bases[j]
                    arrays[f"lay_{j}"] = self._fock_filtration.layer_bases[j]

            np.savez_compressed(path, **arrays)
        finally:
            self._cache_io_in_progress = old_io_flag
        self._fock_cache_dirty = False

    def load_fock_cache(
        self,
        path: Union[str, Path],
        load_generators: bool = True,
        load_hierarchy: bool = True,
    ) -> None:
        """Load persisted Fock generators/hierarchy from ``save_fock_cache``.

        Parameters
        ----------
        path:
            Input ``.npz`` checkpoint path.
        load_generators:
            If ``True``, restore Fock generator cache from checkpoint.
        load_hierarchy:
            If ``True``, restore built Fock hierarchy bases from checkpoint.
        """
        if self._fock_space is None:
            raise NotImplementedError("Fock backend is only available for particle_type='boson'.")
        path = Path(path).expanduser()

        with np.load(path, allow_pickle=False) as data:
            if "meta" not in data:
                raise ValueError("Invalid fock cache file: missing metadata.")
            meta = json.loads(str(data["meta"].item()))
            if meta.get("format") != "photonic_jordan_fock_cache":
                raise ValueError("Invalid fock cache format marker.")
            if int(meta.get("m_ext", -1)) != self.spec.m_ext or int(meta.get("n_particles", -1)) != self.spec.n_particles:
                raise ValueError(
                    "Cache model mismatch: "
                    f"file has (m_ext={meta.get('m_ext')}, n_particles={meta.get('n_particles')}), "
                    f"system has (m_ext={self.spec.m_ext}, n_particles={self.spec.n_particles})."
                )

            if load_generators and int(data.get("has_generators", np.array(0))) == 1:
                gens: Dict[Tuple[int, int], np.ndarray] = {}
                for s in range(self.spec.m_ext):
                    for t in range(self.spec.m_ext):
                        key = f"gen_{s}_{t}"
                        if key not in data:
                            raise ValueError(f"Invalid fock cache: missing generator '{key}'.")
                        gens[(s, t)] = np.asarray(data[key], dtype=complex)
                self._fock_space._generators_cache = gens

            if load_hierarchy and "fock_built_order" in data:
                built_order = int(data["fock_built_order"])
                if built_order >= 0:
                    filtration = JordanFiltration(self._fock_space)
                    filtration.cumulative_bases = {}
                    filtration.layer_bases = {}
                    for j in range(built_order + 1):
                        key_c = f"cum_{j}"
                        key_l = f"lay_{j}"
                        if key_c not in data or key_l not in data:
                            raise ValueError(f"Invalid fock cache: missing hierarchy basis for order {j}.")
                        filtration.cumulative_bases[j] = np.asarray(data[key_c], dtype=complex)
                        filtration.layer_bases[j] = np.asarray(data[key_l], dtype=complex)
                    filtration._clear_projector_cache()
                    self._fock_filtration = filtration
                    self._fock_built_order = built_order
        self._fock_cache_dirty = False

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
            if self._filtration is None:
                raise RuntimeError("Global filtration was not initialized.")
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
