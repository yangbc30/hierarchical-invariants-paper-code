"""High-level photonic state object and invariant views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

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

    def _resolve_scope(
        self,
        sector: Optional[Partition],
        multiplicity: Optional[MultiplicityLabel],
        copy: Optional[MultiplicityLabel],
    ) -> Tuple[Optional[Partition], Optional[MultiplicityLabel]]:
        active = self.state.system._resolve_multiplicity_arg(multiplicity, copy)
        self.state.system._validate_scope_inputs(sector=sector, multiplicity=active)
        return sector, active

    def _use_fock_global(
        self,
        sector: Optional[Partition],
        multiplicity: Optional[MultiplicityLabel],
    ) -> bool:
        return sector is None and multiplicity is None and self.state.has_rep("fock")

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
        sector, multiplicity = self._resolve_scope(sector=sector, multiplicity=multiplicity, copy=copy)
        if self._use_fock_global(sector=sector, multiplicity=multiplicity):
            filtration = self.state.system.ensure_fock_filtration(max_order=order)
            rho_f = self.state.density_matrix(rep="fock", copy=False)
            return filtration.layer_weight(rho_f, order)

        filtration = self.state.system.ensure_scope_filtration(
            max_order=order,
            sector=sector,
            multiplicity=multiplicity,
        )
        return filtration.layer_weight(
            self._density_in_scope(sector=sector, multiplicity=multiplicity, copy=None),
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
        sector, multiplicity = self._resolve_scope(sector=sector, multiplicity=multiplicity, copy=copy)
        if self._use_fock_global(sector=sector, multiplicity=multiplicity):
            filtration = self.state.system.ensure_fock_filtration(max_order=order)
            rho_f = self.state.density_matrix(rep="fock", copy=False)
            return filtration.cumulative_weight(rho_f, order)

        filtration = self.state.system.ensure_scope_filtration(
            max_order=order,
            sector=sector,
            multiplicity=multiplicity,
        )
        return filtration.cumulative_weight(
            self._density_in_scope(sector=sector, multiplicity=multiplicity, copy=None),
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

        sector, multiplicity = self._resolve_scope(sector=sector, multiplicity=multiplicity, copy=copy)
        if self._use_fock_global(sector=sector, multiplicity=multiplicity):
            filtration = self.state.system.ensure_fock_filtration(max_order=max_order)
            rho_f = self.state.density_matrix(rep="fock", copy=False)
            cumulative = {j: filtration.cumulative_weight(rho_f, j) for j in range(max_order + 1)}
            exact = {j: filtration.layer_weight(rho_f, j) for j in range(max_order + 1)}
            sectors = self.state.sector_weights() if include_sectors else None
            return InvariantReport(cumulative=cumulative, exact=exact, sector_weights=sectors)

        filtration = self.state.system.ensure_scope_filtration(
            max_order=max_order,
            sector=sector,
            multiplicity=multiplicity,
        )
        rho_t = self._density_in_scope(sector=sector, multiplicity=multiplicity, copy=None)

        cumulative = {j: filtration.cumulative_weight(rho_t, j) for j in range(max_order + 1)}
        exact = {j: filtration.layer_weight(rho_t, j) for j in range(max_order + 1)}

        sectors = (
            self.state.sector_weights()
            if include_sectors and sector is None and multiplicity is None
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
        May be omitted when ``_cache`` already provides a matrix representation.
    label:
        Optional display label for reports/notebooks.
    _cache:
        Internal cache for alternate representations, keyed by representation name.

    Notes
    -----
    The state itself is representation-independent. Use :meth:`density_matrix`
    to obtain matrices in ``tensor``, ``fock`` (bosons), or ``schur`` representation.
    Invariant and measurement helpers are exposed as ``state.invariant`` and
    ``state.measure``.
    """

    def __init__(
        self,
        system: "PhotonicSystem",
        data: Optional[np.ndarray] = None,
        label: Optional[str] = None,
        _cache: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.system = system
        self.label = label
        self.invariant = StateInvariantView(self)
        self.measure = StateMeasurementView(self)

        if data is None and _cache is None:
            raise ValueError("Provide either data=... or a non-empty _cache with state matrix data.")

        self._cache: Dict[str, np.ndarray] = {}
        if _cache is not None:
            self._cache.update({k: np.asarray(v, dtype=complex) for k, v in _cache.items()})
        if data is not None:
            self._cache["tensor"] = np.asarray(data, dtype=complex)

        if "tensor" in self._cache:
            expected = (self.system.hilbert_dim, self.system.hilbert_dim)
            if self._cache["tensor"].shape != expected:
                raise ValueError(f"Tensor density matrix must have shape {expected}.")

        if "fock" in self._cache:
            if self.system.fock_space is None:
                raise ValueError("Fock cache is only valid for bosonic systems.")
            shape = (self.system.fock_space.dim, self.system.fock_space.dim)
            if self._cache["fock"].shape != shape:
                raise ValueError(f"Fock density matrix must have shape {shape}.")

        if len(self._cache) == 0:
            raise ValueError("State cache is empty after initialization.")

    @classmethod
    def from_modes_and_gram(
        cls,
        ext_modes: Sequence[int],
        gram: GramInput = "indistinguishable",
        *,
        m_ext: Optional[int] = None,
        particle_type: str = "boson",
        rng: Optional[np.random.Generator] = None,
        label: Optional[str] = None,
        system: Optional["PhotonicSystem"] = None,
        auto_cache: Optional[bool] = None,
        cache_dir: Optional[str] = None,
    ) -> "PhotonicState":
        """State-first constructor from external modes and internal Gram matrix.

        Parameters
        ----------
        ext_modes:
            External mode assignment for each labeled particle.
        gram:
            Internal overlap model. Accepts shortcut strings, scalar overlap, or
            explicit Gram matrix.
        m_ext:
            Number of external modes. If omitted and ``system`` is not provided,
            inferred as ``max(ext_modes)+1``.
        particle_type:
            Particle-type label passed to :class:`PhotonicSystem` when ``system``
            is not provided.
        rng:
            Optional RNG passed to :class:`PhotonicSystem` when ``system`` is not
            provided.
        label:
            Optional state label.
        system:
            Existing system context. If provided, no new system is created.
        auto_cache:
            Optional cache switch used only when creating a new system.
        cache_dir:
            Optional cache directory used only when creating a new system.
        """
        modes = [int(x) for x in ext_modes]
        if len(modes) == 0:
            raise ValueError("ext_modes must be non-empty.")
        if any(x < 0 for x in modes):
            raise ValueError("ext_modes must contain nonnegative mode indices.")
        n_particles = len(modes)

        if system is None:
            resolved_m = (max(modes) + 1) if m_ext is None else int(m_ext)
            if resolved_m <= 0:
                raise ValueError("m_ext must be positive.")
            if max(modes) >= resolved_m:
                raise ValueError("ext_modes contains an index >= m_ext.")
            from ..system import PhotonicSystem

            system = PhotonicSystem(
                m_ext=resolved_m,
                n_particles=n_particles,
                particle_type=particle_type,
                rng=rng,
                auto_cache=auto_cache,
                cache_dir=cache_dir,
            )
        else:
            if system.spec.n_particles != n_particles:
                raise ValueError(
                    f"Provided system has n_particles={system.spec.n_particles}, "
                    f"but ext_modes implies n_particles={n_particles}."
                )
            if max(modes) >= system.spec.m_ext:
                raise ValueError("ext_modes contains an index outside provided system.m_ext.")

        return system.state.from_modes_and_gram(modes, gram=gram, label=label)

    @classmethod
    def from_occupation(
        cls,
        occupation: Sequence[int],
        *,
        label: Optional[str] = None,
        system: Optional["PhotonicSystem"] = None,
        particle_type: str = "boson",
        rng: Optional[np.random.Generator] = None,
        auto_cache: Optional[bool] = None,
        cache_dir: Optional[str] = None,
    ) -> "PhotonicState":
        """State-first constructor from bosonic occupation numbers.

        Parameters
        ----------
        occupation:
            Occupation tuple/list ``(n_0, ..., n_{m-1})``.
        label:
            Optional state label.
        system:
            Existing system context. If provided, must match ``m_ext`` and
            ``n_particles`` implied by ``occupation``.
        particle_type:
            Particle-type label used when creating a new system.
        rng:
            Optional RNG used when creating a new system.
        auto_cache:
            Optional cache switch used only when creating a new system.
        cache_dir:
            Optional cache directory used only when creating a new system.
        """
        occ = [int(x) for x in occupation]
        if len(occ) == 0:
            raise ValueError("occupation must be non-empty.")
        if any(x < 0 for x in occ):
            raise ValueError("occupation entries must be nonnegative integers.")

        if sum(occ) == 0:
            raise ValueError("Total particle number must be positive.")

        if system is None:
            from ..system import PhotonicSystem

            system = PhotonicSystem(
                m_ext=len(occ),
                n_particles=int(sum(occ)),
                particle_type=particle_type,
                rng=rng,
                auto_cache=auto_cache,
                cache_dir=cache_dir,
            )
        else:
            if system.spec.m_ext != len(occ):
                raise ValueError(
                    f"Provided system has m_ext={system.spec.m_ext}, "
                    f"but occupation implies m_ext={len(occ)}."
                )
            if system.spec.n_particles != int(sum(occ)):
                raise ValueError(
                    f"Provided system has n_particles={system.spec.n_particles}, "
                    f"but occupation implies n_particles={sum(occ)}."
                )

        return system.state.from_fock(occupation=occ, label=label)

    @classmethod
    def Fock(
        cls,
        *occupation: int,
        label: Optional[str] = None,
        system: Optional["PhotonicSystem"] = None,
        particle_type: str = "boson",
        rng: Optional[np.random.Generator] = None,
        auto_cache: Optional[bool] = None,
        cache_dir: Optional[str] = None,
    ) -> "PhotonicState":
        """State-first constructor for bosonic Fock occupation numbers.

        Parameters
        ----------
        occupation:
            Variadic occupation numbers ``(n_0, n_1, ..., n_{m-1})``.
        label:
            Optional state label.
        system:
            Existing system context. If provided, must match ``m_ext`` and
            ``n_particles`` implied by ``occupation``.
        particle_type:
            Particle-type label used when creating a new system.
        rng:
            Optional RNG used when creating a new system.
        auto_cache:
            Optional cache switch used only when creating a new system.
        cache_dir:
            Optional cache directory used only when creating a new system.
        """
        occ = occupation
        if len(occ) == 1 and isinstance(occ[0], (list, tuple, np.ndarray)):
            occ = tuple(int(x) for x in occ[0])  # backward-compatible fallback
        return cls.from_occupation(
            occupation=occ,
            label=label,
            system=system,
            particle_type=particle_type,
            rng=rng,
            auto_cache=auto_cache,
            cache_dir=cache_dir,
        )

    @classmethod
    def FockMixed(
        cls,
        *terms,
        label: Optional[str] = None,
        system: Optional["PhotonicSystem"] = None,
        m_ext: Optional[int] = None,
        particle_type: str = "boson",
        rng: Optional[np.random.Generator] = None,
        auto_cache: Optional[bool] = None,
        cache_dir: Optional[str] = None,
        normalize: bool = True,
    ) -> "PhotonicState":
        """State-first constructor for classical mixtures of Fock occupations.

        Parameters
        ----------
        terms:
            Mixture terms, each written as either ``(p, n0, n1, ..., n_{m-1})``
            or ``(p, [n0, n1, ..., n_{m-1}])``.
        label:
            Optional state label.
        system:
            Existing system context. If provided, occupations must be compatible.
        m_ext:
            Number of external modes. Used only when ``system`` is not provided.
            If omitted, inferred from the longest occupation term.
        particle_type:
            Particle-type label used when creating a new system.
        rng:
            Optional RNG used when creating a new system.
        auto_cache:
            Optional cache switch used only when creating a new system.
        cache_dir:
            Optional cache directory used only when creating a new system.
        normalize:
            If ``True`` (default), normalize mixture weights to unit trace.
        """
        if len(terms) == 0:
            raise ValueError("FockMixed requires at least one mixture term.")

        parsed_weights: List[float] = []
        parsed_occupations: List[Tuple[int, ...]] = []
        for term in terms:
            if not isinstance(term, (tuple, list)) or len(term) < 2:
                raise ValueError(
                    "Each FockMixed term must be (weight, n0, n1, ...) or (weight, occupation_sequence)."
                )
            weight = float(term[0])
            if len(term) == 2 and isinstance(term[1], (list, tuple, np.ndarray)):
                occ = tuple(int(x) for x in term[1])
            else:
                occ = tuple(int(x) for x in term[1:])
            if len(occ) == 0:
                raise ValueError("Occupation term must contain at least one mode.")
            if any(x < 0 for x in occ):
                raise ValueError("Occupation entries must be nonnegative.")
            parsed_weights.append(weight)
            parsed_occupations.append(occ)

        if system is None:
            resolved_m = max(len(occ) for occ in parsed_occupations) if m_ext is None else int(m_ext)
            if resolved_m <= 0:
                raise ValueError("m_ext must be positive.")
            padded = [occ + (0,) * (resolved_m - len(occ)) for occ in parsed_occupations]
            if any(len(occ) > resolved_m for occ in parsed_occupations):
                raise ValueError("Provided m_ext is smaller than at least one occupation term.")
            n_particles = sum(padded[0])
            if any(sum(occ) != n_particles for occ in padded):
                raise ValueError("All Fock mixture terms must have the same total particle number.")

            from ..system import PhotonicSystem

            system = PhotonicSystem(
                m_ext=resolved_m,
                n_particles=n_particles,
                particle_type=particle_type,
                rng=rng,
                auto_cache=auto_cache,
                cache_dir=cache_dir,
            )
            occupations = padded
        else:
            resolved_m = system.spec.m_ext
            n_particles = system.spec.n_particles
            occupations = []
            for occ in parsed_occupations:
                if len(occ) > resolved_m:
                    raise ValueError("Occupation term length exceeds provided system.m_ext.")
                occ_pad = occ + (0,) * (resolved_m - len(occ))
                if sum(occ_pad) != n_particles:
                    raise ValueError(
                        f"Occupation total {sum(occ_pad)} does not match provided system n_particles={n_particles}."
                    )
                occupations.append(occ_pad)

        return system.state.from_fock_mixture(
            occupations=occupations,
            weights=parsed_weights,
            label=label,
            normalize=normalize,
        )

    @classmethod
    def from_fock_density(
        cls,
        rho_fock: ArrayLike,
        *,
        m_ext: Optional[int] = None,
        n_particles: Optional[int] = None,
        particle_type: str = "boson",
        rng: Optional[np.random.Generator] = None,
        label: Optional[str] = None,
        system: Optional["PhotonicSystem"] = None,
        auto_cache: Optional[bool] = None,
        cache_dir: Optional[str] = None,
    ) -> "PhotonicState":
        """State-first constructor from a Fock-basis density matrix."""
        arr = np.asarray(rho_fock, dtype=complex)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("rho_fock must be a square matrix.")

        if system is None:
            if m_ext is None or n_particles is None:
                raise ValueError(
                    "When system is not provided, from_fock_density requires both m_ext and n_particles."
                )
            from ..system import PhotonicSystem

            system = PhotonicSystem(
                m_ext=int(m_ext),
                n_particles=int(n_particles),
                particle_type=particle_type,
                rng=rng,
                auto_cache=auto_cache,
                cache_dir=cache_dir,
            )

        return system.state.from_fock_density(arr, label=label)

    def _tensor_matrix(self) -> np.ndarray:
        if "tensor" not in self._cache:
            if "fock" not in self._cache:
                raise RuntimeError("No tensor or fock matrix available in state cache.")
            self._cache["tensor"] = self.system.fock_to_tensor_operator(self._cache["fock"])
        return self._cache["tensor"]

    def _fock_matrix(self) -> np.ndarray:
        if self.system.fock_space is None:
            raise NotImplementedError("Fock representation is only available for bosonic systems.")
        if "fock" not in self._cache:
            if "tensor" not in self._cache:
                raise RuntimeError("No tensor or fock matrix available in state cache.")
            self._cache["fock"] = self.system.tensor_to_fock_operator(self._cache["tensor"])
        return self._cache["fock"]

    def _schur_matrix(self) -> np.ndarray:
        if "schur" not in self._cache:
            self._cache["schur"] = self.system.decomposition.to_schur_operator(self._tensor_matrix())
            self.system._mark_schur_cache_dirty()
            self.system._auto_save_schur_cache_if_needed()
        return self._cache["schur"]

    def has_rep(self, rep: str) -> bool:
        """Return whether representation ``rep`` is already cached."""
        return rep.strip().lower() in self._cache

    def density_matrix(self, rep: str = "tensor", copy: bool = True) -> np.ndarray:
        """Return density matrix in requested representation.

        Parameters
        ----------
        rep:
            ``tensor``, ``fock`` (bosons), or ``schur``.
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
        elif rep_key == "fock":
            out = self._fock_matrix()
        elif rep_key == "schur":
            out = self._schur_matrix()
        else:
            raise ValueError("rep must be one of {'tensor', 'fock', 'schur'} in the current implementation.")
        return out.copy() if copy else out

    def pattern_probability(self, occupation: Sequence[int]) -> float:
        """Return output-detection probability for occupation pattern ``(n_0,...,n_{m-1})``.

        Parameters
        ----------
        occupation:
            Output pattern over external modes.

        Returns
        -------
        float
            Probability ``Tr(rho Pi_occ)`` where ``Pi_occ`` projects onto the
            subspace of labeled external basis states with the given occupation.

        Notes
        -----
        This models number-resolving detection that resolves external modes but
        not particle labels or internal states.
        """
        if self.has_rep("fock"):
            fs = self.system.fock_space
            if fs is None:
                raise RuntimeError("State has fock cache but system has no fock backend.")
            idx = fs.index_from_occupation(occupation)
            rho_f = self._fock_matrix()
            return float(np.real(rho_f[idx, idx]))

        indices = self.system.space.indices_for_occupation(occupation)
        rho_t = self._tensor_matrix()
        diag = np.diagonal(rho_t)
        return float(np.real(np.sum(diag[indices])))

    def pattern_distribution(self) -> Dict[Tuple[int, ...], float]:
        """Return all occupation-pattern probabilities as a dictionary."""
        if self.has_rep("fock"):
            fs = self.system.fock_space
            if fs is None:
                raise RuntimeError("State has fock cache but system has no fock backend.")
            rho_f = self._fock_matrix()
            diag = np.real(np.diagonal(rho_f))
            return {
                occ: float(diag[idx])
                for idx, occ in enumerate(fs.basis_states)
            }

        rho_t = self._tensor_matrix()
        diag = np.real(np.diagonal(rho_t))
        return {
            occ: float(np.sum(diag[indices]))
            for occ, indices in self.system.space.occupation_to_indices().items()
        }

    def copy(self) -> "PhotonicState":
        """Return deep-copied state and representation cache."""
        cloned_cache = {k: v.copy() for k, v in self._cache.items()}
        return PhotonicState(system=self.system, data=None, label=self.label, _cache=cloned_cache)

    @staticmethod
    def _spec_signature(state: "PhotonicState") -> Tuple[int, int, str]:
        spec = state.system.spec
        return (int(spec.m_ext), int(spec.n_particles), str(spec.particle_type))

    def _assert_compatible(self, other: "PhotonicState") -> None:
        if not isinstance(other, PhotonicState):
            raise TypeError("Expected `other` to be a PhotonicState.")
        lhs = PhotonicState._spec_signature(self)
        rhs = PhotonicState._spec_signature(other)
        if lhs != rhs:
            raise ValueError(
                "State mismatch: both states must share (m_ext, n_particles, particle_type). "
                f"Got {lhs} vs {rhs}."
            )

    @staticmethod
    def _pure_ket_from_density(rho: np.ndarray, tol: float = 1e-8) -> np.ndarray:
        rho_h = 0.5 * (rho + rho.conj().T)
        tr = complex(np.trace(rho_h))
        if abs(tr) <= tol:
            raise ValueError("Cannot extract a ket from a near-zero-trace operator.")
        rho_n = rho_h / tr
        evals, evecs = np.linalg.eigh(rho_n)
        idx = int(np.argmax(np.real(evals)))
        lam_max = float(np.real(evals[idx]))
        purity = float(np.real(np.trace(rho_n @ rho_n)))
        if abs(lam_max - 1.0) > 10.0 * tol or abs(purity - 1.0) > 10.0 * tol:
            raise ValueError(
                "Coherent superposition requires pure input states. "
                "At least one input is mixed within tolerance."
            )
        ket = evecs[:, idx]
        norm = np.linalg.norm(ket)
        if norm <= tol:
            raise ValueError("Extracted ket has near-zero norm.")
        ket = ket / norm

        # Deterministic gauge: first significant component is made real-positive.
        nz = np.flatnonzero(np.abs(ket) > tol)
        if nz.size > 0:
            ket = ket * np.exp(-1j * np.angle(ket[int(nz[0])]))
        return ket

    @classmethod
    def mixture(
        cls,
        *terms,
        normalize: bool = True,
        label: Optional[str] = None,
        system: Optional["PhotonicSystem"] = None,
    ) -> "PhotonicState":
        """Build classical mixture from arbitrary states.

        Parameters
        ----------
        terms:
            Mixture terms as ``(weight, state)``.
        normalize:
            If ``True`` (default), normalize to unit trace.
        label:
            Optional output label.
        system:
            Optional target system context. If omitted, uses the first state's
            system. All states must be spec-compatible with this system.
        """
        if len(terms) == 0:
            raise ValueError("mixture requires at least one term.")

        parsed: List[Tuple[float, PhotonicState]] = []
        for term in terms:
            if not isinstance(term, (tuple, list)) or len(term) != 2:
                raise ValueError("Each mixture term must be a pair (weight, state).")
            w = float(term[0])
            s = term[1]
            if not isinstance(s, PhotonicState):
                raise TypeError("Each mixture term must provide a PhotonicState as second entry.")
            if w < 0:
                raise ValueError("Mixture weights must be nonnegative.")
            parsed.append((w, s))

        if system is None:
            system = parsed[0][1].system

        sig_target = (
            int(system.spec.m_ext),
            int(system.spec.n_particles),
            str(system.spec.particle_type),
        )
        for _, s in parsed:
            sig = PhotonicState._spec_signature(s)
            if sig != sig_target:
                raise ValueError(
                    "State/system mismatch in mixture: expected "
                    f"{sig_target}, got {sig}."
                )

        total_w = float(sum(w for w, _ in parsed))
        if total_w <= 0:
            raise ValueError("At least one mixture weight must be strictly positive.")

        use_fock = system.fock_space is not None and all(s.has_rep("fock") for _, s in parsed)
        if use_fock:
            assert system.fock_space is not None
            rho = np.zeros((system.fock_space.dim, system.fock_space.dim), dtype=complex)
            for w, s in parsed:
                rho += w * s.density_matrix(rep="fock", copy=False)
            rho = 0.5 * (rho + rho.conj().T)
            if normalize:
                rho = normalize_density(rho)
            return PhotonicState(system=system, data=None, label=label, _cache={"fock": rho})

        rho_t = np.zeros((system.hilbert_dim, system.hilbert_dim), dtype=complex)
        for w, s in parsed:
            rho_t += w * s.density_matrix(rep="tensor", copy=False)
        rho_t = 0.5 * (rho_t + rho_t.conj().T)
        if normalize:
            rho_t = normalize_density(rho_t)
        return PhotonicState(system=system, data=rho_t, label=label, _cache={"tensor": rho_t})

    def mix(
        self,
        other: "PhotonicState",
        weight: float = 0.5,
        normalize: bool = True,
        label: Optional[str] = None,
    ) -> "PhotonicState":
        """Return classical mixture ``weight*self + (1-weight)*other``."""
        self._assert_compatible(other)
        w = float(weight)
        if w < 0 or w > 1:
            raise ValueError("weight must be in [0, 1].")
        return PhotonicState.mixture(
            (w, self),
            (1.0 - w, other),
            normalize=normalize,
            label=label,
            system=self.system,
        )

    def superpose(
        self,
        other: "PhotonicState",
        alpha: complex = 1.0,
        beta: complex = 1.0,
        normalize: bool = True,
        label: Optional[str] = None,
        tol: float = 1e-8,
    ) -> "PhotonicState":
        """Return coherent superposition of two pure states.

        Notes
        -----
        This operation is only defined here for pure-state inputs. For mixed
        states, use :meth:`mix` / :meth:`mixture` instead.
        """
        self._assert_compatible(other)
        use_fock = self.has_rep("fock") and other.has_rep("fock")
        rep = "fock" if use_fock else "tensor"

        ket_a = PhotonicState._pure_ket_from_density(self.density_matrix(rep=rep, copy=False), tol=tol)
        ket_b = PhotonicState._pure_ket_from_density(other.density_matrix(rep=rep, copy=False), tol=tol)
        psi = complex(alpha) * ket_a + complex(beta) * ket_b
        norm_sq = float(np.real(np.vdot(psi, psi)))
        if norm_sq <= tol:
            raise ValueError("Superposition coefficients produce a near-zero state vector.")
        if normalize:
            psi = psi / np.sqrt(norm_sq)
        rho = np.outer(psi, psi.conj())

        if label is None:
            la = self.label or "state_a"
            lb = other.label or "state_b"
            label = f"superpose({la}, {lb})"

        if use_fock:
            return PhotonicState(system=self.system, data=None, label=label, _cache={"fock": rho})
        return PhotonicState(system=self.system, data=rho, label=label, _cache={"tensor": rho})

    def evolve(self, S: ArrayLike) -> "PhotonicState":
        """Evolve state by single-particle unitary ``S``.

        Notes
        -----
        Uses native bosonic Fock evolution when Fock representation is cached,
        otherwise uses tensor-space evolution with ``S^{\\otimes n}``.
        """
        S = self.system.unitary.from_matrix(S)
        label = None if self.label is None else f"{self.label} -> evolved"

        if self.has_rep("fock"):
            if self.system.fock_space is None:
                raise RuntimeError("State has fock cache but system has no fock backend.")
            rho_out_fock = self.system.fock_space.evolve_density(self._fock_matrix(), S)
            return PhotonicState(system=self.system, data=None, label=label, _cache={"fock": rho_out_fock})

        rho_out_tensor = self.system.dynamics.evolve_density(self._tensor_matrix(), S)
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
        multiplicity = self.system._resolve_multiplicity_arg(multiplicity, copy)
        self.system._validate_scope_inputs(sector=sector, multiplicity=multiplicity)

        use_fock = sector is None and multiplicity is None and self.has_rep("fock")
        if use_fock:
            filtration = self.system.ensure_fock_filtration(max_order=order)
            rho_in = self._fock_matrix()
        else:
            filtration = self.system.ensure_scope_filtration(
                max_order=order,
                sector=sector,
                multiplicity=multiplicity,
            )
            rho_in = self.system.project_density_to_scope(
                self._tensor_matrix(),
                sector=sector,
                multiplicity=multiplicity,
            )

        kind = kind.lower()
        if kind in {"exact", "layer", "delta"}:
            data = filtration.apply_projector_layer(rho_in, order)
            tag = f"jordan_exact_{order}"
        elif kind in {"cumulative", "cum", "leq", "<="}:
            data = filtration.apply_projector_cumulative(rho_in, order)
            tag = f"jordan_cumulative_{order}"
        else:
            raise ValueError("kind must be 'exact' or 'cumulative'.")

        if sector is not None:
            tag = f"{tag}_sector_{tuple(sector)}"
        elif multiplicity is not None or copy is not None:
            tag = f"{tag}_multiplicity_{tuple(multiplicity[0])}_{multiplicity[1]}"

        label = None if self.label is None else f"{self.label} | {tag}"
        if use_fock:
            return PhotonicState(system=self.system, data=None, label=label, _cache={"fock": data})
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
        if "tensor" in self._cache:
            return np.trace(self._cache["tensor"])
        if "fock" in self._cache:
            return np.trace(self._cache["fock"])
        return np.trace(self._tensor_matrix())

    def purity(self) -> float:
        """Return purity ``Tr(rho^2)`` in tensor representation."""
        if "tensor" in self._cache:
            rho = self._cache["tensor"]
        elif "fock" in self._cache:
            rho = self._cache["fock"]
        else:
            rho = self._tensor_matrix()
        return float(np.real(np.trace(rho @ rho)))

    def is_physical(self, tol: float = 1e-8) -> bool:
        """Check Hermiticity, unit trace, and positivity within tolerance."""
        if "tensor" in self._cache:
            rho = self._cache["tensor"]
        elif "fock" in self._cache:
            rho = self._cache["fock"]
        else:
            rho = self._tensor_matrix()
        hermitian = np.allclose(rho, rho.conj().T, atol=tol)
        unit_trace = np.allclose(np.trace(rho), 1.0, atol=tol)
        evals = np.linalg.eigvalsh((rho + rho.conj().T) / 2.0)
        positive = np.min(evals) >= -tol
        return bool(hermitian and unit_trace and positive)

    @property
    def matrix(self) -> np.ndarray:
        """Return tensor-basis matrix (same physical state)."""
        return self._tensor_matrix()

    def __array__(self):
        return self._tensor_matrix()

    def __repr__(self) -> str:
        if "tensor" in self._cache:
            rep = "tensor"
            rho = self._cache["tensor"]
        elif "fock" in self._cache:
            rep = "fock"
            rho = self._cache["fock"]
        else:
            rep = "tensor"
            rho = self._tensor_matrix()
        tr = np.trace(rho)
        return (
            f"PhotonicState(rep={rep!r}, shape={rho.shape}, "
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


def from_modes_and_gram(
    ext_modes: Sequence[int],
    gram: GramInput = "indistinguishable",
    *,
    m_ext: Optional[int] = None,
    particle_type: str = "boson",
    rng: Optional[np.random.Generator] = None,
    label: Optional[str] = None,
    system: Optional["PhotonicSystem"] = None,
    auto_cache: Optional[bool] = None,
    cache_dir: Optional[str] = None,
) -> PhotonicState:
    """State-first constructor shortcut.

    This is a top-level functional alias of
    :meth:`PhotonicState.from_modes_and_gram`.
    """
    return PhotonicState.from_modes_and_gram(
        ext_modes=ext_modes,
        gram=gram,
        m_ext=m_ext,
        particle_type=particle_type,
        rng=rng,
        label=label,
        system=system,
        auto_cache=auto_cache,
        cache_dir=cache_dir,
    )


def from_occupation(
    occupation: Sequence[int],
    *,
    label: Optional[str] = None,
    system: Optional["PhotonicSystem"] = None,
    particle_type: str = "boson",
    rng: Optional[np.random.Generator] = None,
    auto_cache: Optional[bool] = None,
    cache_dir: Optional[str] = None,
) -> PhotonicState:
    """State-first Fock occupation shortcut.

    This is a top-level functional alias of
    :meth:`PhotonicState.from_occupation`.
    """
    return PhotonicState.from_occupation(
        occupation=occupation,
        label=label,
        system=system,
        particle_type=particle_type,
        rng=rng,
        auto_cache=auto_cache,
        cache_dir=cache_dir,
    )


def Fock(
    *occupation: int,
    label: Optional[str] = None,
    system: Optional["PhotonicSystem"] = None,
    particle_type: str = "boson",
    rng: Optional[np.random.Generator] = None,
    auto_cache: Optional[bool] = None,
    cache_dir: Optional[str] = None,
) -> PhotonicState:
    """Build a bosonic Fock state directly as ``rho = Fock(n0, n1, ...)``."""
    return PhotonicState.Fock(
        *occupation,
        label=label,
        system=system,
        particle_type=particle_type,
        rng=rng,
        auto_cache=auto_cache,
        cache_dir=cache_dir,
    )


def FockMixed(
    *terms,
    label: Optional[str] = None,
    system: Optional["PhotonicSystem"] = None,
    m_ext: Optional[int] = None,
    particle_type: str = "boson",
    rng: Optional[np.random.Generator] = None,
    auto_cache: Optional[bool] = None,
    cache_dir: Optional[str] = None,
    normalize: bool = True,
) -> PhotonicState:
    """Build a classical mixture of Fock states as a top-level shortcut."""
    return PhotonicState.FockMixed(
        *terms,
        label=label,
        system=system,
        m_ext=m_ext,
        particle_type=particle_type,
        rng=rng,
        auto_cache=auto_cache,
        cache_dir=cache_dir,
        normalize=normalize,
    )


def from_fock_density(
    rho_fock: ArrayLike,
    *,
    m_ext: Optional[int] = None,
    n_particles: Optional[int] = None,
    particle_type: str = "boson",
    rng: Optional[np.random.Generator] = None,
    label: Optional[str] = None,
    system: Optional["PhotonicSystem"] = None,
    auto_cache: Optional[bool] = None,
    cache_dir: Optional[str] = None,
) -> PhotonicState:
    """Wrap a Fock-basis density matrix as a :class:`PhotonicState`."""
    return PhotonicState.from_fock_density(
        rho_fock=rho_fock,
        m_ext=m_ext,
        n_particles=n_particles,
        particle_type=particle_type,
        rng=rng,
        label=label,
        system=system,
        auto_cache=auto_cache,
        cache_dir=cache_dir,
    )


def mix_states(
    *terms,
    normalize: bool = True,
    label: Optional[str] = None,
    system: Optional["PhotonicSystem"] = None,
) -> PhotonicState:
    """Functional alias of :meth:`PhotonicState.mixture`."""
    return PhotonicState.mixture(
        *terms,
        normalize=normalize,
        label=label,
        system=system,
    )


def superpose(
    state_a: PhotonicState,
    state_b: PhotonicState,
    alpha: complex = 1.0,
    beta: complex = 1.0,
    normalize: bool = True,
    label: Optional[str] = None,
    tol: float = 1e-8,
) -> PhotonicState:
    """Functional alias of :meth:`PhotonicState.superpose`."""
    return state_a.superpose(
        state_b,
        alpha=alpha,
        beta=beta,
        normalize=normalize,
        label=label,
        tol=tol,
    )
