from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg as la

from .core import (
    InvariantEngine,
    JordanFiltration,
    LabeledTensorSpace,
    ModelSpec,
    PassiveLODynamics,
    StateFactory,
    SymmetricGroupProjectors,
    haar_random_unitary,
    normalize_density,
    safe_matmul,
)
from .decomposition import SchurWeylDecomposition


ArrayLike = Union[np.ndarray, Sequence[Sequence[complex]]]
GramInput = Union[str, float, complex, np.ndarray, Sequence[Sequence[complex]]]
Partition = Tuple[int, ...]
MultiplicityLabel = Tuple[Partition, int]


@dataclass
class InvariantReport:
    cumulative: Dict[int, float]
    exact: Dict[int, float]
    sector_weights: Optional[Dict[Partition, float]] = None

    def summary(self, digits: int = 8) -> str:
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


class PhotonicSystem:
    """High-level user-facing entry point.

    This keeps the v2 mental model while introducing the v3 Phase-1 internal
    decomposition object for Schur/sector/multiplicity representations.
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

    @property
    def hilbert_dim(self) -> int:
        return self.spec.hilbert_dim

    @property
    def dynamics(self) -> PassiveLODynamics:
        return self._dynamics

    @property
    def invariants_engine(self) -> InvariantEngine:
        return self._invariants

    @property
    def projectors(self) -> Optional[SymmetricGroupProjectors]:
        return self._projectors

    @property
    def decomposition(self) -> SchurWeylDecomposition:
        if self._projectors is None:
            raise NotImplementedError("Schur decomposition is currently demo-supported for n=2,3 only.")
        if self._decomposition is None:
            self._decomposition = SchurWeylDecomposition(self.space, self._projectors)
        return self._decomposition

    def ensure_filtration(self, max_order: Optional[int] = None) -> None:
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

    def available_partitions(self) -> List[Partition]:
        if self._projectors is None:
            return []
        return self.decomposition.partitions()

    def sector_projector(self, lam: Partition) -> np.ndarray:
        if self._projectors is None:
            raise NotImplementedError("Sector projectors are only implemented for n=2,3 in this demo.")
        return self.decomposition.sector_projector(lam)

    def multiplicity_projector(self, lam: Partition, a: int) -> np.ndarray:
        if self._projectors is None:
            raise NotImplementedError("Multiplicity projectors are only implemented for n=2,3 in this demo.")
        return self.decomposition.multiplicity_projector(lam, a)

    # Backward-compatible alias.
    def copy_projector(self, lam: Partition, a: int) -> np.ndarray:
        return self.multiplicity_projector(lam, a)

    def density_state(self, rho: ArrayLike, rep: str = "tensor", label: Optional[str] = None) -> "PhotonicState":
        arr = np.asarray(rho, dtype=complex)
        if arr.shape != (self.hilbert_dim, self.hilbert_dim):
            raise ValueError(f"Density matrix must have shape {(self.hilbert_dim, self.hilbert_dim)}.")
        return PhotonicState(system=self, data=normalize_density(arr), rep=rep, label=label)


class StateBuilder:
    def __init__(self, system: PhotonicSystem):
        self.system = system

    def from_modes_and_gram(
        self,
        ext_modes: Sequence[int],
        gram: GramInput = "indistinguishable",
        label: Optional[str] = None,
    ) -> "PhotonicState":
        G = resolve_gram_input(gram=gram, n_particles=self.system.spec.n_particles)
        rho = self.system._state_factory.from_external_modes_and_gram(ext_modes=ext_modes, gram=G)
        if label is None:
            label = f"modes={list(ext_modes)}, gram={gram_description(gram)}"
        return PhotonicState(system=self.system, data=rho, rep="tensor", label=label)

    def from_density_matrix(
        self,
        rho: ArrayLike,
        rep: str = "tensor",
        label: Optional[str] = None,
    ) -> "PhotonicState":
        return self.system.density_state(rho, rep=rep, label=label)

    def random_sector(self, lam: Partition, label: Optional[str] = None) -> "PhotonicState":
        Q = self.system.sector_projector(lam)
        rho = self.system._state_factory.random_density_in_sector(Q)
        if label is None:
            label = f"random sector state lambda={lam}"
        return PhotonicState(system=self.system, data=rho, rep="tensor", label=label)

    def random_sector_state(self, lam: Partition, label: Optional[str] = None) -> "PhotonicState":
        return self.random_sector(lam=lam, label=label)

    def random_density(self, label: Optional[str] = None) -> "PhotonicState":
        dim = self.system.hilbert_dim
        x = self.system.rng.normal(size=(dim, dim)) + 1j * self.system.rng.normal(size=(dim, dim))
        rho = safe_matmul(x, x.conj().T)
        rho = normalize_density(rho)
        if label is None:
            label = "random full density state"
        return PhotonicState(system=self.system, data=rho, rep="tensor", label=label)

    def random_commutant_state(self, label: Optional[str] = None) -> "PhotonicState":
        """Sample a permutation-commuting external state by permutation twirling."""
        dim = self.system.hilbert_dim
        x = self.system.rng.normal(size=(dim, dim)) + 1j * self.system.rng.normal(size=(dim, dim))
        rho = normalize_density(safe_matmul(x, x.conj().T))

        if self.system.projectors is None:
            raise NotImplementedError("Commutant twirl helper currently supports n=2,3 via demo projectors.")

        twirled = np.zeros_like(rho)
        perms = list(permutations(range(self.system.spec.n_particles)))
        for perm in perms:
            P = self.system.projectors.permutation_matrix(perm)
            twirled += safe_matmul(P, rho, P.conj().T)
        twirled /= float(len(perms))
        twirled = normalize_density(twirled)

        if label is None:
            label = "random commutant state"
        return PhotonicState(system=self.system, data=twirled, rep="tensor", label=label)


class UnitaryFactory:
    def __init__(self, system: PhotonicSystem):
        self.system = system

    def haar(self, seed: Optional[int] = None) -> np.ndarray:
        rng = self.system.rng if seed is None else np.random.default_rng(seed)
        return haar_random_unitary(dim=self.system.spec.m_ext, rng=rng)

    def from_matrix(self, S: ArrayLike) -> np.ndarray:
        S = np.asarray(S, dtype=complex)
        shape = (self.system.spec.m_ext, self.system.spec.m_ext)
        if S.shape != shape:
            raise ValueError(f"Single-particle unitary must have shape {shape}.")
        if not np.allclose(S.conj().T @ S, np.eye(shape[0]), atol=1e-8):
            raise ValueError("Input matrix is not unitary.")
        return S


class StateInvariantView:
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
        if max_order is None:
            max_order = self.state.system.spec.n_particles
        filtration = self.state.system.ensure_scope_filtration(
            max_order=max_order,
            sector=sector,
            multiplicity=multiplicity,
            copy=copy,
        )
        rho_t = self._density_in_scope(sector=sector, multiplicity=multiplicity, copy=copy)
        cumulative = {
            j: filtration.cumulative_weight(rho_t, j)
            for j in range(max_order + 1)
        }
        exact = {
            j: filtration.layer_weight(rho_t, j)
            for j in range(max_order + 1)
        }
        sectors = self.state.sector_weights() if include_sectors and sector is None and multiplicity is None and copy is None else None
        return InvariantReport(cumulative=cumulative, exact=exact, sector_weights=sectors)


class PhotonicState:
    def __init__(
        self,
        system: PhotonicSystem,
        data: np.ndarray,
        rep: str = "tensor",
        label: Optional[str] = None,
        _cache: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.system = system
        self.data = np.asarray(data, dtype=complex)
        self.rep = rep
        self.label = label
        self.invariant = StateInvariantView(self)

        self._cache: Dict[str, np.ndarray] = {} if _cache is None else dict(_cache)
        rep_key = rep.lower()
        if rep_key == "schur":
            self._cache["schur"] = self.data
        else:
            self._cache["tensor"] = self.data

    def _tensor_matrix(self) -> np.ndarray:
        if "tensor" in self._cache:
            return self._cache["tensor"]
        if "schur" in self._cache:
            tensor = self.system.decomposition.to_tensor_operator(self._cache["schur"])
            self._cache["tensor"] = tensor
            return tensor
        raise RuntimeError("No tensor representation available in state cache.")

    def _schur_matrix(self) -> np.ndarray:
        if "schur" in self._cache:
            return self._cache["schur"]
        tensor = self._tensor_matrix()
        schur = self.system.decomposition.to_schur_operator(tensor)
        self._cache["schur"] = schur
        return schur

    def copy(self) -> "PhotonicState":
        cloned_cache = {k: v.copy() for k, v in self._cache.items()}
        return PhotonicState(system=self.system, data=self.data.copy(), rep=self.rep, label=self.label, _cache=cloned_cache)

    def to(self, rep: str) -> "PhotonicState":
        rep_key = rep.strip().lower()
        if rep_key == "tensor":
            tensor = self._tensor_matrix()
            cache = dict(self._cache)
            cache["tensor"] = tensor
            return PhotonicState(system=self.system, data=tensor, rep="tensor", label=self.label, _cache=cache)
        if rep_key == "schur":
            schur = self._schur_matrix()
            cache = dict(self._cache)
            cache["schur"] = schur
            return PhotonicState(system=self.system, data=schur, rep="schur", label=self.label, _cache=cache)
        raise ValueError("rep must be one of {'tensor', 'schur'} in the current implementation.")

    def evolve(self, S: ArrayLike) -> "PhotonicState":
        S = self.system.unitary.from_matrix(S)
        rho_out_tensor = self.system.dynamics.evolve_density(self._tensor_matrix(), S)
        label = None if self.label is None else f"{self.label} -> evolved"

        if self.rep.lower() == "schur":
            rho_out_schur = self.system.decomposition.to_schur_operator(rho_out_tensor)
            cache = {"tensor": rho_out_tensor, "schur": rho_out_schur}
            return PhotonicState(system=self.system, data=rho_out_schur, rep="schur", label=label, _cache=cache)

        cache = dict(self._cache)
        cache["tensor"] = rho_out_tensor
        return PhotonicState(system=self.system, data=rho_out_tensor, rep=self.rep, label=label, _cache=cache)

    def evolve_haar(self, seed: Optional[int] = None) -> "PhotonicState":
        return self.evolve(self.system.unitary.haar(seed=seed))

    def project_jordan(
        self,
        order: int,
        kind: str = "exact",
        sector: Optional[Partition] = None,
        multiplicity: Optional[MultiplicityLabel] = None,
        copy: Optional[MultiplicityLabel] = None,
    ) -> "PhotonicState":
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
            rep = f"jordan_exact_{order}"
        elif kind in {"cumulative", "cum", "leq", "<="}:
            data = filtration.apply_projector_cumulative(rho_t, order)
            rep = f"jordan_cumulative_{order}"
        else:
            raise ValueError("kind must be 'exact' or 'cumulative'.")

        if sector is not None:
            rep = f"{rep}_sector_{tuple(sector)}"
        elif multiplicity is not None or copy is not None:
            active = multiplicity if multiplicity is not None else copy
            rep = f"{rep}_multiplicity_{tuple(active[0])}_{active[1]}"

        label = None if self.label is None else f"{self.label} | {rep}"
        cache = dict(self._cache)
        cache["tensor"] = data
        return PhotonicState(system=self.system, data=data, rep=rep, label=label, _cache=cache)

    def project_sector(self, lam: Partition) -> "PhotonicState":
        Q = self.system.sector_projector(lam)
        data = safe_matmul(Q, self._tensor_matrix(), Q)
        label = None if self.label is None else f"{self.label} | sector {lam}"
        return PhotonicState(system=self.system, data=data, rep=f"sector_{lam}", label=label, _cache={"tensor": data})

    def project_multiplicity(self, lam: Partition, a: int) -> "PhotonicState":
        """Project to multiplicity-local block Q_{lam,a}; basis-dependent by convention."""
        Qa = self.system.multiplicity_projector(lam, a)
        data = safe_matmul(Qa, self._tensor_matrix(), Qa)
        label = None if self.label is None else f"{self.label} | multiplicity ({lam}, {a})"
        return PhotonicState(
            system=self.system,
            data=data,
            rep=f"multiplicity_{lam}_{a}",
            label=label,
            _cache={"tensor": data},
        )

    # Backward-compatible alias.
    def project_copy(self, lam: Partition, a: int) -> "PhotonicState":
        return self.project_multiplicity(lam, a)

    def blocks(self) -> Dict[Partition, "PhotonicState"]:
        return {lam: self.project_sector(lam) for lam in self.system.available_partitions()}

    def block(self, lam: Partition) -> "PhotonicState":
        return self.project_sector(lam)

    def multiplicity_block(self, lam: Partition, a: int) -> "PhotonicState":
        return self.project_multiplicity(lam, a)

    # Backward-compatible alias.
    def copy_block(self, lam: Partition, a: int) -> "PhotonicState":
        return self.multiplicity_block(lam, a)

    def sector_weights(self) -> Optional[Dict[Partition, float]]:
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
        return self.invariant.report(
            max_order=max_order,
            include_sectors=include_sectors,
            sector=sector,
            multiplicity=multiplicity,
            copy=copy,
        )

    def trace(self) -> complex:
        return np.trace(self._tensor_matrix())

    def purity(self) -> float:
        rho_t = self._tensor_matrix()
        return float(np.real(np.trace(rho_t @ rho_t)))

    def is_physical(self, tol: float = 1e-8) -> bool:
        rho_t = self._tensor_matrix()
        hermitian = np.allclose(rho_t, rho_t.conj().T, atol=tol)
        unit_trace = np.allclose(np.trace(rho_t), 1.0, atol=tol)
        evals = np.linalg.eigvalsh((rho_t + rho_t.conj().T) / 2.0)
        positive = np.min(evals) >= -tol
        return bool(hermitian and unit_trace and positive)

    @property
    def matrix(self) -> np.ndarray:
        return self.data

    def to_tensor(self) -> "PhotonicState":
        return self.to("tensor")

    def __array__(self):
        return self.data

    def __repr__(self) -> str:
        tr = np.trace(self._tensor_matrix())
        return (
            f"PhotonicState(rep={self.rep!r}, shape={self.data.shape}, "
            f"trace={tr.real:.6f}{tr.imag:+.2e}j, label={self.label!r})"
        )


def resolve_gram_input(gram: GramInput, n_particles: int) -> np.ndarray:
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
    if isinstance(gram, str):
        return gram
    if np.isscalar(gram):
        return f"pairwise-overlap={complex(gram)}"
    return "custom-matrix"
