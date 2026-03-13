from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from math import factorial
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la


# =========================
# Core specifications
# =========================


@dataclass(frozen=True)
class ModelSpec:
    """Minimal model metadata for the first demo.

    This demo focuses on the external first-quantized space (C^m)^{⊗ n}.
    Internal modes are only used in StateFactory.from_external_modes_and_gram.
    """

    m_ext: int
    n_particles: int
    particle_type: str = "boson"

    @property
    def hilbert_dim(self) -> int:
        return self.m_ext ** self.n_particles


# =========================
# Basic linear algebra utils
# =========================


TOL = 1e-10


def safe_matmul(*ops: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = ops[0]
        for op in ops[1:]:
            out = out @ op
    return out


def orth_columns(mat: np.ndarray, tol: float = TOL) -> np.ndarray:
    """Return an orthonormal basis of the column span of `mat`.

    Use an SVD-based construction through scipy.linalg.orth. This is much more
    robust than a naive non-pivoted QR truncation for the rank-deficient complex
    matrices that appear in Jordan-space builds.
    """
    if mat.size == 0:
        return np.zeros((mat.shape[0], 0), dtype=complex)
    u, s, _ = la.svd(mat, full_matrices=False, check_finite=False)
    if s.size == 0:
        return np.zeros((mat.shape[0], 0), dtype=complex)
    cutoff = max(tol, float(s[0]) * tol)
    keep = s > cutoff
    if not np.any(keep):
        return np.zeros((mat.shape[0], 0), dtype=complex)
    return u[:, keep]



def hs_inner(a: np.ndarray, b: np.ndarray) -> complex:
    return np.trace(safe_matmul(a.conj().T, b))



def vectorize(a: np.ndarray) -> np.ndarray:
    return a.reshape(-1, 1)



def devectorize(v: np.ndarray, dim: int) -> np.ndarray:
    return v.reshape(dim, dim)



def projector_from_basis(basis: np.ndarray) -> np.ndarray:
    if basis.size == 0:
        return np.zeros((basis.shape[0], basis.shape[0]), dtype=complex)
    return safe_matmul(basis, basis.conj().T)



def normalize_density(rho: np.ndarray) -> np.ndarray:
    rho = (rho + rho.conj().T) / 2.0
    tr = np.trace(rho)
    if abs(tr) < TOL:
        raise ValueError("Density matrix has zero trace after projection.")
    return rho / tr



def random_complex_matrix(shape: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)



def haar_random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    z = random_complex_matrix((dim, dim), rng)
    q, r = la.qr(z)
    phases = np.diag(r) / np.abs(np.diag(r))
    phases[np.abs(np.diag(r)) < TOL] = 1.0
    return safe_matmul(q, np.diag(np.conj(phases)))


# =========================
# First-quantized labeled tensor space
# =========================


class LabeledTensorSpace:
    """First-quantized external space (C^m)^{⊗ n}."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.m = spec.m_ext
        self.n = spec.n_particles
        self.dim = spec.hilbert_dim
        self.basis_states = list(product(range(self.m), repeat=self.n))
        self.basis_array = np.asarray(self.basis_states, dtype=int)
        self.index_weights = (self.m ** np.arange(self.n - 1, -1, -1)).astype(int)
        self.state_to_index = {state: idx for idx, state in enumerate(self.basis_states)}
        self.single_ops = self._build_single_particle_ops()
        self.generators = self._build_generators()

    def _build_single_particle_ops(self) -> Dict[Tuple[int, int], np.ndarray]:
        ops: Dict[Tuple[int, int], np.ndarray] = {}
        for s in range(self.m):
            for t in range(self.m):
                op = np.zeros((self.m, self.m), dtype=complex)
                op[s, t] = 1.0
                ops[(s, t)] = op
        return ops

    def lift_one_body_operator(self, h: np.ndarray, slot: int) -> np.ndarray:
        ops = [np.eye(self.m, dtype=complex) for _ in range(self.n)]
        ops[slot] = h
        out = ops[0]
        for k in range(1, self.n):
            out = np.kron(out, ops[k])
        return out

    def total_one_body_operator(self, h: np.ndarray) -> np.ndarray:
        out = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(self.n):
            out += self.lift_one_body_operator(h, i)
        return out

    def _build_generators(self) -> Dict[Tuple[int, int], np.ndarray]:
        gens: Dict[Tuple[int, int], np.ndarray] = {}
        for s in range(self.m):
            for t in range(self.m):
                gens[(s, t)] = self.total_one_body_operator(self.single_ops[(s, t)])
        return gens

    def total_unitary_from_single_particle(self, S: np.ndarray) -> np.ndarray:
        out = S
        for _ in range(self.n - 1):
            out = np.kron(out, S)
        return out


# =========================
# Symmetric-group sector projectors (demo support: n=2,3)
# =========================


class SymmetricGroupProjectors:
    """Isotypic projectors Q_lambda from character sums.

    Demo support is implemented for n=2 and n=3, which is enough to validate the
    block structure numerically before the full Schur-transform machinery is added.
    """

    CHARACTER_TABLE = {
        2: {
            (2,): {"dim": 1, "chars": {"e": 1, "t": 1}},
            (1, 1): {"dim": 1, "chars": {"e": 1, "t": -1}},
        },
        3: {
            (3,): {"dim": 1, "chars": {"e": 1, "t": 1, "c": 1}},
            (2, 1): {"dim": 2, "chars": {"e": 2, "t": 0, "c": -1}},
            (1, 1, 1): {"dim": 1, "chars": {"e": 1, "t": -1, "c": 1}},
        },
    }

    def __init__(self, space: LabeledTensorSpace):
        self.space = space
        self.n = space.n
        if self.n not in self.CHARACTER_TABLE:
            raise NotImplementedError("Demo currently supports n=2 or n=3 for sector projectors.")
        self._permutations = list(permutations(range(self.n)))
        self._perm_matrices: Dict[Tuple[int, ...], np.ndarray] = {}

    @staticmethod
    def parity(perm: Sequence[int]) -> int:
        inv = 0
        p = list(perm)
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                if p[i] > p[j]:
                    inv += 1
        return -1 if inv % 2 else 1

    def conjugacy_type(self, perm: Sequence[int]) -> str:
        p = tuple(perm)
        if self.n == 2:
            return "e" if p == (0, 1) else "t"
        # n == 3
        if p == (0, 1, 2):
            return "e"
        # transpositions are odd, 3-cycles are even non-identity
        return "t" if self.parity(p) == -1 else "c"

    def permutation_matrix(self, perm: Sequence[int]) -> np.ndarray:
        key = tuple(perm)
        if key in self._perm_matrices:
            return self._perm_matrices[key]

        dim = self.space.dim
        permuted = self.space.basis_array[:, list(key)]
        out_idx = permuted @ self.space.index_weights
        in_idx = np.arange(dim)
        P = np.zeros((dim, dim), dtype=complex)
        P[out_idx, in_idx] = 1.0
        self._perm_matrices[key] = P
        return P

    def isotypic_projector(self, partition: Tuple[int, ...]) -> np.ndarray:
        table = self.CHARACTER_TABLE[self.n][partition]
        d_lambda = table["dim"]
        acc = np.zeros((self.space.dim, self.space.dim), dtype=complex)
        for perm in self._permutations:
            cls = self.conjugacy_type(perm)
            chi = table["chars"][cls]
            acc += chi * self.permutation_matrix(perm)
        Q = d_lambda / factorial(self.n) * acc
        Q = (Q + Q.conj().T) / 2.0
        # Numerical clean-up into an exact orthogonal projector.
        vals, vecs = la.eigh(Q)
        keep = vals > 1e-8
        if not np.any(keep):
            return np.zeros_like(Q)
        basis = vecs[:, keep]
        return safe_matmul(basis, basis.conj().T)

    def available_partitions(self) -> List[Tuple[int, ...]]:
        return list(self.CHARACTER_TABLE[self.n].keys())


# =========================
# States
# =========================


class StateFactory:
    """State construction utilities.

    This demo implements two practical constructors:
    1) random_density_in_sector: random PSD state projected into a chosen symmetry sector.
    2) from_external_modes_and_gram: external occupation list + internal Gram matrix.
       This returns rho_ext = Tr_int |Psi><Psi| for a bosonic symmetrized pure state.
    """

    def __init__(self, space: LabeledTensorSpace, rng: Optional[np.random.Generator] = None):
        self.space = space
        self.rng = np.random.default_rng() if rng is None else rng

    def random_density_in_sector(self, Q: np.ndarray) -> np.ndarray:
        x = random_complex_matrix((self.space.dim, self.space.dim), self.rng)
        rho = safe_matmul(x, x.conj().T)
        rho = safe_matmul(Q, rho, Q)
        return normalize_density(rho)

    def from_external_modes_and_gram(self, ext_modes: Sequence[int], gram: np.ndarray) -> np.ndarray:
        """Build rho_ext from external labels and internal Gram matrix.

        ext_modes[i] gives the external mode occupied by labeled particle i.
        gram[i,j] = <phi_j | phi_i> is the internal Gram matrix.

        Construction:
        - realize internal one-particle kets from a Cholesky factor of gram,
        - build the totally symmetrized pure state on ext⊗int,
        - trace out internal degrees of freedom.
        """
        if len(ext_modes) != self.space.n:
            raise ValueError("Length of ext_modes must equal n_particles.")
        gram = np.asarray(gram, dtype=complex)
        if gram.shape != (self.space.n, self.space.n):
            raise ValueError("Gram matrix shape must be (n_particles, n_particles).")
        if not np.allclose(gram, gram.conj().T, atol=1e-9):
            raise ValueError("Gram matrix must be Hermitian.")
        evals = np.linalg.eigvalsh(gram)
        if np.min(evals) < -1e-8:
            raise ValueError("Gram matrix must be positive semidefinite.")
        if not np.allclose(np.diag(gram), 1.0, atol=1e-8):
            raise ValueError("For normalized single-particle internal states, diag(gram) must be 1.")

        # Internal kets: gram = A A^† with row_i = ket_i^† convention.
        w, v = la.eigh(gram)
        keep = w > 1e-12
        A = v[:, keep] @ np.diag(np.sqrt(w[keep]))
        d_int = max(1, A.shape[1])
        internal_kets = [A[i, :].conj() for i in range(self.space.n)]

        ext_dim = self.space.dim
        int_dim = d_int ** self.space.n
        psi = np.zeros(ext_dim * int_dim, dtype=complex)

        ext_basis = self.space.basis_states
        int_basis = list(product(range(d_int), repeat=self.space.n))
        int_index = {state: idx for idx, state in enumerate(int_basis)}

        for perm in permutations(range(self.space.n)):
            ext_state = tuple(ext_modes[perm[k]] for k in range(self.space.n))
            ext_idx = self.space.state_to_index[ext_state]

            # Tensor product of internal labeled kets under the same permutation.
            amp_int = None
            for k in range(self.space.n):
                ket = internal_kets[perm[k]]
                amp_int = ket if amp_int is None else np.kron(amp_int, ket)
            psi[ext_idx * int_dim:(ext_idx + 1) * int_dim] += amp_int

        norm = np.linalg.norm(psi)
        if norm < TOL:
            raise ValueError("Constructed symmetrized state has zero norm.")
        psi /= norm

        rho_tot = np.outer(psi, psi.conj()).reshape(ext_dim, int_dim, ext_dim, int_dim)
        rho_ext = np.trace(rho_tot, axis1=1, axis2=3)
        return normalize_density(rho_ext)


# =========================
# Jordan filtration on operator space
# =========================


class JordanFiltration:
    """Numerical construction of cumulative spaces J_{<=j} and exact layers ΔJ_j.

    Demo choice:
    - J_{<=0} = span{I}
    - J_{<=j} is the span of all words in generators E_st of length <= j.

    This is intentionally a simple, brute-force prototype. It already captures the
    invariance [R_U, Π_{<=j}] = [R_U, Π^Δ_j] = 0 under passive linear optics, which
    is the first property you want to test numerically.
    """

    def __init__(
        self,
        space: LabeledTensorSpace,
        generator_list: Optional[Iterable[np.ndarray]] = None,
        seed_operator: Optional[np.ndarray] = None,
        support_projector: Optional[np.ndarray] = None,
    ):
        self.space = space
        self.dim = space.dim
        self.liou_dim = self.dim ** 2
        self.generator_list = list(space.generators.values()) if generator_list is None else list(generator_list)
        self.seed_operator = (
            np.eye(self.dim, dtype=complex) if seed_operator is None else np.asarray(seed_operator, dtype=complex)
        )
        self.support_projector = (
            None if support_projector is None else np.asarray(support_projector, dtype=complex)
        )
        self.cumulative_bases: Dict[int, np.ndarray] = {}
        self.layer_bases: Dict[int, np.ndarray] = {}
        self._projector_cumulative_cache: Dict[int, np.ndarray] = {}
        self._projector_layer_cache: Dict[int, np.ndarray] = {}

    def _clear_projector_cache(self) -> None:
        self._projector_cumulative_cache.clear()
        self._projector_layer_cache.clear()

    def _project_to_support(self, mats: np.ndarray) -> np.ndarray:
        if self.support_projector is None:
            return mats
        return np.einsum(
            "ab,kbc,cd->kad",
            self.support_projector,
            mats,
            self.support_projector,
            optimize=True,
        )

    def _candidate_columns(self, prev_basis: np.ndarray) -> List[np.ndarray]:
        if prev_basis.size == 0 or len(self.generator_list) == 0:
            return []

        n_prev = prev_basis.shape[1]
        prev_mats = prev_basis.T.reshape(n_prev, self.dim, self.dim)
        out: List[np.ndarray] = []
        for G in self.generator_list:
            products = np.einsum("kij,jl->kil", prev_mats, G, optimize=True)
            products = self._project_to_support(products)
            out.append(products.reshape(n_prev, self.liou_dim).T)
        return out

    def build(self, max_order: int) -> None:
        self.cumulative_bases.clear()
        self.layer_bases.clear()
        self._clear_projector_cache()

        seed = self.seed_operator
        if self.support_projector is not None:
            seed = safe_matmul(self.support_projector, seed, self.support_projector)
        cum_basis = orth_columns(vectorize(seed))
        self.cumulative_bases[0] = cum_basis
        self.layer_bases[0] = cum_basis

        for j in range(1, max_order + 1):
            prev_basis = self.cumulative_bases[j - 1]
            candidates = [prev_basis]
            candidates.extend(self._candidate_columns(prev_basis))

            big = np.column_stack(candidates)
            new_cum = orth_columns(big)
            self.cumulative_bases[j] = new_cum

            P_prev = self.projector_cumulative(j - 1)
            residual = new_cum - safe_matmul(P_prev, new_cum)
            self.layer_bases[j] = orth_columns(residual)

    def projector_cumulative(self, j: int) -> np.ndarray:
        if j not in self._projector_cumulative_cache:
            self._projector_cumulative_cache[j] = projector_from_basis(self.cumulative_bases[j])
        return self._projector_cumulative_cache[j]

    def projector_layer(self, j: int) -> np.ndarray:
        if j not in self._projector_layer_cache:
            self._projector_layer_cache[j] = projector_from_basis(self.layer_bases[j])
        return self._projector_layer_cache[j]

    @staticmethod
    def _apply_basis_projection(v: np.ndarray, basis: np.ndarray) -> np.ndarray:
        if basis.size == 0:
            return np.zeros_like(v)
        coeff = safe_matmul(basis.conj().T, v)
        return safe_matmul(basis, coeff)

    def apply_projector_cumulative(self, rho: np.ndarray, j: int) -> np.ndarray:
        v = vectorize(rho)
        proj_v = self._apply_basis_projection(v, self.cumulative_bases[j])
        return devectorize(proj_v, self.dim)

    def apply_projector_layer(self, rho: np.ndarray, j: int) -> np.ndarray:
        v = vectorize(rho)
        proj_v = self._apply_basis_projection(v, self.layer_bases[j])
        return devectorize(proj_v, self.dim)

    def cumulative_weight(self, rho: np.ndarray, j: int) -> float:
        v = vectorize(rho)
        basis = self.cumulative_bases[j]
        coeff = safe_matmul(basis.conj().T, v)
        return float(np.real(np.vdot(coeff, coeff)))

    def layer_weight(self, rho: np.ndarray, j: int) -> float:
        v = vectorize(rho)
        basis = self.layer_bases[j]
        coeff = safe_matmul(basis.conj().T, v)
        return float(np.real(np.vdot(coeff, coeff)))

    def dimensions_summary(self, max_order: int) -> List[Tuple[int, int, int]]:
        out = []
        for j in range(max_order + 1):
            out.append(
                (
                    j,
                    self.cumulative_bases[j].shape[1],
                    self.layer_bases[j].shape[1],
                )
            )
        return out


# =========================
# Dynamics + invariance checks
# =========================


class PassiveLODynamics:
    def __init__(self, space: LabeledTensorSpace):
        self.space = space

    def random_single_particle_unitary(self, rng: np.random.Generator) -> np.ndarray:
        return haar_random_unitary(self.space.m, rng)

    def evolve_density(self, rho: np.ndarray, S: np.ndarray) -> np.ndarray:
        U = self.space.total_unitary_from_single_particle(S)
        return safe_matmul(U, rho, U.conj().T)

    def heisenberg_conjugate(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        U = self.space.total_unitary_from_single_particle(S)
        return safe_matmul(U, X, U.conj().T)


class InvariantEngine:
    def __init__(self, filtration: JordanFiltration, dynamics: PassiveLODynamics):
        self.filtration = filtration
        self.dynamics = dynamics

    def commutator_error_cumulative(self, rho: np.ndarray, j: int, S: np.ndarray) -> float:
        lhs = self.filtration.apply_projector_cumulative(self.dynamics.evolve_density(rho, S), j)
        rhs = self.dynamics.evolve_density(self.filtration.apply_projector_cumulative(rho, j), S)
        return float(la.norm(lhs - rhs))

    def commutator_error_layer(self, rho: np.ndarray, j: int, S: np.ndarray) -> float:
        lhs = self.filtration.apply_projector_layer(self.dynamics.evolve_density(rho, S), j)
        rhs = self.dynamics.evolve_density(self.filtration.apply_projector_layer(rho, j), S)
        return float(la.norm(lhs - rhs))

    def layer_purities(self, rho: np.ndarray, max_order: int) -> Dict[int, float]:
        return {j: self.filtration.layer_weight(rho, j) for j in range(max_order + 1)}

    def cumulative_purities(self, rho: np.ndarray, max_order: int) -> Dict[int, float]:
        return {j: self.filtration.cumulative_weight(rho, j) for j in range(max_order + 1)}
