"""Bosonic fixed-particle-number Fock space backend."""

from __future__ import annotations

from math import comb, factorial, sqrt
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..math import safe_matmul
from ..specs import ModelSpec
from .labeled_tensor import LabeledTensorSpace


def _compositions(total: int, parts: int) -> Iterable[Tuple[int, ...]]:
    """Yield nonnegative integer tuples of length ``parts`` summing to ``total``."""
    if parts == 1:
        yield (total,)
        return
    for x in range(total + 1):
        for tail in _compositions(total - x, parts - 1):
            yield (x,) + tail


class BosonicFockSpace:
    """Symmetric ``n``-boson subspace in occupation-number basis.

    The basis vectors are occupations ``(n_0, ..., n_{m-1})`` with
    ``sum_k n_k = n``. One-body generators are represented as
    ``E_st = a_s^\\dagger a_t`` in this basis.
    """

    def __init__(self, spec: ModelSpec, tensor_space: Optional[LabeledTensorSpace] = None):
        if spec.particle_type != "boson":
            raise NotImplementedError("BosonicFockSpace is only available for particle_type='boson'.")

        self.spec = spec
        self.m = spec.m_ext
        self.n = spec.n_particles
        self.dim = comb(self.m + self.n - 1, self.n)

        self.basis_states: List[Tuple[int, ...]] = list(_compositions(self.n, self.m))
        self.state_to_index: Dict[Tuple[int, ...], int] = {
            occ: idx for idx, occ in enumerate(self.basis_states)
        }

        self.tensor_space = tensor_space

        self._generators_cache: Optional[Dict[Tuple[int, int], np.ndarray]] = None
        self._isometry_cache: Optional[np.ndarray] = None
        self._multinomial_norm_cache: Dict[Tuple[int, ...], float] = {}
        self._sqrt_factorials = np.array([sqrt(factorial(k)) for k in range(self.n + 1)], dtype=float)

    def occupation_from_modes(self, ext_modes: Sequence[int]) -> Tuple[int, ...]:
        """Return occupation tuple from a labeled mode assignment."""
        if len(ext_modes) != self.n:
            raise ValueError("Length of ext_modes must equal n_particles.")
        occ = [0] * self.m
        for mode in ext_modes:
            idx = int(mode)
            if idx < 0 or idx >= self.m:
                raise ValueError("Mode index out of range.")
            occ[idx] += 1
        return tuple(occ)

    def validate_occupation(self, occupation: Sequence[int]) -> Tuple[int, ...]:
        """Validate and return occupation tuple ``(n_0,...,n_{m-1})``."""
        occ = tuple(int(x) for x in occupation)
        if len(occ) != self.m:
            raise ValueError(f"Occupation must have length m_ext={self.m}.")
        if any(x < 0 for x in occ):
            raise ValueError("Occupation entries must be nonnegative.")
        if sum(occ) != self.n:
            raise ValueError(
                f"Occupation total must equal n_particles={self.n}, got {sum(occ)}."
            )
        return occ

    def index_from_occupation(self, occupation: Sequence[int]) -> int:
        """Return basis index for a validated occupation tuple."""
        occ = self.validate_occupation(occupation)
        return self.state_to_index[occ]

    def pure_density_from_occupation(self, occupation: Sequence[int]) -> np.ndarray:
        """Return pure Fock-basis density for occupation ``(n_0,...,n_{m-1})``."""
        idx = self.index_from_occupation(occupation)
        rho = np.zeros((self.dim, self.dim), dtype=complex)
        rho[idx, idx] = 1.0
        return rho

    def mixed_density_from_occupations(
        self,
        occupations: Sequence[Sequence[int]],
        weights: Sequence[float],
        normalize: bool = True,
    ) -> np.ndarray:
        """Return classical Fock mixture density from occupations and weights."""
        if len(occupations) == 0:
            raise ValueError("occupations must be non-empty.")
        if len(occupations) != len(weights):
            raise ValueError("occupations and weights must have the same length.")

        probs = np.asarray(weights, dtype=float)
        if np.any(probs < -1e-12):
            raise ValueError("weights must be nonnegative (within tolerance).")
        probs[probs < 0.0] = 0.0
        total = float(np.sum(probs))
        if normalize:
            if total <= 0.0:
                raise ValueError("weights must have positive sum.")
            probs = probs / total

        rho = np.zeros((self.dim, self.dim), dtype=complex)
        for occ, p in zip(occupations, probs):
            idx = self.index_from_occupation(occ)
            rho[idx, idx] += float(p)

        if normalize and float(np.real(np.trace(rho))) > 0.0:
            rho = rho / np.trace(rho)
        return rho

    def pure_density_from_modes(self, ext_modes: Sequence[int]) -> np.ndarray:
        """Return pure Fock-basis density for occupation determined by ``ext_modes``."""
        occ = self.occupation_from_modes(ext_modes)
        return self.pure_density_from_occupation(occ)

    def _multinomial_norm(self, occ: Tuple[int, ...]) -> float:
        """Return ``sqrt(n! / prod_k n_k!)`` for occupation ``occ``."""
        if occ not in self._multinomial_norm_cache:
            denom = 1
            for nk in occ:
                denom *= factorial(nk)
            self._multinomial_norm_cache[occ] = sqrt(factorial(self.n) / float(denom))
        return self._multinomial_norm_cache[occ]

    @property
    def isometry_to_tensor(self) -> np.ndarray:
        """Return isometry ``V`` from Fock basis to labeled tensor basis.

        ``V`` has shape ``(m^n, dim_sym)`` and columns are normalized symmetric
        basis vectors in the tensor basis.
        """
        if self._isometry_cache is not None:
            return self._isometry_cache

        if self.tensor_space is None:
            raise RuntimeError("Tensor space is required to build Fock<->tensor isometry.")

        V = np.zeros((self.tensor_space.dim, self.dim), dtype=complex)
        for idx, state in enumerate(self.tensor_space.basis_states):
            occ = [0] * self.m
            for mode in state:
                occ[mode] += 1
            occ_t = tuple(occ)
            col = self.state_to_index[occ_t]
            V[idx, col] = 1.0 / self._multinomial_norm(occ_t)

        self._isometry_cache = V
        return V

    @property
    def generators(self) -> Dict[Tuple[int, int], np.ndarray]:
        """Return one-body generators ``E_st = a_s^\\dagger a_t`` in Fock basis."""
        if self._generators_cache is None:
            self._generators_cache = self._build_generators()
        return self._generators_cache

    def _build_generators(self) -> Dict[Tuple[int, int], np.ndarray]:
        gens: Dict[Tuple[int, int], np.ndarray] = {}
        d = self.dim
        for s in range(self.m):
            for t in range(self.m):
                G = np.zeros((d, d), dtype=complex)
                for col, occ in enumerate(self.basis_states):
                    nt = occ[t]
                    if nt == 0:
                        continue
                    if s == t:
                        G[col, col] = float(nt)
                        continue

                    ns = occ[s]
                    new_occ = list(occ)
                    new_occ[s] += 1
                    new_occ[t] -= 1
                    row = self.state_to_index[tuple(new_occ)]
                    G[row, col] = sqrt((ns + 1.0) * nt)
                gens[(s, t)] = G
        return gens

    def total_one_body_operator(self, h: np.ndarray) -> np.ndarray:
        """Return lifted one-body observable ``sum_{s,t} h_st E_st`` in Fock basis."""
        h = np.asarray(h, dtype=complex)
        if h.shape != (self.m, self.m):
            raise ValueError(f"Single-particle operator must have shape {(self.m, self.m)}.")
        out = np.zeros((self.dim, self.dim), dtype=complex)
        for s in range(self.m):
            for t in range(self.m):
                c = h[s, t]
                if abs(c) > 0.0:
                    out += c * self.generators[(s, t)]
        return out

    def total_unitary_from_single_particle(self, S: np.ndarray) -> np.ndarray:
        """Return bosonic lifted unitary ``Sym^n(S)`` in Fock basis.

        Notes
        -----
        Implementation uses repeated application of transformed creation
        operators to the vacuum in normalized occupation basis, avoiding
        tensor-space construction of ``S^{\\otimes n}``.
        """
        S = np.asarray(S, dtype=complex)
        if S.shape != (self.m, self.m):
            raise ValueError(f"Single-particle unitary must have shape {(self.m, self.m)}.")

        d = self.dim
        U = np.zeros((d, d), dtype=complex)
        vacuum = tuple([0] * self.m)

        for col, occ_in in enumerate(self.basis_states):
            amps: Dict[Tuple[int, ...], complex] = {vacuum: 1.0 + 0.0j}

            for in_mode, count in enumerate(occ_in):
                for _ in range(count):
                    nxt: Dict[Tuple[int, ...], complex] = {}
                    for occ, amp in amps.items():
                        for out_mode in range(self.m):
                            coeff = S[out_mode, in_mode]
                            if abs(coeff) == 0.0:
                                continue
                            occ_next = list(occ)
                            occ_next[out_mode] += 1
                            occ_t = tuple(occ_next)
                            factor = sqrt(float(occ[out_mode] + 1))
                            nxt[occ_t] = nxt.get(occ_t, 0.0 + 0.0j) + amp * coeff * factor
                    amps = nxt

            norm = float(np.prod(self._sqrt_factorials[np.asarray(occ_in, dtype=int)]))
            inv_norm = 1.0 / norm
            for occ_out, amp in amps.items():
                row = self.state_to_index[occ_out]
                U[row, col] = amp * inv_norm

        return U

    def evolve_density(self, rho: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Evolve Fock density by lifted single-particle unitary ``S``."""
        rho = np.asarray(rho, dtype=complex)
        if rho.shape != (self.dim, self.dim):
            raise ValueError(f"Fock density must have shape {(self.dim, self.dim)}.")
        U = self.total_unitary_from_single_particle(S)
        return safe_matmul(U, rho, U.conj().T)
