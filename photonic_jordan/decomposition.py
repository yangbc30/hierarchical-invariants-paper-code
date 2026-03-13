from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as la

from .core import LabeledTensorSpace, SymmetricGroupProjectors, safe_matmul


Partition = Tuple[int, ...]


class SchurWeylDecomposition:
    """Lazy Schur/sector/multiplicity decomposition cache for the current demo scope.

    Notes:
    - Sector objects based on Q_lambda are canonical (basis-independent).
    - Multiplicity objects based on Q_{lambda,a} are basis-dependent because they require
      a chosen multiplicity basis.
    """

    def __init__(self, space: LabeledTensorSpace, projectors: SymmetricGroupProjectors):
        self.space = space
        self._projectors = projectors
        self._partitions: List[Partition] = list(projectors.available_partitions())
        self._sector_projectors: Dict[Partition, np.ndarray] = {}
        self._sector_bases: Dict[Partition, np.ndarray] = {}
        self._multiplicity_projectors: Dict[Partition, Tuple[np.ndarray, ...]] = {}
        self._commutant_bases: Dict[Partition, Tuple[np.ndarray, ...]] = {}
        self._block_slices: Dict[Partition, slice] = {}
        self._W: Optional[np.ndarray] = None

    def partitions(self) -> List[Partition]:
        return list(self._partitions)

    def _build_sector_metadata(self) -> None:
        if self._W is not None:
            return

        cols: List[np.ndarray] = []
        offset = 0

        for lam in self._partitions:
            Q_raw = self._projectors.isotypic_projector(lam)
            vals, vecs = la.eigh((Q_raw + Q_raw.conj().T) / 2.0)
            keep = vals > 1e-8
            basis = vecs[:, keep]
            Q = safe_matmul(basis, basis.conj().T) if basis.size else np.zeros_like(Q_raw)

            self._sector_projectors[lam] = Q
            self._sector_bases[lam] = basis

            rank = basis.shape[1]
            self._block_slices[lam] = slice(offset, offset + rank)
            offset += rank

            if rank > 0:
                cols.append(basis)

        dim = self.space.dim
        W = np.column_stack(cols) if cols else np.zeros((dim, 0), dtype=complex)
        if W.shape != (dim, dim):
            raise RuntimeError(
                "Schur basis construction did not produce a full basis. "
                "Current demo only supports n=2,3 projectors."
            )

        unitary_err = la.norm(safe_matmul(W.conj().T, W) - np.eye(dim, dtype=complex))
        if unitary_err > 1e-6:
            raise RuntimeError("Constructed Schur transform is not unitary within tolerance.")

        self._W = W

    def schur_transform(self) -> np.ndarray:
        self._build_sector_metadata()
        return np.asarray(self._W)

    def block_slice(self, lam: Partition) -> slice:
        self._build_sector_metadata()
        if lam not in self._block_slices:
            raise KeyError(f"Unknown partition {lam}.")
        return self._block_slices[lam]

    def sector_projector(self, lam: Partition) -> np.ndarray:
        self._build_sector_metadata()
        if lam not in self._sector_projectors:
            raise KeyError(f"Unknown partition {lam}.")
        return self._sector_projectors[lam]

    def dim_total(self, lam: Partition) -> int:
        self._build_sector_metadata()
        return int(self._sector_bases[lam].shape[1])

    def dim_mult(self, lam: Partition) -> int:
        table = self._projectors.CHARACTER_TABLE[self.space.n]
        if lam not in table:
            raise KeyError(f"Unknown partition {lam}.")
        return int(table[lam]["dim"])

    def dim_U(self, lam: Partition) -> int:
        total = self.dim_total(lam)
        mult = self.dim_mult(lam)
        if mult <= 0 or total % mult != 0:
            raise RuntimeError(f"Invalid dimensions for partition {lam}: total={total}, mult={mult}.")
        return total // mult

    def to_schur_operator(self, op_tensor: np.ndarray) -> np.ndarray:
        W = self.schur_transform()
        return safe_matmul(W.conj().T, op_tensor, W)

    def to_tensor_operator(self, op_schur: np.ndarray) -> np.ndarray:
        W = self.schur_transform()
        return safe_matmul(W, op_schur, W.conj().T)

    def sector_blocks(self, op: np.ndarray, rep: str = "tensor") -> Dict[Partition, np.ndarray]:
        rep_key = rep.lower()
        op_s = self.to_schur_operator(op) if rep_key == "tensor" else np.asarray(op, dtype=complex)
        blocks: Dict[Partition, np.ndarray] = {}
        for lam in self._partitions:
            sl = self.block_slice(lam)
            blocks[lam] = op_s[sl, sl]
        return blocks

    def _commutant_basis(self, lam: Partition) -> Tuple[np.ndarray, ...]:
        if lam in self._commutant_bases:
            return self._commutant_bases[lam]

        basis = self._sector_bases[lam]
        rank = basis.shape[1]
        if rank == 0:
            self._commutant_bases[lam] = tuple()
            return self._commutant_bases[lam]

        eye_rank = np.eye(rank, dtype=complex)
        constraints = []
        for G in self.space.generators.values():
            G_sub = safe_matmul(basis.conj().T, G, basis)
            constraints.append(np.kron(eye_rank, G_sub) - np.kron(G_sub.T, eye_rank))
        M = np.vstack(constraints)
        null = la.null_space(M)
        mats = tuple(null[:, k].reshape(rank, rank) for k in range(null.shape[1]))
        if not mats:
            raise RuntimeError(f"Could not build commutant basis for partition {lam}.")
        self._commutant_bases[lam] = mats
        return mats

    @staticmethod
    def _cluster_eigenvalues(evals: np.ndarray, tol: float) -> List[List[int]]:
        groups: List[List[int]] = []
        if evals.size == 0:
            return groups
        current = [0]
        for idx in range(1, evals.size):
            if abs(evals[idx] - evals[current[-1]]) <= tol:
                current.append(idx)
            else:
                groups.append(current)
                current = [idx]
        groups.append(current)
        return groups

    @staticmethod
    def _restricted_commutator_error(projectors: List[np.ndarray], generators: List[np.ndarray]) -> float:
        worst = 0.0
        for P in projectors:
            for G in generators:
                worst = max(worst, float(la.norm(safe_matmul(P, G) - safe_matmul(G, P))))
        return worst

    def _build_multiplicity_projectors(self, lam: Partition) -> Tuple[np.ndarray, ...]:
        if lam in self._multiplicity_projectors:
            return self._multiplicity_projectors[lam]

        Q_lam = self.sector_projector(lam)
        d_mult = self.dim_mult(lam)
        if d_mult == 1:
            self._multiplicity_projectors[lam] = (Q_lam,)
            return self._multiplicity_projectors[lam]

        basis = self._sector_bases[lam]
        rank = basis.shape[1]
        if rank % d_mult != 0:
            raise RuntimeError(f"Sector rank {rank} is incompatible with multiplicity {d_mult} for {lam}.")
        d_u = rank // d_mult

        comm_basis = self._commutant_basis(lam)
        generators_sub = [safe_matmul(basis.conj().T, G, basis) for G in self.space.generators.values()]

        seed = self._stable_seed(self.space.m, self.space.n, lam)
        rng = np.random.default_rng(seed)

        n_basis = len(comm_basis)
        coeff_trials: List[np.ndarray] = [
            np.arange(1, n_basis + 1, dtype=float),
            np.exp(1j * np.arange(n_basis)),
            np.linspace(1.0, 2.0, n_basis, dtype=float) + 1j * np.linspace(0.0, 1.0, n_basis, dtype=float),
        ]
        for _ in range(8):
            coeff_trials.append(rng.normal(size=n_basis) + 1j * rng.normal(size=n_basis))

        successful: Optional[List[np.ndarray]] = None
        for coeff in coeff_trials:
            H_sub = np.einsum("k,kab->ab", coeff, np.stack(comm_basis, axis=0), optimize=True)
            H_sub = (H_sub + H_sub.conj().T) / 2.0

            evals, vecs = la.eigh(H_sub)
            groups: Optional[List[List[int]]] = None
            for tol in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5):
                candidate = self._cluster_eigenvalues(evals, tol=tol)
                if len(candidate) == d_mult and all(len(g) == d_u for g in candidate):
                    groups = candidate
                    break
            if groups is None:
                continue

            projectors_sub = []
            for g in groups:
                V = vecs[:, g]
                projectors_sub.append(safe_matmul(V, V.conj().T))

            if self._restricted_commutator_error(projectors_sub, generators_sub) > 5e-6:
                continue

            successful = projectors_sub
            break

        if successful is None:
            raise RuntimeError(
                "Failed to resolve multiplicity projectors from commutant structure. "
                "This demo implementation currently supports robust multiplicity resolution for small n."
            )

        projectors_full: List[np.ndarray] = []
        for P_sub in successful:
            P = safe_matmul(basis, P_sub, basis.conj().T)
            P = (P + P.conj().T) / 2.0
            vals, vecs = la.eigh(P)
            keep = vals > 1e-8
            P_clean = safe_matmul(vecs[:, keep], vecs[:, keep].conj().T) if np.any(keep) else np.zeros_like(P)
            projectors_full.append(P_clean)

        self._multiplicity_projectors[lam] = tuple(projectors_full)
        return self._multiplicity_projectors[lam]

    def multiplicity_projectors(self, lam: Partition) -> Tuple[np.ndarray, ...]:
        self._build_sector_metadata()
        return self._build_multiplicity_projectors(lam)

    def multiplicity_projector(self, lam: Partition, a: int) -> np.ndarray:
        fam = self.multiplicity_projectors(lam)
        if a < 0 or a >= len(fam):
            raise IndexError(f"multiplicity index a={a} is out of range for partition {lam}.")
        return fam[a]

    # Backward-compatible aliases.
    def copy_projectors(self, lam: Partition) -> Tuple[np.ndarray, ...]:
        return self.multiplicity_projectors(lam)

    def copy_projector(self, lam: Partition, a: int) -> np.ndarray:
        return self.multiplicity_projector(lam, a)

    @staticmethod
    def _stable_seed(space_m: int, space_n: int, lam: Partition) -> int:
        payload = f"{space_m}|{space_n}|{','.join(map(str, lam))}".encode("ascii")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)
