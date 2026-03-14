"""Symmetric-group projectors for Schur-Weyl sectors."""

from __future__ import annotations

from itertools import permutations
from math import factorial
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.linalg as la

from ..math import safe_matmul
from .labeled_tensor import LabeledTensorSpace


class SymmetricGroupProjectors:
    """Construct isotypic projectors ``Q_λ`` from character sums.

    Current demo scope supports ``n=2`` and ``n=3`` where explicit character
    tables are hard-coded for transparent validation.

    Method:
    ``Q_λ = d_λ / n! * sum_{π in S_n} χ_λ(π) P_π``.

    References
    ----------
    - B. C. Hall, *Lie Groups, Lie Algebras, and Representations*, 2nd ed.
      (character/projector construction in finite groups).
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
        """Return permutation parity as ``+1`` (even) or ``-1`` (odd)."""
        inv = 0
        p = list(perm)
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                if p[i] > p[j]:
                    inv += 1
        return -1 if inv % 2 else 1

    def conjugacy_type(self, perm: Sequence[int]) -> str:
        """Return conjugacy class tag used by hard-coded character table."""
        p = tuple(perm)
        if self.n == 2:
            return "e" if p == (0, 1) else "t"
        if p == (0, 1, 2):
            return "e"
        return "t" if self.parity(p) == -1 else "c"

    def permutation_matrix(self, perm: Sequence[int]) -> np.ndarray:
        """Return the permutation representation matrix ``P_π`` on tensor basis."""
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
        """Build orthogonal projector onto Schur sector ``lambda``.

        Parameters
        ----------
        partition:
            Young-diagram partition label.

        Returns
        -------
        np.ndarray
            Orthogonal projector ``Q_lambda`` in tensor basis.
        """
        table = self.CHARACTER_TABLE[self.n][partition]
        d_lambda = table["dim"]
        acc = np.zeros((self.space.dim, self.space.dim), dtype=complex)
        for perm in self._permutations:
            cls = self.conjugacy_type(perm)
            chi = table["chars"][cls]
            acc += chi * self.permutation_matrix(perm)

        Q = d_lambda / factorial(self.n) * acc
        Q = (Q + Q.conj().T) / 2.0

        vals, vecs = la.eigh(Q)
        keep = vals > 1e-8
        if not np.any(keep):
            return np.zeros_like(Q)
        basis = vecs[:, keep]
        return safe_matmul(basis, basis.conj().T)

    def available_partitions(self) -> List[Tuple[int, ...]]:
        """Return supported partition labels for current ``n``."""
        return list(self.CHARACTER_TABLE[self.n].keys())
