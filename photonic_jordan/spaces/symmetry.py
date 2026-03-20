"""Symmetric-group projectors for Schur-Weyl sectors."""

from __future__ import annotations

from itertools import combinations, permutations
from math import factorial
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import scipy.linalg as la

from ..math import safe_matmul
from .labeled_tensor import LabeledTensorSpace

Partition = Tuple[int, ...]


def _partitions_of_integer(n: int, max_part: int | None = None) -> Iterable[Partition]:
    """Yield integer partitions of ``n`` in reverse lexicographic order."""
    if max_part is None:
        max_part = n
    if n == 0:
        yield tuple()
        return
    upper = min(max_part, n)
    for first in range(upper, 0, -1):
        for tail in _partitions_of_integer(n - first, first):
            yield (first,) + tail


def _cells_from_partition(partition: Partition) -> Set[Tuple[int, int]]:
    cells: Set[Tuple[int, int]] = set()
    for r, length in enumerate(partition):
        for c in range(length):
            cells.add((r, c))
    return cells


def _is_connected(cells: Set[Tuple[int, int]]) -> bool:
    if not cells:
        return False
    stack = [next(iter(cells))]
    seen = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if nb in cells and nb not in seen:
                stack.append(nb)
    return len(seen) == len(cells)


def _has_2x2_block(cells: Set[Tuple[int, int]]) -> bool:
    for r, c in cells:
        if (r + 1, c) in cells and (r, c + 1) in cells and (r + 1, c + 1) in cells:
            return True
    return False


def _partition_from_cells(cells: Set[Tuple[int, int]], max_rows: int) -> Partition | None:
    row_lengths = [0] * max_rows
    rows: Dict[int, List[int]] = {}
    for r, c in cells:
        rows.setdefault(r, []).append(c)

    for r in range(max_rows):
        cols = sorted(rows.get(r, []))
        if len(cols) == 0:
            row_lengths[r] = 0
            continue
        expected = list(range(len(cols)))
        if cols != expected:
            return None
        row_lengths[r] = len(cols)

    for i in range(max_rows - 1):
        if row_lengths[i] < row_lengths[i + 1]:
            return None

    while row_lengths and row_lengths[-1] == 0:
        row_lengths.pop()
    return tuple(int(x) for x in row_lengths)


class SymmetricGroupProjectors:
    """Construct Schur-sector projectors ``Q_λ`` from character sums.

    Method
    ------
    ``Q_λ = d_λ / n! * sum_{π in S_n} χ_λ(π) P_π``,
    where ``d_λ`` is the symmetric-group irrep dimension and ``χ_λ`` is computed
    by the Murnaghan-Nakayama rule.
    """

    def __init__(self, space: LabeledTensorSpace):
        self.space = space
        self.n = space.n
        self._partitions: List[Partition] = [
            lam for lam in _partitions_of_integer(self.n) if len(lam) <= self.space.m
        ]
        self._perm_matrices: Dict[Tuple[int, ...], np.ndarray] = {}
        self._class_sums: Dict[Partition, np.ndarray] | None = None
        self._char_cache: Dict[Tuple[Partition, Partition], int] = {}
        self._rim_hook_cache: Dict[Tuple[Partition, int], List[Tuple[Partition, int]]] = {}
        self._projector_cache: Dict[Partition, np.ndarray] = {}

    @staticmethod
    def irrep_dimension(partition: Partition) -> int:
        """Return ``dim(V_λ)`` of the symmetric-group irrep via hook-length formula."""
        n = sum(partition)
        if n == 0:
            return 1
        hooks = 1
        for i, row_len in enumerate(partition):
            for j in range(row_len):
                right = row_len - j - 1
                below = sum(1 for r in partition[i + 1 :] if r > j)
                hooks *= right + below + 1
        return factorial(n) // hooks

    @staticmethod
    def _cycle_type(perm: Sequence[int]) -> Partition:
        n = len(perm)
        seen = [False] * n
        lengths: List[int] = []
        for start in range(n):
            if seen[start]:
                continue
            k = start
            length = 0
            while not seen[k]:
                seen[k] = True
                k = perm[k]
                length += 1
            lengths.append(length)
        lengths.sort(reverse=True)
        return tuple(lengths)

    def _rim_hook_removals(self, partition: Partition, length: int) -> List[Tuple[Partition, int]]:
        key = (partition, length)
        if key in self._rim_hook_cache:
            return self._rim_hook_cache[key]

        if length <= 0 or length > sum(partition):
            self._rim_hook_cache[key] = []
            return []

        shape_cells = _cells_from_partition(partition)
        cells_sorted = sorted(shape_cells)
        out: List[Tuple[Partition, int]] = []
        max_rows = len(partition)

        for subset_tuple in combinations(cells_sorted, length):
            subset = set(subset_tuple)
            if not _is_connected(subset):
                continue
            if _has_2x2_block(subset):
                continue

            remaining = shape_cells - subset
            new_partition = _partition_from_cells(remaining, max_rows=max_rows)
            if new_partition is None:
                continue

            height = len({r for r, _ in subset})
            out.append((new_partition, height))

        self._rim_hook_cache[key] = out
        return out

    def character(self, lam: Partition, cycle_type: Partition) -> int:
        """Return irreducible character ``χ_λ(μ)`` via Murnaghan-Nakayama."""
        key = (lam, cycle_type)
        if key in self._char_cache:
            return self._char_cache[key]

        if sum(lam) != sum(cycle_type):
            self._char_cache[key] = 0
            return 0
        if len(cycle_type) == 0:
            val = 1 if len(lam) == 0 else 0
            self._char_cache[key] = val
            return val

        r = cycle_type[0]
        rest = cycle_type[1:]
        total = 0
        for new_lam, height in self._rim_hook_removals(lam, r):
            sign = -1 if (height - 1) % 2 else 1
            total += sign * self.character(new_lam, rest)

        self._char_cache[key] = int(total)
        return int(total)

    def permutation_matrix(self, perm: Sequence[int], cache: bool = True) -> np.ndarray:
        """Return the permutation representation matrix ``P_π`` on tensor basis."""
        key = tuple(perm)
        if cache and key in self._perm_matrices:
            return self._perm_matrices[key]

        dim = self.space.dim
        permuted = self.space.basis_array[:, list(key)]
        out_idx = permuted @ self.space.index_weights
        in_idx = np.arange(dim)
        P = np.zeros((dim, dim), dtype=complex)
        P[out_idx, in_idx] = 1.0
        if cache:
            self._perm_matrices[key] = P
        return P

    def _build_class_sums(self) -> Dict[Partition, np.ndarray]:
        if self._class_sums is not None:
            return self._class_sums

        dim = self.space.dim
        class_sums: Dict[Partition, np.ndarray] = {}
        for perm in permutations(range(self.n)):
            mu = self._cycle_type(perm)
            if mu not in class_sums:
                class_sums[mu] = np.zeros((dim, dim), dtype=complex)
            class_sums[mu] += self.permutation_matrix(perm, cache=False)
        self._class_sums = class_sums
        return class_sums

    def isotypic_projector(self, partition: Partition) -> np.ndarray:
        """Build orthogonal projector onto Schur sector ``lambda``."""
        if partition not in self._partitions:
            raise KeyError(f"Unknown partition {partition}.")
        if partition in self._projector_cache:
            return self._projector_cache[partition]

        d_lambda = self.irrep_dimension(partition)
        acc = np.zeros((self.space.dim, self.space.dim), dtype=complex)
        for mu, class_sum in self._build_class_sums().items():
            chi = self.character(partition, mu)
            if chi != 0:
                acc += chi * class_sum

        Q = d_lambda / factorial(self.n) * acc
        Q = (Q + Q.conj().T) / 2.0

        vals, vecs = la.eigh(Q)
        keep = vals > 1e-8
        if not np.any(keep):
            out = np.zeros_like(Q)
        else:
            basis = vecs[:, keep]
            out = safe_matmul(basis, basis.conj().T)
        self._projector_cache[partition] = out
        return out

    def available_partitions(self) -> List[Partition]:
        """Return partition labels with nontrivial support on ``(C^m)^⊗n``."""
        return list(self._partitions)
