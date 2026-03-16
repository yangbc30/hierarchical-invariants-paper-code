"""Labeled first-quantized tensor space utilities."""

from __future__ import annotations

from itertools import product
from typing import Dict, Optional, Tuple

import numpy as np

from ..specs import ModelSpec


class LabeledTensorSpace:
    """First-quantized external space ``(C^m)^{\\otimes n}``.

    The basis is the computational tensor-product basis indexed by tuples
    ``(x_1, ..., x_n)`` with each ``x_k in {0, ..., m-1}``.

    One-body generators are lifted as
    ``E_st^(tot) = sum_i I^{\otimes(i-1)} \otimes |s><t| \otimes I^{\otimes(n-i)}``.
    These are used to generate Jordan hierarchy subspaces on operator space.
    """

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.m = spec.m_ext
        self.n = spec.n_particles
        self.dim = spec.hilbert_dim

        self.basis_states = list(product(range(self.m), repeat=self.n))
        self.basis_array = np.asarray(self.basis_states, dtype=int)
        self.index_weights = (self.m ** np.arange(self.n - 1, -1, -1)).astype(int)
        self.state_to_index = {state: idx for idx, state in enumerate(self.basis_states)}

        self._single_ops_cache: Optional[Dict[Tuple[int, int], np.ndarray]] = None
        self._generators_cache: Optional[Dict[Tuple[int, int], np.ndarray]] = None

    @property
    def single_ops(self) -> Dict[Tuple[int, int], np.ndarray]:
        """Return cached single-particle basis operators ``|s><t|``."""
        if self._single_ops_cache is None:
            self._single_ops_cache = self._build_single_particle_ops()
        return self._single_ops_cache

    @property
    def generators(self) -> Dict[Tuple[int, int], np.ndarray]:
        """Return cached lifted one-body generators ``E_st``."""
        if self._generators_cache is None:
            self._generators_cache = self._build_generators()
        return self._generators_cache

    def _build_single_particle_ops(self) -> Dict[Tuple[int, int], np.ndarray]:
        ops: Dict[Tuple[int, int], np.ndarray] = {}
        for s in range(self.m):
            for t in range(self.m):
                op = np.zeros((self.m, self.m), dtype=complex)
                op[s, t] = 1.0
                ops[(s, t)] = op
        return ops

    def lift_one_body_operator(self, h: np.ndarray, slot: int) -> np.ndarray:
        """Lift a single-particle operator to slot ``slot`` of the tensor space."""
        ops = [np.eye(self.m, dtype=complex) for _ in range(self.n)]
        ops[slot] = h
        out = ops[0]
        for k in range(1, self.n):
            out = np.kron(out, ops[k])
        return out

    def total_one_body_operator(self, h: np.ndarray) -> np.ndarray:
        """Second-quantization-style sum of one-body action over all particles."""
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
        """Lift single-particle unitary ``S`` to ``S^{\\otimes n}``."""
        out = S
        for _ in range(self.n - 1):
            out = np.kron(out, S)
        return out
