"""Invariant diagnostics for Jordan hierarchy projectors."""

from __future__ import annotations

from typing import Dict

import numpy as np
import scipy.linalg as la

from ..dynamics import PassiveLODynamics
from .jordan import JordanFiltration


class InvariantEngine:
    """Compute commutator diagnostics and layer/cumulative weights."""

    def __init__(self, filtration: JordanFiltration, dynamics: PassiveLODynamics):
        self.filtration = filtration
        self.dynamics = dynamics

    def commutator_error_cumulative(self, rho: np.ndarray, j: int, S: np.ndarray) -> float:
        """Return ``|| Pi_{<=j} R_U(rho) - R_U Pi_{<=j}(rho) ||_2``."""
        lhs = self.filtration.apply_projector_cumulative(self.dynamics.evolve_density(rho, S), j)
        rhs = self.dynamics.evolve_density(self.filtration.apply_projector_cumulative(rho, j), S)
        return float(la.norm(lhs - rhs))

    def commutator_error_layer(self, rho: np.ndarray, j: int, S: np.ndarray) -> float:
        """Return ``|| Pi_j R_U(rho) - R_U Pi_j(rho) ||_2``."""
        lhs = self.filtration.apply_projector_layer(self.dynamics.evolve_density(rho, S), j)
        rhs = self.dynamics.evolve_density(self.filtration.apply_projector_layer(rho, j), S)
        return float(la.norm(lhs - rhs))

    def layer_purities(self, rho: np.ndarray, max_order: int) -> Dict[int, float]:
        """Return map ``j -> I_j`` up to ``max_order``."""
        return {j: self.filtration.layer_weight(rho, j) for j in range(max_order + 1)}

    def cumulative_purities(self, rho: np.ndarray, max_order: int) -> Dict[int, float]:
        """Return map ``j -> I_{<=j}`` up to ``max_order``."""
        return {j: self.filtration.cumulative_weight(rho, j) for j in range(max_order + 1)}
