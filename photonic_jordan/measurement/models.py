"""Measurement result containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ObservableDistribution:
    """Grouped spectral distribution for an observable measurement.

    Parameters
    ----------
    eigenvalues:
        Grouped eigenvalues.
    probabilities:
        Grouped probabilities (or scoped contributions if not conditional).
    scope_weight:
        Scope weight ``Tr(Q rho)`` used for the query.
    conditional:
        Whether scope conditioning was applied.
    degeneracies:
        Optional grouped degeneracies aligned with ``eigenvalues``.
    """

    eigenvalues: np.ndarray
    probabilities: np.ndarray
    scope_weight: float
    conditional: bool
    degeneracies: Optional[np.ndarray] = None

    def mean(self) -> float:
        """Return first spectral moment with stored probabilities."""
        return float(np.real(np.dot(self.eigenvalues, self.probabilities)))

    def variance(self) -> float:
        """Return second centered moment with stored probabilities."""
        m1 = self.mean()
        m2 = float(np.real(np.dot(self.eigenvalues**2, self.probabilities)))
        return m2 - m1**2

    def summary(self, digits: int = 8) -> str:
        """Return human-readable summary."""
        lines = [
            "ObservableDistribution",
            f"  conditional={self.conditional}",
            f"  scope_weight={self.scope_weight:.{digits}f}",
        ]
        for i, (lam, p) in enumerate(zip(self.eigenvalues, self.probabilities)):
            if self.degeneracies is None:
                lines.append(f"  eig[{i}]={lam:.{digits}f}, p={p:.{digits}f}")
            else:
                lines.append(
                    f"  eig[{i}]={lam:.{digits}f}, p={p:.{digits}f}, deg={int(self.degeneracies[i])}"
                )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
