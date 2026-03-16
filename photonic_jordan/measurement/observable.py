"""Single-particle observable models and measurement operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np

from ..math import safe_matmul
from .models import ObservableDistribution

if TYPE_CHECKING:
    from ..state.models import MultiplicityLabel, Partition, PhotonicState
    from ..system.photonic_system import PhotonicSystem


class SingleParticleObservable:
    """System-bound single-particle Hermitian observable.

    Parameters
    ----------
    system:
        Owning system defining ``m_ext`` and generator basis.
    single_matrix:
        Single-particle Hermitian matrix ``A``.
    name:
        Optional display label.

    Notes
    -----
    The lifted labeled external observable is
    ``J_n(A) = sum_{s,t} A_st E_st`` where ``E_st`` are cached one-body
    generators of the current system.
    """

    def __init__(self, system: "PhotonicSystem", single_matrix: np.ndarray, name: Optional[str] = None):
        self.system = system
        self.name = name

        mat = np.asarray(single_matrix, dtype=complex)
        shape = (self.system.spec.m_ext, self.system.spec.m_ext)
        if mat.shape != shape:
            raise ValueError(f"Observable matrix must have shape {shape}.")

        mat_h = 0.5 * (mat + mat.conj().T)
        if not np.allclose(mat_h, mat_h.conj().T, atol=1e-9):
            raise ValueError("Observable matrix must be Hermitian.")
        self.single_matrix = mat_h

        self._total_matrix_cache: Optional[np.ndarray] = None

    def total_matrix(self, copy: bool = True) -> np.ndarray:
        """Return lifted observable matrix ``J_n(A)`` in tensor basis.

        Parameters
        ----------
        copy:
            If ``True``, return a defensive copy.

        Returns
        -------
        np.ndarray
            Tensor-basis lifted matrix.
        """
        if self._total_matrix_cache is None:
            m = self.system.spec.m_ext
            dim = self.system.hilbert_dim
            total = np.zeros((dim, dim), dtype=complex)
            for s in range(m):
                for t in range(m):
                    coeff = self.single_matrix[s, t]
                    if abs(coeff) > 0.0:
                        total += coeff * self.system.space.generators[(s, t)]
            self._total_matrix_cache = 0.5 * (total + total.conj().T)

        out = self._total_matrix_cache
        return out.copy() if copy else out

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

    def _scoped_density(
        self,
        state: "PhotonicState",
        sector: Optional["Partition"],
        multiplicity: Optional["MultiplicityLabel"],
        copy: Optional["MultiplicityLabel"],
        conditional: bool,
        tol: float,
    ) -> Tuple[np.ndarray, float]:
        if state.system is not self.system:
            raise ValueError("Observable and state must belong to the same PhotonicSystem instance.")

        rho = self.system.project_density_to_scope(
            state.density_matrix(rep="tensor", copy=False),
            sector=sector,
            multiplicity=multiplicity,
            copy=copy,
        )
        weight = float(np.real(np.trace(rho)))

        if weight < -tol:
            raise ValueError("Scoped density has negative trace outside tolerance.")
        if abs(weight) <= tol:
            weight = 0.0

        if conditional:
            if weight <= tol:
                raise ValueError("Cannot condition on a zero-weight scope.")
            rho = rho / weight

        return rho, weight

    def expectation(
        self,
        state: "PhotonicState",
        sector: Optional["Partition"] = None,
        multiplicity: Optional["MultiplicityLabel"] = None,
        copy: Optional["MultiplicityLabel"] = None,
        conditional: bool = False,
    ) -> float:
        """Return expectation value of the lifted observable."""
        rho, _ = self._scoped_density(
            state,
            sector=sector,
            multiplicity=multiplicity,
            copy=copy,
            conditional=conditional,
            tol=1e-12,
        )
        O = self.total_matrix(copy=False)
        value = np.trace(safe_matmul(rho, O))
        return float(np.real(value))

    def variance(
        self,
        state: "PhotonicState",
        sector: Optional["Partition"] = None,
        multiplicity: Optional["MultiplicityLabel"] = None,
        copy: Optional["MultiplicityLabel"] = None,
        conditional: bool = False,
    ) -> float:
        """Return variance of the lifted observable."""
        rho, _ = self._scoped_density(
            state,
            sector=sector,
            multiplicity=multiplicity,
            copy=copy,
            conditional=conditional,
            tol=1e-12,
        )
        O = self.total_matrix(copy=False)
        O2 = safe_matmul(O, O)
        mean = float(np.real(np.trace(safe_matmul(rho, O))))
        second = float(np.real(np.trace(safe_matmul(rho, O2))))
        return second - mean**2

    def distribution(
        self,
        state: "PhotonicState",
        sector: Optional["Partition"] = None,
        multiplicity: Optional["MultiplicityLabel"] = None,
        copy: Optional["MultiplicityLabel"] = None,
        conditional: bool = False,
        tol: float = 1e-10,
    ) -> ObservableDistribution:
        """Return grouped spectral distribution of the lifted observable.

        With scoped queries:
        - ``conditional=False`` returns unnormalized scope contribution.
        - ``conditional=True`` returns normalized conditional distribution.
        """
        rho, weight = self._scoped_density(
            state,
            sector=sector,
            multiplicity=multiplicity,
            copy=copy,
            conditional=conditional,
            tol=tol,
        )

        O = self.total_matrix(copy=False)
        O = 0.5 * (O + O.conj().T)

        evals, vecs = np.linalg.eigh(O)
        groups = self._cluster_eigenvalues(evals, tol=tol)

        grouped_eigs: List[float] = []
        grouped_probs: List[float] = []
        grouped_deg: List[int] = []

        for g in groups:
            V = vecs[:, g]
            P = safe_matmul(V, V.conj().T)
            prob = float(np.real(np.trace(safe_matmul(rho, P))))
            if prob < 0.0 and prob > -tol:
                prob = 0.0
            if abs(prob) <= tol:
                prob = 0.0

            grouped_eigs.append(float(np.mean(np.real(evals[g]))))
            grouped_probs.append(prob)
            grouped_deg.append(len(g))

        eig_arr = np.asarray(grouped_eigs, dtype=float)
        prob_arr = np.asarray(grouped_probs, dtype=float)
        deg_arr = np.asarray(grouped_deg, dtype=int)

        if conditional:
            total = float(np.sum(prob_arr))
            if total <= tol:
                raise ValueError("Conditional distribution has zero total probability after grouping.")
            prob_arr = prob_arr / total

        return ObservableDistribution(
            eigenvalues=eig_arr,
            probabilities=prob_arr,
            scope_weight=weight,
            conditional=conditional,
            degeneracies=deg_arr,
        )

    def sample(
        self,
        state: "PhotonicState",
        shots: int = 1,
        rng: Optional[np.random.Generator] = None,
        sector: Optional["Partition"] = None,
        multiplicity: Optional["MultiplicityLabel"] = None,
        copy: Optional["MultiplicityLabel"] = None,
        conditional: bool = False,
    ):
        """Sample observable outcomes from grouped spectral distribution."""
        if shots < 1:
            raise ValueError("shots must be >= 1.")

        dist = self.distribution(
            state,
            sector=sector,
            multiplicity=multiplicity,
            copy=copy,
            conditional=conditional,
        )

        probs = np.asarray(dist.probabilities, dtype=float)
        total = float(np.sum(probs))
        if total <= 0.0:
            raise ValueError("Cannot sample from a zero-probability distribution.")
        probs = probs / total

        if rng is None:
            rng_local = self.system.rng
        elif isinstance(rng, np.random.Generator):
            rng_local = rng
        else:
            rng_local = np.random.default_rng(rng)

        idx = rng_local.choice(len(dist.eigenvalues), size=shots, p=probs)
        out = dist.eigenvalues[idx]
        if shots == 1:
            return float(out[0])
        return out

    def __repr__(self) -> str:
        return (
            f"SingleParticleObservable(name={self.name!r}, "
            f"shape={self.single_matrix.shape}, m_ext={self.system.spec.m_ext})"
        )
