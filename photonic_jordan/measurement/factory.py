"""Observable constructors attached to :class:`PhotonicSystem`."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from .observable import SingleParticleObservable

if TYPE_CHECKING:
    from ..system.photonic_system import PhotonicSystem


class ObservableFactory:
    """Factory for single-particle Hermitian observables."""

    def __init__(self, system: "PhotonicSystem"):
        self.system = system

    def _validate_mode(self, mode: int) -> int:
        if not isinstance(mode, (int, np.integer)):
            raise TypeError("mode index must be an integer.")
        m = self.system.spec.m_ext
        if mode < 0 or mode >= m:
            raise ValueError(f"mode must be in [0, {m - 1}].")
        return int(mode)

    def _validate_two_modes(self, modes: Sequence[int]) -> Tuple[int, int]:
        if len(modes) != 2:
            raise ValueError("modes must contain exactly two entries.")
        a = self._validate_mode(modes[0])
        b = self._validate_mode(modes[1])
        if a == b:
            raise ValueError("modes must be distinct for Pauli-like observables.")
        return a, b

    def from_matrix(self, A: np.ndarray, name: Optional[str] = None) -> SingleParticleObservable:
        """Create observable from single-particle Hermitian matrix.

        Parameters
        ----------
        A:
            Single-particle Hermitian matrix of shape ``(m_ext, m_ext)``.
        name:
            Optional label.
        """
        arr = np.asarray(A, dtype=complex)
        shape = (self.system.spec.m_ext, self.system.spec.m_ext)
        if arr.shape != shape:
            raise ValueError(f"Observable matrix must have shape {shape}.")
        if not np.allclose(arr, arr.conj().T, atol=1e-9):
            raise ValueError("Input observable matrix must be Hermitian.")
        return SingleParticleObservable(system=self.system, single_matrix=arr, name=name)

    def number(self, mode: int) -> SingleParticleObservable:
        """Create number observable lifted from ``|mode><mode|``."""
        idx = self._validate_mode(mode)
        A = np.zeros((self.system.spec.m_ext, self.system.spec.m_ext), dtype=complex)
        A[idx, idx] = 1.0
        return SingleParticleObservable(system=self.system, single_matrix=A, name=f"number(mode={idx})")

    def projector(self, mode: int) -> SingleParticleObservable:
        """Alias of :meth:`number` for the current measurement model."""
        return self.number(mode)

    def sigma_z(self, modes: Sequence[int] = (0, 1)) -> SingleParticleObservable:
        """Create embedded two-level ``sigma_z`` observable."""
        a, b = self._validate_two_modes(modes)
        A = np.zeros((self.system.spec.m_ext, self.system.spec.m_ext), dtype=complex)
        A[a, a] = 1.0
        A[b, b] = -1.0
        return SingleParticleObservable(system=self.system, single_matrix=A, name=f"sigma_z(modes=({a},{b}))")

    def sigma_x(self, modes: Sequence[int] = (0, 1)) -> SingleParticleObservable:
        """Create embedded two-level ``sigma_x`` observable."""
        a, b = self._validate_two_modes(modes)
        A = np.zeros((self.system.spec.m_ext, self.system.spec.m_ext), dtype=complex)
        A[a, b] = 1.0
        A[b, a] = 1.0
        return SingleParticleObservable(system=self.system, single_matrix=A, name=f"sigma_x(modes=({a},{b}))")

    def sigma_y(self, modes: Sequence[int] = (0, 1)) -> SingleParticleObservable:
        """Create embedded two-level ``sigma_y`` observable."""
        a, b = self._validate_two_modes(modes)
        A = np.zeros((self.system.spec.m_ext, self.system.spec.m_ext), dtype=complex)
        A[a, b] = -1j
        A[b, a] = 1j
        return SingleParticleObservable(system=self.system, single_matrix=A, name=f"sigma_y(modes=({a},{b}))")
