"""Convenience builders attached to :class:`PhotonicSystem`."""

from __future__ import annotations

from itertools import permutations
from typing import Optional, Sequence, TYPE_CHECKING

import numpy as np

from ..math import haar_random_unitary, normalize_density, safe_matmul
from .models import (
    ArrayLike,
    GramInput,
    Partition,
    PhotonicState,
    gram_description,
    resolve_gram_input,
)

if TYPE_CHECKING:
    from ..system.photonic_system import PhotonicSystem


class StateBuilder:
    """Factory-style entry point for common state preparations."""

    def __init__(self, system: "PhotonicSystem"):
        self.system = system

    def from_modes_and_gram(
        self,
        ext_modes: Sequence[int],
        gram: GramInput = "indistinguishable",
        label: Optional[str] = None,
    ) -> PhotonicState:
        """Construct state from external occupations and internal Gram matrix.

        Parameters
        ----------
        ext_modes:
            External mode assignment for each labeled particle.
        gram:
            Internal overlap model. Accepts shortcut strings, scalar overlap, or
            explicit Gram matrix.
        label:
            Optional display label.

        Returns
        -------
        PhotonicState
            External density state ``rho_ext``.
        """
        G = resolve_gram_input(gram=gram, n_particles=self.system.spec.n_particles)

        # Fast path: fully indistinguishable bosons live in the symmetric Fock sector.
        if self.system.fock_space is not None and np.allclose(
            G, np.ones((self.system.spec.n_particles, self.system.spec.n_particles), dtype=complex), atol=1e-10
        ):
            rho_f = self.system.fock_space.pure_density_from_modes(ext_modes)
            if label is None:
                label = f"modes={list(ext_modes)}, gram={gram_description(gram)}"
            return PhotonicState(system=self.system, data=None, label=label, _cache={"fock": rho_f})

        rho = self.system._state_factory.from_external_modes_and_gram(ext_modes=ext_modes, gram=G)
        if label is None:
            label = f"modes={list(ext_modes)}, gram={gram_description(gram)}"
        return PhotonicState(system=self.system, data=rho, label=label)

    def from_density_matrix(
        self,
        rho: ArrayLike,
        label: Optional[str] = None,
    ) -> PhotonicState:
        """Wrap user-provided density matrix into :class:`PhotonicState`."""
        return self.system.density_state(rho, label=label)

    def random_sector(self, lam: Partition, label: Optional[str] = None) -> PhotonicState:
        """Sample random PSD state inside sector ``lambda``."""
        Q = self.system.sector_projector(lam)
        rho = self.system._state_factory.random_density_in_sector(Q)
        if label is None:
            label = f"random sector state lambda={lam}"
        return PhotonicState(system=self.system, data=rho, label=label)

    def random_sector_state(self, lam: Partition, label: Optional[str] = None) -> PhotonicState:
        """Alias of :meth:`random_sector`."""
        return self.random_sector(lam=lam, label=label)

    def random_density(self, label: Optional[str] = None) -> PhotonicState:
        """Sample full-space random density matrix using Ginibre construction."""
        dim = self.system.hilbert_dim
        x = self.system.rng.normal(size=(dim, dim)) + 1j * self.system.rng.normal(size=(dim, dim))
        rho = safe_matmul(x, x.conj().T)
        rho = normalize_density(rho)
        if label is None:
            label = "random full density state"
        return PhotonicState(system=self.system, data=rho, label=label)

    def random_commutant_state(self, label: Optional[str] = None) -> PhotonicState:
        """Sample a permutation-commuting external state via permutation twirl."""
        dim = self.system.hilbert_dim
        x = self.system.rng.normal(size=(dim, dim)) + 1j * self.system.rng.normal(size=(dim, dim))
        rho = normalize_density(safe_matmul(x, x.conj().T))

        if self.system.projectors is None:
            raise NotImplementedError("Commutant twirl helper currently supports n=2,3 via demo projectors.")

        twirled = np.zeros_like(rho)
        perms = list(permutations(range(self.system.spec.n_particles)))
        for perm in perms:
            P = self.system.projectors.permutation_matrix(perm)
            twirled += safe_matmul(P, rho, P.conj().T)
        twirled /= float(len(perms))
        twirled = normalize_density(twirled)

        if label is None:
            label = "random commutant state"
        return PhotonicState(system=self.system, data=twirled, label=label)


class UnitaryFactory:
    """Factory for validated single-particle unitaries."""

    def __init__(self, system: "PhotonicSystem"):
        self.system = system

    def haar(self, seed: Optional[int] = None) -> np.ndarray:
        """Return Haar-random single-particle unitary."""
        rng = self.system.rng if seed is None else np.random.default_rng(seed)
        return haar_random_unitary(dim=self.system.spec.m_ext, rng=rng)

    def from_matrix(self, S: ArrayLike) -> np.ndarray:
        """Validate and return single-particle unitary matrix."""
        S = np.asarray(S, dtype=complex)
        shape = (self.system.spec.m_ext, self.system.spec.m_ext)
        if S.shape != shape:
            raise ValueError(f"Single-particle unitary must have shape {shape}.")
        if not np.allclose(S.conj().T @ S, np.eye(shape[0]), atol=1e-8):
            raise ValueError("Input matrix is not unitary.")
        return S
