"""Model-level specifications."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for the labeled external tensor model ``(C^m)^{\\otimes n}``.

    Parameters
    ----------
    m_ext:
        Number of external modes (single-particle Hilbert dimension).
    n_particles:
        Number of labeled particles in first-quantized representation.
    particle_type:
        Reserved for future boson/fermion branching. Current prototype
        implements bosonic construction in state generation.
    """

    m_ext: int
    n_particles: int
    particle_type: str = "boson"

    @property
    def hilbert_dim(self) -> int:
        """Total external Hilbert dimension ``m_ext ** n_particles``."""
        return self.m_ext ** self.n_particles
