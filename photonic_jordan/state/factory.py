"""State-construction routines for external density operators."""

from __future__ import annotations

from itertools import permutations, product
from typing import Optional, Sequence

import numpy as np
import scipy.linalg as la

from ..math import TOL, normalize_density, random_complex_matrix, safe_matmul
from ..spaces import LabeledTensorSpace


class StateFactory:
    """Construct physical states used in hierarchy checks.

    Implemented constructors
    ------------------------
    1) ``random_density_in_sector``: random PSD state compressed into a sector.
    2) ``from_external_modes_and_gram``: construct
       ``rho_ext = Tr_int |Psi><Psi|`` from labeled external occupations and an
       internal Gram matrix of single-particle internal states.

    Physics note
    ------------
    The Gram-matrix formulation is the standard route to partial
    distinguishability in multiphoton interference simulations.

    References
    ----------
    - M. C. Tichy, *Sampling of partially distinguishable bosons and the
      relation to the multidimensional permanent*, Phys. Rev. A 91, 022316 (2015).
    """

    def __init__(self, space: LabeledTensorSpace, rng: Optional[np.random.Generator] = None):
        self.space = space
        self.rng = np.random.default_rng() if rng is None else rng

    def random_density_in_sector(self, Q: np.ndarray) -> np.ndarray:
        """Sample random density matrix supported on projector ``Q``."""
        x = random_complex_matrix((self.space.dim, self.space.dim), self.rng)
        rho = safe_matmul(x, x.conj().T)
        rho = safe_matmul(Q, rho, Q)
        return normalize_density(rho)

    def from_external_modes_and_gram(self, ext_modes: Sequence[int], gram: np.ndarray) -> np.ndarray:
        """Build ``rho_ext`` from external labels and internal Gram matrix.

        Parameters
        ----------
        ext_modes:
            ``ext_modes[i]`` is the external mode occupied by labeled particle ``i``.
        gram:
            Internal overlap matrix with entries
            ``gram[i,j] = <phi_j | phi_i>``.

        Returns
        -------
        np.ndarray
            External density matrix in labeled basis.

        Method
        ------
        1) Realize internal kets from eigendecomposition of ``gram``.
        2) Build fully symmetrized pure state on ``H_ext \\otimes H_int``.
        3) Trace out internal space.
        """
        if len(ext_modes) != self.space.n:
            raise ValueError("Length of ext_modes must equal n_particles.")

        gram = np.asarray(gram, dtype=complex)
        if gram.shape != (self.space.n, self.space.n):
            raise ValueError("Gram matrix shape must be (n_particles, n_particles).")
        if not np.allclose(gram, gram.conj().T, atol=1e-9):
            raise ValueError("Gram matrix must be Hermitian.")

        evals = np.linalg.eigvalsh(gram)
        if np.min(evals) < -1e-8:
            raise ValueError("Gram matrix must be positive semidefinite.")
        if not np.allclose(np.diag(gram), 1.0, atol=1e-8):
            raise ValueError("For normalized single-particle internal states, diag(gram) must be 1.")

        w, v = la.eigh(gram)
        keep = w > 1e-12
        A = v[:, keep] @ np.diag(np.sqrt(w[keep]))

        d_int = max(1, A.shape[1])
        internal_kets = [A[i, :].conj() for i in range(self.space.n)]

        ext_dim = self.space.dim
        int_dim = d_int ** self.space.n
        psi = np.zeros(ext_dim * int_dim, dtype=complex)

        int_basis = list(product(range(d_int), repeat=self.space.n))
        int_index = {state: idx for idx, state in enumerate(int_basis)}

        for perm in permutations(range(self.space.n)):
            ext_state = tuple(ext_modes[perm[k]] for k in range(self.space.n))
            ext_idx = self.space.state_to_index[ext_state]

            amp_int = None
            for k in range(self.space.n):
                ket = internal_kets[perm[k]]
                amp_int = ket if amp_int is None else np.kron(amp_int, ket)

            psi[ext_idx * int_dim : (ext_idx + 1) * int_dim] += amp_int

        norm = np.linalg.norm(psi)
        if norm < TOL:
            raise ValueError("Constructed symmetrized state has zero norm.")
        psi /= norm

        rho_tot = np.outer(psi, psi.conj()).reshape(ext_dim, int_dim, ext_dim, int_dim)
        rho_ext = np.trace(rho_tot, axis1=1, axis2=3)
        return normalize_density(rho_ext)
