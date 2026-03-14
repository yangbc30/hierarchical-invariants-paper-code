"""Linear-algebra helpers used across the package.

The implementation favors numerically stable primitives (SVD/eigendecomposition)
over hand-rolled rank decisions, because Jordan-space construction is sensitive to
near-linear dependence.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.linalg as la

TOL = 1e-10


def safe_matmul(*ops: np.ndarray) -> np.ndarray:
    """Multiply matrices while suppressing floating-point warning spam.

    This is a thin convenience wrapper used heavily in tensor-space algebra.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = ops[0]
        for op in ops[1:]:
            out = out @ op
    return out


def orth_columns(mat: np.ndarray, tol: float = TOL) -> np.ndarray:
    """Return an orthonormal basis of the column span of ``mat``.

    Method:
    - computes a compact SVD ``mat = U diag(s) V^†``;
    - keeps singular vectors above an adaptive cutoff ``max(tol, s_max*tol)``.

    This is robust for the rank-deficient complex matrices generated in
    Jordan-hierarchy builds.
    """
    if mat.size == 0:
        return np.zeros((mat.shape[0], 0), dtype=complex)
    u, s, _ = la.svd(mat, full_matrices=False, check_finite=False)
    if s.size == 0:
        return np.zeros((mat.shape[0], 0), dtype=complex)
    cutoff = max(tol, float(s[0]) * tol)
    keep = s > cutoff
    if not np.any(keep):
        return np.zeros((mat.shape[0], 0), dtype=complex)
    return u[:, keep]


def hs_inner(a: np.ndarray, b: np.ndarray) -> complex:
    """Hilbert-Schmidt inner product ``Tr(a^† b)``."""
    return np.trace(safe_matmul(a.conj().T, b))


def vectorize(a: np.ndarray) -> np.ndarray:
    """Column-vectorize a matrix in row-major NumPy memory convention."""
    return a.reshape(-1, 1)


def devectorize(v: np.ndarray, dim: int) -> np.ndarray:
    """Inverse of :func:`vectorize` for square ``dim x dim`` operators."""
    return v.reshape(dim, dim)


def projector_from_basis(basis: np.ndarray) -> np.ndarray:
    """Build an orthogonal projector ``P = B B^†`` from orthonormal columns ``B``."""
    if basis.size == 0:
        return np.zeros((basis.shape[0], basis.shape[0]), dtype=complex)
    return safe_matmul(basis, basis.conj().T)


def normalize_density(rho: np.ndarray) -> np.ndarray:
    """Hermitize and normalize a density operator to unit trace."""
    rho = (rho + rho.conj().T) / 2.0
    tr = np.trace(rho)
    if abs(tr) < TOL:
        raise ValueError("Density matrix has zero trace after projection.")
    return rho / tr


def random_complex_matrix(shape: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Sample a complex Ginibre matrix with i.i.d. Gaussian entries."""
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def haar_random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-random unitary using QR of a Ginibre matrix.

    References
    ----------
    - F. Mezzadri, *How to Generate Random Matrices from the Classical Compact
      Groups*, Notices of the AMS 54(5), 2007.
    """
    z = random_complex_matrix((dim, dim), rng)
    q, r = la.qr(z)
    phases = np.diag(r) / np.abs(np.diag(r))
    phases[np.abs(np.diag(r)) < TOL] = 1.0
    return safe_matmul(q, np.diag(np.conj(phases)))
