from itertools import permutations

import numpy as np

from photonic_jordan import PhotonicSystem


def _accessible_density_from_symmetrized_products(ext_kets, int_kets):
    n = len(ext_kets)
    if n == 0 or len(int_kets) != n:
        raise ValueError("ext_kets and int_kets must be non-empty and have the same length.")

    m = int(np.asarray(ext_kets[0]).shape[0])
    d = int(np.asarray(int_kets[0]).shape[0])
    ext_dim = m**n
    int_dim = d**n

    psi = np.zeros(ext_dim * int_dim, dtype=complex)
    for perm in permutations(range(n)):
        ext_amp = np.asarray(ext_kets[perm[0]], dtype=complex)
        int_amp = np.asarray(int_kets[perm[0]], dtype=complex)
        for k in range(1, n):
            ext_amp = np.kron(ext_amp, np.asarray(ext_kets[perm[k]], dtype=complex))
            int_amp = np.kron(int_amp, np.asarray(int_kets[perm[k]], dtype=complex))
        psi += np.kron(ext_amp, int_amp)

    psi /= np.linalg.norm(psi)
    rho_tot = np.outer(psi, psi.conj()).reshape(ext_dim, int_dim, ext_dim, int_dim)
    rho_ext = np.trace(rho_tot, axis1=1, axis2=3)
    return rho_ext / np.trace(rho_ext)


def _sorted_nonzero_eigenvalues(op: np.ndarray, tol: float = 1e-10):
    vals = np.linalg.eigvalsh((op + op.conj().T) / 2.0)
    vals = np.real(vals)
    vals[np.abs(vals) < tol] = 0.0
    return np.sort(vals[vals > tol])


def test_adamson_single_term_matches_sector_weights_and_pattern_probabilities():
    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(301), auto_cache=False)

    H = np.array([1.0, 0.0], dtype=complex)
    V = np.array([0.0, 1.0], dtype=complex)
    a = np.array([1.0, 0.0], dtype=complex)
    b = np.array([0.0, 1.0], dtype=complex)

    # Adamson et al. Eq. (28)-(29): a_H^\dagger a_V^\dagger b_V^\dagger.
    rho_ext = _accessible_density_from_symmetrized_products(
        ext_kets=[H, V, V],
        int_kets=[a, a, b],
    )
    rho = sys.state.from_density_matrix(rho_ext)

    dist = rho.pattern_distribution()
    assert abs(dist[(1, 2)] - 1.0) < 1e-12
    assert abs(dist[(3, 0)]) < 1e-12
    assert abs(dist[(2, 1)]) < 1e-12
    assert abs(dist[(0, 3)]) < 1e-12

    weights = rho.sector_weights()
    assert weights is not None
    assert abs(weights[(3,)] - (2.0 / 3.0)) < 1e-10
    assert abs(weights[(2, 1)] - (1.0 / 3.0)) < 1e-10

    rho_s = rho.density_matrix(rep="schur")
    expected_s = np.zeros((8, 8), dtype=complex)
    expected_s[2, 2] = 2.0 / 3.0
    expected_s[5, 5] = 1.0 / 6.0
    expected_s[7, 7] = 1.0 / 6.0
    assert np.allclose(rho_s, expected_s, atol=1e-10)

    rho_sym = sys.tensor_to_fock_operator(rho.matrix)
    i_12 = sys.fock_space.index_from_occupation((1, 2))  # type: ignore[union-attr]
    expected_sym = np.zeros_like(rho_sym)
    expected_sym[i_12, i_12] = 2.0 / 3.0
    assert np.allclose(rho_sym, expected_sym, atol=1e-10)

    block_21 = sys.decomposition.sector_blocks(rho.matrix, rep="tensor")[(2, 1)]
    assert np.allclose(_sorted_nonzero_eigenvalues(block_21), np.array([1.0 / 6.0, 1.0 / 6.0]), atol=1e-10)

    for a_idx in range(2):
        rho_mult = rho.project_multiplicity((2, 1), a_idx)
        vals = _sorted_nonzero_eigenvalues(rho_mult.matrix)
        assert np.allclose(vals, np.array([1.0 / 6.0]), atol=1e-10)


def test_adamson_three_photon_state_matches_reported_schur_structure():
    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(302), auto_cache=False)

    H = np.array([1.0, 0.0], dtype=complex)
    V = np.array([0.0, 1.0], dtype=complex)
    a = np.array([1.0, 0.0], dtype=complex)
    b = np.array([0.0, 1.0], dtype=complex)
    c = (a + b) / np.sqrt(2.0)
    omega = np.exp(2j * np.pi / 3.0)

    # Adamson et al. Eq. (27) and the accessible matrix in Eq. (30).
    rho_ext = _accessible_density_from_symmetrized_products(
        ext_kets=[
            H + V,
            H + omega * V,
            H + (omega**2) * V,
        ],
        int_kets=[a, a, c],
    )
    rho = sys.state.from_density_matrix(rho_ext)

    dist = rho.pattern_distribution()
    assert abs(dist[(3, 0)] - (4.0 / 11.0)) < 1e-10
    assert abs(dist[(2, 1)] - (3.0 / 22.0)) < 1e-10
    assert abs(dist[(1, 2)] - (3.0 / 22.0)) < 1e-10
    assert abs(dist[(0, 3)] - (4.0 / 11.0)) < 1e-10
    assert abs(sum(dist.values()) - 1.0) < 1e-10

    weights = rho.sector_weights()
    assert weights is not None
    assert abs(weights[(3,)] - (8.0 / 11.0)) < 1e-10
    assert abs(weights[(2, 1)] - (3.0 / 11.0)) < 1e-10

    rho_s = rho.density_matrix(rep="schur")
    expected_s = np.zeros((8, 8), dtype=complex)
    expected_s[0, 0] = 4.0 / 11.0
    expected_s[0, 3] = 4.0 / 11.0
    expected_s[3, 0] = 4.0 / 11.0
    expected_s[3, 3] = 4.0 / 11.0

    diag_mixed = 3.0 / 44.0
    off_a = -3.0 / 88.0 + 1j * (3.0 * np.sqrt(3.0) / 88.0)
    off_b = 3.0 / 88.0 - 1j * (3.0 * np.sqrt(3.0) / 88.0)

    expected_s[4, 4] = diag_mixed
    expected_s[5, 5] = diag_mixed
    expected_s[6, 6] = diag_mixed
    expected_s[7, 7] = diag_mixed
    expected_s[4, 5] = off_a
    expected_s[5, 4] = np.conj(off_a)
    expected_s[6, 7] = off_b
    expected_s[7, 6] = np.conj(off_b)

    assert np.allclose(rho_s, expected_s, atol=1e-10)

    rho_sym = sys.tensor_to_fock_operator(rho.matrix)
    i_30 = sys.fock_space.index_from_occupation((3, 0))  # type: ignore[union-attr]
    i_03 = sys.fock_space.index_from_occupation((0, 3))  # type: ignore[union-attr]
    expected_sym = np.zeros_like(rho_sym)
    expected_sym[i_30, i_30] = 4.0 / 11.0
    expected_sym[i_03, i_03] = 4.0 / 11.0
    expected_sym[i_30, i_03] = 4.0 / 11.0
    expected_sym[i_03, i_30] = 4.0 / 11.0
    assert np.allclose(rho_sym, expected_sym, atol=1e-10)

    block_21 = sys.decomposition.sector_blocks(rho.matrix, rep="tensor")[(2, 1)]
    assert np.allclose(
        _sorted_nonzero_eigenvalues(block_21),
        np.array([3.0 / 22.0, 3.0 / 22.0]),
        atol=1e-10,
    )

    for a_idx in range(2):
        rho_mult = rho.project_multiplicity((2, 1), a_idx)
        vals = _sorted_nonzero_eigenvalues(rho_mult.matrix)
        assert np.allclose(vals, np.array([3.0 / 22.0]), atol=1e-10)
        assert abs(float(np.real(np.trace(rho_mult.matrix))) - (3.0 / 22.0)) < 1e-10
