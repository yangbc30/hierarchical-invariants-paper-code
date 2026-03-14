import numpy as np
import pytest

from photonic_jordan import PhotonicSystem, resolve_gram_input


def test_resolve_gram_shortcuts_and_scalar_overlap():
    g_ind = resolve_gram_input("indistinguishable", n_particles=3)
    g_dist = resolve_gram_input("distinguishable", n_particles=3)
    g_scalar = resolve_gram_input(0.25, n_particles=3)

    assert np.allclose(g_ind, np.ones((3, 3)))
    assert np.allclose(g_dist, np.eye(3))
    assert np.allclose(np.diag(g_scalar), np.ones(3))
    assert np.allclose(g_scalar - np.eye(3), 0.25 * (np.ones((3, 3)) - np.eye(3)))


def test_from_modes_and_gram_rejects_non_hermitian_or_non_psd():
    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(21))

    bad_non_hermitian = np.array(
        [
            [1.0, 0.2, 0.0],
            [0.0, 1.0, 0.3],
            [0.0, 0.3, 1.0],
        ],
        dtype=complex,
    )
    with pytest.raises(ValueError, match="Hermitian"):
        _ = sys.state.from_modes_and_gram([0, 1, 0], gram=bad_non_hermitian)

    bad_diag = np.array(
        [
            [1.0, 0.2, 0.1],
            [0.2, 0.9, 0.1],
            [0.1, 0.1, 1.0],
        ],
        dtype=complex,
    )
    with pytest.raises(ValueError, match=r"diag\(gram\)"):
        _ = sys.state.from_modes_and_gram([0, 1, 0], gram=bad_diag)


def test_state_is_physical_detects_negative_eigenvalue():
    sys = PhotonicSystem(m_ext=2, n_particles=2, rng=np.random.default_rng(22))
    bad = np.diag([1.2, -0.2, 0.0, 0.0]).astype(complex)
    rho = sys.state.from_density_matrix(bad)
    assert not rho.is_physical()
