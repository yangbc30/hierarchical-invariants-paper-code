import numpy as np

from photonic_jordan import (
    Fock,
    FockMixed,
    PhotonicState,
    PhotonicSystem,
    from_fock_density,
    from_modes_and_gram,
    from_occupation,
    mix_states,
    superpose,
)


def test_state_first_from_modes_and_gram_matches_system_path():
    ext_modes = [0, 1, 0]
    gram = 0.5

    rho_state_first = PhotonicState.from_modes_and_gram(
        ext_modes=ext_modes,
        gram=gram,
        m_ext=2,
        rng=np.random.default_rng(401),
        auto_cache=False,
    )

    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(401), auto_cache=False)
    rho_system_first = sys.state.from_modes_and_gram(ext_modes=ext_modes, gram=gram)

    assert np.allclose(rho_state_first.matrix, rho_system_first.matrix, atol=1e-10)
    assert rho_state_first.system.spec.m_ext == 2
    assert rho_state_first.system.spec.n_particles == 3


def test_functional_constructor_matches_classmethod():
    rho_func = from_modes_and_gram(
        [0, 1, 0],
        gram=0.5,
        m_ext=2,
        rng=np.random.default_rng(4001),
        auto_cache=False,
    )
    rho_class = PhotonicState.from_modes_and_gram(
        [0, 1, 0],
        gram=0.5,
        m_ext=2,
        rng=np.random.default_rng(4001),
        auto_cache=False,
    )
    assert np.allclose(rho_func.matrix, rho_class.matrix, atol=1e-12)


def test_state_first_from_modes_and_gram_infers_m_ext():
    rho = PhotonicState.from_modes_and_gram(
        ext_modes=[1, 3, 1],
        gram=1,
        rng=np.random.default_rng(402),
        auto_cache=False,
    )
    assert rho.system.spec.m_ext == 4
    assert rho.system.spec.n_particles == 3
    assert rho.has_rep("fock")


def test_state_first_from_occupation_and_Fock_constructor():
    occ = [2, 1, 0]
    rho_occ = PhotonicState.from_occupation(occ, rng=np.random.default_rng(403), auto_cache=False)
    rho_fock = PhotonicState.Fock(2, 1, 0, rng=np.random.default_rng(403), auto_cache=False)

    assert rho_occ.system.spec.m_ext == 3
    assert rho_occ.system.spec.n_particles == 3
    assert rho_occ.has_rep("fock")
    assert np.allclose(rho_occ.density_matrix(rep="fock"), rho_fock.density_matrix(rep="fock"), atol=1e-12)


def test_functional_fock_shortcuts():
    rho_occ = from_occupation([2, 1, 0], rng=np.random.default_rng(4002), auto_cache=False)
    rho_fock = Fock(2, 1, 0, rng=np.random.default_rng(4002), auto_cache=False)
    assert np.allclose(rho_occ.density_matrix(rep="fock"), rho_fock.density_matrix(rep="fock"), atol=1e-12)


def test_Fock_accepts_single_sequence_for_backward_compatibility():
    rho1 = PhotonicState.Fock(2, 1, 0, rng=np.random.default_rng(406), auto_cache=False)
    rho2 = PhotonicState.Fock([2, 1, 0], rng=np.random.default_rng(406), auto_cache=False)
    assert np.allclose(rho1.density_matrix(rep="fock"), rho2.density_matrix(rep="fock"), atol=1e-12)


def test_state_first_uses_provided_system():
    sys = PhotonicSystem(m_ext=3, n_particles=3, rng=np.random.default_rng(404), auto_cache=False)
    rho = PhotonicState.from_occupation([1, 1, 1], system=sys)
    assert rho.system is sys

    obs = sys.observable.sigma_z((0, 1))
    val = rho.measure.expectation(obs)
    assert np.isfinite(val)


def test_state_first_rejects_incompatible_system():
    sys = PhotonicSystem(m_ext=2, n_particles=2, rng=np.random.default_rng(405), auto_cache=False)
    try:
        _ = PhotonicState.from_modes_and_gram([0, 1, 0], gram=1, system=sys)
        assert False, "Expected mismatch error for incompatible n_particles."
    except ValueError as exc:
        assert "n_particles" in str(exc)


def test_FockMixed_builds_expected_diagonal_fock_density():
    rho = PhotonicState.FockMixed(
        (0.7, 2, 1, 0),
        (0.3, 1, 2, 0),
        auto_cache=False,
        rng=np.random.default_rng(407),
    )
    rho_f = rho.density_matrix(rep="fock", copy=False)
    assert abs(float(np.real(np.trace(rho_f))) - 1.0) < 1e-12

    fs = rho.system.fock_space
    assert fs is not None
    i1 = fs.index_from_occupation((2, 1, 0))
    i2 = fs.index_from_occupation((1, 2, 0))
    assert abs(float(np.real(rho_f[i1, i1])) - 0.7) < 1e-12
    assert abs(float(np.real(rho_f[i2, i2])) - 0.3) < 1e-12
    off = rho_f.copy()
    off[i1, i1] = 0.0
    off[i2, i2] = 0.0
    assert np.allclose(off, 0.0, atol=1e-12)


def test_functional_fock_mixed_matches_classmethod():
    rho_a = FockMixed(
        (0.7, 2, 1, 0),
        (0.3, 1, 2, 0),
        auto_cache=False,
        rng=np.random.default_rng(4003),
    )
    rho_b = PhotonicState.FockMixed(
        (0.7, 2, 1, 0),
        (0.3, 1, 2, 0),
        auto_cache=False,
        rng=np.random.default_rng(4003),
    )
    assert np.allclose(rho_a.density_matrix(rep="fock"), rho_b.density_matrix(rep="fock"), atol=1e-12)


def test_FockMixed_with_system_and_sequence_term():
    sys = PhotonicSystem(m_ext=3, n_particles=3, rng=np.random.default_rng(408), auto_cache=False)
    rho = PhotonicState.FockMixed(
        (2.0, [2, 1, 0]),
        (1.0, [1, 2, 0]),
        system=sys,
        normalize=True,
    )
    assert rho.system is sys
    dist = rho.measure.distribution(sys.observable.number(0))
    assert abs(float(np.sum(dist.probabilities)) - 1.0) < 1e-12


def test_from_fock_density_state_first():
    sys = PhotonicSystem(m_ext=3, n_particles=3, rng=np.random.default_rng(409), auto_cache=False)
    fs = sys.fock_space
    assert fs is not None

    d = fs.dim
    rho_f = np.zeros((d, d), dtype=complex)
    i1 = fs.index_from_occupation((2, 1, 0))
    i2 = fs.index_from_occupation((1, 2, 0))
    rho_f[i1, i1] = 0.4
    rho_f[i2, i2] = 0.6

    rho = PhotonicState.from_fock_density(rho_f, system=sys)
    assert rho.system is sys
    assert np.allclose(rho.density_matrix(rep="fock", copy=False), rho_f, atol=1e-12)


def test_functional_from_fock_density_matches_classmethod():
    sys = PhotonicSystem(m_ext=3, n_particles=3, rng=np.random.default_rng(4010), auto_cache=False)
    fs = sys.fock_space
    assert fs is not None

    rho_f = np.zeros((fs.dim, fs.dim), dtype=complex)
    i1 = fs.index_from_occupation((2, 1, 0))
    i2 = fs.index_from_occupation((1, 2, 0))
    rho_f[i1, i1] = 0.1
    rho_f[i2, i2] = 0.9

    rho_func = from_fock_density(rho_f, system=sys)
    rho_class = PhotonicState.from_fock_density(rho_f, system=sys)
    assert np.allclose(rho_func.density_matrix(rep="fock"), rho_class.density_matrix(rep="fock"), atol=1e-12)


def test_pairwise_mix_matches_manual_convex_sum():
    rho_a = from_modes_and_gram([0, 1, 0], gram=0.2, m_ext=2, rng=np.random.default_rng(4101), auto_cache=False)
    rho_b = from_modes_and_gram([0, 1, 0], gram=0.8, m_ext=2, rng=np.random.default_rng(4102), auto_cache=False)
    rho_m = rho_a.mix(rho_b, weight=0.3)

    manual = 0.3 * rho_a.matrix + 0.7 * rho_b.matrix
    manual = manual / np.trace(manual)
    assert np.allclose(rho_m.matrix, manual, atol=1e-10)


def test_functional_mix_states_matches_classmethod():
    rho_a = Fock(1, 0, rng=np.random.default_rng(4103), auto_cache=False)
    rho_b = Fock(0, 1, rng=np.random.default_rng(4104), auto_cache=False)
    rho_f = mix_states((1.0, rho_a), (3.0, rho_b))
    rho_c = PhotonicState.mixture((1.0, rho_a), (3.0, rho_b))
    assert np.allclose(rho_f.density_matrix(rep="fock"), rho_c.density_matrix(rep="fock"), atol=1e-12)


def test_coherent_superposition_two_pure_fock_states():
    rho0 = Fock(1, 0, rng=np.random.default_rng(4105), auto_cache=False)
    rho1 = Fock(0, 1, system=rho0.system)
    rho_plus = rho0.superpose(rho1, alpha=1.0, beta=1.0)
    mat = rho_plus.density_matrix(rep="fock")
    expected = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
    assert np.allclose(mat, expected, atol=1e-10)


def test_functional_superpose_matches_method():
    rho0 = Fock(1, 0, rng=np.random.default_rng(4106), auto_cache=False)
    rho1 = Fock(0, 1, system=rho0.system)
    rho_a = rho0.superpose(rho1, alpha=1.0, beta=1j)
    rho_b = superpose(rho0, rho1, alpha=1.0, beta=1j)
    assert np.allclose(rho_a.density_matrix(rep="fock"), rho_b.density_matrix(rep="fock"), atol=1e-12)


def test_superpose_rejects_mixed_input():
    rho_pure = Fock(1, 0, rng=np.random.default_rng(4107), auto_cache=False)
    rho_mix = FockMixed((0.5, 1, 0), (0.5, 0, 1), system=rho_pure.system)
    try:
        _ = rho_pure.superpose(rho_mix)
        assert False, "Expected ValueError for mixed-state coherent superposition."
    except ValueError as exc:
        assert "pure" in str(exc).lower() or "mixed" in str(exc).lower()
