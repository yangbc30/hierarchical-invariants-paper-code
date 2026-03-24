from math import comb

import numpy as np

from photonic_jordan import PhotonicSystem, safe_matmul


def test_indistinguishable_state_uses_fock_fast_path():
    sys = PhotonicSystem(m_ext=5, n_particles=5, rng=np.random.default_rng(201))
    rho = sys.state.from_modes_and_gram([0, 1, 2, 3, 4], gram=1)

    assert rho.has_rep("fock")
    assert not rho.has_rep("tensor")
    expected_dim = comb(5 + 5 - 1, 5)
    assert rho.density_matrix(rep="fock").shape == (expected_dim, expected_dim)
    assert abs(float(np.real(np.trace(rho.density_matrix(rep="fock", copy=False)))) - 1.0) < 1e-12


def test_fock_tensor_embedding_matches_reference_tensor_construction():
    sys = PhotonicSystem(m_ext=3, n_particles=3, rng=np.random.default_rng(202))
    ext = [0, 1, 0]
    gram = np.ones((3, 3), dtype=complex)

    rho_fast = sys.state.from_modes_and_gram(ext, gram=gram)
    rho_ref_tensor = sys._state_factory.from_external_modes_and_gram(ext_modes=ext, gram=gram)

    assert np.allclose(rho_fast.density_matrix(rep="tensor", copy=False), rho_ref_tensor, atol=1e-10)


def test_fock_observable_path_matches_tensor_path():
    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(203))
    ext = [0, 1, 0]
    gram = np.ones((3, 3), dtype=complex)

    rho_fast = sys.state.from_modes_and_gram(ext, gram=gram)
    rho_ref_tensor = sys._state_factory.from_external_modes_and_gram(ext_modes=ext, gram=gram)
    rho_ref = sys.state.from_density_matrix(rho_ref_tensor)

    obs = sys.observable.sigma_z((0, 1))

    e_fast = rho_fast.measure.expectation(obs)
    e_ref = rho_ref.measure.expectation(obs)
    assert abs(e_fast - e_ref) < 1e-10

    O_tensor = obs.total_matrix(copy=False, rep="tensor")
    O_fock = obs.total_matrix(copy=False, rep="fock")
    V = sys.fock_space.isometry_to_tensor  # type: ignore[union-attr]
    O_projected = safe_matmul(V.conj().T, O_tensor, V)
    assert np.allclose(O_fock, O_projected, atol=1e-10)


def test_fock_global_invariants_match_symmetric_sector_local_invariants():
    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(204))
    ext = [0, 1, 0]
    gram = np.ones((3, 3), dtype=complex)

    rho_fast = sys.state.from_modes_and_gram(ext, gram=gram)
    rho_ref_tensor = sys._state_factory.from_external_modes_and_gram(ext_modes=ext, gram=gram)
    rho_ref = sys.state.from_density_matrix(rho_ref_tensor)

    lam_sym = (3,)
    for j in range(4):
        i_exact_fast = rho_fast.invariant.I_exact(j)
        i_exact_sector = rho_ref.invariant.I_exact(j, sector=lam_sym)
        i_cum_fast = rho_fast.invariant.I_cumulative(j)
        i_cum_sector = rho_ref.invariant.I_cumulative(j, sector=lam_sym)
        assert abs(i_exact_fast - i_exact_sector) < 1e-8
        assert abs(i_cum_fast - i_cum_sector) < 1e-8


def test_native_fock_lifted_unitary_matches_tensor_projection():
    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(205))
    S = sys.unitary.haar(seed=42)

    U_fock = sys.total_unitary_fock_from_single_particle(S)
    U_tensor = sys.space.total_unitary_from_single_particle(S)
    V = sys.fock_space.isometry_to_tensor  # type: ignore[union-attr]
    U_proj = safe_matmul(V.conj().T, U_tensor, V)

    assert np.allclose(U_fock, U_proj, atol=1e-10)
    assert np.allclose(U_fock.conj().T @ U_fock, np.eye(U_fock.shape[0]), atol=1e-10)


def test_state_evolve_uses_native_fock_path_and_matches_tensor_result():
    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(206))
    ext = [0, 1, 0]
    gram = np.ones((3, 3), dtype=complex)
    S = sys.unitary.haar(seed=17)

    rho_fast = sys.state.from_modes_and_gram(ext, gram=gram)
    rho_ref_tensor = sys._state_factory.from_external_modes_and_gram(ext_modes=ext, gram=gram)
    rho_ref = sys.state.from_density_matrix(rho_ref_tensor)

    out_fast = rho_fast.evolve(S)
    out_ref = rho_ref.evolve(S)

    assert out_fast.has_rep("fock")
    assert not out_fast.has_rep("tensor")
    assert np.allclose(out_fast.density_matrix(rep="tensor", copy=False), out_ref.matrix, atol=1e-10)


def test_system_evolve_density_fock_matches_state_evolve():
    sys = PhotonicSystem(m_ext=3, n_particles=3, rng=np.random.default_rng(207))
    rho = sys.state.from_modes_and_gram([0, 1, 2], gram=1)
    S = sys.unitary.haar(seed=5)

    rho_out_state = rho.evolve(S).density_matrix(rep="fock", copy=False)
    rho_out_system = sys.evolve_density_fock(rho.density_matrix(rep="fock", copy=False), S)

    assert np.allclose(rho_out_state, rho_out_system, atol=1e-10)


def test_state_builder_from_fock_mixture():
    sys = PhotonicSystem(m_ext=3, n_particles=3, rng=np.random.default_rng(216), auto_cache=False)
    rho = sys.state.from_fock_mixture(
        occupations=[(2, 1, 0), (1, 2, 0)],
        weights=[0.25, 0.75],
    )

    rho_f = rho.density_matrix(rep="fock", copy=False)
    fs = sys.fock_space
    assert fs is not None
    i1 = fs.index_from_occupation((2, 1, 0))
    i2 = fs.index_from_occupation((1, 2, 0))
    assert abs(float(np.real(rho_f[i1, i1])) - 0.25) < 1e-12
    assert abs(float(np.real(rho_f[i2, i2])) - 0.75) < 1e-12


def test_pattern_distribution_normalizes_for_partial_distinguishability_state():
    sys = PhotonicSystem(m_ext=2, n_particles=2, rng=np.random.default_rng(217), auto_cache=False)
    rho = sys.state.from_modes_and_gram([0, 1], gram=0.8)

    dist = rho.pattern_distribution()
    assert abs(sum(dist.values()) - 1.0) < 1e-12
    assert abs(dist[(1, 1)] - 1.0) < 1e-12
    assert abs(dist[(2, 0)]) < 1e-12
    assert abs(dist[(0, 2)]) < 1e-12


def test_pattern_probability_matches_hom_formula_after_balanced_beamsplitter():
    sys = PhotonicSystem(m_ext=2, n_particles=2, rng=np.random.default_rng(218), auto_cache=False)
    gamma = 0.8
    rho = sys.state.from_modes_and_gram([0, 1], gram=gamma)
    S = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)

    rho_out = rho.evolve(S)

    p11 = rho_out.pattern_probability((1, 1))
    p20 = rho_out.pattern_probability((2, 0))
    p02 = rho_out.pattern_probability((0, 2))

    expected_coin = (1.0 - abs(gamma) ** 2) / 2.0
    expected_bunch = (1.0 + abs(gamma) ** 2) / 4.0

    assert abs(p11 - expected_coin) < 1e-10
    assert abs(p20 - expected_bunch) < 1e-10
    assert abs(p02 - expected_bunch) < 1e-10
    assert abs(p11 + p20 + p02 - 1.0) < 1e-10


def test_pattern_probability_fock_path_matches_tensor_reference():
    sys = PhotonicSystem(m_ext=3, n_particles=3, rng=np.random.default_rng(219), auto_cache=False)
    ext = [0, 1, 0]
    S = sys.unitary.haar(seed=31)

    rho_fast = sys.state.from_modes_and_gram(ext, gram=1)
    rho_ref_tensor = sys._state_factory.from_external_modes_and_gram(ext_modes=ext, gram=np.ones((3, 3), dtype=complex))
    rho_ref = sys.state.from_density_matrix(rho_ref_tensor)

    out_fast = rho_fast.evolve(S)
    out_ref = rho_ref.evolve(S)

    fs = sys.fock_space
    assert fs is not None
    for occ in fs.basis_states:
        p_fast = out_fast.pattern_probability(occ)
        p_ref = out_ref.pattern_probability(occ)
        assert abs(p_fast - p_ref) < 1e-10


def test_save_load_fock_cache_roundtrip(tmp_path):
    path = tmp_path / "fock_cache_m2_n3.npz"

    sys1 = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(208))
    rho1 = sys1.state.from_modes_and_gram([0, 1, 0], gram=1)
    obs1 = sys1.observable.sigma_z((0, 1))

    _ = rho1.measure.expectation(obs1)
    vals1 = [rho1.invariant.I_exact(j) for j in range(3)]
    sys1.save_fock_cache(path, max_order=2, include_generators=True)

    sys2 = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(209))
    sys2.load_fock_cache(path, load_generators=True, load_hierarchy=True)
    rho2 = sys2.state.from_modes_and_gram([0, 1, 0], gram=1)
    obs2 = sys2.observable.sigma_z((0, 1))

    vals2 = [rho2.invariant.I_exact(j) for j in range(3)]
    assert np.allclose(vals1, vals2, atol=1e-12)
    assert abs(rho1.measure.expectation(obs1) - rho2.measure.expectation(obs2)) < 1e-12
    assert sys2.fock_space is not None
    assert sys2.fock_space._generators_cache is not None
    assert sys2._fock_built_order >= 2


def test_load_fock_cache_rejects_model_mismatch(tmp_path):
    path = tmp_path / "fock_cache_m2_n3.npz"
    src = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(210))
    src.save_fock_cache(path, max_order=1, include_generators=False)

    dst = PhotonicSystem(m_ext=3, n_particles=3, rng=np.random.default_rng(211))
    try:
        dst.load_fock_cache(path)
        assert False, "Expected model-mismatch error when loading incompatible cache."
    except ValueError as exc:
        assert "mismatch" in str(exc).lower()


def test_auto_cache_writes_default_model_file_and_autoloads(tmp_path):
    ext = [0, 1, 0]

    sys1 = PhotonicSystem(
        m_ext=2,
        n_particles=3,
        rng=np.random.default_rng(212),
        auto_cache=True,
        cache_dir=tmp_path,
    )
    rho1 = sys1.state.from_modes_and_gram(ext, gram=1)
    vals1 = [rho1.invariant.I_exact(j) for j in range(3)]

    cache_path = sys1.default_fock_cache_path()
    assert cache_path.exists()

    sys2 = PhotonicSystem(
        m_ext=2,
        n_particles=3,
        rng=np.random.default_rng(213),
        auto_cache=True,
        cache_dir=tmp_path,
    )
    assert sys2._fock_built_order >= 2
    rho2 = sys2.state.from_modes_and_gram(ext, gram=1)
    vals2 = [rho2.invariant.I_exact(j) for j in range(3)]
    assert np.allclose(vals1, vals2, atol=1e-12)


def test_auto_cache_disabled_does_not_write_files(tmp_path):
    sys = PhotonicSystem(
        m_ext=2,
        n_particles=3,
        rng=np.random.default_rng(214),
        auto_cache=False,
        cache_dir=tmp_path,
    )
    rho = sys.state.from_modes_and_gram([0, 1, 0], gram=1)
    _ = rho.invariant.I_exact(2)
    assert not sys.default_fock_cache_path().exists()


def test_configure_global_cache_affects_new_instances(tmp_path):
    old_enabled = PhotonicSystem._GLOBAL_AUTO_CACHE
    old_dir = PhotonicSystem._GLOBAL_CACHE_DIR
    try:
        PhotonicSystem.configure_global_cache(enabled=False, cache_dir=tmp_path)
        sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(215))
        assert sys.auto_cache is False
        assert sys.cache_dir == tmp_path
    finally:
        PhotonicSystem.configure_global_cache(enabled=old_enabled, cache_dir=old_dir)
