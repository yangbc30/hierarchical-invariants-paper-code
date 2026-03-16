import numpy as np
import scipy.linalg as la

from photonic_jordan import PhotonicSystem, safe_matmul


def _scope_weight(sys: PhotonicSystem, rho: np.ndarray, sector=None, multiplicity=None) -> float:
    scoped = sys.project_density_to_scope(rho, sector=sector, multiplicity=multiplicity)
    return float(np.real(np.trace(scoped)))


def test_total_matrix_matches_direct_kron_lift():
    sys = PhotonicSystem(m_ext=3, n_particles=3, rng=np.random.default_rng(101))

    X = sys.rng.normal(size=(3, 3)) + 1j * sys.rng.normal(size=(3, 3))
    A = 0.5 * (X + X.conj().T)

    obs = sys.observable.from_matrix(A, name="random-hermitian")
    total_generator = obs.total_matrix(copy=False)

    total_direct = np.zeros_like(total_generator)
    for slot in range(sys.spec.n_particles):
        total_direct += sys.space.lift_one_body_operator(A, slot)

    assert np.allclose(total_generator, total_direct, atol=1e-8)


def test_two_particle_sigma_z_matrix_example():
    sys = PhotonicSystem(m_ext=2, n_particles=2, rng=np.random.default_rng(102))
    obs = sys.observable.sigma_z(modes=(0, 1))
    O = obs.total_matrix(copy=False)

    expected = np.diag([2.0, 0.0, 0.0, -2.0]).astype(complex)
    assert np.allclose(O, expected, atol=1e-8)


def test_expectation_matches_direct_trace():
    sys = PhotonicSystem(m_ext=3, n_particles=2, rng=np.random.default_rng(103))
    rho = sys.state.random_density()

    A = np.array(
        [
            [1.0, 0.2 + 0.1j, -0.3j],
            [0.2 - 0.1j, -0.5, 0.4],
            [0.3j, 0.4, 0.1],
        ],
        dtype=complex,
    )
    A = 0.5 * (A + A.conj().T)
    obs = sys.observable.from_matrix(A, name="A")

    measured = rho.measure.expectation(obs)
    direct = float(np.real(np.trace(safe_matmul(rho.matrix, obs.total_matrix(copy=False)))))
    assert abs(measured - direct) < 1e-8


def test_variance_nonnegative_up_to_tolerance():
    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(104))
    observables = [
        sys.observable.number(0),
        sys.observable.sigma_x((0, 1)),
        sys.observable.sigma_y((0, 1)),
        sys.observable.sigma_z((0, 1)),
    ]

    for _ in range(6):
        rho = sys.state.random_density()
        for obs in observables:
            var = rho.measure.variance(obs)
            assert var >= -1e-8


def test_distribution_normalization_global_and_scoped():
    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(105))
    rho = sys.state.random_density()
    obs = sys.observable.sigma_z((0, 1))

    dist_global = rho.measure.distribution(obs)
    assert abs(float(np.sum(dist_global.probabilities)) - 1.0) < 1e-8

    lam = (2, 1)
    dist_sector_raw = rho.measure.distribution(obs, sector=lam, conditional=False)
    w_sector = _scope_weight(sys, rho.matrix, sector=lam)
    assert abs(float(np.sum(dist_sector_raw.probabilities)) - w_sector) < 1e-8

    if w_sector > 1e-12:
        dist_sector_cond = rho.measure.distribution(obs, sector=lam, conditional=True)
        assert abs(float(np.sum(dist_sector_cond.probabilities)) - 1.0) < 1e-8

    scope = (lam, 0)
    dist_mult_raw = rho.measure.distribution(obs, multiplicity=scope, conditional=False)
    w_mult = _scope_weight(sys, rho.matrix, multiplicity=scope)
    assert abs(float(np.sum(dist_mult_raw.probabilities)) - w_mult) < 1e-8

    if w_mult > 1e-12:
        dist_mult_cond = rho.measure.distribution(obs, multiplicity=scope, conditional=True)
        assert abs(float(np.sum(dist_mult_cond.probabilities)) - 1.0) < 1e-8


def test_observable_commutes_with_sector_and_multiplicity_projectors():
    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(106))
    observables = [
        sys.observable.number(0),
        sys.observable.sigma_x((0, 1)),
        sys.observable.sigma_y((0, 1)),
        sys.observable.sigma_z((0, 1)),
    ]

    for obs in observables:
        O = obs.total_matrix(copy=False)
        for lam in sys.available_partitions():
            Q = sys.sector_projector(lam)
            assert la.norm(safe_matmul(O, Q) - safe_matmul(Q, O)) < 1e-8

            fam = sys.decomposition.multiplicity_projectors(lam)
            for Qa in fam:
                assert la.norm(safe_matmul(O, Qa) - safe_matmul(Qa, O)) < 1e-8


def test_observable_state_dual_api_consistency():
    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(107))
    rho = sys.state.from_modes_and_gram([0, 1, 0], gram=0.5)
    obs = sys.observable.sigma_x((0, 1))

    e1 = obs.expectation(rho)
    e2 = rho.measure.expectation(obs)
    assert abs(e1 - e2) < 1e-10

    d1 = obs.distribution(rho)
    d2 = rho.measure.distribution(obs)
    assert np.allclose(d1.eigenvalues, d2.eigenvalues, atol=1e-10)
    assert np.allclose(d1.probabilities, d2.probabilities, atol=1e-10)


def test_sampling_for_known_eigenstate_is_deterministic():
    sys = PhotonicSystem(m_ext=2, n_particles=2, rng=np.random.default_rng(108))

    rho00 = np.zeros((sys.hilbert_dim, sys.hilbert_dim), dtype=complex)
    rho00[0, 0] = 1.0
    rho = sys.state.from_density_matrix(rho00)

    obs = sys.observable.sigma_z((0, 1))

    one = rho.measure.sample(obs, shots=1, rng=np.random.default_rng(1))
    assert np.isscalar(one)
    assert abs(float(one) - 2.0) < 1e-12

    many = rho.measure.sample(obs, shots=128, rng=np.random.default_rng(2))
    assert many.shape == (128,)
    assert np.allclose(many, 2.0)
