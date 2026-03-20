import numpy as np

from photonic_jordan import PhotonicSystem


def test_save_load_schur_cache_roundtrip(tmp_path):
    path = tmp_path / "schur_cache_m2_n3.npz"

    sys1 = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(301), auto_cache=False)
    W1 = sys1.decomposition.schur_transform()
    Q1 = sys1.sector_projector((2, 1))
    Qa1 = sys1.multiplicity_projector((2, 1), 0)
    sys1.save_schur_cache(path, include_multiplicity=True)

    sys2 = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(302), auto_cache=False)
    sys2.load_schur_cache(path, load_multiplicity=True)

    W2 = sys2.decomposition.schur_transform()
    Q2 = sys2.sector_projector((2, 1))
    Qa2 = sys2.multiplicity_projector((2, 1), 0)

    assert np.allclose(W1, W2, atol=1e-10)
    assert np.allclose(Q1, Q2, atol=1e-10)
    assert np.allclose(Qa1, Qa2, atol=1e-10)


def test_auto_schur_cache_writes_and_loads(tmp_path):
    sys1 = PhotonicSystem(
        m_ext=2,
        n_particles=3,
        rng=np.random.default_rng(303),
        auto_cache=True,
        cache_dir=tmp_path,
    )
    rho = sys1.state.from_modes_and_gram([0, 1, 0], gram=0.5)
    _ = rho.density_matrix(rep="schur")

    cache_path = sys1.default_schur_cache_path()
    assert cache_path.exists()

    sys2 = PhotonicSystem(
        m_ext=2,
        n_particles=3,
        rng=np.random.default_rng(304),
        auto_cache=True,
        cache_dir=tmp_path,
    )
    assert sys2._schur_cache_loaded

    Q1 = sys1.sector_projector((2, 1))
    Q2 = sys2.sector_projector((2, 1))
    assert np.allclose(Q1, Q2, atol=1e-10)
