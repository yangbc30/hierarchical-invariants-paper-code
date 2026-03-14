import numpy as np

from photonic_jordan import PhotonicSystem


def test_hierarchy_dimensions_are_monotone_and_consistent():
    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(31))
    filt = sys.ensure_scope_filtration(max_order=3)

    dims = filt.dimensions_summary(3)
    cumulative_dims = [d[1] for d in dims]
    layer_dims = [d[2] for d in dims]

    assert cumulative_dims == sorted(cumulative_dims)
    for j in range(1, 4):
        assert cumulative_dims[j] == cumulative_dims[j - 1] + layer_dims[j]


def test_cumulative_weights_are_monotone_in_order():
    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(32))
    rho = sys.state.from_modes_and_gram([0, 1, 0], gram=0.4)

    vals = [rho.invariant.I_cumulative(j) for j in range(4)]
    assert vals == sorted(vals)


def test_layer_and_cumulative_projections_match_decomposition():
    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(33))
    rho = sys.state.random_density()

    proj_cum = rho.project_jordan(2, kind="cumulative").matrix
    proj_layer = rho.project_jordan(2, kind="exact").matrix
    prev_cum = rho.project_jordan(1, kind="cumulative").matrix

    assert np.allclose(proj_cum - prev_cum, proj_layer, atol=1e-8)
