import numpy as np
import pytest
import scipy.linalg as la

from photonic_jordan.api import PhotonicSystem
from photonic_jordan.core import safe_matmul


def test_scope_conflict_raises_error():
    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(10))
    rho = sys.state.random_sector((2, 1))
    with pytest.raises(ValueError):
        _ = rho.invariant.I_exact(1, sector=(2, 1), multiplicity=((2, 1), 0))


def test_sector_local_jordan_absorption_and_invariance():
    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(11))
    lam = (2, 1)
    rho = sys.state.random_sector(lam)

    proj = rho.project_jordan(2, kind="exact", sector=lam)
    Q = sys.sector_projector(lam)
    assert la.norm(safe_matmul(Q, proj.matrix, Q) - proj.matrix) < 1e-8

    S = sys.unitary.haar(seed=123)
    rho_out = rho.evolve(S)

    for order in range(3):
        in_exact = rho.invariant.I_exact(order, sector=lam)
        out_exact = rho_out.invariant.I_exact(order, sector=lam)
        assert abs(in_exact - out_exact) < 1e-8

        in_cum = rho.invariant.I_cumulative(order, sector=lam)
        out_cum = rho_out.invariant.I_cumulative(order, sector=lam)
        assert abs(in_cum - out_cum) < 1e-8


def test_multiplicity_local_jordan_absorption_and_invariance_basis_dependent():
    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(12))
    multiplicity_scope = ((2, 1), 0)
    rho = sys.state.random_sector((2, 1)).project_multiplicity(*multiplicity_scope)

    proj = rho.project_jordan(2, kind="cumulative", multiplicity=multiplicity_scope)
    Qa = sys.multiplicity_projector(*multiplicity_scope)
    assert la.norm(safe_matmul(Qa, proj.matrix, Qa) - proj.matrix) < 1e-8

    S = sys.unitary.haar(seed=456)
    rho_out = rho.evolve(S)

    for order in range(3):
        in_exact = rho.invariant.I_exact(order, multiplicity=multiplicity_scope)
        out_exact = rho_out.invariant.I_exact(order, multiplicity=multiplicity_scope)
        assert abs(in_exact - out_exact) < 1e-8

        in_cum = rho.invariant.I_cumulative(order, multiplicity=multiplicity_scope)
        out_cum = rho_out.invariant.I_cumulative(order, multiplicity=multiplicity_scope)
        assert abs(in_cum - out_cum) < 1e-8

    report = rho_out.analyze(max_order=2, include_sectors=True, multiplicity=multiplicity_scope)
    assert report.sector_weights is None
