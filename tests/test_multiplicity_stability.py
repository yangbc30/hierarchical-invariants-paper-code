import numpy as np
import scipy.linalg as la

from photonic_jordan import PhotonicSystem


def _pick_multiplicity_partition(sys: PhotonicSystem):
    for lam in sys.available_partitions():
        if sys.decomposition.dim_mult(lam) > 1:
            return lam
    raise AssertionError("No multiplicity>1 partition available")


def test_multiplicity_projectors_are_deterministic_for_fixed_model():
    sys_a = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(41))
    sys_b = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(999))

    lam = _pick_multiplicity_partition(sys_a)
    fam_a = sys_a.decomposition.multiplicity_projectors(lam)
    fam_b = sys_b.decomposition.multiplicity_projectors(lam)

    assert len(fam_a) == len(fam_b)
    for Pa, Pb in zip(fam_a, fam_b):
        assert la.norm(Pa - Pb) < 1e-8


def test_multiplicity_and_copy_alias_return_same_invariant():
    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(42))
    lam = _pick_multiplicity_partition(sys)
    scope = (lam, 0)

    rho = sys.state.random_sector(lam).project_multiplicity(*scope)
    i_mult = rho.invariant.I_exact(2, multiplicity=scope)
    i_copy = rho.invariant.I_exact(2, copy=scope)
    assert abs(i_mult - i_copy) < 1e-10

    P_mult = sys.multiplicity_projector(*scope)
    P_copy = sys.copy_projector(*scope)
    assert la.norm(P_mult - P_copy) < 1e-10
