import numpy as np

from photonic_jordan import Fock


def test_spagnolo_tritter_three_photon_bosonic_coalescence():
    """Validate the balanced-tritter suppression rule from Spagnolo et al.

    Reference
    ---------
    N. Spagnolo et al., "Three-photon bosonic coalescence in an integrated
    tritter", Nature Communications 4, 1606 (2013).

    For one indistinguishable photon in each input mode of a balanced tritter,
    all output patterns with occupation type (2,1,0) are suppressed.
    """
    omega = np.exp(2j * np.pi / 3.0)
    U = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, omega, omega**2],
            [1.0, omega**2, omega],
        ],
        dtype=complex,
    ) / np.sqrt(3.0)

    rho = Fock(1, 1, 1, auto_cache=False)
    rho_out = rho.evolve(U)

    dist = rho_out.pattern_distribution()

    for occ in ((2, 1, 0), (2, 0, 1), (1, 2, 0), (0, 2, 1), (1, 0, 2), (0, 1, 2)):
        assert abs(dist[occ]) < 1e-12

    assert abs(dist[(1, 1, 1)] - (1.0 / 3.0)) < 1e-10
    assert abs(dist[(3, 0, 0)] - (2.0 / 9.0)) < 1e-10
    assert abs(dist[(0, 3, 0)] - (2.0 / 9.0)) < 1e-10
    assert abs(dist[(0, 0, 3)] - (2.0 / 9.0)) < 1e-10
    assert abs(sum(dist.values()) - 1.0) < 1e-10
