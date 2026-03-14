from itertools import permutations

import numpy as np
import scipy.linalg as la

from photonic_jordan import PhotonicSystem


def _safe_matmul(*ops: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = ops[0]
        for op in ops[1:]:
            out = out @ op
    return out


def test_sector_projectors_and_schur_transform_unitary():
    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(1))
    decomp = sys.decomposition

    dim = sys.hilbert_dim
    W = decomp.schur_transform()
    assert W.shape == (dim, dim)
    assert np.allclose(_safe_matmul(W.conj().T, W), np.eye(dim), atol=1e-8)

    total = np.zeros((dim, dim), dtype=complex)
    for lam in decomp.partitions():
        Q = decomp.sector_projector(lam)
        assert np.allclose(Q, Q.conj().T, atol=1e-8)
        assert np.allclose(_safe_matmul(Q, Q), Q, atol=1e-8)
        total += Q

    assert np.allclose(total, np.eye(dim), atol=1e-8)


def test_multiplicity_projectors_resolve_sector_and_commute_generators():
    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(2))
    decomp = sys.decomposition

    target_lam = None
    for lam in decomp.partitions():
        if decomp.dim_mult(lam) > 1:
            target_lam = lam
            break

    assert target_lam is not None

    fam = decomp.multiplicity_projectors(target_lam)
    d_mult = decomp.dim_mult(target_lam)
    assert len(fam) == d_mult

    Q_lam = decomp.sector_projector(target_lam)
    total = np.zeros_like(Q_lam)

    for Qa in fam:
        assert np.allclose(Qa, Qa.conj().T, atol=1e-7)
        assert np.allclose(_safe_matmul(Qa, Qa), Qa, atol=1e-7)
        total += Qa

        for G in sys.space.generators.values():
            assert la.norm(_safe_matmul(Qa, G) - _safe_matmul(G, Qa)) < 1e-5

    assert np.allclose(total, Q_lam, atol=1e-7)

    for i in range(len(fam)):
        for j in range(i + 1, len(fam)):
            assert la.norm(_safe_matmul(fam[i], fam[j])) < 1e-6


def test_state_representation_access_blocks_and_commutant_sampling():
    sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(3))

    rho = sys.state.from_modes_and_gram([0, 1, 0], gram="indistinguishable")
    rho_t = rho.density_matrix(rep="tensor")
    rho_s = rho.density_matrix(rep="schur")
    rho_back = sys.decomposition.to_tensor_operator(rho_s)
    assert np.allclose(rho_back, rho_t, atol=1e-8)

    blocks = rho.blocks()
    assert set(blocks.keys()) == set(sys.available_partitions())
    for lam, state_lam in blocks.items():
        Q = sys.sector_projector(lam)
        assert np.allclose(state_lam.matrix, _safe_matmul(Q, rho.matrix, Q), atol=1e-8)

    lam_mult = None
    for lam in sys.available_partitions():
        if sys.decomposition.dim_mult(lam) > 1:
            lam_mult = lam
            break
    assert lam_mult is not None

    multiplicity_state = rho.multiplicity_block(lam_mult, 0)
    Qa = sys.multiplicity_projector(lam_mult, 0)
    assert np.allclose(multiplicity_state.matrix, _safe_matmul(Qa, rho.matrix, Qa), atol=1e-8)

    rho_comm = sys.state.random_commutant_state()
    for perm in permutations(range(sys.spec.n_particles)):
        P = sys.projectors.permutation_matrix(perm)
        assert la.norm(_safe_matmul(P, rho_comm.matrix) - _safe_matmul(rho_comm.matrix, P)) < 1e-8
