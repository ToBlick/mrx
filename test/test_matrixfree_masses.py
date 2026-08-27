"""Matrix-free mass and projection applies against a quadrature oracle.

``mrx.mass`` applies every mass-like operator by sum factorisation
with the metric weight formed from ``DF`` and ``det DF`` inside the kernel.
Here the same matrices are built by brute force from the 1-D basis tables at
the quadrature points (``test.dense.dense_mixed_mass``) on a tiny polar
rotating-ellipse sequence, and the applies must agree to roundoff.
"""

import numpy as np
import numpy.testing as npt
import pytest

import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.mass import (build_matrixfree_mass_apply,
                                build_matrixfree_projection_apply)
from mrx.mappings import rotating_ellipse_map
from mrx.operators import (_PROJECTION_SPACES, apply_projection_matrix,
                           new_operators)
from test.dense import dense_from_apply, dense_mixed_mass

# Sum factorisation against the brute-force quadrature sum, relative to the
# largest entry: 1e3 eps = 2.2e-13 f64 / 1.2e-4 f32.
ATOL = mrx.eps(1e3)


@pytest.fixture(scope="module")
def mf_seq():
    """(4, 6, 4) p=2 polar rotating ellipse with its incidence operators."""
    seq = DeRhamSequence((4, 6, 4), (2, 2, 2), 3, ("clamped", "periodic", "periodic"),
                         polar=True)
    seq.set_map(rotating_ellipse_map(eps=0.33, kappa=1.2, R0=1.0, nfp=3))
    return seq, new_operators(seq)


def _n_raw(seq, k):
    return sum(int(np.prod(s)) for s in getattr(seq, f"basis_{k}").shape)


def _mass_weight(seq, k):
    G = seq.geometry
    J = np.asarray(G.jacobian_j)
    if k == 0:
        return J
    if k == 3:
        return 1.0 / J
    if k == 1:
        return np.asarray(G.metric_inv_jkl) * J[:, None, None]
    return np.asarray(G.metric_jkl) / J[:, None, None]


@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_mass_apply_matches_quadrature_oracle(mf_seq, k):
    """``M_k x`` from the in-kernel ``DF``-built weight equals the brute-force matrix."""
    seq, _ = mf_seq
    dense = dense_from_apply(build_matrixfree_mass_apply(seq, k), _n_raw(seq, k))
    oracle = dense_mixed_mass(seq, k, k, _mass_weight(seq, k))
    npt.assert_allclose(dense, oracle, rtol=0.0, atol=ATOL * np.abs(oracle).max())


@pytest.mark.parametrize("pair", sorted(_PROJECTION_SPACES))
def test_projection_apply_matches_quadrature_oracle(mf_seq, pair):
    """The raw projection mass ``int Lambda^row . Lambda^col`` equals the brute-force matrix."""
    seq, _ = mf_seq
    k_row, k_col = _PROJECTION_SPACES[pair]
    dense = dense_from_apply(
        build_matrixfree_projection_apply(seq, k_row, k_col), _n_raw(seq, k_col))
    n_comp = 1 if k_row in (0, 3) else 3
    n_q = seq.geometry.jacobian_j.shape[0]
    weight = np.ones(n_q) if n_comp == 1 else np.broadcast_to(np.eye(3), (n_q, 3, 3))
    oracle = dense_mixed_mass(seq, k_row, k_col, weight)
    npt.assert_allclose(dense, oracle, rtol=0.0, atol=ATOL * np.abs(oracle).max())


@pytest.mark.parametrize("pair,partner", (((2, 1), (1, 2)), ((0, 3), (3, 0))))
@pytest.mark.parametrize("dirichlet", (False, True))
def test_extracted_projection_pairs_are_transposes(mf_seq, pair, partner, dirichlet):
    """``P_12 = P_21^T`` and ``P_30 = P_03^T`` on the extracted spaces."""
    seq, ops = mf_seq

    def dense(p):
        k_in = {(2, 1): 2, (1, 2): 1, (0, 3): 3, (3, 0): 0}[p]
        n = int(seq.n(k_in, dirichlet))
        return dense_from_apply(
            lambda v: apply_projection_matrix(seq, v, *p, dirichlet, dirichlet), n)
    a, b = dense(pair), dense(partner)
    npt.assert_allclose(a, b.T, rtol=0.0, atol=ATOL * np.abs(a).max())
