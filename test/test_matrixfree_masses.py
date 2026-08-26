"""Matrix-free mass and projection applies against a quadrature oracle.

``mrx.local_assembly`` applies every mass-like operator by sum factorisation
with the metric weight formed from ``DF`` and ``det DF`` inside the kernel.
Here the same matrices are built by brute force from the 1-D basis tables at
the quadrature points (``test.dense.dense_mixed_mass``) on a tiny polar
rotating-ellipse sequence, and the applies must agree to roundoff.
"""

import numpy as np
import numpy.testing as npt
import pytest

from mrx.derham_sequence import DeRhamSequence
from mrx.local_assembly import (build_matrixfree_mass_apply,
                                build_matrixfree_projection_apply)
from mrx.mappings import rotating_ellipse_map
from mrx.operators import (_PROJECTION_SPACES, apply_projection_matrix,
                           assemble_incidence_operators)
from test.dense import dense_from_apply, dense_mixed_mass

_SEQ = DeRhamSequence((4, 8, 4), (2, 2, 2), 3, ("clamped", "periodic", "periodic"),
                      polar=True)
_SEQ.evaluate_1d()
_SEQ.set_map(rotating_ellipse_map(eps=0.33, kappa=1.2, R0=1.0, nfp=3))
_OPS = assemble_incidence_operators(_SEQ)
_G = _SEQ.geometry


def _n_raw(k):
    return sum(int(np.prod(s)) for s in getattr(_SEQ, f"basis_{k}").shape)


def _mass_weight(k):
    J = np.asarray(_G.jacobian_j)
    if k == 0:
        return J
    if k == 3:
        return 1.0 / J
    if k == 1:
        return np.asarray(_G.metric_inv_jkl) * J[:, None, None]
    return np.asarray(_G.metric_jkl) / J[:, None, None]


@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_mass_apply_matches_quadrature_oracle(k):
    """``M_k x`` from the in-kernel ``DF``-built weight equals the brute-force matrix."""
    dense = dense_from_apply(build_matrixfree_mass_apply(_SEQ, k), _n_raw(k))
    oracle = dense_mixed_mass(_SEQ, k, k, _mass_weight(k))
    npt.assert_allclose(dense, oracle, rtol=0.0, atol=1e-13 * np.abs(oracle).max())


@pytest.mark.parametrize("pair", sorted(_PROJECTION_SPACES))
def test_projection_apply_matches_quadrature_oracle(pair):
    """The raw projection mass ``int Lambda^row . Lambda^col`` equals the brute-force matrix."""
    k_row, k_col = _PROJECTION_SPACES[pair]
    dense = dense_from_apply(
        build_matrixfree_projection_apply(_SEQ, k_row, k_col), _n_raw(k_col))
    n_comp = 1 if k_row in (0, 3) else 3
    ones = np.ones(_G.jacobian_j.shape[0])
    weight = ones if n_comp == 1 else np.broadcast_to(np.eye(3), (ones.shape[0], 3, 3))
    oracle = dense_mixed_mass(_SEQ, k_row, k_col, weight)
    npt.assert_allclose(dense, oracle, rtol=0.0, atol=1e-13 * np.abs(oracle).max())


@pytest.mark.parametrize("pair,partner", (((2, 1), (1, 2)), ((0, 3), (3, 0))))
@pytest.mark.parametrize("dirichlet", (False, True))
def test_extracted_projection_pairs_are_transposes(pair, partner, dirichlet):
    """``P_12 = P_21^T`` and ``P_30 = P_03^T`` on the extracted spaces."""
    def dense(p):
        k_in = {(2, 1): 2, (1, 2): 1, (0, 3): 3, (3, 0): 0}[p]
        n = int(getattr(_SEQ, f"n{k_in}_dbc" if dirichlet else f"n{k_in}"))
        return dense_from_apply(
            lambda v: apply_projection_matrix(_SEQ, _OPS, v, *p, dirichlet, dirichlet), n)
    a, b = dense(pair), dense(partner)
    npt.assert_allclose(a, b.T, rtol=0.0, atol=1e-13 * np.abs(a).max())
