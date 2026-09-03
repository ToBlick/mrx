"""``MatrixFreeExtraction._apply``: the fused COO apply against its triplets.

The extraction operator ``E`` is applied everywhere in the sequence -- every
mass apply is ``E C E^T`` -- and it is applied as a gather followed by a
``jax.ops.segment_sum``. Those were two eager device calls until
:func:`mrx.extraction_operators._apply_coo` compiled them together, which on
a TPU turned a flat 1.93 ms per apply into something that scales with the
non-zero count. That change has no test of its own, and a fused scatter is
exactly the kind of thing that can silently drop duplicate contributions.

So the oracle here is not another matrix-free apply: it is the dense matrix
assembled from the operator's own ``(rows, cols, vals)`` with ``np.add.at``,
which sums repeated ``(row, col)`` entries by construction. Agreement means
the compiled program implements the COO semantics the class documents. The
accumulation is genuinely exercised: on the ``(4, 6, 4)`` p=2 torus the
polar k=0 operator scatters 160 weighted values into 36 rows, so most rows
are a sum of several contributions rather than a relabelled copy.
"""
import numpy as np
import pytest

import mrx
from mrx.operators import extraction

from test.dense import dense_from_apply


def _dense_oracle(e):
    """Assemble ``E`` densely on the host from its COO triplets."""
    rows = np.asarray(e.rows)
    cols = np.asarray(e.cols)
    vals = np.asarray(e.vals, dtype=np.float64)
    dense = np.zeros(e.forward_shape, dtype=np.float64)
    np.add.at(dense, (rows, cols), vals)
    return dense


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("dirichlet", [False, True])
def test_extraction_apply_matches_its_coo_triplets(tiny_seq, k, dirichlet):
    """``E`` and ``E^T`` densify to the triplets' matrix and its transpose."""
    e = extraction(tiny_seq, k, dirichlet)
    oracle = _dense_oracle(e)
    n_row, n_col = e.forward_shape

    got = dense_from_apply(e._apply, n_col)
    assert got.shape == (n_row, n_col)
    # The apply is a weighted gather and a sum of at most a few terms per
    # row, so the error is a handful of eps times the entry scale.
    scale = max(1.0, np.abs(oracle).max())
    assert np.abs(got - oracle).max() < mrx.eps(1e3) * scale

    got_T = dense_from_apply(e.T._apply, n_row)
    assert got_T.shape == (n_col, n_row)
    assert np.abs(got_T - oracle.T).max() < mrx.eps(1e3) * scale


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_extraction_transpose_is_the_adjoint(tiny_seq, k):
    """``<E x, y> == <x, E^T y>``, the property every solve relies on.

    Densifying cannot catch an apply that is wrong in a way its own
    transpose repeats; this pairs the two orientations against each other on
    random vectors instead of on unit ones.
    """
    import jax
    import jax.numpy as jnp

    e = extraction(tiny_seq, k, True)
    n_row, n_col = e.forward_shape
    key = jax.random.PRNGKey(k)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (n_col,), dtype=mrx.DTYPE)
    y = jax.random.normal(ky, (n_row,), dtype=mrx.DTYPE)

    lhs = float(jnp.dot(e._apply(x), y))
    rhs = float(jnp.dot(x, e.T._apply(y)))
    scale = max(abs(lhs), abs(rhs), 1.0)
    assert abs(lhs - rhs) < mrx.eps(1e3) * scale


@pytest.mark.parametrize("k", [1, 2])
def test_extraction_apply_handles_a_matrix_argument(tiny_seq, k):
    """The apply broadcasts over columns, which ``_apply_coo`` special-cases.

    ``vals`` is reshaped to ``vals[:, None]`` for a 2-D argument, so the
    two-dimensional path is a separate branch of the compiled program and
    has to agree with applying to each column on its own.
    """
    import jax
    import jax.numpy as jnp

    e = extraction(tiny_seq, k, True)
    n_row, n_col = e.forward_shape
    x = jax.random.normal(jax.random.PRNGKey(7 + k), (n_col, 3), dtype=mrx.DTYPE)

    block = np.asarray(e._apply(x))
    assert block.shape == (n_row, 3)
    for j in range(3):
        column = np.asarray(e._apply(jnp.asarray(x[:, j])))
        scale = max(1.0, np.abs(column).max())
        assert np.abs(block[:, j] - column).max() < mrx.eps(1e2) * scale
