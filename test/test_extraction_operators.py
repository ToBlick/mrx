import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy.testing as npt
import pytest

import mrx
from mrx.differential_forms import DifferentialForm
from mrx.extraction_operators import PolarExtractionOperator, get_xi

jax.config.update("jax_enable_x64", True)

NS = (10, 10, 10)
PS = (3, 3, 3)
TYPES = ("clamped", "periodic", "periodic")


@pytest.fixture(params=[0, 1, 2, 3])
def form(request):
    return DifferentialForm(request.param, NS, PS, TYPES)


@pytest.fixture
def xi():
    return get_xi(NS[1])


def _assemble_sparse_legacy(operator: PolarExtractionOperator):
    ncols = operator.Lambda.n
    nrows = operator.n
    col_indices = jnp.arange(ncols)
    max_nnz = 2 * max(operator.nt, operator.dt) + 1

    def process_row(row_idx):
        def compute_element(col_idx):
            return operator._element(row_idx, col_idx)

        row = jax.lax.map(
            compute_element,
            col_indices,
            batch_size=mrx.MAP_BATCH_SIZE_INNER,
        )
        nz_mask = row != 0
        order = jnp.argsort(~nz_mask, stable=True)
        vals = row[order][:max_nnz]
        cols = col_indices[order][:max_nnz]
        nz_count = jnp.sum(nz_mask)
        valid = jnp.arange(max_nnz) < nz_count
        vals = jnp.where(valid, vals, 0.0)
        cols = jnp.where(valid, cols, 0)
        return vals, cols

    all_vals, all_cols = jax.lax.map(
        process_row,
        jnp.arange(nrows),
        batch_size=mrx.MAP_BATCH_SIZE_OUTER,
    )
    row_indices = jnp.broadcast_to(jnp.arange(nrows)[:, None], (nrows, max_nnz))
    indices = jnp.stack([row_indices.ravel(), all_cols.ravel()], axis=-1)
    data = all_vals.ravel()
    return jsparse.BCOO((data, indices), shape=(nrows, ncols))


@pytest.mark.parametrize("zero_bc", [False, True])
def test_polar_sparse_matches_legacy_sparse_reference(form, xi, zero_bc):
    operator = PolarExtractionOperator(form, xi, zero_bc)

    sparse = operator.assemble_sparse()
    legacy_sparse = _assemble_sparse_legacy(operator)

    npt.assert_array_equal(sparse.todense(), legacy_sparse.todense())