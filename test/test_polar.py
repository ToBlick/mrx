"""
Tests for the LazyExtractionOperator, including dense and sparse assembly.
"""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from mrx.differential_forms import DifferentialForm
from mrx.polar import ExtractionOperator, get_xi

jax.config.update("jax_enable_x64", True)

NS = (5, 5, 5)
PS = (3, 3, 3)
TYPES = ("clamped", "periodic", "periodic")


@pytest.fixture(params=[0, 1, 2, 3])
def form(request):
    """DifferentialForm for each degree k."""
    return DifferentialForm(request.param, NS, PS, TYPES)


@pytest.fixture
def xi():
    """Polar mapping coefficients for nt=5."""
    return get_xi(NS[1])


class TestSparseMatchesDense:
    """The sparse matrix should be identical to the dense one."""

    @pytest.mark.parametrize("zero_bc", [True, False])
    def test_sparse_matches_dense(self, form, xi, zero_bc):
        E = ExtractionOperator(form, xi, zero_bc)
        dense = E.assemble()
        sparse = E.assemble_sparse()
        npt.assert_array_equal(sparse.todense(), dense)
