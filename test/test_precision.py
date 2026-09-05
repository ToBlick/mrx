"""``mrx.precision``: the working dtype, its epsilons, and the matmul precision.

The last test is the one that matters on an H100: JAX runs float32
``dot_general`` in TF32 by default (10-bit mantissa, ~5e-4 per term), and a
spline derivative is a cancelling contraction -- the W7-X map's
``dR/dtheta`` on the innermost quadrature ring came out 19% wrong and
``det DF`` went negative (2026-08-26). ``mrx.precision`` sets
``jax_default_matmul_precision = 'highest'``; this reproduces the pattern
in miniature so the setting cannot silently go missing.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

import mrx
from mrx.precision import cast_arrays

_NAME = os.environ.get("MRX_DTYPE", "float32")
_EPS = {"float64": 2.220446049250313e-16, "float32": 1.1920928955078125e-07}


def test_dtype_follows_mrx_dtype():
    assert mrx.DTYPE == jnp.dtype(_NAME)
    assert mrx.EPS == _EPS[_NAME]
    # 64-bit mode is on whatever the working dtype is, and this asserted the
    # opposite until 2026-09-05. A float32 run solves by iterative refinement
    # against a float64 residual, so the wider dtype has to exist at all times
    # (see mrx.precision's module docstring). A fresh array is therefore
    # float64 here and says nothing about the working dtype; what pins the
    # package's arrays to it is cast_arrays, at build time.
    assert jax.config.jax_enable_x64
    assert cast_arrays(jnp.zeros(1, dtype=jnp.float64)).dtype == mrx.DTYPE
    assert cast_arrays(np.zeros(1)).dtype == mrx.DTYPE


def test_default_matmul_precision_is_highest():
    assert jax.config.jax_default_matmul_precision == "highest"


