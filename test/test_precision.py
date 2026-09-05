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
    """The working dtype is the environment's (float32 by default), 64-bit
    mode is on whatever it is (the float64 residual of a refined solve needs
    it), and the caster pins arrays to the working dtype: a fresh JAX array
    is float64 under 64-bit mode and says nothing about the working dtype."""
    assert mrx.DTYPE == jnp.dtype(_NAME)
    assert mrx.EPS == _EPS[_NAME]
    assert jax.config.jax_enable_x64
    assert cast_arrays(jnp.zeros(1, dtype=jnp.float64)).dtype == mrx.DTYPE
    assert cast_arrays(np.zeros(1)).dtype == mrx.DTYPE


def test_default_matmul_precision_is_highest():
    assert jax.config.jax_default_matmul_precision == "highest"


