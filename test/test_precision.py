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
from mrx.spline_bases import SplineBasis

_NAME = os.environ.get("MRX_DTYPE", "float64")
_EPS = {"float64": 2.220446049250313e-16, "float32": 1.1920928955078125e-07}


def test_dtype_follows_mrx_dtype():
    assert mrx.DTYPE == jnp.dtype(_NAME)
    assert jax.config.jax_enable_x64 == (_NAME == "float64")
    # the default dtype of a fresh array is the working dtype
    assert jnp.zeros(1).dtype == mrx.DTYPE
    assert jnp.asarray(np.zeros(1)).dtype == mrx.DTYPE


def test_eps_values():
    assert mrx.EPS == _EPS[_NAME]
    assert mrx.eps() == mrx.EPS
    assert mrx.eps(7.0) == 7.0 * mrx.EPS
    assert mrx.sqrt_eps() == mrx.EPS ** 0.5
    assert mrx.sqrt_eps(3.0) == 3.0 * mrx.EPS ** 0.5


def test_default_matmul_precision_is_highest():
    assert jax.config.jax_default_matmul_precision == "highest"


def test_spline_derivative_contraction_is_not_tf32():
    """``sum_i c_i B_i'(x)`` for ``c_i = R0 + s g_i`` (``g`` the Greville points)
    is exactly ``s``: the terms are ``O(R0 / h)`` and cancel to ``O(s)``.

    Under TF32 each product carries ~5e-4 relative error, ``R0 / h ~ 500``
    here, so the sum is off by O(0.3) against a true value of 0.03 -- the
    W7-X ``dR/dtheta`` failure with the same ratio of term to result. At
    'highest' the error is a few ulps of the largest term, which is the
    bound asserted (measured 2.4e-4 relative on the map at float32).

    The window and the ``'i,j,k,ijk'`` contraction are those of
    ``spline_bases.contract_local``, whose derivative ``jacfwd`` takes for
    the spline map's ``DF``.
    """
    spl = SplineBasis(8, 3, "clamped")
    x = jnp.asarray(0.31, dtype=mrx.DTYPE)
    R0, slope = 60.0, 0.03
    vals = jax.vmap(lambda i: spl(x, i))(spl.ns)
    dvals = jax.vmap(lambda i: jax.grad(spl, argnums=0)(x, i))(spl.ns)
    c = R0 + slope * spl.greville_points()
    window = jnp.broadcast_to(c[:, None, None], (spl.n, spl.n, spl.n))
    got = jnp.einsum("i,j,k,ijk->", dvals, vals, vals, window)
    largest_term = float(jnp.max(jnp.abs(dvals))) * R0
    print(f"\n  derivative {float(got):.6e} vs {slope}, largest term {largest_term:.1f}")
    assert abs(float(got) - slope) <= mrx.eps(10) * (spl.p + 1) * largest_term
