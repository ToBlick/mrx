"""Working precision of the package.

The precision is chosen once, from the environment variable ``MRX_DTYPE``
(``float64``, the default, or ``float32``), before any array is created.
Importing :mod:`mrx` applies it; scripts and tests do not touch
``jax_enable_x64`` themselves.

Every tolerance in the package that depends on roundoff is expressed
through :func:`eps` so it scales with the working precision. Tolerances
that encode a physical or algorithmic choice (a force residual, an ODE
controller, a shift) are ordinary parameters and do not use this module.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

_NAME = os.environ.get("MRX_DTYPE", "float64")
if _NAME not in ("float32", "float64"):
    raise ValueError(
        f"MRX_DTYPE={_NAME!r}; expected 'float32' or 'float64'")

jax.config.update("jax_enable_x64", _NAME == "float64")

# On Ampere-and-later GPUs JAX runs float32 dot products (matmul, einsum,
# dot_general) in TF32 by default: a 10-bit mantissa, relative error ~5e-4
# per term. Every spline derivative in MRX is a cancelling sum of such
# terms -- the W7-X map's dR/dtheta on the innermost quadrature ring came
# out 19% wrong and det DF went negative in float32, while the 1-D basis
# values and the coefficients themselves were accurate to 1e-6 (measured
# 2026-08-26 by bisecting the map evaluation). Nothing in MRX wants a
# 10-bit product; float64 is unaffected by this setting.
jax.config.update("jax_default_matmul_precision", "highest")

#: The working floating-point dtype.
DTYPE = jnp.dtype(_NAME)

#: Machine epsilon of the working dtype, as a Python float.
EPS = float(np.finfo(DTYPE).eps)


def eps(c: float = 1.0) -> float:
    """Return ``c`` times the machine epsilon of the working dtype."""
    return c * EPS


def sqrt_eps(c: float = 1.0) -> float:
    """Return ``c`` times the square root of the machine epsilon.

    The natural stopping tolerance for an iterative solve whose residual is
    limited by roundoff in the matrix-vector product.
    """
    return c * EPS ** 0.5
