"""The floating-point precision MRX runs in, chosen by the environment variable ``MRX_DTYPE``.

It is read once, when :mod:`mrx` is imported, so set it before the import.

- ``float64`` (the default): every array is float64 and JAX 64-bit mode is on.
- ``float32``: for a device without float64. Every array is float32 and 64-bit mode is off, so no float64
  array is created on the device. What float64 computes in float64, float32 computes in double-single, a pair of
  float32 (:mod:`mrx.double_single`) of about 48 bits.

The two run the same algorithm. A solve has an accurate part, in float64 or double-single: its iterate, its
residual and every operator apply that forms them. Everything else steers or measures, and is float32 in both
(:data:`LOW_DTYPE`): the preconditioners, the search directions and coefficients of the conjugate gradients, the
norms, and MINRES. An operator applied to a float array returns a float array, and applied to a
:class:`~mrx.double_single.DS` it returns a :class:`~mrx.double_single.DS`. :func:`lift` takes a vector to the
accurate precision, :func:`to_working` to a float array of :data:`DTYPE` and :func:`low` to :data:`LOW_DTYPE`.

The module also holds the default solve tolerance :data:`SOLVE_TOL`, the cap :data:`MAX_PASSES` on the passes
of the Hodge split, and :func:`eps` and :func:`sqrt_eps`, the round-off scales that tolerances in the library
and the tests are multiples of.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

import mrx.double_single as ds

_NAME = os.environ.get("MRX_DTYPE", "float64")
if _NAME not in ("float32", "float64"):
    raise ValueError(f"MRX_DTYPE={_NAME!r}, expected 'float32' or 'float64'")

jax.config.update("jax_enable_x64", _NAME == "float64")

# By default JAX runs float32 products in TF32 (10-bit mantissa) on Ampere and later GPUs.
# Spline derivatives are cancelling sums that TF32 cannot resolve.
jax.config.update("jax_default_matmul_precision", "highest")

#: The dtype of every array.
DTYPE = jnp.dtype(_NAME)

#: Whether the solves accumulate their solution and form their residual in double-single (``float32``).
PAIRS = _NAME == "float32"

#: The precision of what steers or measures a solve, in both configurations: the preconditioners (stored and
#: applied), the search directions and coefficients of the conjugate gradients, the norms and MINRES. Its round-off
#: costs conjugacy, not attainable accuracy: the accurate part keeps the residual ``b - A x``. A preconditioner
#: rounds its input to this and returns its result in the input's precision.
LOW_DTYPE = jnp.dtype("float32")
#: The storage precision of the preconditioners, :data:`LOW_DTYPE`.
PRECOND_DTYPE = LOW_DTYPE


def to_precond(tree):
    """``tree`` with its floating-point arrays in :data:`PRECOND_DTYPE`, the stored form of a preconditioner."""
    def cast(a):
        return a.astype(PRECOND_DTYPE) if isinstance(a, jax.Array) and jnp.issubdtype(a.dtype, jnp.floating) else a
    return jax.tree_util.tree_map(cast, tree)


#: Machine epsilon of :data:`DTYPE`, as a Python float.
EPS = float(np.finfo(DTYPE).eps)

#: The default relative tolerance of a solve on a sequence.
SOLVE_TOL = 1e-8

#: Passes a solve may take (:func:`mrx.solvers.refine`) before it reports non-convergence.
MAX_PASSES = 6



def lift(x):
    """``x`` in the accurate precision of the solves: a :class:`~mrx.double_single.DS` in float32, float64 in
    float64."""
    return ds.lift(x) if PAIRS else jnp.asarray(x, dtype=DTYPE)


def to_working(x):
    """``x`` as a float array. A :class:`~mrx.double_single.DS` is rounded."""
    return x.round() if isinstance(x, ds.DS) else x


def low(x):
    """``x`` as a float array of :data:`LOW_DTYPE`, rounded."""
    return to_working(x).astype(LOW_DTYPE)


def inner(a, b) -> float:
    """The dot product of two vectors as a Python float, in double-single arithmetic in float32."""
    if PAIRS:
        return float(ds.to_host(ds.dot(a, b)))
    return float(jnp.vdot(a, b))


def eps(c: float = 1.0) -> float:
    """Return ``c`` times the machine epsilon."""
    return c * EPS


def sqrt_eps(c: float = 1.0) -> float:
    """Return ``c`` times the square root of the machine epsilon."""
    return c * EPS ** 0.5
