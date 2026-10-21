"""The floating-point precision MRX runs in, chosen by the environment variable ``MRX_DTYPE``.

It is read once, when :mod:`mrx` is imported, so set it before the import.

- ``float64`` (the default): every array is float64 and JAX 64-bit mode is on.
- ``float32``: for a device without float64. Every array is float32 and 64-bit mode is off, so no float64
  array is created on the device. A float32 Krylov solve stalls above its tolerance, so every solve is an
  iterative refinement (:func:`mrx.solvers.refine`): the solution is accumulated as a double-single number,
  a pair of float32 (:mod:`mrx.double_single`), and the residual is formed in that arithmetic.

There is one rule for both. An operator applied to a float array returns a float array, and applied to a
:class:`~mrx.double_single.DS` it returns a :class:`~mrx.double_single.DS`. :func:`lift` and
:func:`to_working` move a vector between the two, and both are the identity in float64.

The module also holds the default solve tolerance :data:`SOLVE_TOL`, the cap :data:`MAX_PASSES` on the passes
of a solve, and :func:`eps` and :func:`sqrt_eps`, the round-off scales that tolerances in the library and the
tests are multiples of.
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

#: Whether a checked solve in float32 runs its conjugate gradients with the iterate and the residual in
#: double-single, in one pass (:func:`mrx.solvers.preconditioned_cg_pairs`), instead of float32 passes of
#: :func:`mrx.solvers.refine`. Set it with the environment variable ``MRX_DS_KRYLOV`` (``0`` or ``1``).
DS_KRYLOV = PAIRS and os.environ.get("MRX_DS_KRYLOV", "1") == "1"

#: Machine epsilon of :data:`DTYPE`, as a Python float.
EPS = float(np.finfo(DTYPE).eps)

#: The default relative tolerance of a solve on a sequence.
SOLVE_TOL = 1e-8

#: Passes a solve may take (:func:`mrx.solvers.refine`) before it reports non-convergence.
MAX_PASSES = 6


def pass_floor(tol):
    """The strongest reduction of the residual that one pass of a solve at ``tol`` asks of its Krylov solve.
    In float32 it is ``sqrt(tol)``, which a float32 iteration reaches, so that a solve from zero takes two
    passes. In float64 it is ``tol``, one pass."""
    return tol ** 0.5 if PAIRS else tol


def lift(x):
    """``x`` as the solves carry it: a :class:`~mrx.double_single.DS` in float32, unchanged in float64."""
    return ds.lift(x) if PAIRS else x


def to_working(x):
    """``x`` as a float array. A :class:`~mrx.double_single.DS` is rounded."""
    return x.round() if isinstance(x, ds.DS) else x


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
