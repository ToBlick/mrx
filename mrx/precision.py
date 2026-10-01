"""The floating-point precision MRX runs in, chosen by two environment variables.

Both are read once, when :mod:`mrx` is imported, so set them before the import.

- ``MRX_DTYPE`` (``float32``, the default, or ``float64``) is the working dtype :data:`DTYPE`. Every
  array stored on a sequence, its geometry and its operators is in this dtype.
- ``MRX_RESIDUAL_DTYPE`` (``float64``, the default, or ``float32``) is the dtype in which solves measure
  their residual. With a float32 working dtype the default gives the mixed configuration. Every solve is
  then an iterative refinement (:func:`mrx.solvers.refine`) that computes the residual in float64 and the
  corrections with float32 Krylov solves, and so reaches a float64-level tolerance.
  ``MRX_RESIDUAL_DTYPE=float32`` gives plain float32, which is faster but can stall on strongly shaped
  geometries.

The module also sets the default solve tolerance :data:`SOLVE_TOL` and provides :func:`eps` and
:func:`sqrt_eps`, round-off scales of the working dtype. Importing it turns on JAX 64-bit mode, which the
mixed configuration needs.
"""

import functools
import os
import types

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

_NAME = os.environ.get("MRX_DTYPE", "float32")
if _NAME not in ("float32", "float64"):
    raise ValueError(
        f"MRX_DTYPE={_NAME!r}, expected 'float32' or 'float64'")

jax.config.update("jax_enable_x64", True)

# By default JAX runs float32 products in TF32 (10-bit mantissa) on Ampere and later GPUs.
# Spline derivatives are cancelling sums that TF32 cannot resolve.
jax.config.update("jax_default_matmul_precision", "highest")

#: The working floating-point dtype.
DTYPE = jnp.dtype(_NAME)

_RES_NAME = os.environ.get("MRX_RESIDUAL_DTYPE", "float64")
if _RES_NAME not in ("float32", "float64"):
    raise ValueError(
        f"MRX_RESIDUAL_DTYPE={_RES_NAME!r}, expected 'float32' or 'float64'")

#: The dtype in which every solve computes its residual and accumulates its solution. float64 unless
#: ``MRX_RESIDUAL_DTYPE=float32`` selects plain float32.
RESIDUAL_DTYPE = jnp.dtype(_RES_NAME)
if np.finfo(RESIDUAL_DTYPE).eps > np.finfo(DTYPE).eps:
    raise ValueError(
        f"MRX_RESIDUAL_DTYPE={_RES_NAME} is coarser than MRX_DTYPE={_NAME}. "
        "the residual precision is the working precision or finer")

#: Whether the solves are refined (the mixed configuration). When the two dtypes coincide, every solve
#: is a plain Krylov solve.
REFINE = DTYPE != RESIDUAL_DTYPE


def current_precision() -> str:
    """The configuration mrx runs in: ``"mixed"`` (float32 solves refined in float64), ``"float32"`` or
    ``"float64"``."""
    return "mixed" if REFINE else DTYPE.name


#: Machine epsilon of the working dtype, as a Python float.
EPS = float(np.finfo(DTYPE).eps)


def default_tol(dtype, refine) -> float:
    """The default relative tolerance of a solve on a sequence of ``dtype``. It is 1e-8 for refined
    (mixed) solves, 1e-10 for plain float64 solves and 1e-5 for plain float32 solves."""
    if refine:
        return 1e-8
    return 1e-10 if jnp.dtype(dtype) == jnp.dtype("float64") else 1e-5


#: The default relative tolerance of a solve on a sequence, :func:`default_tol` of the configuration.
SOLVE_TOL = default_tol(DTYPE, REFINE)


def inner_tol(tol) -> float:
    """The tolerance of each inner Krylov solve of a refined solve at ``tol``. It is the square root of
    ``tol``, so that a refined solve typically takes two passes."""
    return float(tol) ** 0.5


#: Passes a refined solve may take before it reports non-convergence.
MAX_PASSES = 6


def eps(c: float = 1.0) -> float:
    """Return ``c`` times the machine epsilon of the working dtype."""
    return c * EPS


def sqrt_eps(c: float = 1.0) -> float:
    """Return ``c`` times the square root of the machine epsilon."""
    return c * EPS ** 0.5


def cast_arrays(obj, dtype=DTYPE, _seen=None):
    """Return ``obj`` with every floating-point array reachable from it cast to ``dtype``.

    Under 64-bit mode a stray float64 NumPy array would silently promote float32 computations, and this
    prevents that. Plain objects and dicts are cast in place, pytrees such as Equinox modules are
    rebuilt, and NumPy floating scalars become Python floats. Integer and boolean arrays are left alone,
    and so are functions: a function that captured arrays keeps their dtype and must be rebuilt.
    """
    if _seen is None:
        _seen = {}
    if isinstance(obj, jax.Array):
        return obj.astype(dtype) if jnp.issubdtype(obj.dtype, jnp.floating) else obj
    # NumPy floating data promotes JAX arithmetic under 64-bit mode, a Python float does not.
    if isinstance(obj, np.ndarray):
        return obj.astype(np.dtype(dtype)) if np.issubdtype(obj.dtype, np.floating) else obj
    if isinstance(obj, np.generic):
        return float(obj) if np.issubdtype(obj.dtype, np.floating) else obj
    if isinstance(obj, (str, bytes, int, float, bool, type(None))) or _is_function(obj):
        return obj
    if id(obj) in _seen:
        return _seen[id(obj)]
    _seen[id(obj)] = obj          # a cycle meets the object itself
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = cast_arrays(value, dtype, _seen)
        out = obj
    elif isinstance(obj, (tuple, list)):
        items = [cast_arrays(v, dtype, _seen) for v in obj]
        out = type(obj)(*items) if hasattr(obj, "_fields") else type(obj)(items)
    elif hasattr(obj, "__dict__") and not _is_pytree_module(obj):
        for name, value in vars(obj).items():
            setattr(obj, name, cast_arrays(value, dtype, _seen))
        out = obj
    else:
        out = jax.tree_util.tree_map(
            lambda leaf: cast_arrays(leaf, dtype, _seen), obj,
            is_leaf=lambda leaf: isinstance(leaf, jax.Array)
            or (hasattr(leaf, "__dict__") and not _is_pytree_module(leaf)))
    _seen[id(obj)] = out
    return out


def _is_function(obj):
    """Whether ``obj`` is a function, method, partial or JAX callable, which :func:`cast_arrays` skips.
    An object that merely defines ``__call__`` (such as a spline basis) is still cast."""
    if isinstance(obj, (types.FunctionType, types.MethodType, types.BuiltinFunctionType,
                        types.BuiltinMethodType, functools.partial)):
        return True
    return callable(obj) and type(obj).__module__.split(".")[0] in ("jax", "jaxlib")


def _is_pytree_module(obj):
    """Whether ``obj`` is an Equinox module, whose fields are immutable, so it is rebuilt."""
    return isinstance(obj, eqx.Module)
