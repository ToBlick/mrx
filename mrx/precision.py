"""Working precision of the package, and the residual precision of its solves.

The WORKING precision is chosen once, from the environment variable
``MRX_DTYPE`` (``float32``, the default, or ``float64``), before any array
is created: every stored array of a sequence, a geometry and a
preconditioner bundle is in it (:func:`cast_arrays` pins them at build
time), and so is every field of the relaxation. Importing :mod:`mrx`
applies it; scripts and tests do not touch ``jax_enable_x64`` themselves.

The RESIDUAL precision is float64 whatever the working one: a Krylov solve
in float32 runs as iterative refinement, the residual of the outer
equation evaluated in float64 on a float64 view of the operator
(:attr:`mrx.derham_sequence.DeRhamSequence.residual`), the correction by
the float32 solve to :data:`INNER_TOL`, the solution accumulated in
float64 (:func:`mrx.solvers.refine`). That is what makes a float32 run's
forces accurate beyond the float32 tolerance: the Leray projection's
gradient part is the size of ``J x B`` while the force is a thousandth of
it, so a tolerance relative to ``J x B`` leaves an O(1) error in the
force; measured 2026-09-04 on li383 (16,32,32) p=3, ``|div F| / |F| =
22`` at the old default and 0.04 at 1e-7. With a float64 residual the
float32 solve reaches :data:`SOLVE_TOL` in a few passes, and the force is
formed in float64 before it is stored. 64-bit mode is therefore always
on; Python scalars stay weakly typed and do not promote.
``MRX_RESIDUAL_DTYPE=float32`` is the configuration of a machine without
float64 (a TPU): plain float32 solves, default tolerance 1e-6.

Every tolerance in the package that depends on roundoff is expressed
through :func:`eps` so it scales with the working precision. Tolerances
that encode a physical or algorithmic choice (a force residual, an ODE
controller, a shift) are ordinary parameters and do not use this module.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

_NAME = os.environ.get("MRX_DTYPE", "float32")
if _NAME not in ("float32", "float64"):
    raise ValueError(
        f"MRX_DTYPE={_NAME!r}; expected 'float32' or 'float64'")

jax.config.update("jax_enable_x64", True)

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

_RES_NAME = os.environ.get("MRX_RESIDUAL_DTYPE", "float64")
if _RES_NAME not in ("float32", "float64"):
    raise ValueError(
        f"MRX_RESIDUAL_DTYPE={_RES_NAME!r}; expected 'float32' or 'float64'")

#: The dtype of every solve's residual and accumulated solution: float64
#: unless ``MRX_RESIDUAL_DTYPE=float32`` asks for the float32-only
#: configuration of a machine without float64 (a TPU), where the solves
#: are plain float32 Krylov iterations.
RESIDUAL_DTYPE = jnp.dtype(_RES_NAME)
if np.finfo(RESIDUAL_DTYPE).eps > np.finfo(DTYPE).eps:
    raise ValueError(
        f"MRX_RESIDUAL_DTYPE={_RES_NAME} is coarser than MRX_DTYPE={_NAME}; "
        "the residual precision is the working precision or finer")

#: Whether the solves refine: a float64 residual against a float32 Krylov
#: solve. When the two dtypes coincide the solve is plain.
REFINE = DTYPE != RESIDUAL_DTYPE

#: Machine epsilon of the working dtype, as a Python float.
EPS = float(np.finfo(DTYPE).eps)

def default_tol(dtype, refine) -> float:
    """The default relative residual of a solve on a sequence of ``dtype``
    that refines or not: 1e-8 for a refined float32 solve, 1e-10 for a
    plain float64 one, 1e-6 for a plain float32 one (its iteration reaches
    1e-7 on the production systems). The float64 view of a float32
    sequence solves plainly at 1e-10: the harmonic-form construction on it
    needs that (its k=1 solve's true residual at 1e-8 was 2e-6, and the
    k=2 form's Rayleigh quotient that residual squared)."""
    if refine:
        return 1e-8
    return 1e-10 if jnp.dtype(dtype) == jnp.dtype("float64") else 1e-6


#: Default relative residual of a solve through a sequence, in the residual
#: precision: :func:`default_tol` of the working configuration. Until
#: 2026-09-04 it was sqrt(eps) of the working dtype, 3.5e-4 at float32.
SOLVE_TOL = default_tol(DTYPE, REFINE)

#: Relative tolerance of one float32 pass of a refined solve: each pass
#: takes the residual down by this factor, so a warm start with a 1% defect
#: reaches SOLVE_TOL in two passes.
INNER_TOL = 1e-4

#: Passes a refined solve may take before it reports non-convergence.
MAX_PASSES = 6


def eps(c: float = 1.0) -> float:
    """Return ``c`` times the machine epsilon of the working dtype."""
    return c * EPS


def sqrt_eps(c: float = 1.0) -> float:
    """Return ``c`` times the square root of the machine epsilon."""
    return c * EPS ** 0.5


def solve_tol(c: float = 1.0) -> float:
    """Return ``c`` times :data:`SOLVE_TOL`."""
    return c * SOLVE_TOL


def cast_arrays(obj, dtype=DTYPE, _seen=None):
    """Every floating JAX array reachable from ``obj`` cast to ``dtype``.

    Walks pytrees (Equinox modules, tuples, lists, dicts) and the attributes
    of plain objects; returns the cast object (pytrees are rebuilt, plain
    objects and dicts are cast in place). NumPy floating arrays follow the
    dtype and NumPy floating scalars become Python floats, since either
    promotes JAX arithmetic under 64-bit mode; integer and boolean arrays
    and callables are left alone; a callable that closed over arrays must
    be rebuilt after the cast. An object reachable through several
    attributes is cast once and the same result installed everywhere.
    Built objects are cast once at the end of their construction, so that
    64-bit mode (always on, see the module docstring) never lets a
    NumPy-built array promote a float32 apply.
    """
    if _seen is None:
        _seen = {}
    if isinstance(obj, jax.Array):
        return obj.astype(dtype) if jnp.issubdtype(obj.dtype, jnp.floating) else obj
    # NumPy floating data promotes JAX arithmetic under 64-bit mode (a Python
    # float does not): host arrays follow the dtype, host scalars become floats.
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
    """A function, method, partial or compiled JAX callable: not walked. An
    object of ours that merely defines ``__call__`` (a spline basis) is."""
    import functools  # noqa: PLC0415
    import types  # noqa: PLC0415
    if isinstance(obj, (types.FunctionType, types.MethodType, types.BuiltinFunctionType,
                        types.BuiltinMethodType, functools.partial)):
        return True
    return callable(obj) and type(obj).__module__.split(".")[0] in ("jax", "jaxlib")


def _is_pytree_module(obj):
    """Equinox modules are pytrees whose fields are immutable: rebuild them."""
    import equinox as eqx  # noqa: PLC0415
    return isinstance(obj, eqx.Module)
