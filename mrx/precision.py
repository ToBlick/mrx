"""The floating-point precision of MRX, which is float64 throughout.

Importing :mod:`mrx` turns on JAX 64-bit mode. Every array on a sequence, its geometry and its
operators is float64, and every solve iterates and measures its residual in float64. There is no
switch. MRX needs a backend with float64 (a CPU or a CUDA GPU).

The module holds the default solve tolerance :data:`SOLVE_TOL`, the cap :data:`MAX_PASSES` on the
passes of a solve, and :func:`eps` and :func:`sqrt_eps`, the round-off scales that tolerances in the
library and the tests are multiples of.
"""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

#: Machine epsilon of float64, as a Python float.
EPS = float(np.finfo(np.float64).eps)

#: The default relative tolerance of a solve on a sequence.
SOLVE_TOL = 1e-8

#: Passes a solve may take (:func:`mrx.solvers.refine`) before it reports non-convergence.
MAX_PASSES = 6


def eps(c: float = 1.0) -> float:
    """Return ``c`` times the machine epsilon."""
    return c * EPS


def sqrt_eps(c: float = 1.0) -> float:
    """Return ``c`` times the square root of the machine epsilon."""
    return c * EPS ** 0.5
