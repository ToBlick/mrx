"""MRX: magnetic relaxation with finite element exterior calculus on spline de Rham complexes.

Importing the package fixes the precision (``MRX_DTYPE`` and ``MRX_RESIDUAL_DTYPE``, mixed precision by
default) and turns on JAX's 64-bit mode, so set these environment variables before the first import. See
:mod:`mrx.precision`.
"""
# Imported first because it configures JAX before anything else creates arrays.
from .precision import DTYPE, EPS, eps, sqrt_eps  # noqa: F401

__version__ = "0.1.0"

# Batch size of the loops over quadrature points. 0 evaluates all points at once (fastest, most
# memory), None one point at a time, and a positive integer that many at a time. Lower it when a
# large mesh runs out of GPU memory, since these loops are the main memory cost of the code.
MAP_BATCH_SIZE_INNER = 0
