"""MRX: magnetic relaxation with finite element exterior calculus on spline de Rham complexes.

Importing the package fixes the precision from ``MRX_DTYPE``: float64 (the default), or float32 for a device
without float64. Set it before the first import. See :mod:`mrx.precision`.
"""
# Imported first because it configures JAX before anything else creates arrays.
from .precision import DTYPE, EPS, eps, sqrt_eps  # noqa: F401

__version__ = "0.1.0"

# Batch size of the loops over quadrature points. 0 evaluates all points at once, None one point at a
# time, and a positive integer that many at a time. Lower it when one of these loops runs out of
# device memory. On li383 (64,80,32) and (80,104,40), p = 2, float64, the batch (0, 262144, 65536,
# 8192) changes neither the build time nor the peak memory of the setup (15 and 31 GiB), and below
# 262144 the start field is 6 times slower per call (2026-10-11).
MAP_BATCH_SIZE_INNER = 262144
