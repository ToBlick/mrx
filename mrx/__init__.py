from .precision import DTYPE, EPS, eps, sqrt_eps  # noqa: F401  (sets jax_enable_x64; must come first)

__version__ = "0.0.1"

# Batch size for `lax.map` over quadrature points. 0 is a full `vmap`; a
# positive integer bounds the memory of the quadrature loop (W7-X at high
# resolution needs it).
MAP_BATCH_SIZE_INNER = 0
