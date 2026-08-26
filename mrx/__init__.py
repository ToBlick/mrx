from .precision import DTYPE, EPS, eps, sqrt_eps  # noqa: F401  (sets jax_enable_x64; must come first)
from .derham_sequence import *
from .differential_forms import *
from .extraction_operators import *
from .mappings import *
from .operators import *
from .plotting import *
from .projectors import *
from .quadrature import *
from .solvers import *
from .spline_bases import *
from .utils import *

__version__ = "0.0.1"

# Batch size for `lax.map` over quadrature points and other inner loops.
# 0 is a full `vmap`; set to a positive integer to bound memory.
MAP_BATCH_SIZE_INNER = 0
# Batch size for outer loops (rows in matrix assembly); None is no batching.
MAP_BATCH_SIZE_OUTER = None
