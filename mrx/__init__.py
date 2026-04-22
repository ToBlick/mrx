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

# maximum batch size for map evaluations in inner loops
# most commonly, this is vectorization over quadrature points
MAP_BATCH_SIZE_INNER = 0  # (0 corresponds to vmap)
# maximum batch size for outer loops
# for example, this is used in matrix assembly to batch over rows
# maximum number of concurrent evaluations is thus MAP_BATCH_SIZE_OUTER * MAP_BATCH_SIZE_INNER
MAP_BATCH_SIZE_OUTER = 16384  # (None corresponds to no batching)
