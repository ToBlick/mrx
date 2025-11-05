__version__ = "0.0.1"

from .boundary import *
from .differential_forms import *
from .iterative_solvers import *
from .lazy_matrices import *
from .plotting import *
from .polar import *
from .projectors import *
from .quadrature import *
from .spline_bases import *
from .utils import *
from .mappings import *
from .derham_sequence import *

__all__ = (differential_forms.__all__ + iterative_solvers.__all__ +
           lazy_matrices.__all__ + plotting.__all__ + polar.__all__ +
           projectors.__all__ + quadrature.__all__ + spline_bases.__all__ +
           utils.__all__ + boundary.__all__)
