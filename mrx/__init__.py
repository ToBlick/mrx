__version__ = "0.0.1"

from .BoundaryConditions import *
from .DifferentialForms import *
from .IterativeSolvers import *
from .LazyMatrices import *
from .Plotting import *
from .PolarMapping import *
from .Projectors import *
from .Quadrature import *
from .SplineBases import *
from .Utils import *

__all__ = (DifferentialForms.__all__ + IterativeSolvers.__all__ +
           LazyMatrices.__all__ + Plotting.__all__ + PolarMapping.__all__ +
           Projectors.__all__ + Quadrature.__all__ + SplineBases.__all__ +
           Utils.__all__ + BoundaryConditions.__all__)
