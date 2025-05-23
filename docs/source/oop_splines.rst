Object-Oriented Splines and Differential Forms Demo
===================================================

This tutorial walks through a script that demonstrates the use of spline bases and differential forms for numerical computations in a mapped geometry. The script includes basis construction, field operations, mapping, error analysis, and visualization.

Introduction
------------
We use spline bases and differential forms to perform numerical computations in a mapped geometry. The script demonstrates basis construction, field operations (grad, curl, div), mapping between reference and physical domains, and error analysis.

Setup and Imports
-----------------
We start by importing the necessary libraries and setting up the output directory:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from pathlib import Path
    import time
    from typing import Callable, Any
    from mrx.SplineBases import SplineBasis, DerivativeSpline
    from mrx.PolarMapping import LazyExtractionOperator, get_xi
    from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback
    from mrx.Quadrature import QuadratureRule
    from mrx.Projectors import Projector
    from mrx.LazyMatrices import LazyMassMatrix
    from mrx.Utils import div, curl, grad

    jax.config.update("jax_enable_x64", True)
    output_dir = Path("scripts_output")
    output_dir.mkdir(parents=True, exist_ok=True)

Main Steps
----------
- Construct spline bases and visualize basis functions
- Define and operate on differential forms (grad, curl, div)
- Map between reference and physical domains
- Project and compare continuous and discrete fields
- Visualize results and compute errors

Conclusion
----------
This script demonstrates how to use object-oriented splines and differential forms for numerical computations and visualization in mapped geometries.

**How to run:**
.. code-block:: bash

    python scripts/oop_splines.py 