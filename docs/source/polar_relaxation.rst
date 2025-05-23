Magnetic Field Relaxation in Polar Coordinates
==============================================

This tutorial walks through a script that implements a magnetic field relaxation algorithm in polar coordinates, preserving magnetic helicity while minimizing the Lorentz force. The script demonstrates field setup, relaxation, analysis, and visualization.

Introduction
------------
We use finite element methods and polar mapping to simulate magnetic field relaxation in polar coordinates. The script demonstrates initial field setup, conjugate gradient relaxation, force-free state computation, and conservation analysis.

Setup and Imports
-----------------
We start by importing the necessary libraries and setting up the output directory:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from typing import List
    from pathlib import Path
    from mrx.PolarMapping import LazyExtractionOperator, get_xi
    from mrx.DifferentialForms import DifferentialForm, DiscreteFunction
    from mrx.Quadrature import QuadratureRule
    from mrx.Projectors import Projector, CurlProjection
    from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix, LazyProjectionMatrix, LazyDoubleCurlMatrix
    from mrx.Utils import curl
    from mrx.DifferentialForms import Pullback

    jax.config.update("jax_enable_x64", True)
    output_dir = Path("script_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

Main Steps
----------
- Set up mesh, forms, and operators for polar geometry
- Initialize field and perform perturbation
- Run conjugate gradient relaxation
- Analyze force-free state, helicity, energy, and divergence
- Visualize field evolution and conservation properties

Conclusion
----------
This script demonstrates how to use finite element methods and polar mapping to simulate magnetic field relaxation, analyze conservation properties, and visualize results in polar coordinates.

**How to run:**
.. code-block:: bash

    python scripts/polar_relaxation.py 