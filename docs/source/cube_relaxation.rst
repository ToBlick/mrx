Magnetic Field Relaxation in a 3D Domain
========================================

This tutorial walks through a script that implements a magnetic field relaxation process in a 3D domain using differential forms and finite element methods. The script demonstrates field setup, evolution, analysis, and visualization.

Introduction
------------
We use finite element methods and differential forms to simulate magnetic field relaxation in a 3D domain. The script demonstrates setup of forms and operators, field evolution, helicity and energy analysis, and visualization.

Setup and Imports
-----------------
We start by importing the necessary libraries and setting up the output directory:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import os
    from pathlib import Path
    from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback
    from mrx.Quadrature import QuadratureRule
    from mrx.Projectors import Projector, CurlProjection
    from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix, LazyProjectionMatrix, LazyDoubleCurlMatrix
    from mrx.Utils import curl

    jax.config.update("jax_enable_x64", True)
    output_dir = Path("script_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

Main Steps
----------
- Set up differential forms and operators
- Implement magnetic field evolution equations
- Analyze helicity and energy conservation
- Visualize field evolution and conservation properties

Conclusion
----------
This script demonstrates how to use finite element methods and differential forms to simulate magnetic field relaxation, analyze conservation properties, and visualize results.

**How to run:**
.. code-block:: bash

    python scripts/cube_relaxation.py 