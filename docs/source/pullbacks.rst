Pullback Operations in Differential Forms
=========================================

This tutorial demonstrates pullback operations on differential forms in a polar coordinate system. It shows how functions and vector fields can be transformed between physical and logical spaces using pullback operations.

Introduction
------------
The script implements coordinate mapping between polar and Cartesian coordinates, pullback operations on scalar functions and vector fields, and visualizes the results.

Setup and Imports
-----------------
We start by importing the necessary libraries and setting up the output directory:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from pathlib import Path
    from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback
    from mrx.Quadrature import QuadratureRule
    from mrx.Projectors import Projector
    from mrx.PolarMapping import get_xi, LazyExtractionOperator
    from mrx.LazyMatrices import LazyMassMatrix, LazyProjectionMatrix

    output_dir = Path("script_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

Coordinate Mapping and Pullbacks
--------------------------------
We define forward and inverse mappings between logical (polar) and physical (Cartesian) coordinates, and demonstrate pullbacks for 0-forms, 1-forms, 2-forms, and 3-forms.

.. code-block:: python

    def _R(r, χ):
        return jnp.ones(1) * r * jnp.cos(2 * jnp.pi * χ)
    def _Y(r, χ):
        return jnp.ones(1) * r * jnp.sin(2 * jnp.pi * χ)
    def F(p):
        r, χ, z = p
        return jnp.squeeze(jnp.array([_R(r, χ), _Y(r, χ), jnp.ones(1) * z]))
    def F_inv(p):
        x, y, z = p
        r = jnp.sqrt(x**2 + y**2)
        χ = jnp.arctan2(y, x) / (2 * jnp.pi)
        return jnp.array([r, χ, z])

    # See script for full details and visualization code

Visualization
-------------
The script generates several plots to visualize scalar and vector fields in both logical and physical spaces, including contour and quiver plots.

Main Steps
----------
- Define coordinate mappings
- Demonstrate pullbacks for different form degrees
- Visualize results

Conclusion
----------
This script demonstrates how to use pullback operations to transform and visualize differential forms between logical and physical spaces.

**How to run:**
.. code-block:: bash

    python scripts/pullbacks.py 