2D Poisson Problem in Polar Coordinates with Mixed Formulation
==============================================================

This tutorial walks through a script that solves the 2D Poisson equation in polar coordinates using a mixed finite element formulation. The problem is defined on a circular domain with Dirichlet boundary conditions.

Introduction
------------
We solve the Poisson equation

.. math::

    -\nabla^2 u = f

on a circular domain, using a mixed finite element method. The exact solution is:

.. math::

    u(r,\theta) = -\frac{1}{16}r^4 + \frac{1}{12}r^3

with source term:

.. math::

    f(r,\theta) = r^2 - \frac{3}{4}r

Setup and Imports
-----------------
We start by importing the necessary libraries and setting up the output directory:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import time
    from mrx.PolarMapping import LazyExtractionOperator, get_xi
    from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback
    from mrx.Quadrature import QuadratureRule
    from mrx.Projectors import Projector
    from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix, LazyProjectionMatrix
    from mrx.Utils import l2_product
    from functools import partial

    jax.config.update("jax_enable_x64", True)
    output_dir = Path("script_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

Defining the Problem
--------------------
We define the finite element spaces, mapping, exact solution, and source term:

.. code-block:: python

    def _R(r, χ):
        return jnp.ones(1) * r * jnp.cos(2 * jnp.pi * χ)
    def _Y(r, χ):
        return jnp.ones(1) * r * jnp.sin(2 * jnp.pi * χ)
    def F(p):
        r, χ, z = p
        return jnp.squeeze(jnp.array([_R(r, χ), _Y(r, χ), jnp.ones(1) * z]))
    def u(x):
        r, χ, z = x
        return -jnp.ones(1) * 1/4 * (1/4 * r**4 - 1/3 * r**3 + 1/12)
    def f(x):
        r, χ, z = x
        return jnp.ones(1) * (r - 3/4) * r

Convergence and Timing Analysis
-------------------------------
We run the error computation for a range of mesh sizes, polynomial degrees, and quadrature orders, and measure computation time:

.. code-block:: python

    def get_err(n, p, q):
        # ... see script for details ...

Plotting Results
----------------
The script generates several plots to visualize the convergence of errors and computational time.

Conclusion
----------
This script demonstrates how to use a mixed finite element formulation to solve the 2D Poisson problem in polar coordinates, analyze convergence, and measure performance.

**How to run:**
.. code-block:: bash

    python scripts/polar_poisson_mixed.py 