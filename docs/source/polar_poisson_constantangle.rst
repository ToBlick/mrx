2D Poisson Problem in Polar Coordinates with Constant Angle
===========================================================

This tutorial walks through a script that solves the 2D Poisson equation in polar coordinates using a finite element method with a constant angle approximation. The problem is defined on a circular domain with Dirichlet boundary conditions.

Introduction
------------
We solve the Poisson equation

.. math::

    -\nabla^2 u = f

on a circular domain, using a constant angle approximation. The exact solution is:

.. math::

    u(r,\theta) = \frac{r^3(3\log(r) - 2)}{27} + \frac{2}{27}

with source term:

.. math::

    f(r,\theta) = -r \log(r)

Setup and Imports
-----------------
We start by importing the necessary libraries and setting up the output directory:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import numpy as np
    import time
    import matplotlib.pyplot as plt
    from pathlib import Path
    from mrx.DifferentialForms import DifferentialForm, DiscreteFunction
    from mrx.Quadrature import QuadratureRule
    from mrx.Projectors import Projector
    from mrx.LazyMatrices import LazyStiffnessMatrix
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
        return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * jnp.pi * χ))
    def _Y(r, χ):
        return jnp.ones(1) * (Y0 + a * r * jnp.sin(2 * jnp.pi * χ))
    def F(x):
        r, χ, z = x
        return jnp.ravel(jnp.array([_R(r, χ), _Y(r, χ), jnp.ones(1) * z]))
    def u(x):
        r, χ, z = x
        return jnp.ones(1) * r**3 * (3 * jnp.log(r) - 2) / 27 + 2/27
    def f(x):
        r, χ, z = x
        return -jnp.ones(1) * r * jnp.log(r)

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
This script demonstrates how to use finite element methods to solve the 2D Poisson problem in polar coordinates with a constant angle approximation, analyze convergence, and measure performance.

**How to run:**
.. code-block:: bash

    python scripts/polar_poisson_constantangle.py 