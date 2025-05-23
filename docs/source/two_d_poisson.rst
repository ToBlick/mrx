2D Poisson Problem
==================

This tutorial walks through a script that solves a 2D Poisson problem using finite element methods with Dirichlet boundary conditions. The problem is defined on a square domain [0,1]^2.

Introduction
------------
We solve the Poisson equation

.. math::

    -\Delta u = f

on a square domain, using finite element methods. The exact solution is:

.. math::

    u(x, y) = \sin(2\pi x) \sin(2\pi y)

with source term:

.. math::

    f(x, y) = 2(2\pi)^2 u(x, y)

Setup and Imports
-----------------
We start by importing the necessary libraries and setting up the output directory:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    import os
    from mrx.DifferentialForms import DifferentialForm, DiscreteFunction
    from mrx.Quadrature import QuadratureRule
    from mrx.Projectors import Projector
    from mrx.LazyMatrices import LazyStiffnessMatrix
    from mrx.Utils import l2_product
    from mrx.BoundaryConditions import LazyBoundaryOperator
    from functools import partial

    jax.config.update("jax_enable_x64", True)
    os.makedirs('script_outputs', exist_ok=True)

Defining the Problem
--------------------
We define the finite element spaces, exact solution, and source term:

.. code-block:: python

    ns = (n, n, 1)
    ps = (p, p, 0)
    types = ('clamped', 'clamped', 'constant')
    bcs = ('dirichlet', 'dirichlet', 'none')
    Λ0 = DifferentialForm(0, ns, ps, types)
    Q = QuadratureRule(Λ0, q)
    B0 = LazyBoundaryOperator(Λ0, bcs).M
    K = LazyStiffnessMatrix(Λ0, Q, F=None, E=B0).M
    P0 = Projector(Λ0, Q, E=B0)

    def u(x):
        r, χ, z = x
        return jnp.ones(1) * jnp.sin(2 * jnp.pi * r) * jnp.sin(2 * jnp.pi * χ)
    def f(x):
        return 2 * (2*jnp.pi)**2 * u(x)

Convergence and Timing Analysis
-------------------------------
We run the error computation for a range of mesh sizes, polynomial degrees, and quadrature orders, and measure computation time:

.. code-block:: python

    def run_convergence_analysis():
        # ... see script for details ...

Plotting Results
----------------
The script generates several plots to visualize the convergence of errors and computational time.

Main Function
-------------
The main function runs the convergence analysis and plotting:

.. code-block:: python

    def main():
        err, times, times2 = run_convergence_analysis()
        ns = np.arange(4, 18, 2)
        ps = np.arange(1, 4)
        qs = np.arange(4, 11, 3)
        plot_results(err, times, times2, ns, ps, qs)
        plt.show()
        plt.close('all')

Conclusion
----------
This script demonstrates how to use finite element methods to solve the 2D Poisson problem, analyze convergence, and measure performance. By running the code, you can generate plots that show how the error decreases with mesh refinement and polynomial order, and how JIT compilation speeds up repeated runs.

**How to run:**
.. code-block:: bash

    python scripts/two_d_poisson.py 