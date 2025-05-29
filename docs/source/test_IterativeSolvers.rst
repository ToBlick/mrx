Test: Iterative Solvers
=======================

This walkthrough explains the unit tests for iterative solvers in the MRX package, as implemented in `test_IterativeSolvers.py`.

Introduction
------------
This test module verifies the correctness and convergence of iterative solvers, including Picard iteration and Newton's method, for both scalar and multidimensional problems.

Setup and Imports
-----------------
The test uses Python's `unittest` framework, JAX, and Matplotlib for numerical operations and visualization:

.. code-block:: python

    import unittest
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from mrx.IterativeSolvers import newton_solver, picard_solver
    jax.config.update("jax_enable_x64", True)

Test Structure
--------------
The main test class is `TestIterativeSolvers`, which includes tests for:

- Picard solver (scalar and high-dimensional)
- Newton solver (scalar, multidimensional, and high-dimensional)
- Convergence rates and error analysis

Each test checks for convergence, accuracy, and scaling with problem size.

How to Run
----------
To run this test, execute:

.. code-block:: bash

    python -m unittest test/test_IterativeSolvers.py 