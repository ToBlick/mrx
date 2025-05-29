Test: Lazy Matrices
===================

This walkthrough explains the unit tests for lazy matrix calculations in the MRX package, as implemented in `test_lazy_matrices.py`.

Introduction
------------
This test module verifies the correctness and numerical stability of lazy matrix implementations, focusing on mass, derivative, double curl, and stiffness matrices for different differential form degrees.

Setup and Imports
-----------------
The test uses Python's `unittest` framework, JAX, and NumPy for numerical operations:

.. code-block:: python

    import unittest
    import jax
    import jax.numpy as jnp
    import numpy.testing as npt
    from mrx.DifferentialForms import DifferentialForm
    from mrx.LazyMatrices import LazyDerivativeMatrix, LazyDoubleCurlMatrix, LazyMassMatrix, LazyStiffnessMatrix
    from mrx.Quadrature import QuadratureRule
    jax.config.update("jax_enable_x64", True)

Test Structure
--------------
The main test class is `TestLazyMatrices`, which includes tests for:

- Mass matrix properties (shape, symmetry, positive definiteness)
- Derivative matrices (grad, curl, div)
- Double curl matrix
- Stiffness matrix

Each test checks for NaN/Inf values, matrix properties, and mathematical correctness.

How to Run
----------
To run this test, execute:

.. code-block:: bash

    python -m unittest test/test_lazy_matrices.py 