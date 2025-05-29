Test: Quadrature
================

This walkthrough explains the unit tests for numerical quadrature in the MRX package, as implemented in `test_quadrature.py`.

Introduction
------------
This test module verifies the correctness of quadrature rules for 1D and 3D integration, using various basis functions and domains.

Setup and Imports
-----------------
The test uses Python's `unittest` framework, JAX, and NumPy for numerical operations:

.. code-block:: python

    import unittest
    import jax
    import jax.numpy as jnp
    import numpy.testing as npt
    from mrx.DifferentialForms import DifferentialForm
    from mrx.Quadrature import QuadratureRule
    jax.config.update("jax_enable_x64", True)

Test Structure
--------------
The main test class is `TestQuadrature`, which includes tests for:

- 1D periodic quadrature (integration of trigonometric and exponential functions)
- 3D mixed quadrature (integration over [0,1]^3 with various functions)

Each test checks for numerical accuracy and convergence.

How to Run
----------
To run this test, execute:

.. code-block:: bash

    python -m unittest test/test_quadrature.py 