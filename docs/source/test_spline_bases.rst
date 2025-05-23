Test: Spline Bases
==================

This walkthrough explains the unit tests for spline bases in the MRX package, as implemented in `test_spline_bases.py`.

Introduction
------------
This test module verifies the correct initialization, evaluation, and properties of spline bases, including clamped, periodic, and constant types, as well as tensor product and derivative splines.

Setup and Imports
-----------------
The test uses Python's `unittest` framework, JAX, and NumPy for numerical operations:

.. code-block:: python

    import unittest
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mrx.SplineBases import SplineBasis, TensorBasis, DerivativeSpline
    jax.config.update("jax_enable_x64", True)

Test Structure
--------------
The main test class is `TestSplineBases`, which includes tests for:

- Initialization and error handling
- Spline evaluation and partition of unity
- Derivative and tensor product splines
- Error bounds and edge cases

Each test checks for correct behavior, error conditions, and mathematical properties.

How to Run
----------
To run this test, execute:

.. code-block:: bash

    python -m unittest test/test_spline_bases.py 