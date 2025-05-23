Test: Boundary Conditions
=========================

This walkthrough explains the unit tests for boundary condition operators in the MRX package, as implemented in `test_boundary_conditions.py`.

Introduction
------------
This test module verifies the correct setup and configuration of boundary condition operators for finite element simulations. It ensures that the test environment is initialized with the appropriate parameters for grid size, polynomial degree, and quadrature order.

Setup and Imports
-----------------
The test uses Python's `unittest` framework and JAX for numerical operations:

.. code-block:: python

    import unittest
    import jax
    jax.config.update("jax_enable_x64", True)

Test Structure
--------------
The main test class is `TestBoundaryConditions`, which sets up parameters for the tests:

.. code-block:: python

    class TestBoundaryConditions(unittest.TestCase):
        def setUp(self):
            self.ns = (5, 5, 5)
            self.ps = (3, 3, 3)
            self.quad_order = 5

How to Run
----------
To run this test, execute:

.. code-block:: bash

    python -m unittest test/test_boundary_conditions.py 