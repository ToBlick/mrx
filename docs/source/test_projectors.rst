Test: Projectors
================

This walkthrough explains the unit tests for projector operators in the MRX package, as implemented in `test_projectors.py`.

Introduction
------------
This test module verifies the correct setup and configuration of projector and curl projector classes for finite element simulations.

Setup and Imports
-----------------
The test uses Python's `unittest` framework and JAX for numerical operations:

.. code-block:: python

    import unittest
    import jax
    from mrx.DifferentialForms import DifferentialForm
    from mrx.Quadrature import QuadratureRule
    jax.config.update("jax_enable_x64", True)

Test Structure
--------------
The main test class is `TestProjectors`, which sets up test fixtures for different forms and quadrature rules:

.. code-block:: python

    class TestProjectors(unittest.TestCase):
        def setUp(self):
            self.ns = (8, 8, 1)
            self.ps = (3, 3, 0)
            self.types = ('periodic', 'periodic', 'constant')
            self.Λ0 = DifferentialForm(0, self.ns, self.ps, self.types)
            self.Λ1 = DifferentialForm(1, self.ns, self.ps, self.types)
            self.Λ2 = DifferentialForm(2, self.ns, self.ps, self.types)
            self.Λ3 = DifferentialForm(3, self.ns, self.ps, self.types)
            self.Q = QuadratureRule(self.Λ0, 5)
            self.F = lambda x: x

How to Run
----------
To run this test, execute:

.. code-block:: bash

    python -m unittest test/test_projectors.py 