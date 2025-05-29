Two-Dimensional Helmholtz Decomposition
=======================================

This script demonstrates the Helmholtz decomposition of a vector field into its irrotational (gradient) and solenoidal (curl) components using finite element methods. The decomposition is performed using the Leray projector.

**Key features:**
- Sets up a test vector field and performs Helmholtz decomposition
- Computes and plots the results
- Analyzes the accuracy of the decomposition
- Visualizes fields and singular values

**How to run:**
.. code-block:: bash

    python scripts/two_d_helmholtz_decomposition.py

**Main steps:**
- Set up the finite element problem and test functions
- Perform the decomposition using the Leray projector
- Compute errors and visualize results
- Save plots to `script_outputs/helmholtz/` 