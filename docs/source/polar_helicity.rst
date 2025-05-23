Polar Helicity Analysis
=======================

This tutorial walks through a script that analyzes the convergence properties of magnetic helicity calculations in a polar geometry. The script computes and visualizes errors in vector potential, magnetic helicity, and curl of the vector potential for different polynomial degrees and mesh resolutions.

Introduction
------------
We analyze the convergence of magnetic helicity calculations in a polar geometry using finite element methods. The script demonstrates error convergence rates, computational time scaling, and comparison of exact vs reconstructed fields.

Setup and Imports
-----------------
We start by importing the necessary libraries and setting up the output directory:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import time
    from pathlib import Path
    from mrx.DifferentialForms import DifferentialForm
    from mrx.Quadrature import QuadratureRule
    from mrx.Projectors import Projector
    from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix, LazyProjectionMatrix, LazyDoubleCurlMatrix
    from mrx.Utils import curl
    from mrx.PolarMapping import LazyExtractionOperator, get_xi
    from functools import partial
    jax.config.update("jax_enable_x64", True)
    output_dir = Path("scripts_output")
    output_dir.mkdir(parents=True, exist_ok=True)

Defining the Problem
--------------------
We define the finite element spaces, mapping, and error metrics:

.. code-block:: python

    # See script for details on get_error, mapping, and error computation

Convergence and Timing Analysis
-------------------------------
We run the error computation for a range of mesh sizes and polynomial degrees, and measure computation time:

.. code-block:: python

    def get_error(n, p):
        # ... see script for details ...

Plotting Results
----------------
The script generates several plots to visualize the convergence of errors, computational time, and JIT speedup.

Conclusion
----------
This script demonstrates how to use finite element methods to analyze magnetic helicity in polar geometry, analyze convergence, and measure performance.

**How to run:**
.. code-block:: bash

    python scripts/polar_helicity.py 