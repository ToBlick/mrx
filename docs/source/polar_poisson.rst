Poisson Problem on a disc
==========================

This tutorial demonstrates solving a Poisson problem on a disc using polar coordinates.
The script is located at ``scripts/tutorials/polar_poisson.py``.

For this problem, we consider the source-solution pair $-\Delta u = f$

.. math::

    u(r) = \frac 1 {27} \left( r^3 (3 \log r - 2) + 2 \right),\\
    f(r) = - r \log r

The script demonstrates:

- Setting up finite element spaces with polar coordinates
- Handling the singularity at the axis using polar splines
- Assembling stiffness matrices and projectors
- Solving the Poisson equation and analyzing convergence

To run the script:

.. code-block:: bash

    python scripts/tutorials/polar_poisson.py

The script generates convergence plots showing error vs. mesh size for different polynomial orders.

Code Walkthrough
================

The script is organized into several logical blocks:

**Block 1: Imports and Configuration (lines 1-33)**
   This block imports necessary libraries (JAX, NumPy, Matplotlib) and MRX modules
   for DeRham sequences, discrete functions, and polar mappings. It enables 64-bit
   precision for numerical stability and creates an output directory for generated plots.

**Block 2: Error Computation Function (lines 37-105)**
   The ``get_err`` function is JIT-compiled for efficiency and computes the relative
   L2 error for a given mesh size ``n``, polynomial degree ``p``, and quadrature order ``q``.
   
   It defines the exact solution and source term:
   
   .. math::
   
       u(r) &= \frac{1}{27} \left( r^3 (3 \log r - 2) + 2 \right) \\
       f(r) &= -r \log r
   
   The function sets up a DeRham sequence with polar coordinates,
   assembles the mass matrix :math:`M_0` and Laplacian :math:`\Delta_0`, solves the linear system:
   
   .. math::
   
       M_0 \Delta_0 \hat{u} = P_0(f)
   
   and computes the relative L2 error using quadrature:
   
   .. math::
   
       \text{error} = \frac{\|u - u_h\|_{L^2}}{\|u\|_{L^2}}

**Block 3: Convergence Analysis (lines 108-151)**
   The ``run_convergence_analysis`` function performs two runs over different mesh sizes
   and polynomial degrees. The first run includes JIT compilation overhead, while the
   second run measures pure computation time. This allows comparison of JIT compilation
   impact on performance. Results are stored in arrays for error and timing data.

**Block 4: Plotting Functions (lines 154-224)**
   The ``plot_results`` function generates four plots:
   
   - Error convergence: Log-log plot of error vs. number of elements for each polynomial degree
   - Timing (first run): Shows computation time including JIT compilation
   - Timing (second run): Shows computation time after JIT compilation
   - Speedup factor: Compares first vs. second run to demonstrate JIT compilation benefits

**Block 5: Main Execution (lines 227-243)**
   The ``main`` function orchestrates the analysis by:
   
   1. Defining parameter ranges: ``ns = [6, 8, 10, 12, 14, 16]`` and ``ps = [1, 2, 3, 4]``
   2. Running convergence analysis for all parameter combinations
   3. Generating and saving plots
   4. Displaying figures and cleaning up

The script uses polar splines to handle the coordinate singularity at r=0, which is
essential for accurate solutions on disc domains. The convergence analysis demonstrates
how error decreases with increasing mesh refinement and polynomial degree.

Full script:

.. literalinclude:: ../../scripts/tutorials/polar_poisson.py
   :language: python
   :linenos:
