Mixed Polar Poisson Problem
============================

This tutorial demonstrates solving a mixed formulation of the Poisson problem on a disc.
The script is located at ``scripts/tutorials/mixed_polar_poisson.py``.

The mixed formulation rewrites the Poisson equation as a system:

.. math::

    \nabla \cdot \sigma = f \\
    -\nabla u = \sigma

The script demonstrates:

- Mixed finite element formulation
- Handling polar coordinates and axis singularity
- Convergence analysis for mixed methods
- Performance comparison with standard formulation

To run the script:

.. code-block:: bash

    python scripts/tutorials/mixed_polar_poisson.py

The script generates convergence plots and performance comparisons.

Code Walkthrough
================

The script follows a similar structure to ``polar_poisson.py`` but uses a mixed finite
element formulation:

**Block 1: Imports and Configuration (lines 1-39)**
   Imports necessary libraries and MRX modules. Note that this script uses ``Pushforward``
   in addition to ``DiscreteFunction``, which is needed for the mixed formulation.
   Creates output directory and enables 64-bit precision.

**Block 2: Error Computation Function (lines 42-113)**
   The ``get_err`` function implements the mixed formulation:
   
   .. math::
   
       u(r) &= -\frac{1}{16}r^4 + \frac{1}{12}r^3 + \frac{1}{48} \\
       f(r) &= r^2 - \frac{3}{4}r
   
   Boundary conditions are homogeneous Neumann (:math:`\partial u/\partial n = 0`).
   
   The mixed formulation uses 3-forms (volume forms) instead of 0-forms. It assembles:
   
   - :math:`M_2`: Mass matrix for 2-forms (for the flux variable :math:`\sigma`)
   - :math:`M_3`: Mass matrix for 3-forms (for the solution :math:`u`)
   - :math:`\Delta_3`: Laplacian operator for 3-forms (strong divergence composed with weak gradient)
   
   The system is solved as:
   
   .. math::
   
       M_3 \Delta_3 \hat{u} = P_3(f)
   
   and the solution is pushed forward to physical space using ``Pushforward``. Error is computed using relative L2 norm.

**Block 3: Convergence Analysis (lines 116-159)**
   Identical to ``polar_poisson.py``: runs convergence analysis twice (with and without
   JIT compilation overhead) to measure performance and error convergence.

**Block 4: Plotting Functions (lines 162-231)**
   Generates the same four plots as ``polar_poisson.py``:
   - Error convergence plot
   - Timing plots (first and second run)
   - JIT compilation speedup plot

**Block 5: Main Execution (lines 234-250)**
   Runs the analysis with the same parameter ranges as ``polar_poisson.py`` and generates plots.

The key difference from the standard formulation is that the mixed method solves for both
the solution :math:`u` and the flux :math:`\sigma = -\nabla u` simultaneously, which can be advantageous for
certain types of problems and provides better conservation properties.

Full script:

.. literalinclude:: ../../scripts/tutorials/mixed_polar_poisson.py
   :language: python
   :linenos:

