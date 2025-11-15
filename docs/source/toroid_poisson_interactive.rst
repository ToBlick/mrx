Toroid Poisson (Interactive)
=============================

This script solves a Poisson problem on a toroidal domain interactively.
The script is located at ``scripts/interactive/toroid_poisson.py``.

The script demonstrates:

- Setting up finite element spaces on a toroidal domain
- Solving Poisson equations in toroidal geometry
- Interactive visualization of results

Usage:

.. code-block:: bash

    python scripts/interactive/toroid_poisson.py

Code Walkthrough
================

This script is similar to ``toroid_poisson.py`` but focuses on interactive exploration
and additional diagnostics:

**Block 1: Imports and Setup (lines 1-14)**
   Imports modules and enables 64-bit precision. Uses ``toroid_map`` for domain geometry.

**Block 2: Error and Diagnostics Function (lines 17-100)**
   The ``get_err()`` function computes error and additional diagnostics:
   
   - Exact solution:
   
     .. math::
     
         u(r,\chi,z) = \frac{1}{4}(r^2 - r^4) \cos(2\pi z)
   
   - Source term:
   
     .. math::
     
         f(r,\chi,z) = \cos(2\pi z) \left[ -\frac{1}{a^2}(1-4r^2) - \frac{1}{aR}\left(\frac{r}{2}-r^3\right)\cos(2\pi\chi) + \frac{1}{4}\frac{r^2-r^4}{R^2} \right]
     
     where :math:`R = R_0 + a r \cos(2\pi\chi)` and :math:`\epsilon = a/R_0 = 1/3` is the aspect ratio.
   
   - Sets up DeRham sequence with toroidal mapping
   - Assembles mass matrix :math:`M_0` and Laplacian :math:`\Delta_0`
   - Solves system:
   
     .. math::
     
         M_0 \Delta_0 \hat{u} = P_0(f)
   - Computes relative L2 error
   - **Additional diagnostics**: Condition number and sparsity of system matrix

**Block 3: Convergence Analysis (lines 102-152)**
   Runs convergence study with diagnostics:
   
   - Tests mesh sizes :math:`n \in \{4,6,8,10,12\}` and polynomial degrees :math:`p \in \{1,2,3\}`
   - Tracks error, condition number, and matrix sparsity
   - Reports timing information

**Block 4: Visualization and Analysis (lines 154-152)**
   Generates enhanced visualizations:
   
   - Error convergence plots
   - Condition number vs. mesh size (to check numerical stability)
   - Sparsity patterns of system matrices
   - Solution field visualizations on toroidal cross-sections

The interactive version provides more detailed diagnostics than the tutorial version,
making it useful for understanding numerical properties of the discretization.

Full script:

.. literalinclude:: ../../scripts/interactive/toroid_poisson.py
   :language: python
   :linenos:
