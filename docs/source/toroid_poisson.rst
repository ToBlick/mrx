Toroid Poisson Problem
=======================

This tutorial demonstrates solving a Poisson problem on a toroidal domain.
The script is located at ``scripts/tutorials/toroid_poisson.py``.

The script demonstrates:

- Setting up finite element spaces on a toroidal domain
- Using toroidal mappings
- Solving Poisson equations in toroidal geometry
- Convergence analysis

To run the script:

.. code-block:: bash

    python scripts/tutorials/toroid_poisson.py

The script generates convergence plots showing error vs. mesh size.

Code Walkthrough
================

This script extends the polar Poisson example to 3D toroidal geometry:

**Block 1: Imports and Configuration (lines 1-29)**
   Imports libraries and MRX modules, with ``toroid_map`` instead of ``polar_map``.
   Enables 64-bit precision and creates output directory.

**Block 2: Error Computation Function (lines 32-102)**
   The ``get_err`` function solves a 3D Poisson problem:
   
   .. math::
   
       u(r, \theta, \zeta) &= (r^2 - r^4) \cos(2\pi \zeta) \\
       f(r, \theta, \zeta) &= \cos(2\pi \zeta) \left[ -\frac{4}{\epsilon^2}(1-4r^2) - \frac{4}{\epsilon R}\left(\frac{r}{2}-r^3\right)\cos(2\pi\theta) + \frac{r^2-r^4}{R^2} \right]
   
   where :math:`R = 1 + \epsilon r \cos(2\pi\theta)` is the major radius coordinate
   and :math:`\epsilon = 1/3` is the toroidal aspect ratio. The solution is independent of :math:`\theta`.
   
   The DeRham sequence uses 3D splines with periodic boundary conditions in both
   :math:`\theta` (poloidal) and :math:`\zeta` (toroidal) directions. The system is solved as:
   
   .. math::
   
       M_0 \Delta_0 \hat{u} = P_0(f)

**Block 3: Convergence Analysis (lines 105-150)**
   Similar structure to previous examples, but uses smaller parameter ranges:
   ``ns = [4, 6, 8]`` and ``ps = [1, 2, 3]`` due to the increased computational
   cost of 3D problems.

**Block 4: Plotting Functions (lines 153-223)**
   Generates the same four plots as previous examples, adapted for toroidal geometry.

**Block 5: Main Execution (lines 226-242)**
   Runs the convergence analysis and generates plots.

The toroidal mapping transforms logical coordinates ``(r, θ, ζ)`` to physical
cylindrical coordinates ``(R, φ, Z)``, where the toroidal geometry introduces
curvature effects that must be properly handled by the finite element discretization.

Full script:

.. literalinclude:: ../../scripts/tutorials/toroid_poisson.py
   :language: python
   :linenos:

