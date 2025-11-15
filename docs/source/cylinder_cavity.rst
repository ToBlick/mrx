Cylinder Cavity
===============

This script solves eigenvalue problems for a cylindrical cavity.
The script is located at ``scripts/interactive/cylinder_cavity.py``.

The script demonstrates:

- Computing eigenvalues and eigenmodes for cylindrical cavities
- Visualizing eigenmodes
- Analyzing cavity resonances

Usage:

.. code-block:: bash

    python scripts/interactive/cylinder_cavity.py

The script generates plots showing eigenvalues and eigenmode visualizations.

Code Walkthrough
================

This script computes electromagnetic eigenmodes (TE and TM modes) for a cylindrical
cavity with periodic boundary conditions in the axial direction:

**Block 1: Imports and Configuration (lines 1-20)**
   Imports JAX, NumPy, SciPy, Matplotlib, and MRX modules. Enables 64-bit precision
   and creates output directory. Sets up parameters for cylinder geometry (radius ``a=1``,
   height ``h=1``) and discretization (``ns=(15,15,1)``, ``ps=(3,3,0)``).

**Block 2: DeRham Sequence Setup (lines 22-68)**
   Creates DeRham sequence with:
   
   - Boundary conditions: clamped in radial direction, periodic in azimuthal and axial
   - Assembles mass matrices: :math:`M_0` (0-forms), :math:`M_1` (1-forms)
   - Assembles gradient operator: :math:`D_0` (strong gradient)
   - Constructs double curl matrix:
   
     .. math::
     
         C = M_1 (\Delta_1 + \nabla \nabla \cdot)
     
   - Builds block matrices :math:`Q` and :math:`P` for generalized eigenvalue problem:
   
     .. math::
     
         Q = \begin{bmatrix} C & D_0 \\ D_0^T & 0 \end{bmatrix}, \quad
         P = \begin{bmatrix} M_1 & 0 \\ 0 & 0 \end{bmatrix}

**Block 3: Eigenvalue Computation (lines 70-83)**
   Solves generalized eigenvalue problem:
   
   .. math::
   
       Q \mathbf{v} = \lambda P \mathbf{v}
   
   using SciPy:
   
   - Extracts real parts of eigenvalues and eigenvectors
   - Filters out infinite eigenvalues
   - Sorts eigenvalues in ascending order

**Block 4: Analytical Comparison (lines 88-221)**
   Defines function ``calculate_cylindrical_periodic_TE_TM_eigenvalues()`` that computes
   analytical eigenvalues for comparison:
   
   For TE modes:
   
   .. math::
   
       k^2 = \left(\frac{j'_{nm}}{a}\right)^2 + \left(\frac{2\pi k_{\text{axial}}}{h}\right)^2
   
   where :math:`j'_{nm}` is the :math:`m`-th positive root of :math:`J'_n(x) = 0`.
   
   For TM modes:
   
   .. math::
   
       k^2 = \left(\frac{j_{nm}}{a}\right)^2 + \left(\frac{2\pi k_{\text{axial}}}{h}\right)^2
   
   where :math:`j_{nm}` is the :math:`m`-th positive root of :math:`J_n(x) = 0`.
   
   The function accounts for mode multiplicities (azimuthal and axial symmetries)
   and computes eigenvalues for modes :math:`n \in [0,7]`, :math:`m \in [1,7]`, :math:`k_{\text{axial}} \in [0]`.

**Block 5: Visualization (lines 224-378)**
   Generates plots:
   
   - Eigenvalue comparison: Computed vs. analytical eigenvalues (first 40 modes)
   - Eigenmode visualization: Plots norm of pushforward of first 25 eigenvectors
     on a 2D cross-section (``z=0.5``) using contour plots
   - Uses ``plot_eigenvectors_grid()`` function to create a grid of eigenmode plots

The script validates the numerical method by comparing computed eigenvalues with
analytical solutions for cylindrical cavity modes, demonstrating the accuracy of
the finite element discretization.

Full script:

.. literalinclude:: ../../scripts/interactive/cylinder_cavity.py
   :language: python
   :linenos:
