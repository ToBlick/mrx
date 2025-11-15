Toroid Cavity
==============

This script solves eigenvalue problems for a toroidal cavity.
The script is located at ``scripts/interactive/toroid_cavity.py``.

The script demonstrates:

- Computing eigenvalues and eigenmodes for toroidal cavities
- Visualizing eigenmodes
- Analyzing cavity resonances

Usage:

.. code-block:: bash

    python scripts/interactive/toroid_cavity.py

The script generates plots showing eigenvalues and eigenmode visualizations.

Code Walkthrough
================

This script computes electromagnetic eigenmodes for a toroidal cavity (2D cross-section
with periodic boundary in toroidal direction):

**Block 1: Imports and Configuration (lines 1-21)**
   Imports modules and sets up output directory. Configures discretization parameters:
   ``n=8`` elements, ``p=3`` polynomial degree, creating a 2D toroidal cross-section.

**Block 2: Domain Setup (lines 23-38)**
   Defines toroidal geometry:
   
   - Minor radius: :math:`a=1`
   - Major radius: :math:`R_0=2.1`
   - Aspect ratio: :math:`\epsilon = a/R_0 \approx 0.476`
   - Uses ``toroid_map`` for coordinate transformation

**Block 3: DeRham Sequence and Operators (lines 40-56)**
   Sets up finite element spaces:
   
   - Boundary conditions: clamped in radial, periodic in poloidal, constant in toroidal
   - Assembles mass matrices: :math:`M_0, M_1, M_2, M_3`
   - Constructs gradient operator: :math:`D_0 = \nabla` (strong gradient)
   - Builds double curl matrix:
   
     .. math::
     
         C = M_1 (\Delta_1 + \nabla \nabla \cdot)
     
     This represents the curl-curl operator :math:`\nabla \times \nabla \times` for electromagnetic modes.

**Block 4: Eigenvalue Computation (lines 58-68)**
   Solves generalized eigenvalue problem:
   
   .. math::
   
       C \mathbf{v} = \lambda M_1 \mathbf{v}
   
   - Extracts real eigenvalues and eigenvectors
   - Filters finite eigenvalues
   - Sorts in ascending order

**Block 5: Visualization (lines 71-215)**
   Generates plots:
   
   - Eigenvalue spectrum: First 40 eigenvalues normalized by ``π²``
   - Eigenmode visualization: Plots pushforward of eigenvectors on 2D cross-section
     (``ζ=0.5``) showing field magnitude as contour plots
   - Creates grid of eigenmode plots (first 25 modes) using ``plot_eigenvectors_grid()``

The toroidal geometry introduces curvature effects that modify the eigenmode structure
compared to cylindrical cavities, making this a more complex test case for the
electromagnetic solver.

Full script:

.. literalinclude:: ../../scripts/interactive/toroid_cavity.py
   :language: python
   :linenos:
