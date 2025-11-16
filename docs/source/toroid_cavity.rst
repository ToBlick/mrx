Toroid Cavity
==============

.. note::
   For general information about finite element discretization, basis functions, mesh parameters,
   polynomial degrees, boundary conditions, and matrix/operator dimensions, see :doc:`overview`.

This script solves eigenvalue problems for a toroidal cavity.
The script is located at ``scripts/interactive/toroid_cavity.py``.

Mathematical Problem
====================

This script computes electromagnetic eigenmodes for a toroidal cavity (2D cross-section
with periodic boundary in toroidal direction). The eigenvalue problem is:

.. math::

    \nabla \times \nabla \times \mathbf{E} = k^2 \mathbf{E}

where:
- :math:`\mathbf{E}: \Omega \to \mathbb{R}^3` is the electric field (1-form)
- :math:`k^2` is the eigenvalue (square of the wavenumber)
- :math:`\Omega` is a toroidal domain

**Toroidal Geometry**

The toroidal domain is parameterized by:
- Minor radius: :math:`a=1`
- Major radius: :math:`R_0=2.1`
- Aspect ratio: :math:`\epsilon = a/R_0 \approx 0.476`

The mapping from logical coordinates :math:`(r, \chi, \zeta)` to physical coordinates is:

.. math::

    F(r, \chi, \zeta) = \begin{bmatrix}
        (R_0 + \epsilon r \cos(2\pi\chi)) \cos(2\pi\zeta) \\
        (R_0 + \epsilon r \cos(2\pi\chi)) \sin(2\pi\zeta) \\
        \epsilon r \sin(2\pi\chi)
    \end{bmatrix}

**Boundary Conditions**

- Radial direction: Clamped (perfect conductor boundary)
- Poloidal direction: Periodic (rotational symmetry)
- Toroidal direction: Constant (2D cross-section)

The script demonstrates:

- Computing eigenvalues and eigenmodes for toroidal cavities
- Visualizing eigenmodes
- Analyzing cavity resonances

Usage:

.. code-block:: bash

    python scripts/interactive/toroid_cavity.py

The script generates plots showing eigenvalues and eigenmode visualizations.

Mathematical Formulation
=========================

**Finite Element Discretization**

The electric field is represented as a 1-form:

.. math::

    V_1 = \text{span}\{\Lambda_1^i\}_{i=1}^{N_1}

where :math:`N_1` is the number of 1-form DOFs.

**Mass Matrix**

The 1-form mass matrix :math:`M_1 \in \mathbb{R}^{N_1 \times N_1}`:

.. math::

    (M_1)_{ij} = \int_\Omega \Lambda_1^i(x) \cdot G^{-1}(x) \Lambda_1^j(x) \det(DF(x)) \, dx

Dimensions: :math:`N_1 \times N_1`.

**Derivative Operators**

The strong gradient operator :math:`D_0: V_0 \to V_1`:

.. math::

    (D_0)_{ij} = \int_\Omega \Lambda_1^i(x) \cdot G^{-1}(x) \nabla \Lambda_0^j(x) \det(DF(x)) \, dx

Dimensions: :math:`N_1 \times N_0`.

**Double Curl Matrix**

The double curl operator is constructed as:

.. math::

    C = M_1 (\Delta_1 + \text{strong\_grad} \circ \text{weak\_div})

where:
- :math:`\Delta_1` is the 1-form Laplacian
- :math:`\text{strong\_grad} \circ \text{weak\_div}` is the gradient-divergence term

This represents the curl-curl operator :math:`\nabla \times \nabla \times` for electromagnetic modes.

**Eigenvalue Problem**

The eigenvalue problem is:

.. math::

    C \mathbf{v} = \lambda M_1 \mathbf{v}

where:
- :math:`C \in \mathbb{R}^{N_1 \times N_1}` is the double curl matrix
- :math:`M_1 \in \mathbb{R}^{N_1 \times N_1}` is the mass matrix
- :math:`\mathbf{v} \in \mathbb{R}^{N_1}` is the eigenvector
- :math:`\lambda = k^2` is the eigenvalue

**Eigenvalue Extraction**

The eigenvalue problem is solved using SciPy's ``eig`` function:

.. math::

    \text{eig}(C, M_1) = (\lambda_i, \mathbf{v}_i)_{i=1}^{N_1}

Finite eigenvalues are extracted and sorted in ascending order.

**Toroidal Curvature Effects**

The toroidal geometry introduces curvature through:
- **Jacobian determinant**: :math:`J(x) = \det(DF(x)) = \epsilon (R_0 + \epsilon r \cos(2\pi\chi))`
- **Metric tensor**: :math:`G(x) = DF(x)^T DF(x)` (accounts for non-orthogonal coordinates)
- **Inverse metric**: :math:`G^{-1}(x)` (used in mass matrix computation)

These geometric factors modify the eigenmode structure compared to cylindrical cavities.

Code Walkthrough
----------------

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
