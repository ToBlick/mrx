Toroid Cavity
==============

.. note::
   For general information about finite element discretization, basis functions, mesh parameters,
   polynomial degrees, boundary conditions, and matrix/operator dimensions, see :doc:`overview`.

This script solves eigenvalue problems for a toroidal cavity.
The script is located at ``scripts/interactive/toroid_cavity.py``.

**Problem Statement**

This script computes electromagnetic eigenmodes for a toroidal cavity (2D cross-section
with periodic boundary in toroidal direction). The eigenvalue problem is:

.. math::

    \nabla \times \nabla \times \mathbf{E} = k^2 \mathbf{E} \quad \text{in } \Omega

with the constraint:

.. math::

    \nabla \cdot \mathbf{E} = 0 \quad \text{in } \Omega

and boundary conditions:

.. math::

    \mathbf{E} \times \mathbf{n} = 0 \quad \text{on } \partial\Omega

where:
- :math:`\mathbf{E}: \Omega \to \mathbb{R}^3` is the electric field (1-form)
- :math:`k^2` is the eigenvalue (square of the wavenumber)
- :math:`\Omega` is a toroidal domain
- :math:`\mathbf{n}` is the outward unit normal vector on the boundary

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

**Finite Element Discretization**

The electric field is represented as a 1-form:

.. math::

    V_1 = \text{span}\{\Lambda_1^i\}_{i=1}^{N_1}

where :math:`N_1` is the number of 1-form DOFs.

**Matrix and Operator Dimensions**

The 1-form mass matrix :math:`M_1 \in \mathbb{R}^{N_1 \times N_1}` is used.

The double curl operator is constructed as:

.. math::

    C = M_1 (\Delta_1 + \nabla_h \circ (\nabla \cdot)_h) = (\nabla \times)_h^T M_2 (\nabla \times)_h

This represents the curl-curl operator :math:`\nabla \times (\nabla \times)` for electromagnetic modes.

**Eigenvalue Problem**

The eigenvalue problem is:

.. math::

    C \mathbf{v} = \lambda M_1 \mathbf{v}

where :math:`C \in \mathbb{R}^{N_1 \times N_1}` is the double curl matrix, :math:`M_1 \in \mathbb{R}^{N_1 \times N_1}` is the mass matrix,
:math:`\mathbf{v} \in \mathbb{R}^{N_1}` is the eigenvector, and :math:`\lambda = k^2` is the eigenvalue.

**Toroidal Curvature Effects**

The toroidal geometry introduces curvature through:
- **Jacobian determinant**: :math:`J(x) = \det(DF(x)) = \epsilon (R_0 + \epsilon r \cos(2\pi\chi))`
- **Metric tensor**: :math:`G(x) = DF(x)^T DF(x)` (accounts for non-orthogonal coordinates)
- **Inverse metric**: :math:`G^{-1}(x)` (used in mass matrix computation)

These geometric factors modify the eigenmode structure compared to cylindrical cavities.

Code Walkthrough
----------------

**Block 1: Imports and Configuration (lines 1-21)**

Imports modules and sets up output directory. Configures discretization parameters:
:math:`n=8` elements, :math:`p=3` polynomial degree, creating a 2D toroidal cross-section.

.. literalinclude:: ../../scripts/interactive/toroid_cavity.py
   :language: python
   :lines: 1-21

**Block 2: Domain Setup (lines 23-38)**

Defines toroidal geometry:

- Minor radius: :math:`a=1`
- Major radius: :math:`R_0=2.1`
- Aspect ratio: :math:`\epsilon = a/R_0 \approx 0.476`
- Uses ``toroid_map`` for coordinate transformation

.. literalinclude:: ../../scripts/interactive/toroid_cavity.py
   :language: python
   :lines: 23-38

**Block 3: DeRham Sequence and Operators (lines 40-56)**

Sets up finite element spaces:

- Boundary conditions: clamped in radial, periodic in poloidal, constant in toroidal
- Assembles mass matrices: :math:`M_0, M_1, M_2, M_3`
- Constructs gradient operator: :math:`D_0 = \nabla_h` (strong gradient)
- Builds double curl matrix:

.. math::

    C = M_1 (\Delta_1 + \nabla_h \circ (\nabla \cdot)_h) = (\nabla \times)_h^T M_2 (\nabla \times)_h

This represents the curl-curl operator :math:`\nabla \times (\nabla \times)` for electromagnetic modes.

.. literalinclude:: ../../scripts/interactive/toroid_cavity.py
   :language: python
   :lines: 40-56

**Block 4: Eigenvalue Computation (lines 58-68)**

Solves generalized eigenvalue problem:

.. math::

    C \mathbf{v} = \lambda M_1 \mathbf{v}

- Extracts real eigenvalues and eigenvectors
- Filters finite eigenvalues
- Sorts in ascending order

.. literalinclude:: ../../scripts/interactive/toroid_cavity.py
   :language: python
   :lines: 58-68

**Block 5: Visualization (lines 70-215)**

Generates plots:

- Eigenvalue spectrum: First 40 eigenvalues normalized by :math:`\pi^2`
- Eigenmode visualization: Plots pushforward of eigenvectors on 2D cross-section
  (:math:`\zeta=0.5`) showing field magnitude as contour plots
- Creates grid of eigenmode plots (first 25 modes) using ``plot_eigenvectors_grid()``

The toroidal geometry introduces curvature effects that modify the eigenmode structure
compared to cylindrical cavities, making this a more complex test case for the
electromagnetic solver.

.. literalinclude:: ../../scripts/interactive/toroid_cavity.py
   :language: python
   :lines: 70-214
