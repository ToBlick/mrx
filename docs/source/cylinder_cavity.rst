Cylinder Cavity
===============

.. note::
   For general information about finite element discretization, basis functions, mesh parameters,
   polynomial degrees, boundary conditions, and matrix/operator dimensions, see :doc:`overview`.

This script solves eigenvalue problems for a cylindrical cavity.
The script is located at ``scripts/interactive/cylinder_cavity.py``.

Mathematical Problem
====================

The script computes electromagnetic eigenmodes (TE and TM modes) for a cylindrical cavity
with periodic boundary conditions in the axial direction. The eigenvalue problem is:

.. math::

    \nabla \times \nabla \times \mathbf{E} = k^2 \mathbf{E}

where:
- :math:`\mathbf{E}: \Omega \to \mathbb{R}^3` is the electric field (1-form)
- :math:`k^2` is the eigenvalue (square of the wavenumber)
- :math:`\Omega` is a cylinder of radius :math:`a=1` and height :math:`h=1`

**Boundary Conditions**

- Radial direction: Clamped (perfect conductor boundary)
- Azimuthal direction: Periodic (rotational symmetry)
- Axial direction: Periodic (periodic boundary conditions)

**Analytical Solutions**

For TE modes (transverse electric):

.. math::

    k^2 = \left(\frac{j'_{nm}}{a}\right)^2 + \left(\frac{2\pi k_{\text{axial}}}{h}\right)^2

where :math:`j'_{nm}` is the :math:`m`-th positive root of :math:`J'_n(x) = 0` (derivative of Bessel function).

For TM modes (transverse magnetic):

.. math::

    k^2 = \left(\frac{j_{nm}}{a}\right)^2 + \left(\frac{2\pi k_{\text{axial}}}{h}\right)^2

where :math:`j_{nm}` is the :math:`m`-th positive root of :math:`J_n(x) = 0` (Bessel function).

The script demonstrates:

- Computing eigenvalues and eigenmodes for cylindrical cavities
- Visualizing eigenmodes
- Analyzing cavity resonances

Usage:

.. code-block:: bash

    python scripts/interactive/cylinder_cavity.py

The script generates plots showing eigenvalues and eigenmode visualizations.

Mathematical Formulation
=========================

**Finite Element Discretization**

The electric field is represented as a 1-form:

.. math::

    V_1 = \text{span}\{\Lambda_1^i\}_{i=1}^{N_1}

where :math:`N_1` is the number of 1-form DOFs.

**Mass Matrices**

The 1-form mass matrix :math:`M_1 \in \mathbb{R}^{N_1 \times N_1}`:

.. math::

    (M_1)_{ij} = \int_\Omega \Lambda_1^i(x) \cdot G^{-1}(x) \Lambda_1^j(x) \det(DF(x)) \, dx

Dimensions: :math:`N_1 \times N_1`.

The 0-form mass matrix :math:`M_0 \in \mathbb{R}^{N_0 \times N_0}`:

.. math::

    (M_0)_{ij} = \int_\Omega \Lambda_0^i(x) \Lambda_0^j(x) \det(DF(x)) \, dx

Dimensions: :math:`N_0 \times N_0`.

**Derivative Operators**

The strong gradient operator :math:`D_0: V_0 \to V_1`:

.. math::

    (D_0)_{ij} = \int_\Omega \Lambda_1^i(x) \cdot G^{-1}(x) \nabla \Lambda_0^j(x) \det(DF(x)) \, dx

Dimensions: :math:`N_1 \times N_0`.

The 1-form Laplacian :math:`\Delta_1 \in \mathbb{R}^{N_1 \times N_1}`:

.. math::

    \Delta_1 = M_1^{-1} \text{curl\_curl} - \text{strong\_grad} \circ \text{weak\_div}

where:

.. math::

    (\text{curl\_curl})_{ij} = \int_\Omega \nabla \times \Lambda_1^i(x) \cdot G(x) \nabla \times \Lambda_1^j(x) \frac{1}{\det(DF(x))} \, dx

**Double Curl Matrix**

The double curl operator is constructed as:

.. math::

    C = M_1 (\Delta_1 + \text{strong\_grad} \circ \text{weak\_div})

Since :math:`\Delta_1 = M_1^{-1} \text{curl\_curl} - \text{strong\_grad} \circ \text{weak\_div}`, adding :math:`\text{strong\_grad} \circ \text{weak\_div}`
cancels the gradient-divergence term, leaving :math:`C = M_1 \cdot M_1^{-1} \text{curl\_curl} = \text{curl\_curl}`.

This represents the curl-curl operator :math:`\nabla \times \nabla \times` for electromagnetic modes.
The matrix :math:`C` has dimensions :math:`N_1 \times N_1`.

**Generalized Eigenvalue Problem**

The eigenvalue problem is formulated as a generalized eigenvalue problem:

.. math::

    Q \mathbf{v} = \lambda P \mathbf{v}

where:

.. math::

    Q = \begin{bmatrix} C & D_0 \\ D_0^T & 0 \end{bmatrix}, \quad
    P = \begin{bmatrix} M_1 & 0 \\ 0 & 0 \end{bmatrix}

Dimensions:
- :math:`Q \in \mathbb{R}^{(N_1+N_0) \times (N_1+N_0)}`
- :math:`P \in \mathbb{R}^{(N_1+N_0) \times (N_1+N_0)}`
- :math:`\mathbf{v} \in \mathbb{R}^{N_1+N_0}` is the eigenvector
- :math:`\lambda = k^2` is the eigenvalue

The block structure enforces the constraint :math:`\nabla \cdot \mathbf{E} = 0` (divergence-free condition).

**Eigenvalue Extraction**

The generalized eigenvalue problem is solved using SciPy's ``eig`` function:

.. math::

    \text{eig}(Q, P) = (\lambda_i, \mathbf{v}_i)_{i=1}^{N_1+N_0}

Finite eigenvalues are extracted (filtering out infinite eigenvalues corresponding to
the null space of :math:`P`), and eigenvalues are sorted in ascending order.

**Mode Classification**

Eigenmodes are classified as:
- **TE modes**: Transverse electric modes (electric field perpendicular to axis)
- **TM modes**: Transverse magnetic modes (magnetic field perpendicular to axis)

The classification is based on comparison with analytical solutions.

Code Walkthrough
----------------

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
