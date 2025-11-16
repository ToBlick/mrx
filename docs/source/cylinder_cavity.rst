Cylinder Cavity
===============

.. note::
   For general information about finite element discretization, basis functions, mesh parameters,
   polynomial degrees, boundary conditions, and matrix/operator dimensions, see :doc:`overview`.

This script solves eigenvalue problems for a cylindrical cavity.
The script is located at ``scripts/interactive/cylinder_cavity.py``.

**Problem Statement**

The script computes electromagnetic eigenmodes (TE and TM modes) for a cylindrical cavity.
The eigenvalue problem is:

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
- :math:`\Omega` is a cylinder of radius :math:`a=1` and height :math:`h=1`
- :math:`\mathbf{n}` is the outward unit normal vector on the boundary

**Boundary Conditions**

- Radial direction: Clamped (perfect conductor boundary :math:`\mathbf{E} \times \mathbf{n} = 0`)
- Azimuthal direction: Periodic (rotational symmetry)
- Axial direction: Periodic (periodic boundary conditions)

**Analytical Solutions**

For TE modes (transverse electric):

.. math::

    k^2 = \left(\frac{j'_{nm}}{a}\right)^2 + \left(\frac{2\pi k_{\mathrm{axial}}}{h}\right)^2

where :math:`j'_{nm}` is the :math:`m`-th positive root of :math:`J'_n(x) = 0` (derivative of Bessel function).

For TM modes (transverse magnetic):

.. math::

    k^2 = \left(\frac{j_{nm}}{a}\right)^2 + \left(\frac{2\pi k_{\mathrm{axial}}}{h}\right)^2

where :math:`j_{nm}` is the :math:`m`-th positive root of :math:`J_n(x) = 0` (Bessel function).

The script demonstrates:

- Computing eigenvalues and eigenmodes for cylindrical cavities
- Visualizing eigenmodes
- Analyzing cavity resonances

Usage:

.. code-block:: bash

    python scripts/interactive/cylinder_cavity.py

The script generates plots showing eigenvalues and eigenmode visualizations.

**Finite Element Discretization**

The electric field is represented as a 1-form:

.. math::

    V_1 = \text{span}\{\Lambda_1^i\}_{i=1}^{N_1}

where :math:`N_1` is the number of 1-form DOFs.

**Matrix and Operator Dimensions**

The 1-form mass matrix :math:`M_1 \in \mathbb{R}^{N_1 \times N_1}` and 0-form mass matrix :math:`M_0 \in \mathbb{R}^{N_0 \times N_0}` are used.

The double curl operator is constructed as:

.. math::

    C = M_1 (\Delta_1 + \nabla_h \circ (\nabla \cdot)_h) = (\nabla \times)_h^T M_2 (\nabla \times)_h

This represents the curl-curl operator :math:`\nabla \times (\nabla \times)` for electromagnetic modes.

**Generalized Eigenvalue Problem**

The eigenvalue problem is formulated as a generalized eigenvalue problem:

.. math::

    Q \mathbf{v} = \lambda P \mathbf{v}

where:

.. math::

    Q = \begin{bmatrix} C & D_0 \\ D_0^T & 0 \end{bmatrix}, \quad
    P = \begin{bmatrix} M_1 & 0 \\ 0 & 0 \end{bmatrix}

The block structure enforces the constraint :math:`\nabla \cdot \mathbf{E} = 0` (divergence-free condition).

Code Walkthrough
----------------

**Block 1: Imports and Configuration (lines 1-43)**

Imports JAX, NumPy, SciPy, Matplotlib, and MRX modules. Enables 64-bit precision
and creates output directory. Sets up parameters for cylinder geometry (radius :math:`a=1`,
height :math:`h=1`) and discretization (:math:`ns=(15,15,1)`, :math:`ps=(3,3,0)`).

.. literalinclude:: ../../scripts/interactive/cylinder_cavity.py
   :language: python
   :lines: 1-43

**Block 2: DeRham Sequence Setup (lines 45-67)**

Creates DeRham sequence with boundary conditions (clamped in radial direction, periodic in azimuthal and axial),
assembles mass matrices :math:`M_0` (0-forms) and :math:`M_1` (1-forms),
assembles gradient operator :math:`D_0` (strong gradient),
and constructs double curl matrix:

.. math::

    C = M_1 (\Delta_1 + \nabla_h \circ (\nabla \cdot)_h) = (\nabla \times)_h^T M_2 (\nabla \times)_h

Builds block matrices :math:`Q` and :math:`P` for generalized eigenvalue problem:

.. math::

    Q = \begin{bmatrix} C & D_0 \\ D_0^T & 0 \end{bmatrix}, \quad P = \begin{bmatrix} M_1 & 0 \\ 0 & 0 \end{bmatrix}

.. literalinclude:: ../../scripts/interactive/cylinder_cavity.py
   :language: python
   :lines: 45-67

**Block 3: Eigenvalue Computation (lines 69-83)**

Solves generalized eigenvalue problem:

.. math::

    Q \mathbf{v} = \lambda P \mathbf{v}

using SciPy:
- Extracts real parts of eigenvalues and eigenvectors
- Filters out infinite eigenvalues
- Sorts eigenvalues in ascending order

.. literalinclude:: ../../scripts/interactive/cylinder_cavity.py
   :language: python
   :lines: 69-83

**Block 4: Analytical Comparison (lines 88-225)**

Defines function ``calculate_cylindrical_periodic_TE_TM_eigenvalues()`` that computes
analytical eigenvalues for comparison.

For TE modes:

.. math::

    k^2 = \left(\frac{j'_{nm}}{a}\right)^2 + \left(\frac{2\pi k_{\mathrm{axial}}}{h}\right)^2

where :math:`j'_{nm}` is the :math:`m`-th positive root of :math:`J'_n(x) = 0`.

For TM modes:

.. math::

    k^2 = \left(\frac{j_{nm}}{a}\right)^2 + \left(\frac{2\pi k_{\mathrm{axial}}}{h}\right)^2

where :math:`j_{nm}` is the :math:`m`-th positive root of :math:`J_n(x) = 0`.

The function accounts for mode multiplicities (azimuthal and axial symmetries)
and computes eigenvalues for modes :math:`n \in [0,7]`, :math:`m \in [1,7]`, :math:`k_{\mathrm{axial}} \in [0]`.

.. literalinclude:: ../../scripts/interactive/cylinder_cavity.py
   :language: python
   :lines: 88-225

**Block 5: Visualization (lines 227-378)**

Generates plots:

- Eigenvalue comparison: Computed vs. analytical eigenvalues (first 40 modes)
- Eigenmode visualization: Plots norm of pushforward of first 25 eigenvectors
  on a 2D cross-section (:math:`z=0.5`) using contour plots
- Uses ``plot_eigenvectors_grid()`` function to create a grid of eigenmode plots

The script validates the numerical method by comparing computed eigenvalues with
analytical solutions for cylindrical cavity modes, demonstrating the accuracy of
the finite element discretization.

.. literalinclude:: ../../scripts/interactive/cylinder_cavity.py
   :language: python
   :lines: 227-378
