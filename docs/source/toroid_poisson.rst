Toroid Poisson Problem
=======================

.. note::
   For general information about finite element discretization, basis functions, mesh parameters,
   polynomial degrees, boundary conditions, and matrix/operator dimensions, see :doc:`overview`.

This tutorial demonstrates solving a Poisson problem on a toroidal domain.
The script is located at ``scripts/tutorials/toroid_poisson.py``.

**Problem Statement**

We solve the Poisson equation on a toroidal domain :math:`\Omega`:

.. math::

    -\Delta u = f \quad \text{in } \Omega

with homogeneous Dirichlet boundary conditions:

.. math::

    u|_{\partial\Omega} = 0

where:
- :math:`u: \Omega \to \mathbb{R}` is the unknown scalar field (0-form)
- :math:`f: \Omega \to \mathbb{R}` is the given source term
- :math:`\Delta = \nabla \cdot \nabla` is the scalar Laplacian operator
- :math:`\partial\Omega` denotes the boundary of the toroidal domain

The toroidal domain is parameterized by logical coordinates :math:`(r, \chi, \zeta) \in [0,1]^3`:
- :math:`r`: Radial coordinate (minor radius direction)
- :math:`\chi`: Poloidal angle coordinate
- :math:`\zeta`: Toroidal angle coordinate

The mapping :math:`F: [0,1]^3 \to \mathbb{R}^3` transforms logical to physical cylindrical coordinates:

.. math::

    F(r, \chi, \zeta) = (R, \phi, Z)

where:
- :math:`R = R_0 + \epsilon r \cos(2\pi\chi)` is the major radius
- :math:`\phi = 2\pi\zeta` is the toroidal angle
- :math:`Z = \epsilon r \sin(2\pi\chi)` is the vertical coordinate
- :math:`R_0` is the major radius of the torus
- :math:`\epsilon = a/R_0` is the inverse aspect ratio (minor radius :math:`a` divided by major radius)

For this problem, we use:
- :math:`R_0 = 1.0`
- :math:`\epsilon = 1/3` (aspect ratio :math:`A = R_0/a = 3`)

**Exact Solution and Source Term**

The exact solution is:

.. math::

    u(r, \chi, \zeta) = (r^2 - r^4) \cos(2\pi\zeta)

which is independent of the poloidal angle :math:`\chi`.

The corresponding source term is:

.. math::

    f(r, \chi, \zeta) = \cos(2\pi\zeta) \left[ -\frac{4}{\epsilon^2}(1-4r^2) - \frac{4}{\epsilon R}\left(\frac{r}{2}-r^3\right)\cos(2\pi\chi) + \frac{r^2-r^4}{R^2} \right]

where :math:`R = R_0 + \epsilon r \cos(2\pi\chi)`.

The script demonstrates:

- Setting up finite element spaces on a toroidal domain
- Using toroidal mappings
- Solving Poisson equations in toroidal geometry
- Convergence analysis

To run the script:

.. code-block:: bash

    python scripts/tutorials/toroid_poisson.py

The script generates convergence plots showing error vs. mesh size.

**Discretization Parameters**

This script uses:
- **Mesh parameters**: :math:`n_r = n_\chi = n_\zeta = n` elements in each direction
- **Polynomial degrees**: :math:`p_r = p_\chi = p_\zeta = p`
- **Boundary conditions**: Clamped in radial direction, periodic in poloidal and toroidal directions

Following the general formulas in :doc:`overview`, the number of DOFs are:
- **0-forms**: :math:`N_0 = n_r \cdot n_\chi \cdot n_\zeta = n^3`
- **1-forms**: :math:`N_1 = d_r \cdot n_\chi \cdot n_\zeta + n_r \cdot d_\chi \cdot n_\zeta + n_r \cdot n_\chi \cdot d_\zeta = n^2(3n-1)` where :math:`d_r = n-1` (clamped), :math:`d_\chi = d_\zeta = n` (periodic)
- **2-forms**: :math:`N_2 = n_r \cdot d_\chi \cdot d_\zeta + d_r \cdot n_\chi \cdot d_\zeta + d_r \cdot d_\chi \cdot n_\zeta = n^2(3n-2)`
- **3-forms**: :math:`N_3 = d_r \cdot d_\chi \cdot d_\zeta = n^2(n-1)`

**Matrix and Operator Dimensions**

The 0-form mass matrix :math:`M_0 \in \mathbb{R}^{N_0 \times N_0}` where :math:`N_0 = n^3`:

.. math::

    (M_0)_{ij} = \int_\Omega \Lambda_0^i(x) \Lambda_0^j(x) \det(DF(x)) \, dx

The 0-form Laplacian :math:`\Delta_0 \in \mathbb{R}^{N_0 \times N_0}`:

.. math::

    \Delta_0 = M_0^{-1} \nabla_h^T M_1 \nabla_h

where the gradient-gradient matrix :math:`\nabla_h^T M_1 \nabla_h` represents :math:`\nabla \cdot \nabla`.

The discrete Poisson equation:

.. math::

    M_0 \Delta_0 \hat{u} = P_0(f)

where :math:`\hat{u} \in \mathbb{R}^{N_0}` are the solution coefficients and :math:`P_0(f) \in \mathbb{R}^{N_0}` is the projection of the source term.

**Toroidal Geometry Effects**

The toroidal mapping introduces curvature through:
- **Jacobian determinant**: :math:`J(x) = \det(DF(x)) = \epsilon R` (varies with position)
- **Metric tensor**: :math:`G(x) = DF(x)^T DF(x)` (accounts for non-orthogonal coordinates)
- **Inverse metric**: :math:`G^{-1}(x)` (used in Laplacian computation)

These geometric factors must be properly accounted for in the finite element discretization
to maintain accuracy in curved geometries.

Code Walkthrough
----------------

**Block 1: Imports and Configuration (lines 1-29)**

Imports libraries and MRX modules, with ``toroid_map`` instead of ``polar_map``.
Enables 64-bit precision and creates output directory.

.. literalinclude:: ../../scripts/tutorials/toroid_poisson.py
   :language: python
   :lines: 1-29

**Block 2: Error Computation Function (lines 32-102)**

The ``get_err`` function solves a 3D Poisson problem with the exact solution and source term:

.. math::

    u(r, \theta, \zeta) &= (r^2 - r^4) \cos(2\pi \zeta) \\
    f(r, \theta, \zeta) &= \cos(2\pi \zeta) \left[ -\frac{4}{\epsilon^2}(1-4r^2) - \frac{4}{\epsilon R}\left(\frac{r}{2}-r^3\right)\cos(2\pi\theta) + \frac{r^2-r^4}{R^2} \right]

where :math:`R = 1 + \epsilon r \cos(2\pi\theta)` is the major radius coordinate
and :math:`\epsilon = 1/3` is the toroidal aspect ratio. The solution is independent of :math:`\theta`.

The DeRham sequence uses 3D splines with periodic boundary conditions in both
:math:`\theta` (poloidal) and :math:`\zeta` (toroidal) directions. The system is solved as:

.. math::

    M_0 \Delta_0 \hat{u} = P_0(f)

Error is computed using relative L2 norm.

.. literalinclude:: ../../scripts/tutorials/toroid_poisson.py
   :language: python
   :lines: 32-102

**Block 3: Convergence Analysis (lines 105-150)**

Runs convergence analysis twice (with and without JIT compilation overhead) to measure performance and error convergence.
Uses smaller parameter ranges: ``ns = [4, 6, 8]`` and ``ps = [1, 2, 3]`` due to the increased computational
cost of 3D problems.

.. literalinclude:: ../../scripts/tutorials/toroid_poisson.py
   :language: python
   :lines: 105-150

**Block 4: Plotting Functions (lines 153-223)**

Generates four plots:
- Error convergence: Log-log plot of error vs. number of elements for each polynomial degree
- Timing (first run): Shows computation time including JIT compilation
- Timing (second run): Shows computation time after JIT compilation
- Speedup factor: Compares first vs. second run to demonstrate JIT compilation benefits

.. literalinclude:: ../../scripts/tutorials/toroid_poisson.py
   :language: python
   :lines: 153-223

**Block 5: Main Execution (lines 226-242)**

Runs the convergence analysis with parameter ranges ``ns = [4, 6, 8]`` and ``ps = [1, 2, 3]``,
generates plots, displays figures, and cleans up.

The toroidal mapping transforms logical coordinates :math:`(r, \chi, \zeta)` to physical
cylindrical coordinates :math:`(R, \phi, Z)`, where the toroidal geometry introduces
curvature effects that must be properly handled by the finite element discretization.

.. literalinclude:: ../../scripts/tutorials/toroid_poisson.py
   :language: python
   :lines: 226-242
