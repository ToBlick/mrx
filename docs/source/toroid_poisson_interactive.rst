Toroid Poisson (Interactive)
============================

.. note::
   For general information about finite element discretization, basis functions, mesh parameters,
   polynomial degrees, boundary conditions, and matrix/operator dimensions, see :doc:`overview`.

This script solves a Poisson problem on a toroidal domain interactively.
The script is located at ``scripts/interactive/toroid_poisson.py``.

**Problem Statement**

This script is similar to ``toroid_poisson.py`` but focuses on interactive exploration
and additional diagnostics. It solves the Poisson equation:

.. math::

    -\Delta u = f \quad \text{in } \Omega

with homogeneous Dirichlet boundary conditions:

.. math::

    u|_{\partial\Omega} = 0

where:
- :math:`u: \Omega \to \mathbb{R}` is the scalar solution (0-form)
- :math:`f: \Omega \to \mathbb{R}` is the source term (0-form)
- :math:`\Delta = \nabla \cdot \nabla` is the scalar Laplacian operator
- :math:`\Omega` is a toroidal domain
- :math:`\partial\Omega` denotes the boundary of the toroidal domain

**Toroidal Geometry**

The toroidal domain is parameterized by:
- Minor radius: :math:`a=1/3`
- Major radius: :math:`R_0 = 1.0`
- Aspect ratio: :math:`\epsilon = a/R_0 = 1/3`

The mapping from logical coordinates :math:`(r, \chi, \zeta)` to physical coordinates is:

.. math::

    F(r, \chi, \zeta) = \begin{bmatrix}
        (R_0 + \epsilon r \cos(2\pi\chi)) \cos(2\pi\zeta) \\
        (R_0 + \epsilon r \cos(2\pi\chi)) \sin(2\pi\zeta) \\
        \epsilon r \sin(2\pi\chi)
    \end{bmatrix}

**Exact Solution**

The exact solution is:

.. math::

    u(r,\chi,\zeta) = \frac{1}{4}(r^2 - r^4) \cos(2\pi\zeta)

which is independent of the poloidal angle :math:`\chi`.

**Source Term**

The corresponding source term is:

.. math::

    f(r,\chi,\zeta) = \cos(2\pi\zeta) \left[ -\frac{1}{a^2}(1-4r^2) - \frac{1}{aR}\left(\frac{r}{2}-r^3\right)\cos(2\pi\chi) + \frac{1}{4}\frac{r^2-r^4}{R^2} \right]

where :math:`R = R_0 + a r \cos(2\pi\chi)`.

The script demonstrates:

- Setting up finite element spaces on a toroidal domain
- Solving Poisson equations in toroidal geometry
- Interactive visualization of results
- Computing condition numbers and matrix sparsity for diagnostics

Usage:

.. code-block:: bash

    python scripts/interactive/toroid_poisson.py <n> <p>

where ``n`` is the number of elements and ``p`` is the polynomial degree.

**Finite Element Discretization**

The domain is discretized using a DeRham sequence with:
- **Mesh parameters**: :math:`n_r = n_\chi = n_\zeta = n` elements in each direction
- **Polynomial degrees**: :math:`p_r = p_\chi = p_\zeta = p`
- **Quadrature order**: :math:`q = p`
- **Boundary conditions**: Clamped in radial direction, periodic in poloidal and toroidal directions

**Matrix and Operator Dimensions**

The 0-form mass matrix :math:`M_0 \in \mathbb{R}^{N_0 \times N_0}` and Laplacian :math:`\Delta_0 \in \mathbb{R}^{N_0 \times N_0}` are used.

The discrete Poisson equation:

.. math::

    M_0 \Delta_0 \hat{u} = P_0(f)

where :math:`\hat{u} \in \mathbb{R}^{N_0}` are the solution coefficients.

**Diagnostics**

The script computes:
- **Condition number**: :math:`\kappa(A) = \sigma_{\max}(A)/\sigma_{\min}(A)` where :math:`A = M_0 \Delta_0`
- **Sparsity**: Fraction of non-zero entries in the system matrix

Code Walkthrough
----------------

**Block 1: Imports and Setup (lines 1-14)**

Imports modules and enables 64-bit precision. Uses ``toroid_map`` for domain geometry.

.. literalinclude:: ../../scripts/interactive/toroid_poisson.py
   :language: python
   :lines: 1-14

**Block 2: Error and Diagnostics Function (lines 17-105)**

The ``get_err()`` function computes error and additional diagnostics.

Exact solution:

.. math::

    u(r,\chi,z) = \frac{1}{4}(r^2 - r^4) \cos(2\pi z)

Source term:

.. math::

    f(r,\chi,z) = \cos(2\pi z) \left[ -\frac{1}{a^2}(1-4r^2) - \frac{1}{aR}\left(\frac{r}{2}-r^3\right)\cos(2\pi\chi) + \frac{1}{4}\frac{r^2-r^4}{R^2} \right]

where :math:`R = R_0 + a r \cos(2\pi\chi)` and :math:`\epsilon = a/R_0 = 1/3` is the aspect ratio.

Sets up DeRham sequence with toroidal mapping,
assembles mass matrix :math:`M_0` and Laplacian :math:`\Delta_0`,
solves system:

.. math::

    M_0 \Delta_0 \hat{u} = P_0(f)

Computes relative L2 error using ``jax.lax.scan`` to avoid memory issues with large arrays.
Also computes condition number and sparsity of the system matrix :math:`M_0 \Delta_0`.

.. literalinclude:: ../../scripts/interactive/toroid_poisson.py
   :language: python
   :lines: 17-105

**Block 3: Main Function (lines 108-149)**

Parses command-line arguments for mesh size :math:`n` and polynomial degree :math:`p`,
computes error, condition number, and sparsity,
and saves results to a text file.

The interactive version provides more detailed diagnostics than the tutorial version,
making it useful for understanding numerical properties of the discretization.

.. literalinclude:: ../../scripts/interactive/toroid_poisson.py
   :language: python
   :lines: 108-149
