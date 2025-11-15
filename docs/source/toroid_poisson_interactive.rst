Toroid Poisson (Interactive)
============================

This script solves a Poisson problem on a toroidal domain interactively.
The script is located at ``scripts/interactive/toroid_poisson.py``.

Mathematical Problem
====================

This script is similar to ``toroid_poisson.py`` but focuses on interactive exploration
and additional diagnostics. It solves the Poisson equation:

.. math::

    -\Delta u = f \quad \text{in } \Omega

where:
- :math:`u: \Omega \to \mathbb{R}` is the scalar solution (0-form)
- :math:`f: \Omega \to \mathbb{R}` is the source term (0-form)
- :math:`\Omega` is a toroidal domain

**Toroidal Geometry**

The toroidal domain is parameterized by:
- Minor radius: :math:`a=1`
- Major radius: :math:`R_0 = 3a = 3`
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

Usage:

.. code-block:: bash

    python scripts/interactive/toroid_poisson.py

Mathematical Formulation
=========================

**Finite Element Discretization**

The domain is discretized using a DeRham sequence with:
- **Mesh parameters**: :math:`n_r = n_\chi = n_\zeta = n` elements in each direction
- **Polynomial degrees**: :math:`p_r = p_\chi = p_\zeta = p`
- **Quadrature order**: :math:`q = p + 2`
- **Boundary conditions**: Clamped in radial direction, periodic in poloidal and toroidal directions

**Basis Functions**

The 0-form basis functions :math:`\{\Lambda_0^i\}_{i=1}^{N_0}` are tensor products:

.. math::

    \Lambda_0^i(r,\chi,\zeta) = \Lambda_r^{i_r}(r) \Lambda_\chi^{i_\chi}(\chi) \Lambda_\zeta^{i_\zeta}(\zeta)

where :math:`N_0 = n_r \cdot n_\chi \cdot n_\zeta` is the total number of 0-form DOFs.

**Mass Matrix**

The 0-form mass matrix :math:`M_0 \in \mathbb{R}^{N_0 \times N_0}`:

.. math::

    (M_0)_{ij} = \int_\Omega \Lambda_0^i(x) \Lambda_0^j(x) \det(DF(x)) \, dx

Dimensions: :math:`N_0 \times N_0`.

**Laplacian Operator**

The 0-form Laplacian :math:`\Delta_0 \in \mathbb{R}^{N_0 \times N_0}`:

.. math::

    (\Delta_0)_{ij} = \int_\Omega \nabla \Lambda_0^i(x) \cdot G^{-1}(x) \nabla \Lambda_0^j(x) \det(DF(x)) \, dx

where :math:`G(x) = DF(x)^T DF(x)` is the metric tensor.

**Linear System**

The discrete Poisson equation:

.. math::

    M_0 \Delta_0 \hat{u} = P_0(f)

where:
- :math:`\hat{u} \in \mathbb{R}^{N_0}`: Solution coefficients
- :math:`M_0 \in \mathbb{R}^{N_0 \times N_0}`: Mass matrix
- :math:`\Delta_0 \in \mathbb{R}^{N_0 \times N_0}`: Laplacian operator
- :math:`P_0(f) \in \mathbb{R}^{N_0}`: Projection of source term

**Condition Number**

The condition number of the system matrix :math:`A = M_0 \Delta_0`:

.. math::

    \kappa(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}

where :math:`\sigma_{\max}` and :math:`\sigma_{\min}` are the largest and smallest singular values.

**Sparsity**

The sparsity of a matrix :math:`A` is defined as:

.. math::

    \text{sparsity}(A) = 1 - \frac{\text{nnz}(A)}{N^2}

where :math:`\text{nnz}(A)` is the number of non-zero entries and :math:`N` is the matrix size.

**Error Computation**

The relative L2 error:

.. math::

    \text{error} = \frac{\|u - u_h\|_{L^2(\Omega)}}{\|u\|_{L^2(\Omega)}}

is computed using quadrature:

.. math::

    \|u - u_h\|_{L^2(\Omega)}^2 = \int_\Omega (u(x) - u_h(x))^2 \det(DF(x)) \, dx \approx \sum_{j=1}^{n_q} (u(x_j) - u_h(x_j))^2 J_j w_j

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
