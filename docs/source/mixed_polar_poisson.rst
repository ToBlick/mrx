Mixed Polar Poisson Problem
============================

.. note::
   For general information about finite element discretization, basis functions, mesh parameters,
   polynomial degrees, boundary conditions, and matrix/operator dimensions, see :doc:`overview`.

This tutorial demonstrates solving a mixed formulation of the Poisson problem on a disc.
The script is located at ``scripts/tutorials/mixed_polar_poisson.py``.

Mathematical Problem
====================

The mixed formulation rewrites the Poisson equation :math:`-\Delta u = f` as a first-order system:

.. math::

    \nabla \cdot \sigma &= f \\
    -\nabla u &= \sigma

where:
- :math:`u: \Omega \to \mathbb{R}` is the scalar solution (3-form, volume form)
- :math:`\sigma: \Omega \to \mathbb{R}^2` is the flux variable (2-form, area form)
- :math:`f: \Omega \to \mathbb{R}` is the source term

This formulation is equivalent to the standard Poisson equation but solves for both the solution
and its gradient simultaneously, which can provide better conservation properties and is useful
for problems where flux conservation is important.

For this problem, we use the exact solution and source term:

.. math::

    u(r) &= -\frac{1}{16}r^4 + \frac{1}{12}r^3 + \frac{1}{48} \\
    f(r) &= r^2 - \frac{3}{4}r

Boundary conditions are homogeneous Neumann: :math:`\partial u/\partial n = 0` on :math:`\partial\Omega`.

The script demonstrates:

- Mixed finite element formulation
- Handling polar coordinates and axis singularity
- Convergence analysis for mixed methods
- Performance comparison with standard formulation

To run the script:

.. code-block:: bash

    python scripts/tutorials/mixed_polar_poisson.py

The script generates convergence plots and performance comparisons.

Mathematical Formulation
=========================

**Discretization Parameters**

This script uses:
- **Mesh parameters**: :math:`n_r = n_\theta = n`, :math:`n_\zeta = 1` (2D problem with trivial third dimension)
- **Polynomial degrees**: :math:`p_r = p_\theta = p`, :math:`p_\zeta = 0` (constant in third direction)
- **Boundary conditions**: Clamped in radial direction, periodic in poloidal direction, constant in toroidal direction

Following the general formulas in :doc:`overview`, the number of DOFs are:
- **2-forms** (flux variable :math:`\sigma`): :math:`N_2 = n_r \cdot d_\theta \cdot d_\zeta + d_r \cdot n_\theta \cdot d_\zeta + d_r \cdot d_\theta \cdot n_\zeta = n \cdot n \cdot 1 + (n-1) \cdot n \cdot 1 + (n-1) \cdot n \cdot 1 = 3n^2 - 2n` where :math:`d_r = n-1` (clamped), :math:`d_\theta = n` (periodic), :math:`d_\zeta = 1` (constant)
- **3-forms** (solution :math:`u`): :math:`N_3 = d_r \cdot d_\theta \cdot d_\zeta = (n-1) \cdot n \cdot 1 = n(n-1)`

**Finite Element Spaces**

The mixed formulation uses a 3D DeRham sequence (with trivial third dimension) to solve a 2D problem:
- **3-forms** (volume forms) for the solution :math:`u`: :math:`V_3 = \text{span}\{\Lambda_3^i\}_{i=1}^{N_3}` where :math:`N_3 = n(n-1)`
- **2-forms** (area forms) for the flux :math:`\sigma`: :math:`V_2 = \text{span}\{\Lambda_2^i\}_{i=1}^{N_2}` where :math:`N_2 = 3n^2 - 2n`

Note: Although the problem is 2D (disc domain), the code uses a 3D DeRham sequence with the third dimension
having a single element and zero polynomial degree, effectively reducing to 2D.

**Matrix and Operator Dimensions**

All matrices and operators are defined with explicit dimensions:

**Mass Matrices**

The 2-form mass matrix :math:`M_2 \in \mathbb{R}^{N_2 \times N_2}` where :math:`N_2 = 3n^2 - 2n`:

.. math::

    (M_2)_{ij} = \int_\Omega \Lambda_2^i(x) \cdot G(x) \Lambda_2^j(x) \frac{1}{\det(DF(x))} \, dx

where :math:`G(x) = DF(x)^T DF(x)` is the metric tensor.
Dimensions: :math:`M_2 \in \mathbb{R}^{(3n^2-2n) \times (3n^2-2n)}`.

The 3-form mass matrix :math:`M_3 \in \mathbb{R}^{N_3 \times N_3}` where :math:`N_3 = n(n-1)`:

.. math::

    (M_3)_{ij} = \int_\Omega \Lambda_3^i(x) \Lambda_3^j(x) \frac{1}{\det(DF(x))} \, dx

Dimensions: :math:`M_3 \in \mathbb{R}^{n(n-1) \times n(n-1)}`.

**Derivative Operators**

The strong divergence operator :math:`\text{strong\_div}: V_2 \to V_3`:

.. math::

    (\text{strong\_div})_{ij} = \int_\Omega \Lambda_3^i(x) \text{div} \Lambda_2^j(x) \frac{1}{\det(DF(x))} \, dx

Dimensions: :math:`\text{strong\_div} \in \mathbb{R}^{n(n-1) \times (3n^2-2n)}`.

The weak gradient operator :math:`\text{weak\_grad}: V_3 \to V_2`:

.. math::

    (\text{weak\_grad})_{ij} = -\int_\Omega \Lambda_2^i(x) \cdot G^{-1}(x) \nabla \Lambda_3^j(x) \det(DF(x)) \, dx

Dimensions: :math:`\text{weak\_grad} \in \mathbb{R}^{(3n^2-2n) \times n(n-1)}`.

**3-Form Laplacian**

The 3-form Laplacian :math:`\Delta_3 \in \mathbb{R}^{N_3 \times N_3}` is constructed as:

.. math::

    \Delta_3 = -\text{strong\_div} \circ \text{weak\_grad}

This operator satisfies:

.. math::

    -(\text{div} \xi, \mu) = (\delta d \rho, \mu) \quad \forall \mu \in V_3

where :math:`\xi = \text{weak\_grad} \rho` and :math:`\rho \in V_3`.
Dimensions: :math:`\Delta_3 \in \mathbb{R}^{n(n-1) \times n(n-1)}`.

**Projection Operator**

The 3-form projection operator :math:`P_3: L^2(\Omega) \to V_3`:

.. math::

    P_3(f) = \arg\min_{v_h \in V_3} \|f - v_h\|_{L^2(\Omega)}

Dimensions: :math:`P_3(f) \in \mathbb{R}^{n(n-1)}`.

**Linear System**

The discrete mixed formulation becomes:

.. math::

    M_3 \Delta_3 \hat{u} = P_3(f)

where:
- :math:`\hat{u} \in \mathbb{R}^{N_3}` are the solution coefficients, :math:`\hat{u} \in \mathbb{R}^{n(n-1)}`
- :math:`M_3 \in \mathbb{R}^{N_3 \times N_3}` is the 3-form mass matrix, :math:`M_3 \in \mathbb{R}^{n(n-1) \times n(n-1)}`
- :math:`\Delta_3 \in \mathbb{R}^{N_3 \times N_3}` is the 3-form Laplacian, :math:`\Delta_3 \in \mathbb{R}^{n(n-1) \times n(n-1)}`
- :math:`P_3(f) \in \mathbb{R}^{N_3}` is the projection of the source term, :math:`P_3(f) \in \mathbb{R}^{n(n-1)}`

**Pushforward Operation**

The solution is pushed forward from logical to physical space:

.. math::

    u_h(x) = \text{Pushforward}(u_{\text{logical}}, F, 3)(x)

where the pushforward of a 3-form transforms the discrete function representation
to account for the coordinate mapping :math:`F`.

Code Walkthrough
----------------

The script follows a similar structure to ``polar_poisson.py`` but uses a mixed finite
element formulation:

**Block 1: Imports and Configuration (lines 1-39)**
   Imports necessary libraries and MRX modules. Note that this script uses ``Pushforward``
   in addition to ``DiscreteFunction``, which is needed for the mixed formulation.
   Creates output directory and enables 64-bit precision.

**Block 2: Error Computation Function (lines 42-113)**
   The ``get_err`` function implements the mixed formulation:
   
   .. math::
   
       u(r) &= -\frac{1}{16}r^4 + \frac{1}{12}r^3 + \frac{1}{48} \\
       f(r) &= r^2 - \frac{3}{4}r
   
   Boundary conditions are homogeneous Neumann (:math:`\partial u/\partial n = 0`).
   
   The mixed formulation uses 3-forms (volume forms) instead of 0-forms. It assembles:
   
   - :math:`M_2`: Mass matrix for 2-forms (for the flux variable :math:`\sigma`)
   - :math:`M_3`: Mass matrix for 3-forms (for the solution :math:`u`)
   - :math:`\Delta_3`: Laplacian operator for 3-forms (strong divergence composed with weak gradient)
   
   The system is solved as:
   
   .. math::
   
       M_3 \Delta_3 \hat{u} = P_3(f)
   
   and the solution is pushed forward to physical space using ``Pushforward``. Error is computed using relative L2 norm.

**Block 3: Convergence Analysis (lines 116-159)**
   Identical to ``polar_poisson.py``: runs convergence analysis twice (with and without
   JIT compilation overhead) to measure performance and error convergence.

**Block 4: Plotting Functions (lines 162-231)**
   Generates the same four plots as ``polar_poisson.py``:
   - Error convergence plot
   - Timing plots (first and second run)
   - JIT compilation speedup plot

**Block 5: Main Execution (lines 234-250)**
   Runs the analysis with the same parameter ranges as ``polar_poisson.py`` and generates plots.

The key difference from the standard formulation is that the mixed method solves for both
the solution :math:`u` and the flux :math:`\sigma = -\nabla u` simultaneously, which can be advantageous for
certain types of problems and provides better conservation properties.

Full script:

.. literalinclude:: ../../scripts/tutorials/mixed_polar_poisson.py
   :language: python
   :linenos:
