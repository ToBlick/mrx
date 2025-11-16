Poisson Problem on a disc
==========================

.. note::
   For general information about finite element discretization, basis functions, mesh parameters,
   polynomial degrees, boundary conditions, and matrix/operator dimensions, see :doc:`overview`.

This tutorial demonstrates solving a Poisson problem on a disc using polar coordinates.
The script is located at ``scripts/tutorials/polar_poisson.py``.

**Problem Statement**

We solve the Poisson equation on a disc domain :math:`\Omega = \{ (r,\theta) : 0 \leq r \leq 1, 0 \leq \theta < 2\pi \}`:

.. math::

    -\Delta u = f \quad \text{in } \Omega

with homogeneous Dirichlet boundary conditions:

.. math::

    u|_{\partial\Omega} = 0

where:
- :math:`u: \Omega \to \mathbb{R}` is the unknown scalar field (0-form)
- :math:`f: \Omega \to \mathbb{R}` is the given source term
- :math:`\Delta = \nabla \cdot \nabla` is the scalar Laplacian operator
- :math:`\partial\Omega` denotes the boundary of the domain

For this problem, we consider the source-solution pair:

.. math::

    u(r) &= \frac{1}{27} \left( r^3 (3 \log r - 2) + 2 \right) \\
    f(r) &= -r \log r

Note that :math:`u \in H^s(\Omega)` for all :math:`s < 4`, limiting the convergence rate.

The script demonstrates:

- Setting up finite element spaces with polar coordinates
- Handling the singularity at the axis using polar splines
- Assembling stiffness matrices and projectors
- Solving the Poisson equation and analyzing convergence

To run the script:

.. code-block:: bash

    python scripts/tutorials/polar_poisson.py

The script generates convergence plots showing error vs. mesh size for different polynomial orders.

**Finite Element Discretization**

The domain is discretized using a DeRham sequence with:
- **Mesh parameters**: :math:`n_r = n_\theta = n` elements in radial and poloidal directions
- **Polynomial degrees**: :math:`p_r = p_\theta = p` (B-spline degree)
- **Quadrature order**: :math:`q = p + 2` (Gauss-Legendre quadrature)
- **Boundary conditions**: Clamped in radial direction (:math:`r=0,1`), periodic in poloidal direction (:math:`\theta`)

Following the general formulas in :doc:`overview`, the number of DOFs are:
- **0-forms**: :math:`N_0 = n_r \cdot n_\theta = n^2`
- **1-forms**: :math:`N_1 = d_r \cdot n_\theta + n_r \cdot d_\theta = (n-1) \cdot n + n \cdot n = n(2n-1)` where :math:`d_r = n-1` (clamped), :math:`d_\theta = n` (periodic)
- **2-forms**: :math:`N_2 = n_r \cdot d_\theta + d_r \cdot n_\theta = n^2 + n(n-1) = n(2n-1)`
- **3-forms**: :math:`N_3 = d_r \cdot d_\theta = n(n-1)`

**Matrix and Operator Dimensions**

The 0-form mass matrix :math:`M_0 \in \mathbb{R}^{N_0 \times N_0}` where :math:`N_0 = n^2`:

.. math::

    (M_0)_{ij} = \int_\Omega \Lambda_0^i(x) \Lambda_0^j(x) \det(DF(x)) \, dx

The 0-form Laplacian :math:`\Delta_0 \in \mathbb{R}^{N_0 \times N_0}`:

.. math::

    \Delta_0 = M_0^{-1} \nabla_h^T M_1 \nabla_h

where the gradient-gradient matrix :math:`\nabla_h^T M_1 \nabla_h` represents :math:`\nabla \cdot \nabla` (the scalar Laplacian operator).

The discrete Poisson equation becomes:

.. math::

    M_0 \Delta_0 \hat{u} = P_0(f)

where :math:`\hat{u} \in \mathbb{R}^{N_0}` are the solution coefficients and :math:`P_0(f) \in \mathbb{R}^{N_0}` is the projection of the source term.

Code Walkthrough
----------------

**Block 1: Imports and Configuration (lines 1-33)**

This block imports necessary libraries (JAX, NumPy, Matplotlib) and MRX modules
for DeRham sequences, discrete functions, and polar mappings. It enables 64-bit
precision for numerical stability and creates an output directory for generated plots.

.. literalinclude:: ../../scripts/tutorials/polar_poisson.py
   :language: python
   :lines: 1-33

**Block 2: Error Computation Function (lines 37-105)**

The ``get_err`` function is JIT-compiled for efficiency and computes the relative
L2 error for a given mesh size :math:`n`, polynomial degree :math:`p`, and quadrature order :math:`q`.

It defines the exact solution and source term:

.. math::

    u(r) &= \frac{1}{27} \left( r^3 (3 \log r - 2) + 2 \right) \\
    f(r) &= -r \log r

The function sets up a DeRham sequence with polar coordinates,
assembles the mass matrix :math:`M_0` and Laplacian :math:`\Delta_0`, solves the linear system:

.. math::

    M_0 \Delta_0 \hat{u} = P_0(f)

and computes the relative L2 error using quadrature:

.. math::

    \text{error} = \frac{\|u - u_h\|_{L^2}}{\|u\|_{L^2}}

where the L2 norm is computed using:

.. math::

    \|u - u_h\|_{L^2(\Omega)}^2 = \int_\Omega (u(x) - u_h(x))^2 \det(DF(x)) \, dx \approx \sum_{j=1}^{n_q} (u(x_j) - u_h(x_j))^2 J_j w_j

.. literalinclude:: ../../scripts/tutorials/polar_poisson.py
   :language: python
   :lines: 37-105

**Block 3: Convergence Analysis (lines 108-151)**

The ``run_convergence_analysis`` function performs two runs over different mesh sizes
and polynomial degrees. The first run includes JIT compilation overhead, while the
second run measures pure computation time. This allows comparison of JIT compilation
impact on performance. Results are stored in arrays for error and timing data.

.. literalinclude:: ../../scripts/tutorials/polar_poisson.py
   :language: python
   :lines: 108-151

**Block 4: Plotting Functions (lines 154-224)**

The ``plot_results`` function generates four plots:

- Error convergence: Log-log plot of error vs. number of elements for each polynomial degree
- Timing (first run): Shows computation time including JIT compilation
- Timing (second run): Shows computation time after JIT compilation
- Speedup factor: Compares first vs. second run to demonstrate JIT compilation benefits

.. literalinclude:: ../../scripts/tutorials/polar_poisson.py
   :language: python
   :lines: 154-224

**Block 5: Main Execution (lines 227-243)**

The ``main`` function orchestrates the analysis by:

1. Defining parameter ranges: ``ns = [6, 8, 10, 12, 14, 16]`` and ``ps = [1, 2, 3, 4]``
2. Running convergence analysis for all parameter combinations
3. Generating and saving plots
4. Displaying figures and cleaning up

The script uses polar splines to handle the coordinate singularity at r=0, which is
essential for accurate solutions on disc domains. The convergence analysis demonstrates
how error decreases with increasing mesh refinement and polynomial degree.

.. literalinclude:: ../../scripts/tutorials/polar_poisson.py
   :language: python
   :lines: 227-243
