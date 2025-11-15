Hopf Configuration
==================

This script runs simulations for Hopf configurations.
The script is located at ``scripts/config_scripts/hopf.py``.

Usage:

.. code-block:: bash

    python scripts/config_scripts/hopf.py run_name=test_run

The script generates output files with Hopf configuration results.

Code Walkthrough
================

This script implements a magnetic relaxation solver for the Hopf fibration, which is
an exact solution to the force-free field equations:
   
   .. math::
   
       \mathbf{J} \times \mathbf{B} = 0
   
   in 3D Cartesian space, where :math:`\mathbf{J} = \nabla \times \mathbf{B}`.

**Block 1: Imports and Configuration (lines 1-18)**
   Imports modules including ``picard_solver`` for nonlinear iterations and
   ``CrossProductProjection`` for computing ``J × B`` terms. Creates output directory.

**Block 2: Default Configuration (lines 20-62)**
   Defines comprehensive configuration dictionary with parameters:
   
   - ``omega_1, omega_2``: Winding numbers for the Hopf fibration
   - ``s``: Scale parameter
   - Discretization parameters: ``n_r, n_chi, n_zeta, p_r, p_chi, p_zeta``
   - Relaxation parameters: ``gamma`` (regularization), ``eps`` (initial condition smoothing),
     ``dt`` (time step), ``force_tol`` (convergence tolerance)
   - Solver parameters: ``max_iter``, ``solver_tol``

**Block 3: Setup and Operators (lines 65-136)**
   The ``run()`` function:
   
   - Sets up a simple Cartesian mapping:
   
     .. math::
     
         F(x, y, z) = (8x-4, 8y-4, 20z-10)
   - Creates DeRham sequence with clamped boundary conditions (no polar coordinates)
   - Assembles all mass matrices (M0, M1, M2, M3) and derivative operators
   - Constructs gradient, curl, and divergence operators (both strong and weak forms)
   - Builds Laplacian operators for each form degree:
     * ``laplace_0``: Standard scalar Laplacian
     * ``laplace_1``: Vector Laplacian with correction term
     * ``laplace_2``: 2-form Laplacian (has 1D kernel for tunnel topology)
     * ``laplace_3``: 3-form Laplacian (has 1D kernel for constants)
   - Creates cross-product projections :math:`P_{\mathbf{J}\times\mathbf{H}}` and :math:`P_{\mathbf{u}\times\mathbf{H}}` for computing
     :math:`\mathbf{J} \times \mathbf{B}` and :math:`\mathbf{u} \times \mathbf{B}` terms
   - Constructs Leray projection :math:`P_{\text{Leray}}` to enforce divergence-free condition:
   
     .. math::
     
         P_{\text{Leray}} = I + \nabla(-\Delta)^{-1}\nabla\cdot

**Block 4: Initial Condition (lines 141-180)**
   Sets up initial magnetic field from analytical Hopf fibration:
   
   - Analytical formula for the Hopf fibration magnetic field:
   
     .. math::
     
         \mathbf{B}(x,y,z) = \frac{4\sqrt{s}}{\pi(1+r^2)^3\sqrt{\omega_1^2+\omega_2^2}} \begin{bmatrix}
             2\omega_2 y - 2\omega_1 x z \\
             -2\omega_2 x - 2\omega_1 y z \\
             \omega_1(x^2+y^2-z^2-1)
         \end{bmatrix}
     
     where :math:`r^2 = x^2 + y^2 + z^2` and :math:`\omega_1, \omega_2` are winding numbers.
   - Projects into FEM space and applies Leray projection
   - Performs one step of resistive relaxation to satisfy boundary conditions
   - Extracts harmonic component for helicity computation

**Block 5: Time-stepping Loop (lines 182-249)**
   Implements implicit time-stepping with Picard iteration:
   
   - ``implicit_update()``: Computes :math:`\mathbf{B}_{n+1}` from :math:`\mathbf{B}_n` using midpoint rule:
   
     .. math::
     
         \mathbf{B}_{\text{mid}} &= \frac{\mathbf{B}_{n+1} + \mathbf{B}_n}{2} \\
         \mathbf{J} &= \nabla \times \mathbf{B}_{\text{mid}} \\
         \mathbf{H} &= M_{12} \mathbf{B}_{\text{mid}} \\
         \mathbf{u} &= \begin{cases}
             \mathbf{J} \times \mathbf{H} & \text{(force-free)} \\
             P_{\text{Leray}}(\mathbf{J} \times \mathbf{H}) & \text{(with pressure)}
         \end{cases} \\
         \mathbf{E} &= \mathbf{u} \times \mathbf{H} - \eta \mathbf{J} \\
         \mathbf{B}_{n+1} &= \mathbf{B}_n + \Delta t \nabla \times \mathbf{E}
     
     Regularization is applied if :math:`\gamma > 0`: :math:`\mathbf{u} \leftarrow (M_2 + \Delta_2)^{-1} M_2 \mathbf{u}`.
   - ``picard_solver()``: Solves the implicit equation iteratively
   - Tracks convergence metrics: force norm, energy, helicity, divergence
   - Stops when force tolerance is reached or maximum iterations exceeded

**Block 6: Post-processing (lines 251-300)**
   After convergence:
   
   - Computes pressure field:
   
     .. math::
     
         p = \begin{cases}
             -\Delta^{-1}(\nabla \cdot (\mathbf{J} \times \mathbf{B})) & \text{(not force-free)} \\
             \frac{\mathbf{J} \cdot \mathbf{B}}{|\mathbf{B}|^2} & \text{(force-free, computed pointwise)}
         \end{cases}
   - Saves all data to HDF5: magnetic field, pressure, traces, configuration

The Hopf fibration provides an exact solution for testing the numerical method,
as it satisfies :math:`\nabla \cdot \mathbf{B} = 0` and :math:`\mathbf{J} \times \mathbf{B} = 0` analytically.

Full script:

.. literalinclude:: ../../scripts/config_scripts/hopf.py
   :language: python
   :linenos:
