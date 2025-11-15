Hopf Configuration
==================

This script runs simulations for Hopf configurations.
The script is located at ``scripts/config_scripts/hopf.py``.

Mathematical Problem
====================

The Hopf fibration is an exact solution to the force-free field equations:

.. math::

    \mathbf{J} \times \mathbf{B} = 0

where:
- :math:`\mathbf{B}: \Omega \to \mathbb{R}^3` is the magnetic field (2-form)
- :math:`\mathbf{J} = \nabla \times \mathbf{B}` is the current density (1-form)

The force-free condition implies :math:`\mathbf{J} \parallel \mathbf{B}`, meaning the current
is parallel to the magnetic field everywhere.

**Hopf Fibration Field**

The analytical Hopf fibration magnetic field is:

.. math::

    \mathbf{B}(x,y,z) = \frac{4\sqrt{s}}{\pi(1+r^2)^3\sqrt{\omega_1^2+\omega_2^2}} \begin{bmatrix}
        2\omega_2 y - 2\omega_1 x z \\
        -2\omega_2 x - 2\omega_1 y z \\
        \omega_1(x^2+y^2-z^2-1)
    \end{bmatrix}

where:
- :math:`r^2 = x^2 + y^2 + z^2` is the distance from the origin
- :math:`\omega_1, \omega_2 \in \mathbb{Z}` are winding numbers
- :math:`s > 0` is a scale parameter

This field satisfies:
- :math:`\nabla \cdot \mathbf{B} = 0` (divergence-free)
- :math:`\mathbf{J} \times \mathbf{B} = 0` (force-free)
- :math:`|\mathbf{B}|^2` decays as :math:`O(r^{-4})` for large :math:`r`

**Domain Mapping**

The domain uses a simple Cartesian mapping:

.. math::

    F(x, y, z) = (8x-4, 8y-4, 20z-10)

which maps the unit cube :math:`[0,1]^3` to :math:`[-4,4] \times [-4,4] \times [-10,10]`.

Usage:

.. code-block:: bash

    python scripts/config_scripts/hopf.py run_name=test_run

The script generates output files with Hopf configuration results.

Mathematical Formulation
=========================

**Finite Element Spaces**

The magnetic field is represented as a 2-form:

.. math::

    V_2 = \text{span}\{\Lambda_2^i\}_{i=1}^{N_2}

where :math:`N_2` is the number of 2-form DOFs.

**Mass Matrices**

The 2-form mass matrix :math:`M_2 \in \mathbb{R}^{N_2 \times N_2}`:

.. math::

    (M_2)_{ij} = \int_\Omega \Lambda_2^i(x) \cdot G(x) \Lambda_2^j(x) \frac{1}{\det(DF(x))} \, dx

The 1-form mass matrix :math:`M_1 \in \mathbb{R}^{N_1 \times N_1}`:

.. math::

    (M_1)_{ij} = \int_\Omega \Lambda_1^i(x) \cdot G^{-1}(x) \Lambda_1^j(x) \det(DF(x)) \, dx

**Derivative Operators**

The weak curl operator :math:`\text{weak\_curl}: V_2 \to V_1`:

.. math::

    (\text{weak\_curl})_{ij} = \int_\Omega \Lambda_1^i(x) \cdot \nabla \times \Lambda_2^j(x) \det(DF(x)) \, dx

Dimensions: :math:`N_1 \times N_2`.

The strong curl operator :math:`\text{strong\_curl}: V_1 \to V_2`:

.. math::

    (\text{strong\_curl})_{ij} = \int_\Omega \Lambda_2^i(x) \cdot G(x) \nabla \times \Lambda_1^j(x) \frac{1}{\det(DF(x))} \, dx

Dimensions: :math:`N_2 \times N_1`.

**Laplacian Operators**

The 2-form Laplacian :math:`\Delta_2 \in \mathbb{R}^{N_2 \times N_2}`:

.. math::

    \Delta_2 = M_2^{-1} \text{div\_div} + \text{strong\_curl} \circ \text{weak\_curl}

where:

.. math::

    (\text{div\_div})_{ij} = \int_\Omega \text{div} \Lambda_2^i(x) \text{div} \Lambda_2^j(x) \frac{1}{\det(DF(x))} \, dx

The 2-form Laplacian has a 1D kernel corresponding to harmonic 2-forms (tunnel topology).

**Cross-Product Projections**

The cross-product projection :math:`P_{2 \times 1 \to 2}` computes:

.. math::

    (P_{2 \times 1 \to 2}(\mathbf{B}, \mathbf{J}))_i = \int_\Omega (\mathbf{B} \times \mathbf{J}) \cdot \Lambda_2^i(x) \frac{1}{\det(DF(x))} \, dx

This is used to compute :math:`\mathbf{J} \times \mathbf{B}`.

**Leray Projection**

The Leray projection :math:`P_{\text{Leray}}: V_2 \to V_2^{\text{div-free}}` projects onto the divergence-free subspace.
The standard form is:

.. math::

    P_{\text{Leray}} = I - \text{strong\_grad} \circ (\text{weak\_div} \circ M_1^{-1} \circ \text{strong\_grad})^{-1} \circ \text{weak\_div}

which ensures :math:`\nabla \cdot (P_{\text{Leray}} \mathbf{u}) = 0` for all :math:`\mathbf{u} \in V_2`.
The implementation may use equivalent forms involving the 3-form Laplacian.

**Time-Stepping Scheme**

The implicit time-stepping uses a midpoint rule:

.. math::

    \mathbf{B}^{n+1} = \mathbf{B}^n + \Delta t \nabla \times \mathbf{E}(\mathbf{B}^{\text{mid}})

where :math:`\mathbf{B}^{\text{mid}} = (\mathbf{B}^n + \mathbf{B}^{n+1})/2` and:

.. math::

    \mathbf{E} = \mathbf{u} \times \mathbf{B} - \eta \mathbf{J}

The velocity is:

.. math::

    \mathbf{u} = \begin{cases}
        \mathbf{J} \times \mathbf{H} & \text{(force-free)} \\
        P_{\text{Leray}}(\mathbf{J} \times \mathbf{H}) & \text{(with pressure)}
    \end{cases}

where :math:`\mathbf{H} = M_{12} \mathbf{B}` and :math:`M_{12}` is the 1-2 form coupling matrix.

**Regularization**

If :math:`\gamma > 0`, regularization is applied:

.. math::

    \mathbf{u} \leftarrow (M_2 + \gamma \Delta_2)^{-1} M_2 \mathbf{u}

This smooths the velocity field and can improve stability.

**Pressure Computation**

For force-free fields, pressure is computed pointwise:

.. math::

    p = \frac{\mathbf{J} \cdot \mathbf{B}}{|\mathbf{B}|^2}

For non-force-free fields:

.. math::

    p = -\Delta_0^{-1}(\nabla \cdot (\mathbf{J} \times \mathbf{B}))

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
