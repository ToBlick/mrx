Hopf Configuration
==================

.. note::
   For general information about finite element discretization, basis functions, mesh parameters,
   polynomial degrees, boundary conditions, and matrix/operator dimensions, see :doc:`overview`.

This script runs simulations for Hopf configurations.
The script is located at ``scripts/config_scripts/hopf.py``.

**Problem Statement**

The Hopf fibration is an exact solution to the force-free field equations:

.. math::

    \mathbf{J} \times \mathbf{B} = 0 \quad \text{in } \Omega

with the constraint:

.. math::

    \nabla \cdot \mathbf{B} = 0 \quad \text{in } \Omega

and boundary conditions:

.. math::

    \mathbf{B} \cdot \mathbf{n} = 0 \quad \text{on } \partial\Omega

where:
- :math:`\mathbf{B}: \Omega \to \mathbb{R}^3` is the magnetic field (2-form)
- :math:`\mathbf{J} = \nabla \times \mathbf{B}` is the current density (1-form)
- :math:`\Omega = [-4,4] \times [-4,4] \times [-10,10]` is the Cartesian domain
- :math:`\mathbf{n}` is the outward unit normal vector on the boundary

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

**Finite Element Spaces**

The magnetic field is represented as a 2-form:

.. math::

    V_2 = \text{span}\{\Lambda_2^i\}_{i=1}^{N_2}

where :math:`N_2` is the number of 2-form DOFs.

**Matrix and Operator Dimensions**

The 2-form mass matrix :math:`M_2 \in \mathbb{R}^{N_2 \times N_2}` and 1-form mass matrix :math:`M_1 \in \mathbb{R}^{N_1 \times N_1}` are used.

The 2-form Laplacian :math:`\Delta_2 \in \mathbb{R}^{N_2 \times N_2}` has a 1D kernel corresponding to harmonic 2-forms (tunnel topology).

The Leray projection :math:`P_{\mathrm{Leray}} \in \mathbb{R}^{N_2 \times N_2}` projects onto the divergence-free subspace.

**Time-Stepping Scheme**

The implicit time-stepping uses a midpoint rule:

.. math::

    \mathbf{B}^{n+1} = \mathbf{B}^n + \Delta t \nabla \times \mathbf{E}(\mathbf{B}^{\mathrm{mid}})

where :math:`\mathbf{B}^{\mathrm{mid}} = (\mathbf{B}^n + \mathbf{B}^{n+1})/2` and:

.. math::

    \mathbf{E} = \mathbf{u} \times \mathbf{B} - \eta \mathbf{J}

The velocity is:

.. math::

    \mathbf{u} = \begin{cases}
        \mathbf{J} \times \mathbf{H} & \text{(force-free)} \\
        P_{\mathrm{Leray}}(\mathbf{J} \times \mathbf{H}) & \text{(with pressure)}
    \end{cases}

where :math:`\mathbf{H} = M_{12} \mathbf{B}` and :math:`M_{12}` is the 1-2 form coupling matrix.

Code Walkthrough
----------------

**Block 1: Imports and Configuration (lines 1-18)**

Imports modules including ``picard_solver`` for nonlinear iterations and
``CrossProductProjection`` for computing :math:`\mathbf{J} \times \mathbf{B}` terms. Creates output directory.

.. literalinclude:: ../../scripts/config_scripts/hopf.py
   :language: python
   :lines: 1-18

**Block 2: Default Configuration (lines 20-62)**

Defines comprehensive configuration dictionary with parameters:

- ``omega_1, omega_2``: Winding numbers for the Hopf fibration
- ``s``: Scale parameter
- Discretization parameters: ``n_r, n_chi, n_zeta, p_r, p_chi, p_zeta``
- Relaxation parameters: ``gamma`` (regularization), ``eps`` (initial condition smoothing),
  ``dt`` (time step), ``force_tol`` (convergence tolerance)
- Solver parameters: ``max_iter``, ``solver_tol``

.. literalinclude:: ../../scripts/config_scripts/hopf.py
   :language: python
   :lines: 20-62

**Block 3: Setup and Operators (lines 65-172)**

The ``run()`` function sets up a simple Cartesian mapping:

.. math::

    F(x, y, z) = (8x-4, 8y-4, 20z-10)

Creates DeRham sequence with clamped boundary conditions (no polar coordinates),
assembles all mass matrices (M0, M1, M2, M3) and derivative operators,
constructs gradient, curl, and divergence operators (both strong and weak forms),
builds Laplacian operators for each form degree,
creates cross-product projections :math:`P_{\mathbf{J}\times\mathbf{H}}` and :math:`P_{\mathbf{u}\times\mathbf{H}}` for computing
:math:`\mathbf{J} \times \mathbf{B}` and :math:`\mathbf{u} \times \mathbf{B}` terms,
and constructs Leray projection :math:`P_{\mathrm{Leray}}` to enforce divergence-free condition:

.. math::

    P_{\mathrm{Leray}} = I + \nabla(-\Delta)^{-1}\nabla\cdot

.. literalinclude:: ../../scripts/config_scripts/hopf.py
   :language: python
   :lines: 65-172

**Block 4: Initial Condition (lines 174-180)**

Sets up initial magnetic field from analytical Hopf fibration:

- Analytical formula for the Hopf fibration magnetic field (see Problem Statement above)
- Projects into FEM space and applies Leray projection
- Performs one step of resistive relaxation to satisfy boundary conditions
- Extracts harmonic component for helicity computation

.. literalinclude:: ../../scripts/config_scripts/hopf.py
   :language: python
   :lines: 174-180

**Block 5: Time-stepping Loop (lines 182-249)**

Implements implicit time-stepping with Picard iteration.

The ``implicit_update()`` function computes :math:`\mathbf{B}_{n+1}` from :math:`\mathbf{B}_n` using midpoint rule:

.. math::

    \mathbf{B}_{\mathrm{mid}} &= \frac{\mathbf{B}_{n+1} + \mathbf{B}_n}{2} \\
    \mathbf{J} &= \nabla \times \mathbf{B}_{\mathrm{mid}} \\
    \mathbf{H} &= M_{12} \mathbf{B}_{\mathrm{mid}} \\
    \mathbf{u} &= \begin{cases}
        \mathbf{J} \times \mathbf{H} & \text{(force-free)} \\
        P_{\mathrm{Leray}}(\mathbf{J} \times \mathbf{H}) & \text{(with pressure)}
    \end{cases} \\
    \mathbf{E} &= \mathbf{u} \times \mathbf{H} - \eta \mathbf{J} \\
    \mathbf{B}_{n+1} &= \mathbf{B}_n + \Delta t \nabla \times \mathbf{E}

Regularization is applied if :math:`\gamma > 0`: :math:`\mathbf{u} \leftarrow (M_2 + \gamma \Delta_2)^{-1} M_2 \mathbf{u}`.

The ``picard_solver()`` solves the implicit equation iteratively.
Tracks convergence metrics: force norm, energy, helicity, divergence.
Stops when force tolerance is reached or maximum iterations exceeded.

.. literalinclude:: ../../scripts/config_scripts/hopf.py
   :language: python
   :lines: 182-249

**Block 6: Post-processing (lines 251-300)**

After convergence:

- Computes pressure field:

  .. math::

      p = \begin{cases}
          -\Delta^{-1}(\nabla \cdot (\mathbf{J} \times \mathbf{B})) & \text{(not force-free)} \\
          \frac{\mathbf{J} \cdot \mathbf{B}}{|\mathbf{B}|^2} & \text{(force-free, computed pointwise)}
      \end{cases}

- Saves all data to HDF5: magnetic field, pressure, traces, configuration

.. literalinclude:: ../../scripts/config_scripts/hopf.py
   :language: python
   :lines: 251-300

**Block 7: Main Function (lines 305-319)**

Parses command-line arguments and calls ``run()`` with the updated configuration.

.. literalinclude:: ../../scripts/config_scripts/hopf.py
   :language: python
   :lines: 305-319

The Hopf fibration provides an exact solution for testing the numerical method,
as it satisfies :math:`\nabla \cdot \mathbf{B} = 0` and :math:`\mathbf{J} \times \mathbf{B} = 0` analytically.
