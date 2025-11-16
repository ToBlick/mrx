Solovev Configuration
====================

.. note::
   For general information about finite element discretization, basis functions, mesh parameters,
   polynomial degrees, boundary conditions, and matrix/operator dimensions, see :doc:`overview`.

This script runs a magnetic relaxation simulation for a Solovev configuration.
The script is located at ``scripts/config_scripts/solovev.py``.

Mathematical Problem
====================

The Solovev configuration is a simple tokamak equilibrium solution that can be used
as a starting point for magnetic relaxation simulations. The script solves the MHD
(magnetohydrodynamic) force balance equation:

.. math::

    \mathbf{J} \times \mathbf{B} = \nabla p

where:
- :math:`\mathbf{B}: \Omega \to \mathbb{R}^3` is the magnetic field (2-form, area form)
- :math:`\mathbf{J} = \nabla \times \mathbf{B}` is the current density (1-form)
- :math:`p: \Omega \to \mathbb{R}` is the plasma pressure (0-form, scalar)

For the force-free case (:math:`\nabla p = 0`), the equation becomes:

.. math::

    \mathbf{J} \times \mathbf{B} = 0

which implies :math:`\mathbf{J} \parallel \mathbf{B}` (force-free condition).

**Magnetic Relaxation Method**

The relaxation method solves the time-dependent MHD equations:

.. math::

    \frac{\partial \mathbf{B}}{\partial t} = -\nabla \times \mathbf{E}

where the electric field :math:`\mathbf{E}` is given by:

.. math::

    \mathbf{E} = \mathbf{u} \times \mathbf{B} - \eta \mathbf{J}

and:
- :math:`\mathbf{u}` is the velocity field (divergence-free)
- :math:`\eta` is the resistivity parameter

The velocity is chosen to drive the system toward equilibrium:

.. math::

    \mathbf{u} = \mathbf{f} = \mathbf{J} \times \mathbf{B} - \nabla p

where :math:`\mathbf{f}` is the force imbalance. The velocity is projected onto the
divergence-free space using the Leray projection :math:`P_{\text{Leray}}`.

**Time-Stepping Scheme**

The time-stepping uses a midpoint rule (Crank-Nicolson):

.. math::

    \mathbf{B}^{n+1} = \mathbf{B}^n + \Delta t \nabla \times \mathbf{E}(\mathbf{B}^{\text{mid}})

where :math:`\mathbf{B}^{\text{mid}} = (\mathbf{B}^n + \mathbf{B}^{n+1})/2` is evaluated at the midpoint.

The system is solved using a Picard iteration scheme with adaptive time-stepping.

The Solovev configuration provides a simple analytical equilibrium that serves as a
test case for the relaxation solver. The script demonstrates how to set up and run
magnetic relaxation simulations for various boundary geometries.

Usage:

.. code-block:: bash

    python scripts/config_scripts/solovev.py run_name=test_run boundary_type=rotating_ellipse n_r=16 n_theta=16 n_zeta=8

The script accepts various parameters:

- ``run_name``: Name for the output files
- ``boundary_type``: Type of boundary (e.g., ``rotating_ellipse``, ``cerfon``)
- ``n_r``, ``n_theta``, ``n_zeta``: Number of splines in each direction
- ``p_r``, ``p_theta``, ``p_zeta``: Polynomial degrees
- ``maxit``: Maximum number of iterations
- And many more (see script for full list)

The script generates HDF5 files with simulation data and PDF plots showing:

- Force trace over iterations
- Pressure contours
- Magnetic field structure

Mathematical Formulation
=========================

**Discretization Parameters**

This script uses a 3D DeRham sequence with mesh parameters :math:`n_r, n_\theta, n_\zeta`, polynomial degrees :math:`p_r, p_\theta, p_\zeta`,
and boundary conditions (typically clamped in radial direction, periodic in poloidal and toroidal directions).
For details on computing the number of DOFs :math:`N_0, N_1, N_2, N_3` from these parameters, see :doc:`overview`.

**Finite Element Spaces**

The magnetic field :math:`\mathbf{B}` is represented as a 2-form (area form) in the finite element space:

.. math::

    V_2 = \text{span}\{\Lambda_2^i\}_{i=1}^{N_2}

where:
- :math:`N_2` is the number of 2-form degrees of freedom
- :math:`\{\Lambda_2^i\}` are 2-form basis functions (vector fields with 2 components)

The current density :math:`\mathbf{J} = \nabla \times \mathbf{B}` is a 1-form:

.. math::

    \mathbf{J} = \text{weak\_curl}(\mathbf{B})

where :math:`\text{weak\_curl}: V_2 \to V_1` is the weak curl operator.

**Matrix and Operator Dimensions**

All matrices and operators have explicit dimensions as described in :doc:`overview`:

**Mass Matrices**

The 2-form mass matrix :math:`M_2 \in \mathbb{R}^{N_2 \times N_2}`:

.. math::

    (M_2)_{ij} = \int_\Omega \Lambda_2^i(x) \cdot G(x) \Lambda_2^j(x) \frac{1}{\det(DF(x))} \, dx

where :math:`G(x) = DF(x)^T DF(x)` is the metric tensor.

The 1-form mass matrix :math:`M_1 \in \mathbb{R}^{N_1 \times N_1}`:

.. math::

    (M_1)_{ij} = \int_\Omega \Lambda_1^i(x) \cdot G^{-1}(x) \Lambda_1^j(x) \det(DF(x)) \, dx

**Derivative Operators**

The weak curl operator :math:`\text{weak\_curl}: V_2 \to V_1`:

.. math::

    (\text{weak\_curl})_{ij} = \int_\Omega \Lambda_1^i(x) \cdot \nabla \times \Lambda_2^j(x) \det(DF(x)) \, dx

Dimensions: :math:`\text{weak\_curl} \in \mathbb{R}^{N_1 \times N_2}`.

The strong curl operator :math:`\text{strong\_curl}: V_1 \to V_2`:

.. math::

    (\text{strong\_curl})_{ij} = \int_\Omega \Lambda_2^i(x) \cdot G(x) \nabla \times \Lambda_1^j(x) \frac{1}{\det(DF(x))} \, dx

Dimensions: :math:`\text{strong\_curl} \in \mathbb{R}^{N_2 \times N_1}`.

**Cross-Product Projections**

The cross-product projection :math:`P_{2 \times 1 \to 2}` computes:

.. math::

    (P_{2 \times 1 \to 2}(\mathbf{B}, \mathbf{J}))_i = \int_\Omega (\mathbf{B} \times \mathbf{J}) \cdot \Lambda_2^i(x) \frac{1}{\det(DF(x))} \, dx

This is used to compute :math:`\mathbf{J} \times \mathbf{B}` in the finite element space.
Dimensions: :math:`P_{2 \times 1 \to 2}(\mathbf{B}, \mathbf{J}) \in \mathbb{R}^{N_2}`.

**Leray Projection**

The Leray projection :math:`P_{\text{Leray}}: V_2 \to V_2^{\text{div-free}}` projects onto the divergence-free subspace:

.. math::

    P_{\text{Leray}} = I - \text{strong\_grad} \circ (\text{weak\_div} \circ M_1^{-1} \circ \text{strong\_grad})^{-1} \circ \text{weak\_div}

This ensures :math:`\nabla \cdot (P_{\text{Leray}} \mathbf{u}) = 0` for all :math:`\mathbf{u} \in V_2`.
Dimensions: :math:`P_{\text{Leray}} \in \mathbb{R}^{N_2 \times N_2}`.

**Initial Magnetic Field**

The initial magnetic field is constructed from a harmonic eigenmode by solving:

.. math::

    (M_2 \Delta_2) \mathbf{v} = \lambda \mathbf{v}

where :math:`M_2` is the 2-form mass matrix and :math:`\Delta_2` is the 2-form Laplacian.
The eigenvector :math:`\mathbf{v}` corresponding to the smallest eigenvalue :math:`\lambda` is extracted:

.. math::

    \mathbf{B}_{\text{harm}} = \mathbf{v}_0

where :math:`\mathbf{v}_0` is the first eigenvector from ``jnp.linalg.eigh(M_2 \Delta_2)[1][:, 0]`.
The field is normalized to unit L2 norm:

.. math::

    \hat{\mathbf{B}} = \frac{\mathbf{B}_{\text{harm}}}{\|\mathbf{B}_{\text{harm}}\|_2}

where :math:`\|\mathbf{B}\|_2^2 = \mathbf{B}^T M_2 \mathbf{B}`.

**Diagnostics**

**Magnetic Energy**:

.. math::

    E = \frac{1}{2} \int_\Omega |\mathbf{B}|^2 \, dx = \frac{1}{2} \mathbf{B}^T M_2 \mathbf{B}

**Magnetic Helicity**:

.. math::

    H = \int_\Omega \mathbf{A} \cdot \mathbf{B} \, dx

where :math:`\mathbf{A}` is the vector potential satisfying :math:`\nabla \times \mathbf{A} = \mathbf{B}`.

**Force Norm**:

.. math::

    \|\mathbf{f}\|_2 = \|\mathbf{J} \times \mathbf{B} - \nabla p\|_2

**Divergence Norm**:

.. math::

    \|\nabla \cdot \mathbf{B}\|_2

Code Walkthrough
----------------

The script implements a magnetic relaxation solver for MHD equilibrium:

**Block 1: Imports and Configuration (lines 1-16)**
   Imports JAX, NumPy, and MRX modules for DeRham sequences, I/O, mappings, relaxation,
   and plotting. Enables 64-bit precision for numerical accuracy.

**Block 2: Main Function (lines 18-33)**
   The ``main`` function parses command-line arguments using ``parse_args()`` and updates
   the default configuration. It then calls ``run()`` with the updated configuration.
   Command-line parameters override defaults using the format ``parameter_name=value``.

**Block 3: Setup and Initialization (lines 36-83)**
   The ``run()`` function:
   
   - Creates output directory structure: ``script_outputs/solovev/{run_name}/``
   - Initializes trace dictionary for storing simulation data
   - Selects boundary mapping based on ``boundary_type``:
     * ``tokamak``: Uses ``cerfon_map`` (circular cross-section)
     * ``helix``: Uses ``helical_map`` (helical boundary)
     * ``rotating_ellipse``: Uses ``rotating_ellipse_map`` (rotating elliptical cross-section)
   - Sets up DeRham sequence with polar coordinates and Dirichlet boundary conditions
   - Assembles all necessary matrices (mass, derivative, cross-product projections, Leray projection)
   - Initializes magnetic field from harmonic eigenmode:
   
     .. math::
     
         \mathbf{B}_{\text{harm}} = \text{eigh}(M_2 \Delta_2)[1][:, 0]
     
     where :math:`M_2` is the 2-form mass matrix, :math:`\Delta_2` is the 2-form Laplacian,
     and ``eigh`` computes the eigenvector corresponding to the smallest eigenvalue of the
     generalized eigenvalue problem :math:`(M_2 \Delta_2) \mathbf{v} = \lambda \mathbf{v}`.
   - Normalizes the initial field to unit L2 norm: :math:`\hat{\mathbf{B}} = \mathbf{B}_{\text{harm}} / \|\mathbf{B}_{\text{harm}}\|_2`

**Block 4: Magnetic Relaxation Loop (lines 85-86)**
   Calls ``run_relaxation_loop()`` which performs the time-stepping iteration to relax
   the magnetic field toward equilibrium. This solves the MHD force balance equation:
   
   .. math::
   
       \mathbf{J} \times \mathbf{B} = \nabla p
   
   or for the force-free case:
   
   .. math::
   
       \mathbf{J} \times \mathbf{B} = 0
   
   where :math:`\mathbf{J} = \nabla \times \mathbf{B}` is the current density. The solver uses an implicit time-stepping scheme.

**Block 5: Post-processing and Output (lines 88-114)**
   After the relaxation loop completes:
   
   - Computes final pressure field using ``diagnostics.pressure()``
   - Saves all trace data (iterations, forces, energies, etc.) to HDF5 file
   - Generates plots using ``generate_solovev_plots()`` which creates:
     * Pressure contour plots
     * Force and energy convergence traces
     * Magnetic field visualizations (if ``save_B=True``)

The Solovev configuration provides a simple analytical equilibrium that serves as a
test case for the relaxation solver. The script demonstrates how to set up and run
magnetic relaxation simulations for various boundary geometries.

Full script:

.. literalinclude:: ../../scripts/config_scripts/solovev.py
   :language: python
   :linenos:
