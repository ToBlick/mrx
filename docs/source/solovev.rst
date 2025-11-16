Solovev Configuration
======================

.. note::
   For general information about finite element discretization, basis functions, mesh parameters,
   polynomial degrees, boundary conditions, and matrix/operator dimensions, see :doc:`overview`.

This script runs a magnetic relaxation simulation for a Solovev configuration.
The script is located at ``scripts/config_scripts/solovev.py``.

**Problem Statement**

The Solovev configuration solves the magnetohydrostatic (MHS) equilibrium equation:

.. math::

    \mathbf{J} \times \mathbf{B} = \nabla p \quad \text{in } \Omega

with boundary conditions:

.. math::

    \mathbf{B} \cdot \mathbf{n} = 0 \quad \text{on } \partial\Omega

and the constraint:

.. math::

    \nabla \cdot \mathbf{B} = 0 \quad \text{in } \Omega

where:
- :math:`\mathbf{B}: \Omega \to \mathbb{R}^3` is the magnetic field (2-form, area form)
- :math:`\mathbf{J} = \nabla \times \mathbf{B}` is the current density (1-form)
- :math:`p: \Omega \to \mathbb{R}` is the plasma pressure (0-form, scalar)
- :math:`\mathbf{n}` is the outward unit normal vector on the boundary
- :math:`\Omega` is the toroidal plasma domain

For the force-free case (:math:`\nabla p = 0`), the equation becomes:

.. math::

    \mathbf{J} \times \mathbf{B} = 0 \quad \text{in } \Omega

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
divergence-free space using the Leray projection :math:`P_{\mathrm{Leray}}`.

**Time-Stepping Scheme**

The time-stepping uses a midpoint rule (Crank-Nicolson):

.. math::

    \mathbf{B}^{n+1} = \mathbf{B}^n + \Delta t \nabla \times \mathbf{E}(\mathbf{B}^{\mathrm{mid}})

where :math:`\mathbf{B}^{\mathrm{mid}} = (\mathbf{B}^n + \mathbf{B}^{n+1})/2` is evaluated at the midpoint.

The system is solved using a Picard iteration scheme with adaptive time-stepping.

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

**Discretization Parameters**

This script uses a 3D DeRham sequence with mesh parameters :math:`n_r, n_\theta, n_\zeta`, polynomial degrees :math:`p_r, p_\theta, p_\zeta`,
and boundary conditions (typically clamped in radial direction, periodic in poloidal and toroidal directions).
For details on computing the number of DOFs :math:`N_0, N_1, N_2, N_3` from these parameters, see :doc:`overview`.

**Finite Element Spaces**

The magnetic field :math:`\mathbf{B}` is represented as a 2-form (area form) in the finite element space:

.. math::

    V_2 = \text{span}\{\Lambda_2^i\}_{i=1}^{N_2}

The current density :math:`\mathbf{J} = (\nabla \times)_h \mathbf{B}` is a 1-form, where :math:`(\nabla \times)_h: V_2 \to V_1` is the curl operator (weak form).

**Matrix and Operator Dimensions**

The 2-form mass matrix :math:`M_2 \in \mathbb{R}^{N_2 \times N_2}` and 1-form mass matrix :math:`M_1 \in \mathbb{R}^{N_1 \times N_1}` are used.
The Leray projection :math:`P_{\mathrm{Leray}} \in \mathbb{R}^{N_2 \times N_2}` ensures divergence-free velocity fields.
Cross-product projections compute :math:`\mathbf{J} \times \mathbf{B}` in the finite element space.

**Initial Magnetic Field**

The initial magnetic field is constructed from a harmonic eigenmode by solving:

.. math::

    (M_2 \Delta_2) \mathbf{v} = \lambda \mathbf{v}

The eigenvector :math:`\mathbf{v}` corresponding to the smallest eigenvalue is extracted and normalized to unit L2 norm.

Code Walkthrough
----------------

**Block 1: Imports and Configuration (lines 1-17)**

Imports JAX, NumPy, and MRX modules for DeRham sequences, I/O, mappings, relaxation,
and plotting. Enables 64-bit precision for numerical accuracy.

.. literalinclude:: ../../scripts/config_scripts/solovev.py
   :language: python
   :lines: 1-17

**Block 2: Main Function (lines 19-41)**

The ``main`` function parses command-line arguments using ``parse_args()`` and updates
the default configuration. It then calls ``run()`` with the updated configuration.
Command-line parameters override defaults using the format ``parameter_name=value``.

.. literalinclude:: ../../scripts/config_scripts/solovev.py
   :language: python
   :lines: 19-41

**Block 3: Setup and Initialization (lines 44-91)**

The ``run()`` function creates output directory structure, initializes trace dictionary,
and selects boundary mapping based on ``boundary_type``:

- ``tokamak``: Uses ``cerfon_map`` (circular cross-section)
- ``helix``: Uses ``helical_map`` (helical boundary)
- ``rotating_ellipse``: Uses ``rotating_ellipse_map`` (rotating elliptical cross-section)

Sets up DeRham sequence with polar coordinates and Dirichlet boundary conditions,
assembles all necessary matrices (mass, derivative, cross-product projections, Leray projection),
and initializes magnetic field from harmonic eigenmode:

.. math::

    \mathbf{B}_{\mathrm{harm}} = \text{eigh}(M_2 \Delta_2)[1][:, 0]

The field is normalized to unit L2 norm: :math:`\hat{\mathbf{B}} = \mathbf{B}_{\mathrm{harm}} / \|\mathbf{B}_{\mathrm{harm}}\|_2`

.. literalinclude:: ../../scripts/config_scripts/solovev.py
   :language: python
   :lines: 44-91

**Block 4: Magnetic Relaxation Loop (lines 93-94)**

Calls ``run_relaxation_loop()`` which performs the time-stepping iteration to relax
the magnetic field toward equilibrium. This solves the MHD force balance equation:

.. math::

    \mathbf{J} \times \mathbf{B} = \nabla p

or for the force-free case:

.. math::

    \mathbf{J} \times \mathbf{B} = 0

where :math:`\mathbf{J} = \nabla \times \mathbf{B}` is the current density. The solver uses an implicit time-stepping scheme with Picard iteration.

.. literalinclude:: ../../scripts/config_scripts/solovev.py
   :language: python
   :lines: 93-94

**Block 5: Post-processing and Output (lines 96-122)**

After the relaxation loop completes:

- Computes final pressure field using ``diagnostics.pressure()``
- Saves all trace data (iterations, forces, energies, etc.) to HDF5 file
- Generates plots using ``generate_solovev_plots()`` which creates:
  * Pressure contour plots
  * Force and energy convergence traces
  * Magnetic field visualizations (if ``save_B=True``)

.. literalinclude:: ../../scripts/config_scripts/solovev.py
   :language: python
   :lines: 96-122

The Solovev configuration provides a simple analytical equilibrium that serves as a
test case for the relaxation solver. The script demonstrates how to set up and run
magnetic relaxation simulations for various boundary geometries.
