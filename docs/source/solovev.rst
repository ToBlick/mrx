Solovev Configuration
====================

This script runs a magnetic relaxation simulation for a Solovev configuration.
The script is located at ``scripts/config_scripts/solovev.py``.

The Solovev configuration is a simple tokamak equilibrium solution that can be used
as a starting point for magnetic relaxation simulations.

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

Code Walkthrough
================

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
     
         \mathbf{B}_{\text{harm}} = \text{eig}(M_2 \Delta_2)[1][:, 0]
     
     where :math:`M_2` is the 2-form mass matrix and :math:`\Delta_2` is the 2-form Laplacian.
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
