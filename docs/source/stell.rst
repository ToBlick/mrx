Stellarator Configuration
=========================

This script runs a magnetic relaxation simulation for a stellarator configuration.
The script is located at ``scripts/config_scripts/stell.py``.

Usage:

.. code-block:: bash

    python scripts/config_scripts/stell.py run_name=test_run n_r=16 n_theta=16 n_zeta=8

The script accepts various parameters similar to the Solovev script.
It generates HDF5 files with simulation data and PDF plots.

Code Walkthrough
================

This script is similar to ``solovev.py`` but configured specifically for stellarator
geometries with rotating elliptical cross-sections:

**Block 1: Imports and Configuration (lines 1-15)**
   Imports necessary modules. Note that this script uses ``rotating_ellipse_map``
   specifically for stellarator boundaries.

**Block 2: Main Function (lines 17-33)**
   Parses command-line arguments and sets default configuration parameters:
   
   - ``boundary_type="rotating_ellipse"``: Fixed for stellarator geometry
   - ``eps=0.33``: Ellipticity parameter
   - ``kappa=1.1``: Elongation parameter
   - ``q_star=1.54``: Safety factor
   - ``n_fp=3``: Number of field periods (rotational symmetry)
   - Default discretization: ``n_r=8, n_theta=8, n_zeta=4``
   - ``dt=1e-4``: Time step size

**Block 3: Setup and Initial Condition (lines 36-108)**
   The ``run()`` function:
   
   - Creates output directory: ``script_outputs/stell/{run_name}/``
   - Sets up rotating ellipse mapping with specified parameters
   - Configures DeRham sequence with polar coordinates
   - Defines initial magnetic field :math:`\mathbf{B}_{\text{xyz}}(p)` in physical space:
   
     .. math::
     
         B_R &= z R \\
         B_\phi &= \frac{\tau}{R} \\
         B_z &= -\left(\frac{1}{2}(R^2 - 1^2) + z^2\right)
     
     where :math:`\tau = q_*` is the safety factor and :math:`R = \sqrt{x^2 + y^2}`.
   
   - Defines perturbation function :math:`\delta\mathbf{B}_{\text{xyz}}(p)` for adding magnetic islands:
   
     .. math::
     
         \delta\mathbf{B}_r = a(r) \sin(2\pi\theta m_{\text{pol}}) \sin(2\pi\zeta n_{\text{tor}}) \frac{\partial F}{\partial r}
     
     where :math:`a(r) = \exp(-(r - r_0)^2/(2\sigma^2))` is a Gaussian radial profile.
   - Projects initial field into FEM space and applies Leray projection to ensure
     divergence-free condition
   - Optionally applies perturbation if ``apply_pert_after=0`` and ``pert_strength>0``

**Block 4: Magnetic Relaxation Loop (lines 111-113)**
   Runs the relaxation loop with perturbation function stored in CONFIG for later use.

**Block 5: Post-processing (lines 115-142)**
   Computes final pressure, saves data to HDF5, and generates trace plots.
   Note: This script uses ``trace_plot()`` directly instead of ``generate_solovev_plots()``,
   so it only generates convergence traces, not pressure contour plots.

The stellarator configuration uses a rotating elliptical boundary that creates 3D
magnetic field structure with inherent rotational transform, making it distinct from
axisymmetric tokamak configurations.

Full script:

.. literalinclude:: ../../scripts/config_scripts/stell.py
   :language: python
   :linenos:
