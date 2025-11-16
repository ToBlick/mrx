Stellarator Configuration
=========================

.. note::
   For general information about finite element discretization, basis functions, mesh parameters,
   polynomial degrees, boundary conditions, and matrix/operator dimensions, see :doc:`overview`.

This script runs a magnetic relaxation simulation for a stellarator configuration.
The script is located at ``scripts/config_scripts/stell.py``.

**Problem Statement**

The stellarator configuration solves the magnetohydrostatic (MHS) equilibrium equation:

.. math::

    \mathbf{J} \times \mathbf{B} = \nabla p \quad \text{in } \Omega

with boundary conditions:

.. math::

    \mathbf{B} \cdot \mathbf{n} = 0 \quad \text{on } \partial\Omega

and the constraint:

.. math::

    \nabla \cdot \mathbf{B} = 0 \quad \text{in } \Omega

where:
- :math:`\mathbf{B}: \Omega \to \mathbb{R}^3` is the magnetic field (2-form)
- :math:`\mathbf{J} = \nabla \times \mathbf{B}` is the current density (1-form)
- :math:`p: \Omega \to \mathbb{R}` is the plasma pressure (0-form)
- :math:`\mathbf{n}` is the outward unit normal vector on the boundary
- :math:`\Omega` is the stellarator plasma domain

The stellarator geometry uses a rotating elliptical boundary with 3D structure:

.. math::

    F(r, \chi, \zeta) = \begin{bmatrix}
        R(r, \chi, \zeta) \cos(\phi) \\
        R(r, \chi, \zeta) \sin(\phi) \\
        Z(r, \chi, \zeta)
    \end{bmatrix}

where:
- :math:`R(r, \chi, \zeta) = R_0 + \epsilon r \cos(2\pi\chi + n_{\mathrm{fp}} \zeta)` is the major radius
- :math:`Z(r, \chi, \zeta) = \epsilon \kappa r \sin(2\pi\chi + n_{\mathrm{fp}} \zeta)` is the vertical coordinate
- :math:`\phi = 2\pi\zeta` is the toroidal angle
- :math:`R_0 = 1.0` is the major radius
- :math:`\epsilon = 0.33` is the inverse aspect ratio
- :math:`\kappa = 1.1` is the elongation parameter
- :math:`n_{\mathrm{fp}} = 3` is the number of field periods (rotational symmetry)

**Initial Magnetic Field**

The initial magnetic field is defined in physical cylindrical coordinates:

.. math::

    B_R &= z R \\
    B_\phi &= \frac{\tau}{R} \\
    B_z &= -\left(\frac{1}{2}(R^2 - R_0^2) + z^2\right)

where :math:`\tau = q_* = 1.54` is the safety factor.

**Perturbation Function**

A perturbation can be added to introduce magnetic islands:

.. math::

    \delta\mathbf{B}_r = a(r) \sin(2\pi\theta m_{\mathrm{pol}}) \sin(2\pi\zeta n_{\mathrm{tor}}) \frac{\partial F}{\partial r}

where:
- :math:`a(r) = \exp(-(r - r_0)^2/(2\sigma^2))` is a Gaussian radial profile
- :math:`m_{\mathrm{pol}}` is the poloidal mode number
- :math:`n_{\mathrm{tor}}` is the toroidal mode number
- :math:`r_0` is the radial location of the perturbation
- :math:`\sigma` is the radial width

Usage:

.. code-block:: bash

    python scripts/config_scripts/stell.py run_name=test_run n_r=16 n_theta=16 n_zeta=8

The script accepts various parameters similar to the Solovev script.
It generates HDF5 files with simulation data and PDF plots.

Code Walkthrough
----------------

**Block 1: Imports and Configuration (lines 1-15)**

Imports necessary modules. Note that this script uses ``rotating_ellipse_map``
specifically for stellarator boundaries.

.. literalinclude:: ../../scripts/config_scripts/stell.py
   :language: python
   :lines: 1-15

**Block 2: Main Function (lines 17-33)**

Parses command-line arguments and sets default configuration parameters:

- ``boundary_type="rotating_ellipse"``: Fixed for stellarator geometry
- ``eps=0.33``: Ellipticity parameter
- ``kappa=1.1``: Elongation parameter
- ``q_star=1.54``: Safety factor
- ``n_fp=3``: Number of field periods (rotational symmetry)
- Default discretization: ``n_r=8, n_theta=8, n_zeta=4``
- ``dt=1e-4``: Time step size

.. literalinclude:: ../../scripts/config_scripts/stell.py
   :language: python
   :lines: 17-33

**Block 3: Setup and Initial Condition (lines 36-108)**

The ``run()`` function:

- Creates output directory: ``script_outputs/stell/{run_name}/``
- Sets up rotating ellipse mapping with specified parameters
- Configures DeRham sequence with polar coordinates
- Defines initial magnetic field :math:`\mathbf{B}_{\mathrm{xyz}}(p)` in physical space
- Defines perturbation function :math:`\delta\mathbf{B}_{\mathrm{xyz}}(p)` for adding magnetic islands
- Projects initial field into FEM space and applies Leray projection to ensure
  divergence-free condition
- Optionally applies perturbation if ``apply_pert_after=0`` and ``pert_strength>0``

.. literalinclude:: ../../scripts/config_scripts/stell.py
   :language: python
   :lines: 36-108

**Block 4: Magnetic Relaxation Loop (lines 111-113)**

Runs the relaxation loop with perturbation function stored in CONFIG for later use.

.. literalinclude:: ../../scripts/config_scripts/stell.py
   :language: python
   :lines: 111-113

**Block 5: Post-processing (lines 115-139)**

Computes final pressure, saves data to HDF5, and generates trace plots.
Note: This script uses ``trace_plot()`` directly instead of ``generate_solovev_plots()``,
so it only generates convergence traces, not pressure contour plots.

The stellarator configuration uses a rotating elliptical boundary that creates 3D
magnetic field structure with inherent rotational transform, making it distinct from
axisymmetric tokamak configurations.

.. literalinclude:: ../../scripts/config_scripts/stell.py
   :language: python
   :lines: 115-139
