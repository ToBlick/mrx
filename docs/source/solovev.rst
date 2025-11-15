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

Full script:

.. literalinclude:: ../../scripts/config_scripts/solovev.py
   :language: python
   :linenos:

