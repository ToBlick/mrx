Stellarator Configuration
=========================

This script runs a magnetic relaxation simulation for a stellarator configuration.
The script is located at ``scripts/config_scripts/stell.py``.

Usage:

.. code-block:: bash

    python scripts/config_scripts/stell.py run_name=test_run n_r=16 n_theta=16 n_zeta=8

The script accepts various parameters similar to the Solovev script.
It generates HDF5 files with simulation data and PDF plots.

Full script:

.. literalinclude:: ../../scripts/config_scripts/stell.py
   :language: python
   :linenos:

