GVEC Stellarator Interface Test
================================

This script tests the interface with GVEC data for stellarator configurations.
The script is located at ``scripts/interactive/test_gvec_stellarator.py``.

The script demonstrates:

- Loading GVEC data from HDF5 files
- Interpolating mappings from GVEC data
- Testing projection accuracy for stellarator geometries
- Visualizing stellarator configurations

Usage:

.. code-block:: bash

    python scripts/interactive/test_gvec_stellarator.py

The script generates plots showing the stellarator configuration and projection errors.

Full script:

.. literalinclude:: ../../scripts/interactive/test_gvec_stellarator.py
   :language: python
   :linenos:

