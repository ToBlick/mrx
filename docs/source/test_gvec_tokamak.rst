GVEC Tokamak Interface Test
============================

This script tests the interface with GVEC data for tokamak configurations.
The script is located at ``scripts/interactive/test_gvec_tokamak.py``.

The script demonstrates:

- Loading GVEC data from HDF5 files
- Interpolating mappings from GVEC data
- Testing projection accuracy
- Visualizing tokamak configurations

Usage:

.. code-block:: bash

    python scripts/interactive/test_gvec_tokamak.py

The script generates plots showing the tokamak configuration and projection errors.

Full script:

.. literalinclude:: ../../scripts/interactive/test_gvec_tokamak.py
   :language: python
   :linenos:

