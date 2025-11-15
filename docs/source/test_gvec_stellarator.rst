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

Code Walkthrough
================

This script tests the GVEC interface for 3D stellarator geometries (non-axisymmetric):

**Block 1: Imports and Data Loading (lines 1-34)**
   Imports modules and loads GVEC stellarator data from ``data/gvec_stellarator.h5``:
   
   - :math:`\theta^*`: Modified poloidal angle :math:`\theta^*(\rho,\theta,\zeta)` (3D array)
   - :math:`\rho, \theta, \zeta`: Radial, poloidal, and toroidal coordinate grids
   - :math:`X_1, X_2`: Physical coordinates as functions of :math:`(\rho, \theta^*, \zeta)`
   - :math:`n_{\text{fp}}=3`: Number of field periods (rotational symmetry)

**Block 2: Mapping Interpolation (lines 36-57)**
   Interpolates 3D GVEC mapping:
   
   - Creates 3D DeRham sequence for approximating coordinate functions
   - Evaluates basis functions on 3D meshgrid
   - Solves least-squares problem to find spline coefficients
   - Constructs ``gvec_stellarator_map`` with ``nfp=3``

**Block 3: Magnetic Field Interpolation (lines 59-150)**
   Tests interpolation of magnetic field ``B`` from GVEC data:
   
   - Creates DeRham sequence using GVEC mapping
   - Defines physical 2-form basis functions (pushforward of logical basis functions):
   
     .. math::
     
         \Lambda_2^{\text{phys}}(i,x) = \frac{D\Phi \Lambda_2[i]}{\det(D\Phi)}
   
   - Builds design matrix by evaluating physical basis at GVEC grid points
   - Solves least-squares:
   
     .. math::
     
         M \mathbf{c} = \mathbf{B}_{\text{data}}
     
     to find field coefficients :math:`\mathbf{c}`
   - Tests projection accuracy for various mesh sizes

**Block 4: Field Line Integration (lines 152-303)**
   Validates magnetic field representation:
   
   - Integrates field lines using the interpolated field
   - Compares with GVEC field line data
   - Visualizes field line structure

**Block 5: Visualization (lines 305-303)**
   Generates plots showing:
   
   - Stellarator boundary shape (3D visualization)
   - Magnetic field magnitude contours
   - Field line trajectories
   - Projection error convergence

This script demonstrates that MRX can handle complex 3D stellarator geometries from
GVEC, enabling MHD equilibrium and stability calculations on experimentally relevant
configurations.

Full script:

.. literalinclude:: ../../scripts/interactive/test_gvec_stellarator.py
   :language: python
   :linenos:
