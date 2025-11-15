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

Code Walkthrough
================

This script tests the interface between MRX and GVEC (Generalized Variational Equilibrium Code)
data for tokamak geometries:

**Block 1: Imports and Data Loading (lines 1-27)**
   Imports xarray for reading HDF5/NetCDF data and MRX modules. Loads GVEC tokamak
   equilibrium data from ``data/gvec_tokamak.h5``:
   
   - :math:`\theta^*`: Modified poloidal angle mapping :math:`\theta^*(\rho,\theta)`
   - :math:`\rho, \theta`: Radial and poloidal coordinate grids
   - :math:`X_1, X_2`: Physical coordinates :math:`(R, Z)` as functions of :math:`(\rho, \theta^*)`

**Block 2: Mapping Interpolation (lines 29-59)**
   Interpolates GVEC coordinate mapping into MRX spline space:
   
   - Creates DeRham sequence ``mapSeq`` for approximating :math:`X_1(\rho,\theta^*)` and :math:`X_2(\rho,\theta^*)`
   - Builds design matrix :math:`M` by evaluating 0-form basis functions at GVEC grid points
   - Solves least-squares problem:
   
     .. math::
     
         M \mathbf{c} = [X_1, X_2]
     
     to find spline coefficients :math:`\mathbf{c}`
   - Constructs ``gvec_stellarator_map`` from interpolated coordinates (note: uses
     stellarator map function but with ``nfp=1`` for axisymmetric tokamak)

**Block 3: Projection Accuracy Test (lines 61-240)**
   Tests projection accuracy for various mesh sizes:
   
   - Defines test function:
   
     .. math::
     
         f(r,\theta,\zeta) = \sin(2\pi\theta) \sin(\pi r)
   
   - Creates DeRham sequence using GVEC mapping
   - Projects function into finite element space: :math:`\hat{f} = P_0(f)`
   - Reconstructs function: :math:`f_h = \text{DiscreteFunction}(\hat{f}, \Lambda_0, E_0)`
   - Computes L2 projection error:
   
     .. math::
     
         \text{error} = \frac{\|f - f_h\|_{L^2}}{\|f\|_{L^2}}
   
   - Tests convergence as mesh is refined (:math:`n \in \{4,6,8,\ldots,18\}`)

**Block 4: Visualization (lines 242-303)**
   Generates plots:
   
   - GVEC coordinate mapping visualization
   - Projection error convergence plot
   - Comparison of exact vs. projected function

This script validates that MRX can accurately represent geometries from external
equilibrium codes like GVEC, enabling the use of MRX solvers on experimentally
relevant configurations.

Full script:

.. literalinclude:: ../../scripts/interactive/test_gvec_tokamak.py
   :language: python
   :linenos:
