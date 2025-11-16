GVEC Tokamak Interface Test
============================

.. note::
   For general information about finite element discretization, basis functions, mesh parameters,
   polynomial degrees, boundary conditions, and matrix/operator dimensions, see :doc:`overview`.

This script tests the interface with GVEC data for tokamak configurations.
The script is located at ``scripts/interactive/test_gvec_tokamak.py``.

**Problem Statement**

This script tests the interface between MRX and GVEC (Generalized Variational Equilibrium Code)
data for tokamak geometries. The script does not solve a PDE directly, but rather:

1. Loads GVEC equilibrium data from HDF5 files
2. Interpolates the GVEC coordinate mapping into MRX spline space
3. Tests projection accuracy for functions defined on the GVEC geometry

The test function is:

.. math::

    f(r,\theta,\zeta) = \sin(2\pi\theta) \sin(\pi r)

which is projected onto the 0-form finite element space and the projection error is computed.

**GVEC Data Structure**

GVEC provides equilibrium data in the form:
- :math:`\rho \in [0,1]`: Normalized radial coordinate
- :math:`\theta \in [0,2\pi]`: Poloidal angle
- :math:`\theta^*(\rho,\theta)`: Modified poloidal angle mapping
- :math:`X_1(\rho,\theta^*), X_2(\rho,\theta^*)`: Physical coordinates :math:`(R, Z)` in cylindrical coordinates

The mapping from logical coordinates :math:`(\rho, \theta^*, \zeta)` to physical coordinates is:

.. math::

    F(\rho, \theta^*, \zeta) = \begin{bmatrix}
        X_1(\rho, \theta^*) \cos(2\pi\zeta) \\
        X_1(\rho, \theta^*) \sin(2\pi\zeta) \\
        X_2(\rho, \theta^*)
    \end{bmatrix}

**Least-Squares Interpolation**

The functions :math:`X_1(\rho,\theta^*)` and :math:`X_2(\rho,\theta^*)` are interpolated into MRX spline space
using least-squares fitting:

.. math::

    \min_{\mathbf{c}_1, \mathbf{c}_2} \sum_{j=1}^{m_\rho m_\theta} \left\| \sum_{i=1}^{N_0} c_{1,i} \Lambda_0^i(\rho_j, \theta^*_j, 0) - X_1(\rho_j, \theta_j) \right\|^2 + \left\| \sum_{i=1}^{N_0} c_{2,i} \Lambda_0^i(\rho_j, \theta^*_j, 0) - X_2(\rho_j, \theta_j) \right\|^2

This leads to the linear system:

.. math::

    M \mathbf{c} = \mathbf{y}

where :math:`M \in \mathbb{R}^{(m_\rho m_\theta) \times N_0}` is the design matrix and :math:`\mathbf{c} \in \mathbb{R}^{N_0 \times 2}` contains coefficients for both :math:`X_1` and :math:`X_2`.

**Projection Error Test**

The projection error is computed as:

.. math::

    \text{error} = \frac{\|f - f_h\|_{L^2(\Omega)}}{\|f\|_{L^2(\Omega)}}

where :math:`f_h = \sum_{i=1}^{N_0} \hat{f}_i \Lambda_0^i` is the projected function and :math:`\hat{f} = P_0(f)` is the projection of :math:`f` onto the 0-form space.

Usage:

.. code-block:: bash

    python scripts/interactive/test_gvec_tokamak.py

The script generates plots showing the tokamak configuration and projection errors.

Code Walkthrough
----------------

**Block 1: Imports and Data Loading (lines 1-27)**

Imports xarray for reading HDF5/NetCDF data and MRX modules. Loads GVEC tokamak
equilibrium data from ``data/gvec_tokamak.h5`:

- :math:`\theta^*`: Modified poloidal angle mapping :math:`\theta^*(\rho,\theta)`
- :math:`\rho, \theta`: Radial and poloidal coordinate grids
- :math:`X_1, X_2`: Physical coordinates :math:`(R, Z)` as functions of :math:`(\rho, \theta^*)`

.. literalinclude:: ../../scripts/interactive/test_gvec_tokamak.py
   :language: python
   :lines: 1-27

**Block 2: Mapping Interpolation (lines 29-59)**

Interpolates GVEC coordinate mapping into MRX spline space:

- Creates DeRham sequence ``mapSeq`` for approximating :math:`X_1(\rho,\theta^*)` and :math:`X_2(\rho,\theta^*)`
- Builds design matrix :math:`M` by evaluating 0-form basis functions at GVEC grid points
- Solves least-squares problem:

.. math::

    M \mathbf{c} = [X_1, X_2]

to find spline coefficients :math:`\mathbf{c}`.
- Constructs ``gvec_stellarator_map`` from interpolated coordinates (note: uses
  stellarator map function but with ``nfp=1`` for axisymmetric tokamak)

.. literalinclude:: ../../scripts/interactive/test_gvec_tokamak.py
   :language: python
   :lines: 29-59

**Block 3: Projection Accuracy Test (lines 61-140)**

Tests projection accuracy for various mesh sizes:

- Defines test function:

.. math::

    f(r,\theta,\zeta) = \sin(2\pi\theta) \sin(\pi r)

- Creates DeRham sequence using GVEC mapping
- Projects function into finite element space: :math:`\hat{f} = P_0(f)`
- Reconstructs function: :math:`f_h = \text{DiscreteFunction}(\hat{f}, \Lambda_0, E_0)`
- Computes L2 projection error using ``jax.lax.scan`` to avoid memory issues
- Tests convergence as mesh is refined (:math:`n \in \{4,6,8,\ldots,18\}`)

.. literalinclude:: ../../scripts/interactive/test_gvec_tokamak.py
   :language: python
   :lines: 61-140

**Block 4: Visualization (lines 142-241)**

Generates plots:

- GVEC coordinate mapping visualization
- Projection error convergence plot
- Comparison of exact vs. projected function

This script validates that MRX can accurately represent geometries from external
equilibrium codes like GVEC, enabling the use of MRX solvers on experimentally
relevant configurations.

.. literalinclude:: ../../scripts/interactive/test_gvec_tokamak.py
   :language: python
   :lines: 142-241
