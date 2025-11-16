GVEC Tokamak Interface Test
============================

.. note::
   For general information about finite element discretization, basis functions, mesh parameters,
   polynomial degrees, boundary conditions, and matrix/operator dimensions, see :doc:`overview`.

This script tests the interface with GVEC data for tokamak configurations.
The script is located at ``scripts/interactive/test_gvec_tokamak.py``.

Mathematical Problem
====================

This script tests the interface between MRX and GVEC (Generalized Variational Equilibrium Code)
data for tokamak geometries. The goal is to:

1. Load GVEC equilibrium data from HDF5 files
2. Interpolate the GVEC coordinate mapping into MRX spline space
3. Test projection accuracy for functions defined on the GVEC geometry

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

where:
- :math:`\{\Lambda_0^i\}_{i=1}^{N_0}` are 0-form basis functions
- :math:`\mathbf{c}_1, \mathbf{c}_2 \in \mathbb{R}^{N_0}` are spline coefficients
- :math:`(\rho_j, \theta_j)` are GVEC grid points

This leads to the linear system:

.. math::

    M \mathbf{c} = \mathbf{y}

where:
- :math:`M \in \mathbb{R}^{(m_\rho m_\theta) \times N_0}` is the design matrix: :math:`M_{ji} = \Lambda_0^i(\rho_j, \theta^*_j, 0)`
- :math:`\mathbf{c} \in \mathbb{R}^{(m_\rho m_\theta) \times 2}` are the coefficients (one column for :math:`X_1`, one for :math:`X_2`)
- :math:`\mathbf{y} \in \mathbb{R}^{(m_\rho m_\theta) \times 2}` are the target values (one column for :math:`X_1`, one for :math:`X_2`)

The least-squares problem is solved separately for each coordinate function, or jointly as a matrix equation.

**Projection Error Test**

A test function is defined:

.. math::

    f(r,\theta,\zeta) = \sin(2\pi\theta) \sin(\pi r)

The projection error is computed as:

.. math::

    \text{error} = \frac{\|f - f_h\|_{L^2(\Omega)}}{\|f\|_{L^2(\Omega)}}

where:
- :math:`f_h = \sum_{i=1}^{N_0} \hat{f}_i \Lambda_0^i` is the projected function
- :math:`\hat{f} = P_0(f)` is the projection of :math:`f` onto the 0-form space

The script demonstrates:

- Loading GVEC data from HDF5 files
- Interpolating mappings from GVEC data
- Testing projection accuracy
- Visualizing tokamak configurations

Usage:

.. code-block:: bash

    python scripts/interactive/test_gvec_tokamak.py

The script generates plots showing the tokamak configuration and projection errors.

Mathematical Formulation
=========================

**Design Matrix Construction**

The design matrix :math:`M \in \mathbb{R}^{(m_\rho m_\theta) \times N_0}` is constructed by evaluating
basis functions at GVEC grid points:

.. math::

    M_{ji} = \Lambda_0^i(\rho_j, \theta^*_j, 0)

where:
- :math:`j = 1, \ldots, m_\rho m_\theta` indexes grid points
- :math:`i = 1, \ldots, N_0` indexes basis functions
- :math:`(\rho_j, \theta^*_j)` are GVEC grid coordinates

**Least-Squares Solution**

The least-squares problem:

.. math::

    \min_{\mathbf{c}} \|M \mathbf{c} - \mathbf{y}\|_2^2

is solved using:

.. math::

    \mathbf{c} = (M^T M)^{-1} M^T \mathbf{y}

or using the pseudo-inverse:

.. math::

    \mathbf{c} = M^+ \mathbf{y}

**Mapping Construction**

The interpolated functions are used to construct the mapping:

.. math::

    X_1^h(\rho, \theta^*) = \sum_{i=1}^{N_0} c_{1,i} \Lambda_0^i(\rho, \theta^*, 0)

.. math::

    X_2^h(\rho, \theta^*) = \sum_{i=1}^{N_0} c_{2,i} \Lambda_0^i(\rho, \theta^*, 0)

The full mapping is then:

.. math::

    F(\rho, \theta^*, \zeta) = \begin{bmatrix}
        X_1^h(\rho, \theta^*) \cos(2\pi\zeta) \\
        X_1^h(\rho, \theta^*) \sin(2\pi\zeta) \\
        X_2^h(\rho, \theta^*)
    \end{bmatrix}

**Projection Operator**

The 0-form projection operator :math:`P_0: L^2(\Omega) \to V_0`:

.. math::

    P_0(f) = \arg\min_{v_h \in V_0} \|f - v_h\|_{L^2(\Omega)}

The projection coefficients satisfy:

.. math::

    M_0 \hat{f} = \mathbf{b}, \quad b_i = \int_\Omega f(x) \Lambda_0^i(x) \det(DF(x)) \, dx

where :math:`M_0 \in \mathbb{R}^{N_0 \times N_0}` is the 0-form mass matrix.

**Error Computation**

The relative L2 error:

.. math::

    \text{error} = \frac{\|f - f_h\|_{L^2(\Omega)}}{\|f\|_{L^2(\Omega)}}

is computed using quadrature:

.. math::

    \|f - f_h\|_{L^2(\Omega)}^2 = \int_\Omega (f(x) - f_h(x))^2 \det(DF(x)) \, dx \approx \sum_{j=1}^{n_q} (f(x_j) - f_h(x_j))^2 J_j w_j

where:
- :math:`n_q` is the number of quadrature points
- :math:`x_j` are quadrature points
- :math:`J_j = \det(DF(x_j))` are Jacobian determinants
- :math:`w_j` are quadrature weights

Code Walkthrough
----------------

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
