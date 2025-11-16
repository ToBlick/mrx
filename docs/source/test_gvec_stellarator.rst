GVEC Stellarator Interface Test
================================

.. note::
   For general information about finite element discretization, basis functions, mesh parameters,
   polynomial degrees, boundary conditions, and matrix/operator dimensions, see :doc:`overview`.

This script tests the interface with GVEC data for stellarator configurations.
The script is located at ``scripts/interactive/test_gvec_stellarator.py``.

Mathematical Problem
====================

This script tests the GVEC interface for 3D stellarator geometries (non-axisymmetric).
The goal is to:

1. Load GVEC equilibrium data from HDF5 files
2. Interpolate the GVEC coordinate mapping into MRX spline space
3. Interpolate magnetic field data from GVEC into MRX finite element space
4. Test projection accuracy and field line integration

**GVEC Data Structure**

GVEC provides 3D equilibrium data:
- :math:`\rho \in [0,1]`: Normalized radial coordinate
- :math:`\theta \in [0,2\pi]`: Poloidal angle
- :math:`\zeta \in [0,2\pi/n_{\text{fp}}]`: Toroidal angle (one field period)
- :math:`\theta^*(\rho,\theta,\zeta)`: Modified poloidal angle mapping (3D)
- :math:`X_1(\rho,\theta^*,\zeta), X_2(\rho,\theta^*,\zeta)`: Physical coordinates :math:`(R, Z)`
- :math:`n_{\text{fp}} = 3`: Number of field periods (rotational symmetry)

The mapping from logical coordinates :math:`(\rho, \theta^*, \zeta)` to physical coordinates is:

.. math::

    F(\rho, \theta^*, \zeta) = \begin{bmatrix}
        X_1(\rho, \theta^*, \zeta) \cos(2\pi n_{\text{fp}} \zeta) \\
        X_1(\rho, \theta^*, \zeta) \sin(2\pi n_{\text{fp}} \zeta) \\
        X_2(\rho, \theta^*, \zeta)
    \end{bmatrix}

**Least-Squares Interpolation**

The coordinate functions are interpolated using least-squares fitting:

.. math::

    \min_{\mathbf{c}_1, \mathbf{c}_2} \sum_{j=1}^{m_\rho m_\theta m_\zeta} \left\| \sum_{i=1}^{N_0} c_{1,i} \Lambda_0^i(\rho_j, \theta^*_j, \zeta_j) - X_1(\rho_j, \theta_j, \zeta_j) \right\|^2 + \left\| \sum_{i=1}^{N_0} c_{2,i} \Lambda_0^i(\rho_j, \theta^*_j, \zeta_j) - X_2(\rho_j, \theta_j, \zeta_j) \right\|^2

**Magnetic Field Interpolation**

The magnetic field :math:`\mathbf{B}` is interpolated using pushforward basis functions:

.. math::

    \Lambda_2^{\text{phys}}(i,x) = \frac{D\Phi(x) \Lambda_2[i](x)}{\det(D\Phi(x))}

where :math:`\Phi` is the mapping from logical to physical coordinates.

The least-squares problem is:

.. math::

    \min_{\mathbf{c}} \sum_{j=1}^{m_\rho m_\theta m_\zeta} \left\| \sum_{i=1}^{N_2} c_i \Lambda_2^{\text{phys}}(i, x_j) - \mathbf{B}_{\text{data}}(x_j) \right\|^2

The script demonstrates:

- Loading GVEC data from HDF5 files
- Interpolating mappings from GVEC data
- Testing projection accuracy for stellarator geometries
- Visualizing stellarator configurations

Usage:

.. code-block:: bash

    python scripts/interactive/test_gvec_stellarator.py

The script generates plots showing the stellarator configuration and projection errors.

Mathematical Formulation
=========================

**3D Coordinate Interpolation**

The design matrix :math:`M \in \mathbb{R}^{(m_\rho m_\theta m_\zeta) \times N_0}` is constructed by evaluating
basis functions at GVEC grid points:

.. math::

    M_{ji} = \Lambda_0^i(\rho_j, \theta^*_j, \zeta_j)

where:
- :math:`j = 1, \ldots, m_\rho m_\theta m_\zeta` indexes grid points
- :math:`i = 1, \ldots, N_0` indexes basis functions

**Least-Squares Solution**

The least-squares problem:

.. math::

    \min_{\mathbf{c}} \|M \mathbf{c} - \mathbf{y}\|_2^2

is solved using:

.. math::

    \mathbf{c} = (M^T M)^{-1} M^T \mathbf{y}

**Pushforward Basis Functions**

For magnetic field interpolation, the physical basis functions are constructed using
pushforward:

.. math::

    \Lambda_2^{\text{phys}}(i,x) = \frac{D\Phi(x) \Lambda_2[i](x)}{\det(D\Phi(x))}

where:
- :math:`D\Phi(x)` is the Jacobian matrix of the mapping
- :math:`\det(D\Phi(x))` is the Jacobian determinant
- :math:`\Lambda_2[i]` are logical 2-form basis functions

This ensures that the interpolated field satisfies the correct transformation properties
under coordinate changes.

**Design Matrix for Field Interpolation**

The design matrix :math:`M_{\text{field}} \in \mathbb{R}^{(m_\rho m_\theta m_\zeta) \times N_2}`:

.. math::

    (M_{\text{field}})_{ji} = \Lambda_2^{\text{phys}}(i, x_j)

**Field Line Integration**

Field lines are integrated by solving:

.. math::

    \frac{d\mathbf{x}}{dt} = \mathbf{B}(\mathbf{x})

using a differential equation solver (e.g., Runge-Kutta).

**Projection Error**

The projection error is computed as:

.. math::

    \text{error} = \frac{\|\mathbf{B} - \mathbf{B}_h\|_{L^2(\Omega)}}{\|\mathbf{B}\|_{L^2(\Omega)}}

where:
- :math:`\mathbf{B}` is the exact field from GVEC data
- :math:`\mathbf{B}_h` is the interpolated field in MRX space

Code Walkthrough
----------------

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
