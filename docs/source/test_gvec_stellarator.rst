GVEC Stellarator Interface Test
================================

.. note::
   For general information about finite element discretization, basis functions, mesh parameters,
   polynomial degrees, boundary conditions, and matrix/operator dimensions, see :doc:`overview`.

This script tests the interface with GVEC data for stellarator configurations.
The script is located at ``scripts/interactive/test_gvec_stellarator.py``.

**Problem Statement**

This script tests the GVEC interface for 3D stellarator geometries (non-axisymmetric).
The script does not solve a PDE directly, but rather:

1. Loads GVEC equilibrium data from HDF5 files
2. Interpolates the GVEC coordinate mapping into MRX spline space
3. Interpolates magnetic field data from GVEC into MRX finite element space
4. Tests projection accuracy and field line integration

The magnetic field interpolation solves a least-squares problem:

.. math::

    \min_{\mathbf{c}} \sum_{j=1}^{m_\rho m_\theta m_\zeta} \left\| \sum_{i=1}^{N_2} c_i \Lambda_2^{\mathrm{phys}}(i, x_j) - \mathbf{B}_{\mathrm{data}}(x_j) \right\|^2

where :math:`\Lambda_2^{\mathrm{phys}}` are pushforward basis functions and :math:`\mathbf{B}_{\mathrm{data}}` is the GVEC magnetic field data.

**GVEC Data Structure**

GVEC provides 3D equilibrium data:
- :math:`\rho \in [0,1]`: Normalized radial coordinate
- :math:`\theta \in [0,2\pi]`: Poloidal angle
- :math:`\zeta \in [0,2\pi/n_{\mathrm{fp}}]`: Toroidal angle (one field period)
- :math:`\theta^*(\rho,\theta,\zeta)`: Modified poloidal angle mapping (3D)
- :math:`X_1(\rho,\theta^*,\zeta), X_2(\rho,\theta^*,\zeta)`: Physical coordinates :math:`(R, Z)`
- :math:`n_{\mathrm{fp}} = 3`: Number of field periods (rotational symmetry)

The mapping from logical coordinates :math:`(\rho, \theta^*, \zeta)` to physical coordinates is:

.. math::

    F(\rho, \theta^*, \zeta) = \begin{bmatrix}
        X_1(\rho, \theta^*, \zeta) \cos(2\pi n_{\mathrm{fp}} \zeta) \\
        X_1(\rho, \theta^*, \zeta) \sin(2\pi n_{\mathrm{fp}} \zeta) \\
        X_2(\rho, \theta^*, \zeta)
    \end{bmatrix}

**Magnetic Field Interpolation**

The magnetic field :math:`\mathbf{B}` is interpolated using pushforward basis functions:

.. math::

    \Lambda_2^{\mathrm{phys}}(i,x) = \frac{D\Phi(x) \Lambda_2[i](x)}{\det(D\Phi(x))}

where :math:`\Phi` is the mapping from logical to physical coordinates.

Usage:

.. code-block:: bash

    python scripts/interactive/test_gvec_stellarator.py

The script generates plots showing the stellarator configuration and projection errors.

Code Walkthrough
----------------

**Block 1: Imports and Data Loading (lines 1-34)**

Imports modules and loads GVEC stellarator data from ``data/gvec_stellarator.h5`:

- :math:`\theta^*`: Modified poloidal angle :math:`\theta^*(\rho,\theta,\zeta)` (3D array)
- :math:`\rho, \theta, \zeta`: Radial, poloidal, and toroidal coordinate grids
- :math:`X_1, X_2`: Physical coordinates as functions of :math:`(\rho, \theta^*, \zeta)`
- :math:`n_{\mathrm{fp}}=3`: Number of field periods (rotational symmetry)

.. literalinclude:: ../../scripts/interactive/test_gvec_stellarator.py
   :language: python
   :lines: 1-34

**Block 2: Mapping Interpolation (lines 36-57)**

Interpolates 3D GVEC mapping:

- Creates 3D DeRham sequence for approximating coordinate functions
- Evaluates basis functions on 3D meshgrid
- Solves least-squares problem to find spline coefficients
- Constructs ``gvec_stellarator_map`` with ``nfp=3``

.. literalinclude:: ../../scripts/interactive/test_gvec_stellarator.py
   :language: python
   :lines: 36-57

**Block 3: Sequence Assembly (lines 59-65)**

Creates DeRham sequence using GVEC mapping for subsequent field interpolation and testing.

.. literalinclude:: ../../scripts/interactive/test_gvec_stellarator.py
   :language: python
   :lines: 59-65

**Block 4: Magnetic Field Interpolation (lines 68-100)**

Tests interpolation of magnetic field ``B`` from GVEC data:

- Defines physical 2-form basis functions (pushforward of logical basis functions):

.. math::

    \Lambda_2^{\mathrm{phys}}(i,x) = \frac{D\Phi \Lambda_2[i]}{\det(D\Phi)}

- Builds design matrix by evaluating physical basis at GVEC grid points (avoiding axis singularity)
- Solves least-squares:

.. math::

    M \mathbf{c} = \mathbf{B}_{\mathrm{data}}

to find field coefficients :math:`\mathbf{c}`.

.. literalinclude:: ../../scripts/interactive/test_gvec_stellarator.py
   :language: python
   :lines: 68-100

**Block 5: Field Validation and Diagnostics (lines 103-151)**

Validates magnetic field representation:

- Creates pushforward of interpolated field
- Checks divergence-free property
- Validates exactness identities (curl∘grad = 0, div∘curl = 0)
- Checks Laplacian eigenvalue patterns

.. literalinclude:: ../../scripts/interactive/test_gvec_stellarator.py
   :language: python
   :lines: 103-151

**Block 6: Projection Error Test (lines 154-206)**

Tests projection accuracy for various mesh sizes:

- Defines test function:

.. math::

    f(r,\theta,\zeta) = \sin(2\pi\theta) \sin(2\pi\zeta) \sin(\pi r)

- Projects function into finite element space
- Computes L2 projection error using ``jax.lax.scan``
- Tests convergence as mesh is refined
- Validates convergence rate matches expected order

.. literalinclude:: ../../scripts/interactive/test_gvec_stellarator.py
   :language: python
   :lines: 154-206

**Block 7: Visualization (lines 208-302)**

Generates plots showing:

- Projection error convergence
- Stellarator boundary shape (deformed polar grid visualization)
- Constant :math:`\rho` and :math:`\theta^*` coordinate lines

This script demonstrates that MRX can handle complex 3D stellarator geometries from
GVEC, enabling MHD equilibrium and stability calculations on experimentally relevant
configurations.

.. literalinclude:: ../../scripts/interactive/test_gvec_stellarator.py
   :language: python
   :lines: 208-302
