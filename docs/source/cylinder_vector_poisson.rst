Cylinder Vector Poisson
========================

.. note::
   For general information about finite element discretization, basis functions, mesh parameters,
   polynomial degrees, boundary conditions, and matrix/operator dimensions, see :doc:`overview`.

This script solves vector Poisson problems on a cylindrical domain.
The script is located at ``scripts/interactive/cylinder_vector_poisson.py``.

**Problem Statement**

This script solves a vector Poisson problem:

.. math::

    -\Delta \mathbf{u} = \mathbf{f} \quad \text{in } \Omega

with boundary conditions:

.. math::

    \mathbf{u} = 0 \quad \text{on } \partial\Omega

where:
- :math:`\mathbf{u}: \Omega \to \mathbb{R}^3` is the vector solution (1-form)
- :math:`\mathbf{f}: \Omega \to \mathbb{R}^3` is the vector source term (1-form)
- :math:`\Delta` is the vector Laplacian operator
- :math:`\Omega` is a cylindrical domain of radius :math:`a=1` and height :math:`h=1`

**Vector Laplacian**

The vector Laplacian is defined as:

.. math::

    \Delta \mathbf{u} = \nabla(\nabla \cdot \mathbf{u}) - \nabla \times (\nabla \times \mathbf{u})

This can be decomposed into:
- Gradient of divergence: :math:`\nabla(\nabla \cdot \mathbf{u})`
- Curl of curl: :math:`\nabla \times (\nabla \times \mathbf{u})`

**Exact Solution**

The exact solution has only an azimuthal component:

.. math::

    \mathbf{u}(r,\chi,z) = (0, r^2(1-r)^2\cos(2\pi z), 0)

in cylindrical coordinates :math:`(r, \chi, z)`.

**Source Term**

The corresponding source term is:

.. math::

    \mathbf{f}(r,\chi,z) = \left(0, \left[4\pi^2 r^2(1-r)^2 - (3-16r+15r^2)\right]\cos(2\pi z), 0\right)

The script demonstrates:

- Solving vector Poisson equations
- Handling vector fields in cylindrical coordinates
- Error analysis and convergence studies

Usage:

.. code-block:: bash

    python scripts/interactive/cylinder_vector_poisson.py

The script generates plots showing solution fields and error convergence.

**Finite Element Spaces**

The vector field is represented as a 1-form:

.. math::

    V_1 = \text{span}\{\Lambda_1^i\}_{i=1}^{N_1}

where :math:`N_1` is the number of 1-form DOFs.

**Matrix and Operator Dimensions**

The 1-form mass matrix :math:`M_1 \in \mathbb{R}^{N_1 \times N_1}` and 2-form mass matrix :math:`M_2 \in \mathbb{R}^{N_2 \times N_2}` are used.

The vector Laplacian is discretized as:

.. math::

    \Delta_1 = M_1^{-1} ((\nabla \times)_h^T M_2 (\nabla \times)_h + \nabla_h \circ (\nabla \cdot)_h)

where the curl-curl term represents :math:`\nabla \times (\nabla \times)`.

The discrete vector Poisson equation becomes:

.. math::

    M_1 \Delta_1 \hat{\mathbf{u}} = P_1(\mathbf{f})

where :math:`\hat{\mathbf{u}} \in \mathbb{R}^{N_1}` are the solution coefficients and :math:`P_1(\mathbf{f}) \in \mathbb{R}^{N_1}` is the projection of the source term.

Code Walkthrough
----------------

**Block 1: Imports and Setup (lines 1-17)**

Imports modules and sets up output directory. The script uses ``cylinder_map``
for the domain geometry.

.. literalinclude:: ../../scripts/interactive/cylinder_vector_poisson.py
   :language: python
   :lines: 1-17

**Block 2: Error Computation Function (lines 19-122)**

The ``get_err()`` function is JIT-compiled and computes relative L2 error.

Exact solution (only azimuthal component):

.. math::

    \mathbf{u}(r,\chi,z) = (0, r^2(1-r)^2\cos(2\pi z), 0)

Source term:

.. math::

    \mathbf{f}(r,\chi,z) = \left(0, \left[4\pi^2 r^2(1-r)^2 - (3-16r+15r^2)\right]\cos(2\pi z), 0\right)

Sets up DeRham sequence with clamped/periodic/periodic boundary conditions,
assembles mass matrices :math:`M_1` (1-forms) and :math:`M_2` (2-forms),
constructs curl operator :math:`C = (\nabla \times)_h` (strong curl),
and builds double divergence operator:

.. math::

    \text{divdiv} = M_2 (\Delta_2 - (\nabla \times)_h^T M_1 (\nabla \times)_h)

Solves block system for vector field components and computes relative L2 error using quadrature.

.. literalinclude:: ../../scripts/interactive/cylinder_vector_poisson.py
   :language: python
   :lines: 19-122

**Block 3: Convergence Analysis (lines 124-180)**

Runs convergence study over mesh sizes :math:`n \in [4,6,8,10,12]` and polynomial
degrees :math:`p \in [1,2,3]`:

- First run includes JIT compilation overhead
- Second run measures pure computation time
- Stores error and timing data

.. literalinclude:: ../../scripts/interactive/cylinder_vector_poisson.py
   :language: python
   :lines: 124-180

**Block 4: Plotting (lines 182-222)**

Generates convergence plots:

- Error vs. mesh size (log-log scale)
- Computation time vs. mesh size
- JIT compilation speedup factor

The vector Poisson problem requires solving a coupled system due to the curl-curl
structure of the vector Laplacian, which is more complex than the scalar case.

.. literalinclude:: ../../scripts/interactive/cylinder_vector_poisson.py
   :language: python
   :lines: 166-221
