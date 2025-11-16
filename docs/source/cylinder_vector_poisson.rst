Cylinder Vector Poisson
========================

.. note::
   For general information about finite element discretization, basis functions, mesh parameters,
   polynomial degrees, boundary conditions, and matrix/operator dimensions, see :doc:`overview`.

This script solves vector Poisson problems on a cylindrical domain.
The script is located at ``scripts/interactive/cylinder_vector_poisson.py``.

Mathematical Problem
====================

This script solves a vector Poisson problem:

.. math::

    -\Delta \mathbf{u} = \mathbf{f}

for vector fields in cylindrical geometry, where:
- :math:`\mathbf{u}: \Omega \to \mathbb{R}^3` is the vector solution (1-form)
- :math:`\mathbf{f}: \Omega \to \mathbb{R}^3` is the vector source term (1-form)
- :math:`\Delta` is the vector Laplacian

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

Mathematical Formulation
=========================

**Finite Element Spaces**

The vector field is represented as a 1-form:

.. math::

    V_1 = \text{span}\{\Lambda_1^i\}_{i=1}^{N_1}

where :math:`N_1` is the number of 1-form DOFs.

**Mass Matrices**

The 1-form mass matrix :math:`M_1 \in \mathbb{R}^{N_1 \times N_1}`:

.. math::

    (M_1)_{ij} = \int_\Omega \Lambda_1^i(x) \cdot G^{-1}(x) \Lambda_1^j(x) \det(DF(x)) \, dx

Dimensions: :math:`N_1 \times N_1`.

The 2-form mass matrix :math:`M_2 \in \mathbb{R}^{N_2 \times N_2}`:

.. math::

    (M_2)_{ij} = \int_\Omega \Lambda_2^i(x) \cdot G(x) \Lambda_2^j(x) \frac{1}{\det(DF(x))} \, dx

Dimensions: :math:`N_2 \times N_2`.

**Derivative Operators**

The strong curl operator :math:`\text{strong\_curl}: V_1 \to V_2`:

.. math::

    (\text{strong\_curl})_{ij} = \int_\Omega \Lambda_2^i(x) \cdot G(x) \nabla \times \Lambda_1^j(x) \frac{1}{\det(DF(x))} \, dx

Dimensions: :math:`N_2 \times N_1`.

The weak divergence operator :math:`\text{weak\_div}: V_1 \to V_0`:

.. math::

    (\text{weak\_div})_{ij} = -\int_\Omega \Lambda_0^i(x) \nabla \cdot \Lambda_1^j(x) \det(DF(x)) \, dx

Dimensions: :math:`N_0 \times N_1`.

**Vector Laplacian Discretization**

The vector Laplacian is discretized as:

.. math::

    \Delta_1 = M_1^{-1} (\text{curl\_curl} + \text{strong\_grad} \circ \text{weak\_div})

where:

.. math::

    (\text{curl\_curl})_{ij} = \int_\Omega \nabla \times \Lambda_1^i(x) \cdot G(x) \nabla \times \Lambda_1^j(x) \frac{1}{\det(DF(x))} \, dx

**Block System**

The discrete vector Poisson equation becomes:

.. math::

    M_1 \Delta_1 \hat{\mathbf{u}} = P_1(\mathbf{f})

where:
- :math:`\hat{\mathbf{u}} \in \mathbb{R}^{N_1}` are the solution coefficients
- :math:`M_1 \in \mathbb{R}^{N_1 \times N_1}` is the 1-form mass matrix
- :math:`\Delta_1 \in \mathbb{R}^{N_1 \times N_1}` is the 1-form Laplacian
- :math:`P_1(\mathbf{f}) \in \mathbb{R}^{N_1}` is the projection of the source term

**Error Computation**

The relative L2 error is computed using quadrature:

.. math::

    \text{error} = \frac{\|\mathbf{u} - \mathbf{u}_h\|_{L^2(\Omega)}}{\|\mathbf{u}\|_{L^2(\Omega)}}

where:

.. math::

    \|\mathbf{u} - \mathbf{u}_h\|_{L^2(\Omega)}^2 = \int_\Omega |\mathbf{u}(x) - \mathbf{u}_h(x)|^2 \det(DF(x)) \, dx

and :math:`\mathbf{u}_h = \sum_{i=1}^{N_1} \hat{u}_i \Lambda_1^i` is the discrete solution.

Code Walkthrough
----------------

This script solves a vector Poisson problem:
   
   .. math::
   
       -\Delta \mathbf{u} = \mathbf{f}
   
   for vector fields in cylindrical geometry:

**Block 1: Imports and Setup (lines 1-17)**
   Imports modules and sets up output directory. The script uses ``cylinder_map``
   for the domain geometry.

**Block 2: Error Computation Function (lines 19-122)**
   The ``get_err()`` function is JIT-compiled and computes relative L2 error:
   
   - Exact solution (only azimuthal component):
   
     .. math::
     
         \mathbf{u}(r,\chi,z) = (0, r^2(1-r)^2\cos(2\pi z), 0)
     
   - Source term:
   
     .. math::
     
         \mathbf{f}(r,\chi,z) = \left(0, \left[4\pi^2 r^2(1-r)^2 - (3-16r+15r^2)\right]\cos(2\pi z), 0\right)
   
   - Sets up DeRham sequence with clamped/periodic/periodic boundary conditions
   - Assembles mass matrices :math:`M_1` (1-forms) and :math:`M_2` (2-forms)
   - Constructs curl operator :math:`C = \nabla \times` (strong curl)
   - Builds double divergence operator:
   
     .. math::
     
         \text{divdiv} = M_2 (\Delta_2 - (\nabla \times)(\nabla \times))
   - Solves block system for vector field components
   - Computes relative L2 error using quadrature

**Block 3: Convergence Analysis (lines 124-180)**
   Runs convergence study over mesh sizes ``n ∈ [4,6,8,10,12]`` and polynomial
   degrees ``p ∈ [1,2,3]``:
   
   - First run includes JIT compilation overhead
   - Second run measures pure computation time
   - Stores error and timing data

**Block 4: Plotting (lines 182-222)**
   Generates convergence plots:
   
   - Error vs. mesh size (log-log scale)
   - Computation time vs. mesh size
   - JIT compilation speedup factor

The vector Poisson problem requires solving a coupled system due to the curl-curl
structure of the vector Laplacian, which is more complex than the scalar case.

Full script:

.. literalinclude:: ../../scripts/interactive/cylinder_vector_poisson.py
   :language: python
   :linenos:
