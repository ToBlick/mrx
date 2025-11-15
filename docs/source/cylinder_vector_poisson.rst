Cylinder Vector Poisson
========================

This script solves vector Poisson problems on a cylindrical domain.
The script is located at ``scripts/interactive/cylinder_vector_poisson.py``.

The script demonstrates:

- Solving vector Poisson equations
- Handling vector fields in cylindrical coordinates
- Error analysis and convergence studies

Usage:

.. code-block:: bash

    python scripts/interactive/cylinder_vector_poisson.py

The script generates plots showing solution fields and error convergence.

Code Walkthrough
================

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
