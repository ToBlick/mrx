Drum Shape Optimization
========================

This script demonstrates shape optimization for drum-like configurations.
The script is located at ``scripts/interactive/drumshape.py``.

The script demonstrates:

- Shape optimization using finite element methods
- Target shape reconstruction
- Iterative optimization algorithms

Usage:

.. code-block:: bash

    python scripts/interactive/drumshape.py

The script generates plots showing the optimization progress and final shape.

Code Walkthrough
================

This script implements an inverse problem: given a target eigenvalue spectrum, find
the drum shape that produces those eigenvalues ("hearing the shape of a drum"):

**Block 1: Imports and Setup (lines 1-30)**
   Imports JAX, Optax (for optimization), and MRX modules. Sets up output directory.
   Uses ``drumshape_map`` which defines a 2D domain with variable radius ``r(χ)``.

**Block 2: Generalized Eigenvalue Solver (lines 34-59)**
   Defines ``generalized_eigh()`` function:
   
   - Solves generalized eigenvalue problem:
   
     .. math::
     
         A \mathbf{v} = \lambda B \mathbf{v}
     
     using Cholesky decomposition: :math:`B = LL^T`, then transforms to standard form:
   
     .. math::
     
         C = L^{-1} A L^{-T}, \quad C \mathbf{v}' = \lambda \mathbf{v}'
   - Returns eigenvalues and eigenvectors in original basis

**Block 3: Eigenvalue Computation (lines 64-200)**
   The ``get_evs()`` function computes eigenvalues for a given shape:
   
   - Takes shape parameters ``a_hat`` (discrete radius function)
   - Constructs ``drumshape_map`` from radius function
   - Manually assembles mass and stiffness matrices (to enable JAX transformations)
   - Solves generalized eigenvalue problem for Laplacian eigenmodes
   - Returns eigenvalues and eigenvectors

**Block 4: Optimization Setup (lines 202-350)**
   Sets up optimization problem:
   
   - Defines target eigenvalues :math:`\{\lambda_i^{\text{target}}\}` (from a known shape)
   - Creates loss function:
   
     .. math::
     
         L = \sum_i (\lambda_i - \lambda_i^{\text{target}})^2
     
     where :math:`\{\lambda_i\}` are the computed eigenvalues for the current shape.
   - Uses Optax optimizer (e.g., Adam) to minimize loss
   - Implements gradient descent with JAX automatic differentiation

**Block 5: Optimization Loop (lines 352-450)**
   Runs iterative optimization:
   
   - Computes eigenvalues for current shape
   - Evaluates loss function
   - Computes gradients using ``jax.grad()``
   - Updates shape parameters using optimizer
   - Tracks convergence and saves intermediate results

**Block 6: Visualization (lines 452-547)**
   Generates plots showing:
   
   - Initial vs. target vs. optimized shape
   - Eigenvalue convergence during optimization
   - Final optimized drum shape

This inverse problem demonstrates how shape optimization can be used to design
structures with desired spectral properties, which has applications in acoustics,
electromagnetics, and structural engineering.

Full script:

.. literalinclude:: ../../scripts/interactive/drumshape.py
   :language: python
   :linenos:
