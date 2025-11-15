Drum Shape Optimization
========================

This script demonstrates shape optimization for drum-like configurations.
The script is located at ``scripts/interactive/drumshape.py``.

Mathematical Problem
====================

This script implements an inverse problem: given a target eigenvalue spectrum, find
the drum shape that produces those eigenvalues ("hearing the shape of a drum").

The forward problem is to solve the Laplacian eigenvalue problem:

.. math::

    -\Delta u = \lambda u \quad \text{in } \Omega

with homogeneous Dirichlet boundary conditions :math:`u|_{\partial\Omega} = 0`.

The inverse problem is: given eigenvalues :math:`\{\lambda_i^{\text{target}}\}`, find
the domain :math:`\Omega` such that the eigenvalues :math:`\{\lambda_i\}` of the Laplacian
match the target eigenvalues.

**Drum Shape Parameterization**

The drum shape is parameterized by a radius function :math:`r(\chi)` in polar coordinates:

.. math::

    F(\rho, \chi) = \begin{bmatrix}
        \rho r(\chi) \cos(2\pi\chi) \\
        \rho r(\chi) \sin(2\pi\chi)
    \end{bmatrix}

where:
- :math:`\rho \in [0,1]` is the radial coordinate
- :math:`\chi \in [0,1]` is the angular coordinate
- :math:`r(\chi)` is the radius function (discretized as :math:`\hat{r} \in \mathbb{R}^{n_\chi}`)

**Optimization Problem**

The optimization problem is:

.. math::

    \min_{\hat{r}} L(\hat{r}) = \sum_{i=1}^{N} (\lambda_i(\hat{r}) - \lambda_i^{\text{target}})^2

where:
- :math:`\lambda_i(\hat{r})` are the computed eigenvalues for shape :math:`\hat{r}`
- :math:`\lambda_i^{\text{target}}` are the target eigenvalues
- :math:`N` is the number of eigenvalues to match

The script demonstrates:

- Shape optimization using finite element methods
- Target shape reconstruction
- Iterative optimization algorithms

Usage:

.. code-block:: bash

    python scripts/interactive/drumshape.py

The script generates plots showing the optimization progress and final shape.

Mathematical Formulation
=========================

**Generalized Eigenvalue Problem**

The Laplacian eigenvalue problem is discretized as:

.. math::

    K \mathbf{v} = \lambda M \mathbf{v}

where:
- :math:`K \in \mathbb{R}^{N_0 \times N_0}` is the stiffness matrix (Laplacian)
- :math:`M \in \mathbb{R}^{N_0 \times N_0}` is the mass matrix
- :math:`\mathbf{v} \in \mathbb{R}^{N_0}` is the eigenvector
- :math:`\lambda` is the eigenvalue

**Stiffness Matrix**

The stiffness matrix is:

.. math::

    K_{ij} = \int_\Omega \nabla \Lambda_0^i(x) \cdot G^{-1}(x) \nabla \Lambda_0^j(x) \det(DF(x)) \, dx

where:
- :math:`\{\Lambda_0^i\}_{i=1}^{N_0}` are 0-form basis functions
- :math:`G(x) = DF(x)^T DF(x)` is the metric tensor
- :math:`F` is the mapping from logical to physical coordinates

**Mass Matrix**

The mass matrix is:

.. math::

    M_{ij} = \int_\Omega \Lambda_0^i(x) \Lambda_0^j(x) \det(DF(x)) \, dx

**Eigenvalue Solver**

The generalized eigenvalue problem is solved using Cholesky decomposition:

.. math::

    M = LL^T

Then transform to standard form:

.. math::

    C = L^{-1} K L^{-T}, \quad C \mathbf{v}' = \lambda \mathbf{v}'

where :math:`\mathbf{v}' = L^T \mathbf{v}`.

**Loss Function**

The optimization loss function is:

.. math::

    L(\hat{r}) = \sum_{i=1}^{N} w_i (\lambda_i(\hat{r}) - \lambda_i^{\text{target}})^2

where :math:`w_i` are optional weights (typically :math:`w_i = 1`).

**Gradient Computation**

The gradient is computed using JAX automatic differentiation:

.. math::

    \nabla_{\hat{r}} L = \sum_{i=1}^{N} 2(\lambda_i(\hat{r}) - \lambda_i^{\text{target}}) \frac{\partial \lambda_i}{\partial \hat{r}}

The derivative :math:`\partial \lambda_i / \partial \hat{r}` is computed using the chain rule
through the eigenvalue solver.

**Optimization Algorithm**

The optimization uses gradient descent (e.g., Adam optimizer):

.. math::

    \hat{r}^{(k+1)} = \hat{r}^{(k)} - \alpha \nabla_{\hat{r}} L(\hat{r}^{(k)})

where :math:`\alpha` is the learning rate (adaptive in Adam).

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
