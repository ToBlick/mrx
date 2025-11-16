Drum Shape Optimization
========================

.. note::
   For general information about finite element discretization, basis functions, mesh parameters,
   polynomial degrees, boundary conditions, and matrix/operator dimensions, see :doc:`overview`.

This script demonstrates shape optimization for drum-like configurations.
The script is located at ``scripts/interactive/drumshape.py``.

**Problem Statement**

This script implements an inverse problem: given a target eigenvalue spectrum, find
the drum shape that produces those eigenvalues ("hearing the shape of a drum").

The forward problem is to solve the Laplacian eigenvalue problem:

.. math::

    -\Delta u = \lambda u \quad \text{in } \Omega

with homogeneous Dirichlet boundary conditions:

.. math::

    u|_{\partial\Omega} = 0

where:
- :math:`u: \Omega \to \mathbb{R}` is the eigenfunction (0-form)
- :math:`\lambda` is the eigenvalue
- :math:`\Delta = \nabla \cdot \nabla` is the scalar Laplacian operator
- :math:`\Omega` is the drum domain (shape to be optimized)
- :math:`\partial\Omega` denotes the boundary of the domain

The inverse problem is: given eigenvalues :math:`\{\lambda_i^{\mathrm{target}}\}`, find
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

    \min_{\hat{r}} L(\hat{r}) = \sum_{i=1}^{N} (\lambda_i(\hat{r}) - \lambda_i^{\mathrm{target}})^2

where:
- :math:`\lambda_i(\hat{r})` are the computed eigenvalues for shape :math:`\hat{r}`
- :math:`\lambda_i^{\mathrm{target}}` are the target eigenvalues
- :math:`N` is the number of eigenvalues to match

Usage:

.. code-block:: bash

    python scripts/interactive/drumshape.py

The script generates plots showing the optimization progress and final shape.

**Generalized Eigenvalue Problem**

The Laplacian eigenvalue problem is discretized as:

.. math::

    K \mathbf{v} = \lambda M \mathbf{v}

where:
- :math:`K \in \mathbb{R}^{N_0 \times N_0}` is the stiffness matrix (Laplacian)
- :math:`M \in \mathbb{R}^{N_0 \times N_0}` is the mass matrix
- :math:`\mathbf{v} \in \mathbb{R}^{N_0}` is the eigenvector
- :math:`\lambda` is the eigenvalue

Code Walkthrough
----------------

**Block 1: Imports and Setup (lines 1-30)**

Imports JAX, Optax (for optimization), and MRX modules. Sets up output directory.
Uses ``drumshape_map`` which defines a 2D domain with variable radius ``r(χ)``.

.. literalinclude:: ../../scripts/interactive/drumshape.py
   :language: python
   :lines: 1-30

**Block 2: Generalized Eigenvalue Solver (lines 34-59)**

Defines ``generalized_eigh()`` function:

- Solves generalized eigenvalue problem:

.. math::

    A \mathbf{v} = \lambda B \mathbf{v}

using Cholesky decomposition: :math:`B = LL^T`, then transforms to standard form:

.. math::

    C = L^{-1} A L^{-T}, \quad C \mathbf{v}' = \lambda \mathbf{v}'`

- Returns eigenvalues and eigenvectors in original basis

.. literalinclude:: ../../scripts/interactive/drumshape.py
   :language: python
   :lines: 34-59

**Block 3: Eigenvalue Computation (lines 64-132)**

The ``get_evs()`` function computes eigenvalues for a given shape:

- Takes shape parameters ``a_hat`` (discrete radius function)
- Constructs ``drumshape_map`` from radius function
- Manually assembles mass and stiffness matrices (to enable JAX transformations)
- Solves generalized eigenvalue problem for Laplacian eigenmodes
- Returns eigenvalues and eigenvectors

.. literalinclude:: ../../scripts/interactive/drumshape.py
   :language: python
   :lines: 64-132

**Block 4: Target Shape Setup (lines 137-199)**

Sets up target elliptical shape:

- Defines elliptical radius function
- Projects target shape into discrete representation
- Computes target eigenvalue spectrum

.. literalinclude:: ../../scripts/interactive/drumshape.py
   :language: python
   :lines: 137-199

**Block 5: Plotting Function (lines 204-406)**

Generates multi-panel visualization:

- Radius function comparison (target vs. fitted)
- First eigenfunction contour plot
- Eigenvalue spectrum comparison
- Relative eigenvalue error
- Loss history over iterations

.. literalinclude:: ../../scripts/interactive/drumshape.py
   :language: python
   :lines: 204-406

**Block 6: Main Optimization Loop (lines 411-545)**

Sets up and runs optimization:

- Defines loss function:

.. math::

    L = \sum_k \left(\frac{\lambda_k - \lambda_k^{\mathrm{target}}}{\lambda_k^{\mathrm{target}}}\right)^2

- Uses Optax Adam optimizer
- JIT-compiles loss and gradient computation
- Runs iterative optimization with periodic plotting
- Tracks convergence

This inverse problem demonstrates how shape optimization can be used to design
structures with desired spectral properties, which has applications in acoustics,
electromagnetics, and structural engineering.

.. literalinclude:: ../../scripts/interactive/drumshape.py
   :language: python
   :lines: 411-545
