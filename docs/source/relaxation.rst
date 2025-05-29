Magnetic Relaxation Simulation
==============================

This tutorial walks through a script that implements a magnetic relaxation simulation using finite element methods with JAX for automatic differentiation. The simulation models the evolution of a magnetic field under the influence of forces while preserving certain invariants like helicity.

Introduction
------------
We want to simulate the evolution of a magnetic field in a 2D domain, using finite element methods and JAX for fast computation. The simulation preserves magnetic helicity and analyzes energy minimization, force, and divergence.

Setup and Imports
-----------------
We start by importing the necessary libraries and setting up the output directory:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback
    from mrx.Quadrature import QuadratureRule
    from mrx.Projectors import Projector, CurlProjection
    from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix, LazyProjectionMatrix, LazyDoubleCurlMatrix, LazyStiffnessMatrix
    from mrx.Utils import curl

    output_dir = Path("script_outputs")
    output_dir.mkdir(exist_ok=True)
    jax.config.update("jax_enable_x64", True)

Defining Differential Forms and Operators
-----------------------------------------
We define the finite element spaces and assemble the necessary matrices and operators:

.. code-block:: python

    ns = (8, 8, 1)
    ps = (3, 3, 1)
    types = ('periodic', 'periodic', 'constant')
    Λ0 = DifferentialForm(0, ns, ps, types)
    Λ1 = DifferentialForm(1, ns, ps, types)
    Λ2 = DifferentialForm(2, ns, ps, types)
    Λ3 = DifferentialForm(3, ns, ps, types)
    Q = QuadratureRule(Λ0, 10)
    def F(x): return x
    M0, M1, M2, M3 = [LazyMassMatrix(Λ, Q).M for Λ in [Λ0, Λ1, Λ2, Λ3]]
    P0, P1, P2, P3 = [Projector(Λ, Q) for Λ in [Λ0, Λ1, Λ2, Λ3]]
    Pc = CurlProjection(Λ1, Q)
    D0, D1, D2 = [LazyDerivativeMatrix(Λk, Λkplus1, Q).M for Λk, Λkplus1 in zip([Λ0, Λ1, Λ2], [Λ1, Λ2, Λ3])]
    M12 = LazyProjectionMatrix(Λ1, Λ2, Q, F).M.T
    M03 = LazyProjectionMatrix(Λ0, Λ3, Q, F).M.T
    C = LazyDoubleCurlMatrix(Λ1, Q).M
    K = LazyStiffnessMatrix(Λ0, Q).M

Helper Functions
----------------
We define a helper function to compute the L2 inner product of two functions over the domain:

.. code-block:: python

    def l2_product(f, g, Q):
        return jnp.einsum("ij,ij,i->", jax.vmap(f)(Q.x), jax.vmap(g)(Q.x), Q.w)

Initial Field Setup and Visualization
-------------------------------------
We define the initial magnetic field configuration and visualize it:

.. code-block:: python

    def E(x, m, n):
        r, χ, z = x
        h = (1 + 0.0 * jnp.exp(-((r - 0.5)**2 + (χ - 0.5)**2) / 0.3**2))
        a1 = jnp.sin(m * jnp.pi * r) * jnp.cos(n * jnp.pi * χ) * jnp.sqrt(n**2/(n**2 + m**2))
        a2 = -jnp.cos(m * jnp.pi * r) * jnp.sin(n * jnp.pi * χ) * jnp.sqrt(m**2/(n**2 + m**2))
        a3 = jnp.sin(m * jnp.pi * r) * jnp.sin(n * jnp.pi * χ)
        return jnp.array([a1, a2, a3]) * h
    def A(x):
        return E(x, 2, 2)

    # Visualization code omitted for brevity

Field Projection and Error Analysis
-----------------------------------
We project the initial field onto the finite element space and compute the error:

.. code-block:: python

    A_hat = jnp.linalg.solve(M1, P1(A))
    A_h = DiscreteFunction(A_hat, Λ1)
    def compute_A_error(x): return A(x) - A_h(x)
    (l2_product(compute_A_error, compute_A_error, Q) / l2_product(A, A, Q))**0.5

Magnetic Field and Error Analysis
---------------------------------
We compute the magnetic field from the vector potential and analyze errors:

.. code-block:: python

    B0 = curl(A)
    B0_hat = jnp.linalg.solve(M2, P2(B0))
    B_h = DiscreteFunction(B0_hat, Λ2)
    def compute_B_error(x): return B0(x) - B_h(x)
    (l2_product(compute_B_error, compute_B_error, Q) / l2_product(B0, B0, Q))**0.5

Magnetic Field Visualization
----------------------------
We visualize the initial magnetic field configuration and its evolution (see script for plotting code).

Conclusion
----------
This script demonstrates how to use finite element methods and JAX to simulate magnetic field relaxation, analyze helicity and energy conservation, and visualize the results. By running the code, you can generate plots that show the evolution of the field and key physical quantities. 