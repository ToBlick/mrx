Two-Dimensional Helicity Analysis
=================================

This tutorial walks through a script that analyzes the convergence properties of magnetic helicity calculations in two dimensions using finite element methods. The script demonstrates error analysis for vector potential reconstruction, helicity conservation, convergence rates for different polynomial orders, and computational performance scaling.

Introduction
------------
We want to study how well we can compute magnetic helicity and related quantities using finite element methods. The script uses JAX for fast computation and automatic differentiation, and saves several plots to the `script_outputs/` directory.

Setup and Imports
-----------------
We start by importing the necessary libraries and setting up the output directory:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    from pathlib import Path
    from mrx.DifferentialForms import DifferentialForm
    from mrx.Quadrature import QuadratureRule
    from mrx.Projectors import Projector
    from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix, LazyProjectionMatrix, LazyDoubleCurlMatrix
    from mrx.Utils import curl
    from functools import partial

    jax.config.update("jax_enable_x64", True)
    output_dir = Path("script_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

Defining the Analytical Vector Potential
----------------------------------------
We define the analytical vector potential function A(x) that we want to reconstruct and analyze:

.. code-block:: python

    m1 = 2
    m2 = 2

    @jax.jit
    def A(x):
        r, χ, z = x
        a1 = jnp.sin(m1 * jnp.pi * r) * jnp.cos(m2 * jnp.pi * χ) * jnp.sqrt(m2**2/(m2**2 + m1**2))
        a2 = -jnp.cos(m1 * jnp.pi * r) * jnp.sin(m2 * jnp.pi * χ) * jnp.sqrt(m1**2/(m2**2 + m1**2))
        a3 = jnp.sin(m1 * jnp.pi * r) * jnp.sin(m2 * jnp.pi * χ)
        return jnp.array([a1, a2, a3])

Error Computation Function
--------------------------
The function `get_error(n, p)` sets up the finite element spaces, assembles the necessary matrices, and computes the errors in vector potential reconstruction, helicity, and curl for a given mesh size `n` and polynomial degree `p`:

.. code-block:: python

    @partial(jax.jit, static_argnames=['n', 'p'])
    def get_error(n, p):
        types = ('periodic', 'periodic', 'constant')
        ns = (n, n, 1)
        ps = (p, p, 0)
        Λ0 = DifferentialForm(0, ns, ps, types)
        Λ1 = DifferentialForm(1, ns, ps, types)
        Λ2 = DifferentialForm(2, ns, ps, types)
        Q = QuadratureRule(Λ0, 10)
        M2 = LazyMassMatrix(Λ2, Q).M
        M1 = LazyMassMatrix(Λ1, Q).M
        C = LazyDoubleCurlMatrix(Λ1, Q).M
        D = LazyDerivativeMatrix(Λ1, Λ2, Q).M
        P2 = Projector(Λ2, Q)
        P1 = Projector(Λ1, Q)
        M12 = LazyProjectionMatrix(Λ1, Λ2, Q).M.T
        B = curl(A)
        A_hat = jnp.linalg.solve(M1, P1(A))
        B_hat = jnp.linalg.solve(M2, P2(B))
        U, S, Vh = jnp.linalg.svd(C)
        S_inv = jnp.where(S/S[0] > 1e-12, 1/S, 0)
        A_hat_recon = Vh.T @ jnp.diag(S_inv) @ U.T @ D.T @ B_hat
        A_err = ((A_hat - A_hat_recon) @ M1 @ (A_hat - A_hat_recon) / (A_hat @ M1 @ A_hat))**0.5
        H_err = (A_hat - A_hat_recon) @ M12 @ B_hat / (A_hat @ M12 @ B_hat)
        curl_A_err = (jnp.linalg.solve(M2, D @ (A_hat - A_hat_recon)) @ M2 @ jnp.linalg.solve(M2, D @ (A_hat - A_hat_recon)) / (jnp.linalg.solve(M2, D @ A_hat) @ M2 @ jnp.linalg.solve(M2, D @ A_hat)))**0.5
        return A_err, H_err, curl_A_err

Convergence Analysis Loop
-------------------------
We run the error computation for a range of mesh sizes and polynomial degrees, storing the results for plotting:

.. code-block:: python

    ns = np.arange(4, 15, 2)
    ps = np.arange(1, 4)
    A_err = np.zeros((len(ns), len(ps)))
    H_err = np.zeros((len(ns), len(ps)))
    curl_A_err = np.zeros((len(ns), len(ps)))
    times = np.zeros((len(ns), len(ps)))
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            start = time.time()
            _A_err, _H_err, _curl_A_err = get_error(n, p)
            A_err[i, j] = _A_err
            H_err[i, j] = _H_err
            curl_A_err[i, j] = _curl_A_err
            end = time.time()
            times[i, j] = end - start
            print(f"n={n}, p={p}, A_err={A_err[i, j]:.2e}, H_err={H_err[i, j]:.2e}, curl_A_err={curl_A_err[i, j]:.2e}, time={times[i, j]:.2f}s")

Plotting Results
----------------
The script generates several plots to visualize the convergence of errors and computational time. For example, to plot the vector potential error convergence:

.. code-block:: python

    plt.figure(figsize=(10, 6))
    plt.plot(ns, A_err[:, 0], label='p=1', marker='o')
    plt.plot(ns, A_err[:, 1], label='p=2', marker='*')
    plt.plot(ns, A_err[:, 2], label='p=3', marker='s')
    plt.plot(ns, A_err[-1, 0] * (ns/ns[-1])**(-2), label='O(n^-2)', linestyle='--')
    plt.plot(ns, A_err[-1, 1] * (ns/ns[-1])**(-4), label='O(n^-4)', linestyle='--')
    plt.plot(ns, A_err[-1, 2] * (ns/ns[-1])**(-6), label='O(n^-6)', linestyle='--')
    plt.loglog()
    plt.xlabel('Number of Elements (n)')
    plt.ylabel('Relative Error in A')
    plt.title('Vector Potential Error Convergence')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / 'vector_potential_error.png', dpi=300, bbox_inches='tight')

Similar code is used to plot the curl error, helicity error, and computational time scaling.

Singular Value Spectrum Analysis
--------------------------------
The script also analyzes the singular value spectrum of the double curl operator, which is important for understanding the conditioning of the problem:

.. code-block:: python

    # Compute double curl operator and its SVD
    C = D.T @ jnp.linalg.solve(M2, D)
    U, S, Vh = jnp.linalg.svd(C)
    plt.figure(figsize=(10, 6))
    plt.plot(S/S[0], marker='o')
    plt.yscale('log')
    plt.xlabel('Index')
    plt.ylabel('Normalized Singular Value')
    plt.title('Singular Value Spectrum')
    plt.grid(True)
    plt.savefig(output_dir / 'singular_values.png', dpi=300, bbox_inches='tight')

Conclusion
----------
This script demonstrates how to use finite element methods to analyze the accuracy and convergence of magnetic helicity calculations in 2D. By running the code, you can generate plots that show how the error decreases with mesh refinement and polynomial order, and gain insight into the numerical properties of the double curl operator. 