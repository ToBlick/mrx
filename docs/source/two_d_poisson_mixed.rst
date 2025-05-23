2D Mixed Poisson Problem
========================

This tutorial walks through a script that solves a 2D Poisson problem using a mixed finite element formulation. The problem is defined on a square domain [0,1]^2 with Dirichlet boundary conditions. We analyze convergence, error, and performance.

Introduction
------------
We want to solve the Poisson equation

.. math::

    -\Delta u = f

on a square domain, using a mixed finite element method. The exact solution is:

.. math::

    u(x, y) = \sin(2\pi x) \sin(2\pi y)

with source term:

.. math::

    f(x, y) = 2(2\pi)^2 u(x, y)

Setup and Imports
-----------------
We start by importing the necessary libraries and setting up the output directory:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    import os
    from mrx.DifferentialForms import DifferentialForm, DiscreteFunction
    from mrx.Quadrature import QuadratureRule
    from mrx.Projectors import Projector
    from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix
    from mrx.Utils import l2_product
    from functools import partial

    jax.config.update("jax_enable_x64", True)
    os.makedirs('script_outputs', exist_ok=True)

Defining the Mixed Formulation
------------------------------
The mixed formulation uses:
- Λ2: 2-forms for the flux
- Λ3: 3-forms for the potential
- Λ0: 0-forms for quadrature

We define a function to compute the error for a given mesh size and polynomial degree:

.. code-block:: python

    @partial(jax.jit, static_argnames=['n', 'p'])
    def get_err(n, p):
        ns = (n, n, 1)
        ps = (p, p, 0)
        types = ('clamped', 'clamped', 'constant')
        def u(x):
            r, χ, z = x
            return jnp.ones(1) * jnp.sin(2 * jnp.pi * r) * jnp.sin(2 * jnp.pi * χ)
        def f(x):
            return 2 * (2*jnp.pi)**2 * u(x)
        Λ0 = DifferentialForm(0, ns, ps, types)
        Λ2 = DifferentialForm(2, ns, ps, types)
        Λ3 = DifferentialForm(3, ns, ps, types)
        Q = QuadratureRule(Λ0, 3)
        D = LazyDerivativeMatrix(Λ2, Λ3, Q).M
        M2 = LazyMassMatrix(Λ2, Q).M
        reg = 1e-10 * jnp.eye(M2.shape[0])
        M2_reg = M2 + reg
        K = D @ jnp.linalg.solve(M2_reg, D.T)
        P3 = Projector(Λ3, Q)
        u_hat = jnp.linalg.solve(K, P3(f))
        u_h = DiscreteFunction(u_hat, Λ3)
        def err(x): return u(x) - u_h(x)
        return (l2_product(err, err, Q) / l2_product(u, u, Q))**0.5

Convergence Analysis
--------------------
We run the error computation for a range of mesh sizes and polynomial degrees, and also measure computation time:

.. code-block:: python

    def run_convergence_analysis():
        ns = np.arange(7, 21, 2)
        ps = np.arange(1, 4)
        err = np.zeros((len(ns), len(ps)))
        times = np.zeros((len(ns), len(ps)))
        print("First run (with JIT compilation):")
        for i, n in enumerate(ns):
            for j, p in enumerate(ps):
                start = time.time()
                err[i, j] = get_err(n, p)
                end = time.time()
                times[i, j] = end - start
                print(f"n={n}, p={p}, err={err[i, j]:.2e}, time={times[i, j]:.2f}s")
        print("\nSecond run (after JIT compilation):")
        times2 = np.zeros((len(ns), len(ps)))
        for i, n in enumerate(ns):
            for j, p in enumerate(ps):
                start = time.time()
                _ = get_err(n, p)
                end = time.time()
                times2[i, j] = end - start
                print(f"n={n}, p={p}, time={times2[i, j]:.2f}s")
        return err, times, times2

Plotting Results
----------------
The script generates several plots to visualize the convergence of errors and computational time. For example, to plot the error convergence:

.. code-block:: python

    fig1 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        plt.loglog(ns, err[:, j], label=f'p={p}', marker='o')
    plt.loglog(ns, err[-1, 0] * (ns/ns[-1])**(-1), label='O(n^-1)', linestyle='--')
    plt.loglog(ns, err[-1, 1] * (ns/ns[-1])**(-2), label='O(n^-2)', linestyle='--')
    plt.loglog(ns, err[-1, 2] * (ns/ns[-1])**(-4), label='O(n^-4)', linestyle='--')
    plt.xlabel('Number of elements (n)')
    plt.ylabel('Relative L2 error')
    plt.title('Error Convergence')
    plt.grid(True)
    plt.legend()
    plt.savefig('script_outputs/2d_poisson_mixed_error.png', dpi=300, bbox_inches='tight')

Similar code is used to plot computation time and JIT speedup.

Main Function
-------------
The main function runs the convergence analysis and plotting:

.. code-block:: python

    def main():
        err, times, times2 = run_convergence_analysis()
        ns = np.arange(7, 21, 2)
        ps = np.arange(1, 4)
        plot_results(err, times, times2, ns, ps)
        plt.show()
        plt.close('all')

Conclusion
----------
This script demonstrates how to use a mixed finite element formulation to solve the 2D Poisson problem, analyze convergence, and measure performance. By running the code, you can generate plots that show how the error decreases with mesh refinement and polynomial order, and how JIT compilation speeds up repeated runs.

**Key features:**
- Solution of the 2D Poisson equation using mixed formulation
- Convergence analysis with respect to number of elements and polynomial degree
- JIT compilation speedup comparison
- Error and timing analysis

**Exact solution:**
.. math::

    u(x, y) = \sin(2\pi x) \sin(2\pi y)

with source term:

.. math::

    f(x, y) = 2(2\pi)^2 u(x, y)

**How to run:**
.. code-block:: bash

    python scripts/two_d_poisson_mixed.py

**Main steps:**
- Set up finite element spaces and operators
- Solve the mixed Poisson system
- Run convergence and timing analysis
- Plot and save results in `script_outputs/` 