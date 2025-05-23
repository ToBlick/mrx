Beltrami Field Analysis
=======================

This tutorial walks through a script that analyzes Beltrami fields in a 3D domain with homogeneous Dirichlet boundary conditions. The script computes magnetic helicity, performs error analysis for different modes, and visualizes results.

Introduction
------------
This script demonstrates the analysis of Beltrami fields using finite element methods. It computes the magnetic helicity for different mode numbers, analyzes convergence, and visualizes field components.

Setup and Imports
-----------------
We start by importing the necessary libraries and setting up the output directory:

.. code-block:: python

    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import os
    from typing import List, Tuple
    from mrx.DifferentialForms import DifferentialForm
    from mrx.Quadrature import QuadratureRule

    os.makedirs('script_outputs', exist_ok=True)

Defining the Beltrami Field
---------------------------
We define the eigenvalue, field components, and weight function:

.. code-block:: python

    def mu(m, n):
        return jnp.pi * jnp.sqrt(m**2 + n**2)

    def u(A_0, x, m, n):
        x_1, x_2, x_3 = x
        return jnp.array([
            ((A_0 * n) / (jnp.sqrt(m**2 + n**2))) * jnp.sin(jnp.pi * m * x_1) * jnp.cos(jnp.pi * n * x_2),
            ((A_0 * m * -1) / (jnp.sqrt(m**2 + n**2))) * jnp.cos(jnp.pi * m * x_1) * jnp.sin(jnp.pi * n * x_2),
            jnp.sin(jnp.pi * m * x_1) * jnp.sin(jnp.pi * n * x_2)
        ])

    def eta(x):
        x_1, x_2, x_3 = x
        return ((x_1**2) * ((1-x_1))**2) * ((x_2**2) * ((1-x_2))**2) * ((x_3**2) * ((1-x_3))**2)

    def integrand(m, n, x, A_0):
        field = u(A_0, x, m, n)
        mu_val = mu(m, n)
        return eta(x) * jnp.dot(field, field) * mu_val

    def compute_helicity(m, n, A_0, Q):
        integrand_values = jnp.array([integrand(m, n, x, A_0) for x in Q.x])
        integral = jnp.sum(integrand_values * Q.w)
        return integral * mu(m, n)

Field Visualization
-------------------
We visualize the field components for a given mode:

.. code-block:: python

    def plot_field_components(m, n, A_0, nx=100):
        # ... see script for details ...
        plt.savefig('script_outputs/beltrami_components_m{m}_n{n}.png')

Convergence Analysis
--------------------
We analyze the convergence of magnetic helicity for different modes:

.. code-block:: python

    def analyze_convergence(m_range, n_range, A_0, Q):
        # ... see script for details ...
        plt.savefig('script_outputs/beltrami_helicity.png')

Main Function
-------------
The main function runs the analysis and plotting:

.. code-block:: python

    def main():
        n = 5
        p = 3
        ns = (n, n, n)
        ps = (p, p, p)
        types = ('clamped', 'clamped', 'constant')
        Λ0 = DifferentialForm(0, ns, ps, types)
        Q = QuadratureRule(Λ0, 15)
        A_0 = 1.0
        m_range = [1, 2, 3]
        n_range = [1, 2, 3]
        plot_field_components(m_range[0], n_range[0], A_0)
        analyze_convergence(m_range, n_range, A_0, Q)
        for m in m_range:
            for n in n_range:
                H = compute_helicity(m, n, A_0, Q)
                print(f"Mode ({m},{n}): H = {float(H):.6f}")

Conclusion
----------
This script demonstrates how to analyze Beltrami fields, compute magnetic helicity, and visualize results using finite element methods. By running the code, you can generate plots for field components and helicity for different modes.

**How to run:**
.. code-block:: bash

    python scripts/Beltrami.py 