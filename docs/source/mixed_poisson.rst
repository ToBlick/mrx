Poisson Problem - mixed formulation
===================================

The Poisson equation can be re-written as

.. math::

    \nabla \cdot \sigma = f \\
    -\nabla u = \sigma

Note that only one of these can hold in a strong sense, since $\sigma$ cannot be both a two- and a one-form.

When the first equation is fulfilled in strong form and the second weakly, this leads to the system

.. math::

    (\nabla \cdot \sigma^2, \phi^3) = (f, \phi^3) \\
    (u^3, \nabla \cdot \psi^2) = (\sigma^2, \psi^2),

where we use subscripts to denote what forms the variables are interpreted as. This formulation automatically implies the boundary condition $u = 0$ on $\partial \Omega$.

In matrix-vector form, the equations read

.. math::

    \mathbb D \sigma^2 = \Pi_3 f \\
    \mathbb D^T u = \mathbb M_2 \sigma^2

where $\mathbb D$ denotes the divergence matrix. This can be re-written as

.. math::

    \begin{bmatrix}
        \mathbb M_2 & - \mathbb D^T \\
        \mathbb D & 0
    \end{bmatrix}
    \begin{bmatrix}
        \sigma \\ u
    \end{bmatrix}
    = 
    \begin{bmatrix}
        0 \\ \Pi_3 f
    \end{bmatrix}

and back-substituted to get

.. math::

    \mathbb D \mathbb M_2^{-1} \mathbb D^T u = \Pi_3 f.

Note that

.. math::

    (\Pi_3 f)_i = \int_{\Omega} f(x) \frac{\Lambda^3 \circ F^{-1} (x)}{\det DF \circ F^{-1}(x)} \, \mathrm d x \\
    = \int_{\hat \Omega} \left( f \circ F (\hat x) \right) \frac{\Lambda^3(\hat x)}{\det DF(\hat x)} \det DF(\hat x) \, \mathrm d \hat x.

To test this case, we need a solution to Poisson's equation on a disc that satisfies $\int_{\partial \Omega} \nabla u \cdot n \, \mathrm d \sigma = \int_\Omega f \mathrm d x = 0$:

.. math::

    u(r) = \frac 1 4 \left( \frac 1 3 r^3 - \frac 1 4 r^4 - \frac 1 {12} \right) \\
    f(r) = r \left( r - \frac 3 4 \right)

The constant in $u$ is chosen such that $u(1) = 0$.

In code, the mixed Poisson problem is solved as follows: We begin by defining the mapping

.. code-block:: python

    def _R(r, χ):
        return jnp.ones(1) * r * jnp.cos(2 * jnp.pi * χ)
    def _Y(r, χ):
        return jnp.ones(1) * r * jnp.sin(2 * jnp.pi * χ)
    def F(p):
        r, χ, z = p
        return jnp.squeeze(jnp.array([_R(r, χ), _Y(r, χ), jnp.ones(1) * z]))
    def F_inv(p):
        x, y, z = p 
        r = jnp.sqrt(x**2 + y**2)
        χ = jnp.arctan2(y, x)
        χ = jnp.where(χ < 0, χ + 2 * jnp.pi, χ) / (2 * jnp.pi)
        return jnp.array([r, χ, z])

Next, we assemble the required matrices and also compute $\mathbb K$:

.. code-block:: python

    ns = (n, p, 1)
    ps = (p, p, 0)
    types = ('clamped', 'periodic', 'constant')
    Λ0, Λ2, Λ3 = [DifferentialForm(i, ns, ps, types) for i in [0, 2, 3]]
    Q = QuadratureRule(Λ0, q)
    ξ, R_hat, Y_hat, Λ, τ = get_xi(_R, _Y, Λ0, Q)
    E0, E2, E3 = [LazyExtractionOperator(Λ, ξ, False).M for Λ in [Λ0, Λ2, Λ3]]
    D = LazyDerivativeMatrix(Λ2, Λ3, Q, F, E2, E3).M
    M2 = LazyMassMatrix(Λ2, Q, F, E2).M
    K = D @ jnp.linalg.solve(M2, D.T)

After assembly, we can project the source term and compute the degrees of freedom for ``u_h``.

.. code-block:: python

    def u(x):
        r, χ, z = x
        return -jnp.ones(1) * 1/4 * (1/4 * r**4 - 1/3 * r**3 + 1/12)
    def f(x):
        r, χ, z = x
        return jnp.ones(1) * (r - 3/4) * r
    P3 = Projector(Λ3, Q, F, E3)
    u_hat = jnp.linalg.solve(K, P3(f))

Note that ``u_hat`` are degrees of freedom for a discrete three-form, while the analytical solution is given in terms of a zero-form. To compare the two, we can project ``u_h`` to the space of zero-forms:

.. code-block:: python

    M03 = LazyProjectionMatrix(Λ0, Λ3, Q, F, E0, E3).M
    M0 = LazyMassMatrix(Λ0, Q, F, E0).M
    u_hat = jnp.linalg.solve(M0, M03.T @ u_hat)
    u_h = DiscreteFunction(u_hat, Λ0, E0) 