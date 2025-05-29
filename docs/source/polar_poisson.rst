Poisson Problem on a disc
=========================

For this problem, we consider the source-solution pair $-\Delta u = f$

.. math::

    u(r) = \frac 1 {27} \left( r^3 (3 \log r - 2) + 2 \right),\\
    f(r) = - r \log r

The initial setup is analogous to the case on a square domain. The $\chi$ variable is treated as periodic.

.. code-block:: python

    ns = (n, n, 1)
    ps = (p, p, 0)
    types = ('clamped', 'periodic', 'constant')
    Λ0 = DifferentialForm(0, ns, ps, types)
    Q = QuadratureRule(Λ0, q)

Now, we need to deal with the singularity at the axis. This is done by constructing a tensor $\xi$ with shape `(3, 2, nχ)` that is used to create new basis functions around the line $r = 0$, replacing the inner rings of cartesian splines.

We define the mapping (a cylinder in this example) and use it to calculate $\xi$.

.. code-block:: python

    a = 1
    R0 = 3.0
    Y0 = 0.0

    def _R(r, χ):
        return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * jnp.pi * χ))

    def _Y(r, χ):
        return jnp.ones(1) * (Y0 + a * r * jnp.sin(2 * jnp.pi * χ))

    def F(x):
        r, χ, z = x
        return jnp.ravel(jnp.array([_R(r, χ),
                                    _Y(r, χ),
                                    jnp.ones(1) * z]))

    ξ, R_hat, Y_hat, Λ, τ = get_xi(_R, _Y, Λ0, Q)

Since polar splines are linear combination of cartesian ones, they can be evaluated using an extraction matrix $\mathbb E$ that functions much the same as the boundary matrix we already know:

.. math::

    f_h(x) = \sum_{i=0}^{n-1} \mathtt{f}_i \Lambda_i(x) \quad \text{(cartesian splines)} \\
    = \sum_{j=0}^{m-1} {\mathtt{f}}^{\mathrm{pol}}_j \Lambda_j^{\mathrm{pol}}(x) = \sum_{j=0}^{m-1} {\mathtt{f}}_j^{\mathrm{pol}} \sum_{i=0}^{n-1} \mathbb E_{ji} \Lambda_i(x).

.. code-block:: python

    E0 = LazyExtractionOperator(Λ0, ξ, zero_bc=True).M

As we can see, the ``LazyExtractionOperator`` constructor also allows us to drop the outer "ring" of basis functions to implement dirichlet boundary conditions there.

The stiffness matrix and projector are built as before, with the difference that now, the non-trivial mapping ``F`` and the extraction operator ``E0`` need to be passed to them.

.. code-block:: python

    K = LazyStiffnessMatrix(Λ0, Q, F=F, E=E0).M
    P0 = Projector(Λ0, Q, F=F, E=E0)
    u_hat = jnp.linalg.solve(K, P0(f))
    u_h = DiscreteFunction(u_hat, Λ0, E0)

The discrete function ``u_h`` is defined on the logical domain, i.e. it represents $\hat u_h: \hat \Omega \to \mathbb R$. To get the solution in the physical domain, we need to apply a push-forward, $u_h(x) := \hat u_h \circ F^{-1}(x)$.

Lastly, note that the solution ``u`` is not smooth, we only have $u \in H^s(\Omega)$ for all $s < 4$. This limits the order of convergence we can expect to see. 