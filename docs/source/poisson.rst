Two-dimensional Poisson Problem
===============================

We begin by setting the number and degree of basis functions in the three spatial dimensions. All code is written in 3D, hence solving a 2D problem is done by setting one of the basis functions constant:

.. code-block:: python

    ns = (n, n, 1)
    ps = (p, p, 0)

Next, we define the source function $f(r, \chi, z) = 2 (2 \pi)^2 \sin(2 \pi r) \sin(2 \pi \chi)$ and the solution $u$ where $-\Delta u = f$.

.. code-block:: python

    def u(x):
        r, χ, z = x
        return jnp.ones(1) * jnp.sin(2 * jnp.pi * r) * jnp.sin(2 * jnp.pi * χ)

    def f(x):
        return 2 * (2*jnp.pi)**2 * u(x)

Note that we use channel dimensions: A scalar function is represented as an array with a single element.

We will solve the Poisson problem using zero-forms. This allows us to apply Dirichlet boundary conditions.

.. code-block:: python

    types = ('clamped', 'clamped', 'constant')
    Λ0 = DifferentialForm(0, ns, ps, types)

The ``DifferentialForm`` object support indexing and evaluation - to evaluate the i-th basis function at a point ``x``, we would call ``Λ0[i](x)`` where the ``shape`` of ``x`` is ``(3,)``.

Next, we get the corresponding boundary operator. The way that boundary conditions are implemented is by multiplying the basis functions evaluation with a rectangular matrix that "removes" the boundary splines. 

In general, discrete functions with boundary conditions applied have a lower amount of degrees of freedom $m < n$. For example, in one spatial dimension $m = n - 2$. In general:

.. math::

    f_h(x) = \sum_{i=0}^{n-1} \mathtt{f}_i \Lambda_i(x) \quad \text{(no boundary conditions applied)} \\
    \mathring{f}_h(x) = \sum_{j=0}^{m-1} {\mathring{\mathtt{f}}}_j \mathring\Lambda_j(x) = \sum_{j=0}^{m-1} {\mathring{\mathtt{f}}_j} \sum_{i=0}^{n-1} \mathbb B_{ji} \Lambda_i(x).

In code, this is

.. code-block:: python

    bcs = ('dirichlet', 'dirichlet', 'none')
    B0 = LazyBoundaryOperator(Λ0, bcs).M

where ``.M`` already returns the assembled matrix.

With this, we can assemble the stiffness matrix $\mathring {\mathbb K}$. Its $i,j$-th element is

.. math::

    \mathring{\mathbb K}_{ij} = \int_{\hat \Omega} \hat \nabla \mathring\Lambda_i \cdot (DF)^{-1} (DF)^{-T} \hat \nabla \mathring\Lambda_j \, \det DF \, \mathrm d \hat x

In practice, it is assembled by computing $\mathbb K$ - the stiffness matrix with no boundary conditions applied - and then contracting it on both sides with $\mathbb B$ as $\mathring{\mathbb K} = \mathbb B \mathbb K \mathbb B^T$:

.. code-block:: python

    K = LazyStiffnessMatrix(Λ0, Q, F=None, E=B0).M

The choice ``F = None`` defaults to ``F = lambda x: x`` or $F(\hat x) = \hat x$.

To solve the Poisson problem itself, we follow the usual arguments.

.. math::

    \sum_{i=0}^{m-1} \mathring{\mathtt{u}}_i \int_{\hat \Omega} \hat \nabla \mathring\Lambda_i \cdot (DF)^{-1} (DF)^{-T} \hat \nabla \mathring\Lambda_j \, \det DF \, \mathrm d \hat x = \int_{\hat \Omega} \hat f \mathring\Lambda_j \, \det DF \, \mathrm d \hat x 

The function $\hat f$ is the pull-back of $f$ into the logical domain where $f$ is treated as a zero-form, i.e. $\hat f(\hat x) = f \circ F(\hat x)$. This right-hand-side is evaluated using a ``Projector`` object that corresponds to the operation 

.. math::

    \mathring\Pi_0: \hat f \mapsto \left( \int_{\hat \Omega} \hat f \mathring\Lambda_j \, \det DF \, \mathrm d \hat x \right)_{j = 0, \dots, m-1}

.. code-block:: python

    P0 = Projector(Λ0, Q, E=B0)

Internally, again, this is evaluated as $\mathring\Pi_0 = \mathbb B \Pi_0$. 

With all this in place, we can solve for $\mathring{\mathtt u}$ and create a ``DiscreteFunction`` object that supports evaluation as ``u_h(x)``:

.. code-block:: python

    u_hat = jnp.linalg.solve(K, P0(f))
    u_h = DiscreteFunction(u_hat, Λ0, B0) 