# MRX primer

This note explains the building blocks of `mrx`: the discrete function
spaces, how tensor-product structure is exploited during assembly, the
extraction operators that enforce boundary conditions and polar
geometries, the Hodge Laplacians, the iterative solvers, and the main
data structures and how they fit together.

All code paths are JIT-compilable and AD-friendly: dense / sparse tensors
are `jax.numpy` or `jax.experimental.sparse` arrays, the main container
types are `equinox.Module` pytrees, and linear solves are built on
callable matvecs so we can differentiate through them.

---

## 1. FEM: the finite element spaces we use

`MRX` discretises the de Rham sequence on the unit cube $[0,1]^3$ in
logical coordinates $\hat x = (r,\theta,\zeta)$:

$$
V^0 \xrightarrow{\mathrm{grad}}
V^1 \xrightarrow{\mathrm{curl}}
V^2 \xrightarrow{\mathrm{div}}
V^3.
$$

Each $V^k$ is built from tensor products of **1D B-spline bases** along
each logical coordinate, with per-direction type `"clamped"`,
`"periodic"`, `"constant"`, or `"fourier"` (see
[mrx/spline_bases.py](mrx/spline_bases.py), `fourier` is currently not supported).

Given 1D B-spline bases $\{\lambda^r_a\}$, $\{\lambda^\theta_b\}$,
$\{\lambda^\zeta_c\}$ of degrees $p_r, p_\theta, p_\zeta$, the 3D 0-form
basis functions are products

$$
\Lambda^0_{abc}(\hat x) = \lambda^r_a(r)\,\lambda^\theta_b(\theta)\,\lambda^\zeta_c(\zeta).
$$

For $k=1,2$ we take one derivative in $k$ of the three directions,
producing three vector components:

$$
\Lambda^{1,r}_{abc}(\hat x) = \partial_r \lambda^r_a(r)  \; \lambda^\theta_b(\theta) \;\lambda^\zeta_c(\zeta) \\
\Lambda^{1,\theta}_{abc}(\hat x) = \lambda^r_a(r)  \; \partial_\theta \lambda^\theta_b(\theta) \;\lambda^\zeta_c(\zeta) \\
\Lambda^{1,\zeta}_{abc}(\hat x) = \lambda^r_a(r)  \; \lambda^\theta_b(\theta) \;\partial_\zeta \lambda^\zeta_c(\zeta)
$$

In practice, these index sets are flattened into a single index $I$ that runs over all three components and all $(a,b,c)$. The same applies to $k=2$ with two derivatives and three components:

$$
\Lambda^{2,r}_{abc}(\hat x) = \lambda^r_a(r)  \; \partial_\theta \lambda^\theta_b(\theta) \;\partial_\zeta \lambda^\zeta_c(\zeta) \\
\Lambda^{2,\theta}_{abc}(\hat x) = \partial_r \lambda^r_a(r)  \; \lambda^\theta_b(\theta) \;\partial_\zeta \lambda^\zeta_c(\zeta) \\
\Lambda^{2,\zeta}_{abc}(\hat x) = \partial_r \lambda^r_a(r)  \; \partial_\theta \lambda^\theta_b(\theta) \;\lambda^\zeta_c(\zeta).
$$

Lastly, $k=3$ basis functions are again simple products of the 1D bases, but with all three factors differentiated:

$$
\Lambda^3_{abc}(\hat x) = \partial_r \lambda^r_a(r)  \; \partial_\theta \lambda^\theta_b(\theta) \;\partial_\zeta \lambda^\zeta_c(\zeta).
$$

These discrete spaces reproduce the continuous de Rham complex exactly. The gradient of any 0-form basis function is exactly represented in the 1-form space, the curl of a 1-form is exactly represented in the 2-form space, and the divergence of a 2-form is exactly represented in the 3-form space.

A discrete $k$-form

$$
u_h^k(\hat x) = \sum_I \texttt{u}_I\, \Lambda^k_I(\hat x)
$$

is stored as a coefficient vector `u` of length `basis_k.n` (with boundary conditions applied later via extraction — see §3). 
In [mrx/differential_forms.py](mrx/differential_forms.py) the `DifferentialForm` class owns the 1D bases and the multi-index bookkeeping; `DiscreteFunction` pairs a `DifferentialForm` with a
coefficient vector.

---

## 2. Tensor-product structure and sum factorization

A basis function on the 3D grid factors into
a product of three 1D functions, and this makes both **assembly** and
**geometry evaluation** dramatically cheaper than a black-box
element-by-element loop.

### 2.1 Sum factorization at quadrature

Every mass / stiffness matrix entry is of the form

$$
\mathbb{M}_{IJ}^k = \int_\Omega \Lambda^k_I(\hat x) \cdot W^k(\hat x) \Lambda^k_J(\hat x)\, d\hat x
$$

for some weight tensor $W$ (scalar for mass/pressure forms, $3\times 3$
for 1-forms and 2-forms). 

Concretely, $W$ is built from the coordinate mapping Jacobian $D\Phi$:
$$
\begin{align*}
\mathbb{M}^0_{IJ} &= \int_\Omega \Lambda^0_I(\hat x) \Lambda^0_J(\hat x) \det D\Phi(\hat x) \, d\hat x \\
\mathbb{M}^1_{IJ} &= \int_\Omega \Lambda^1_I(\hat x) \cdot (D\Phi)^{-1} (D\Phi)^{-T} \Lambda^1_J(\hat x) \det D\Phi(\hat x) \, d\hat x \\
\mathbb{M}^2_{IJ} &= \int_\Omega \Lambda^2_I(\hat x) \cdot (D\Phi)^{T} (D\Phi) \Lambda^2_J(\hat x) (\det D\Phi(\hat x))^{-1} \, d\hat x \\
\mathbb{M}^3_{IJ} &= \int_\Omega \Lambda^3_I(\hat x) \Lambda^3_J(\hat x) (\det D\Phi(\hat x))^{-1} \, d\hat x
\end{align*}
$$

The entries of stiffness/derivative matrices are similar but with one or two of the $\Lambda$ swapped for their derivatives. Importantly, since gradients of 0-forms transform like 1-forms, curls of 1-forms transform like 2-forms, and divergences of 2-forms transform like 3-forms, the same $W$ tensors appear.

The central metric quantities are hence the metric and inverse metric tensors $g = (D\Phi)^T D\Phi$ and $g^{-1} = (D\Phi)^{-1} (D\Phi)^{-T}$, and $\sqrt{\det g} = \det D\Phi$,  which are stored on the `SequenceGeometry` as `jacobian_j`, `metric_jkl`, and `metric_inv_jkl` respectively.

Using Gauss quadrature with points and weights $(\hat x_q, w_q)$, the integral becomes

$$
\mathbb{M}_{IJ}^k = \sum_q \Lambda^k_I(\hat x_q) \cdot W^k(\hat x_q) \Lambda^k_J(\hat x_q)\, w_q.
$$

At this point, we exploit the tensor-product structure: each $\Lambda^k_I$ factors into three 1D functions and the quadrature grid itself is a tensor product of 1D quadrature nodes in $r$, $\theta$, and $\zeta$. For example, for $k=0$ we have

$$
\mathbb{M}_{IJ}^0 = \sum_{q_r, q_\theta, q_\zeta} \lambda^r_{a}(r_{q_r}) \lambda^\theta_{b}(\theta_{q_\theta}) \lambda^\zeta_{c}(\zeta_{q_\zeta}) \lambda^r_{d}(r_{q_r}) \lambda^\theta_{e}(\theta_{q_\theta}) \lambda^\zeta_{f}(\zeta_{q_\zeta}) \det D\Phi(r_{q_r}, \theta_{q_\theta}, \zeta_{q_\zeta}) \, w_{q_r} w_{q_\theta} w_{q_\zeta},
$$
where $I = \text{flatten}(a,b,c), J = \text{flatten}(d,e,f)$.

We evaluate the 1D basis functions at the 1D quadrature nodes once, store them (this takes only $O(N^2)$ memory) and then contract them against $W$.

For vector-valued forms, the same idea applies but with the correct pattern of derivatives in the basis and the correct $W$ tensor (the basis functions are vector-valued).

The actual implementation in
[mrx/assembly.py](mrx/assembly.py) caches the 1D basis evaluations at
1D quadrature nodes once (on the sequence as `seq.basis_r_jk`,
`seq.basis_t_jk`, `seq.basis_z_jk`, and their derivative counterparts
`seq.d_basis_*_jk`) and then contracts them against $W$ using
`jnp.einsum` / `jax.experimental.sparse` with the correct sparsity
pattern (independent of the geometry, some basis functions never overlap). The dispatcher functions are:

- `assemble_scalar_tp(...)` — for scalar mass matrices ($k=0$ and
  $k=3$).
- `assemble_vectorial_tp(...)` — for vector mass matrices
  ($k=1, 2$), with a list of terms per component describing which
  directions use the derivative basis.
- `assemble_stiffness_scalar_tp(...)` — for the $k=0$ $\mathrm{grad}$-$\mathrm{grad}$ Laplacian.

The cost for an $N^3$ grid drops from the naive $O(N^6)$ to
$O(N^4)$ per matrix. All further sparse operations are
$O(\text{nnz})$.

### 2.2 Sum factorization of spline geometry

The same idea applies to the physical map $\Phi:\hat\Omega\to\Omega$ when
$\Phi$ is itself a tensor-product spline (a `SplineMap`). Each Cartesian
component factors as

$$
 \Phi_\alpha(\hat x) = \sum_{abc} c^{\alpha}_{abc}\,
\lambda^r_a(r)\,\lambda^\theta_b(\theta)\,\lambda^\zeta_c(\zeta), \quad \alpha \in \{x,y,z\},
$$

and its partials $\partial_{\{r, \theta, \zeta\}} \Phi_\alpha$ differ only by swapping one of
the three factors for its derivative. See [mrx/spline_geometry.py](mrx/spline_geometry.py):

 `spline_map_Phi_DPhi_at_quad(coefficients, extraction_T, seq)` returns
  $\Phi(x_q)$ and $D\Phi(x_q)$ at all quadrature points via three
  `einsum`s.
- `spline_map_jacobian_j_at_quad(...)` — only the Jacobian
  determinant.
- `compute_geometry_terms_from_spline(...)` — full
  $(g, g^{-1}, \sqrt{\det g})$ triple.
- `min_jacobian_from_coeffs(...)` — cheap mesh-folding check.

This path is faster than calling `jax.jacfwd` on the map pointwise and,
crucially, has a compact AD signature, which is what makes shape
optimization practical.

When the geometry changes throughout optimization, this corresponds to changing the `SplineMap.coefficients` leaf, which triggers a recomputation of the geometry terms and the operators, but not the 1D basis evaluations or the extraction sparsity pattern. The only computations that take place are tensor contractions and inversion/determinant of small $3\times 3$ matrices at quadrature points.

---

## 3. Extraction operators

B-spline bases are always assembled on an unconstrained tensor-product
basis ("raw" basis). Extraction operators restrict this raw basis
onto the actual discrete space of the problem: they enforce boundary
conditions and polar / toroidal compatibility.

Conceptually, if $\Lambda^k_{\text{raw}}$ is the raw tensor-product
basis of size $n^k_{\text{raw}}$, and $\Lambda^k$ is the extracted basis
of size $n^k$, the extraction operator is a sparse rectangular matrix
$\mathbb{E}^k \in \mathbb{R}^{n^k \times n^k_{\text{raw}}}$ with

$$
\Lambda^k_I(\hat x) = \sum_J \mathbb{E}^k_{IJ}\,\Lambda^k_{\text{raw},J}(\hat x).
$$

Equivalently, matrices $\mathbb{A}$ in the extracted basis are $\mathbb{E}^k \mathbb{A}_{\text{raw}} (\mathbb{E}^k)^T$.

Two kinds of extraction are implemented:

1. Boundary-condition extraction in
   [mrx/extraction_operators.py](mrx/extraction_operators.py), via
   `bc_extraction_op(...)`. This drops the boundary DOFs in clamped
   directions to realise homogeneous Dirichlet conditions, and it
   identifies DOFs across periodic edges. It is available for all
   $k = 0, 1, 2, 3$.
2. Polar (axis) extraction, via `PolarExtractionOperator`. In
   toroidal domains with a polar axis (e.g. a tokamak), the radial
   coordinate degenerates at $r = 0$: the physical map is not
   injective there. The polar extraction enforces the correct
   $C^1$ (for $k=0$) conditions at the axis by fusing the first "rings" of radial DOFs in a single z-slice into 3 "axis modes" ($c_0, c_\theta, s_\theta$).

A `DeRhamSequence` stores two flavors per degree:

- `seq.e0, seq.e1, seq.e2, seq.e3` — extraction without Dirichlet
  BCs (periodic / polar compatibility only).
- `seq.e0_dbc, seq.e1_dbc, seq.e2_dbc, seq.e3_dbc` — extraction
  that additionally removes boundary DOFs in clamped directions.

All of these are `BCSR` sparse matrices. Their transposes
(`seq.e0_T`, `seq.e0_dbc_T`, ...) are precomputed because they appear
in every matvec (and JAX's `BCSR` does not support transposes).

---

## 4. Hodge Laplacians

The **Hodge Laplacian** of degree $k$ is

$$
\mathbb{L}^k = d^k \delta^{k} + \delta^{k-1} d^{k-1}.
$$

Discretely these are saddle-point systems that couple $k$-forms and $(k-1)$-forms.

Concretely in [mrx/operators.py](mrx/operators.py), these bilinear forms are built:

- $k=0$: $\mathbb{L}^0$ = -div grad, built as $\mathrm{grad grad}$
  (this is the standard stiffness matrix).
- $k=1$: $\mathbb{L}^1$ = curl-curl - grad-div, built as
  $\mathrm{curl curl} + \mathbb{D}^{0} (\mathbb{M}^0)^{-1} (\mathbb{D}^0)^T$.
- $k=2$: $\mathbb{L}^2$ = curl-curl - grad-div, built as
  $\mathrm{div div} + \mathbb{D}^{1} (\mathbb{M}^1)^{-1} (\mathbb{D}^1)^T$.
- $k=3$: $\mathbb{L}^3$ = -div grad, built as $\mathbb{D}^{2} (\mathbb{M}^2)^{-1} (\mathbb{D}^2)^T$. This is a pure Schur complement because there are no 4-forms.

Key implementation points:

- Each block $\mathbb{D}^k$ and each `(grad-grad / curl-curl / div-div)` stiffness
  block is assembled once via sum-factorization, using the
  metric / inverse-metric-weighted $W$ tensor built from
  `geometry.metric_jkl`, `geometry.metric_inv_jkl`,
  `geometry.jacobian_j`.
- The Schur-complement contributions $\mathbb{D}^{k-1} (\mathbb{M}^{k-1})^{-1} (\mathbb{D}^{k-1})^\top$
  are not assembled. The systems are solved with matrix-free MINRES.
- Everything is available in two flavors: "plain" (extraction with
  `e*`) and "Dirichlet" (extraction with `e*_dbc`).

High-level user-facing matvecs in `mrx.operators`:

- `apply_mass_matrix(seq, ops, v, k, dirichlet=True)`
- `apply_stiffness(seq, ops, v, k, dirichlet=True)`
- `apply_hodge_laplacian(seq, ops, v, k, dirichlet=True)`
- `apply_inverse_*` counterparts that wrap the iterative solvers
  (see §5).

---

## 5. Solvers

All inverse operators are Krylov solvers that take a matvec callback
rather than an assembled dense operator. This keeps memory usage low
and lets us compose preconditioners on the fly. See
[mrx/solvers.py](mrx/solvers.py).

| Function                              | Purpose                                                             |
|---------------------------------------|---------------------------------------------------------------------|
| `preconditioned_cg`                   | Preconditioned CG for SPD systems. Default for mass-matrix solves.  |
| `solve_singular_cg`                   | CG on a singular SPD system with a known null space (deflation).    |
| `minres`                              | MINRES for symmetric indefinite systems.                            |
| `solve_saddle_point_minres`           | Block-preconditioned MINRES for $2\times 2$ saddle-point systems.   |
| `picard_solver`, `newton_solver`      | Fixed-point / Newton nonlinear solvers (used by relaxation).        |
| `backtracking_line_search`            | Armijo backtracking with an optional feasibility filter. Used by optimisation scripts. |

Preconditioners are usually diagonal: we precompute
$\mathrm{diag}(\mathbb{E} \mathbb{A} \mathbb{E}^\top)$ once per Hodge assembly (`diag_EAET`
in `mrx/utils.py`) and store its inverse on the `SequenceOperators` as
`*_sp_diaginv` (with `_dbc` variants). For the Hodge Laplacians we
additionally add the Schur-complement diagonal.

The Hodge-Laplacian inverse dispatches to the right solver per $k$:

- $k = 0, 3$: `preconditioned_cg` (non-singular after Dirichlet or
  average-zero conditions).
- $k = 1, 2$: `solve_singular_cg` with the known harmonic null space
  deflated out, using the diagonal Schur preconditioner.

---

## 6. Data structures and what they hold

### 6.1 Static topology

This is the data that depends only on the grid topology and the choice of 1D bases, and not on the physical map. We do not assume that we can compile code that runs on different topologies/resolutions without recompilation, so this data lives on the static `DeRhamSequence`:

- **`SplineBasis` / `TensorBasis`** — 1D and 3D B-spline bases. Just
  knots, degrees, type. No arrays that depend on the physical map.
- **`DifferentialForm`** — a $k$-form discrete space. Bundles the three
  1D bases plus bookkeeping (`n`, `nr`, `nt`, `nz`, `shape`, `types`).
- **`QuadratureRule`** — tensor product of 1D Gauss rules: `x` (points),
  `w` (weights), and per-direction counts.
- **Extraction operators** — sparse `BCSR` matrices plus their
  transposes. Computed once from topology; not dynamic.
- **`DeRhamSequence`** (plain Python class,
  [mrx/derham_sequence.py](mrx/derham_sequence.py)) — the canonical
  "setup" object. It holds all the above, plus cached 1D basis
  evaluations at quadrature (`basis_*_jk`, `d_basis_*_jk`), tolerances
  for default solvers, and a reference mass matrix (used to generate spline 
  representations of logical-to-physical maps) if assembled. **It
  is deliberately a static Python class, not a pytree**, because all of
  its contents are topology, and we want it captured once by closure
  (and therefore frozen) across JIT traces.

### 6.2 Dynamic geometry data

This is the data that depends on the physical map and must be recomputed when the geometry changes. It lives on dynamic `eqx.Module`s that can be passed into JIT-traced functions to do shape optimization and AD:

- **`SplineMap`** ([mrx/mappings.py](mrx/mappings.py)) — a
  logical-to-physical map expressed in the scalar spline basis.
  `eqx.Module`. Coefficients are a dynamic leaf so `jax.grad` can
  differentiate through them; the basis, extraction, and precomputed
  transpose are normal attributes.
- **`SequenceGeometry`** ([mrx/derham_sequence.py](mrx/derham_sequence.py)) —
  the three arrays that appear in every weighted integral:
  `metric_jkl`, `metric_inv_jkl`, `jacobian_j`. `eqx.Module`; built by
  `SequenceGeometry.from_map(F, seq.quad.x)` (generic path) or
  `SequenceGeometry.from_spline_map(spline_map, seq)` (sum-factorised
  path, used automatically when the map is a `SplineMap`).
- **`SequenceOperators`** ([mrx/operators.py](mrx/operators.py)) — the
  assembled geometry-dependent operators: sparse $\mathbb{M}^k$, sparse
  $\mathbb{D}^k$, sparse stiffness (grad-grad, curl-curl, div-div), all with
  their diagonal preconditioners for plain and `dbc` extraction.
  `eqx.Module`; every field is `Optional[...]` so you only pay for the
  blocks you actually assemble. Build it with
  `assemble_mass_operators / assemble_derivative_operators /
  assemble_hodge_operators(seq, geometry, ks=(0,))` or
  `operators_from_coeffs(seq, coeffs, ks, kinds)` when the map is a
  `SplineMap`.

### 6.3 Why the split (static vs. dynamic)

The static / dynamic split matches what JAX's staging model needs:

- Anything inside a `@jax.jit` traced function, or anything we want to
  differentiate w.r.t. the geometry, should be a **pytree leaf of a
  dynamic object** (`SplineMap.coefficients`,
  `SequenceGeometry.metric_*`, `SequenceOperators.m0_sp.data`, etc.).
- Anything that should not cause recompilation, and should be identical
  across all traces (bases, knot vectors, extraction sparsity pattern,
  quadrature points and weights), lives on the static
  `DeRhamSequence`.

A typical workflow therefore looks like:

```python
seq = DeRhamSequence(ns, ps, p_quad, types, lambda x: x, polar=True)
seq.evaluate_1d()                              # cache 1D basis @ quad

coeffs = ...                                    # (3, n_dof) from user
ops, geom = operators_from_coeffs(seq, coeffs, ks=(0,))  # dynamic
u = apply_inverse_hodge_laplacian_ops(seq, ops, rhs, k=0)
```

and, when differentiating:

```python
def scalar(coeffs, u, lam):
    ops, _ = operators_from_coeffs(seq, coeffs, ks=(0,), kinds=("hodge",))
    return jnp.dot(lam, apply_stiffness(seq, ops, u, 0))

grad_coeffs = jax.grad(scalar, argnums=0)(coeffs, u, lam)
```

The forward / adjoint CG solves sit outside the `grad` trace (via
`jax.lax.stop_gradient` on `u` and `lam`); the only thing the trace
sees is the pure, sum-factorised path
`coeffs → geometry → operator blocks → scalar`.

---

## 7. Assembly order and nullspace computation

Operators and preconditioners have a strict build order — each layer
reads results from the previous one. The high-level builders in
[mrx/operators.py](mrx/operators.py) already call things in the right
sequence, but the dependency chain is worth understanding when you
assemble things by hand (e.g. to skip degrees you don't need).

### 7.1 Dependency graph

```
DeRhamSequence (static)
      │   topology, 1D bases, extraction operators,
      │   cached 1D basis-at-quadrature tables
      ▼
SequenceGeometry            ←  from_map(F, quad.x)  or  from_spline_map(spline_map, seq)
      │   metric_jkl, metric_inv_jkl, jacobian_j
      ▼
SequenceOperators
      │   1. mass         (assemble_mass_operators)        — needs geometry
      │   2. derivative   (assemble_derivative_operators)  — needs geometry (for sign / curl-perm only)
      │   3. hodge        (assemble_hodge_operators)       — needs geometry; the grad-grad /
      │                                                      curl-curl / div-div stiffness blocks
      │                                                      and their diagonal preconditioners
      │   4. projection   (assemble_projection_operators)  — topology-only, no geometry
      ▼
Nullspaces (compute_nullspaces / compute_nullspaces_iterative)
            needs fully assembled mass, derivative, and Hodge operators
            because it calls apply_inverse_mass_matrix,
            apply_leray_projection, and apply_inverse_hodge_laplacian.
```

Concretely, the canonical sequence is:

```python
seq = DeRhamSequence(ns, ps, p_quad, types, F, polar=True)
seq.evaluate_1d()                              # cache 1D basis @ quadrature

geom = SequenceGeometry.from_map(F, seq.quad.x)           # or .from_spline_map(...)
ops  = assemble_mass_operators(seq, geom)                 # M^k, diag(M^k)^{-1}
ops  = assemble_derivative_operators(seq, geom, ops)      # D^k (+ transposes)
ops  = assemble_hodge_operators(seq, geom, ops)           # grad-grad / curl-curl /
                                                          # div-div, with Schur diag preconds
ops  = assemble_projection_operators(seq, ops)            # optional, topology only
```

`assemble_all_operators(seq, geom)` packages all four steps. When the
map is a `SplineMap`, `operators_from_coeffs(seq, coeffs, ks, kinds)`
additionally does the `SequenceGeometry` rebuild in one call and is
the preferred entry point for AD.

Each builder is **incremental**: it takes an optional existing
`SequenceOperators` and fills in the missing blocks for the requested
`ks`. You can assemble only what you need (e.g. `ks=(0,)`, `kinds=("hodge",)`)
and the unused fields stay `None`.

### 7.2 Why this order

- **Mass before Hodge.** The diagonal preconditioners for the
  Schur-complement contributions $\mathbb{D}^{k-1}(\mathbb{M}^{k-1})^{-1}(\mathbb{D}^{k-1})^\top$
  in `dd*_sp_diaginv` combine `diag(stiffness)` with
  `diag(M^{-1})`-weighted $\mathbb{D}\mathbb{D}^\top$ diagonals, so the
  mass-matrix diagonal inverse must already be available when the
  Hodge block is assembled.
- **Derivative before Hodge.** The Schur term also needs the sparse
  $\mathbb{D}^k$ block (and its transpose) to build the diagonal
  preconditioner.

### 7.3 Nullspaces

Hodge Laplacians generally have a non-trivial kernel and must be
deflated in CG. There are two ways to populate the null vectors on
`seq`, both in [mrx/nullspace.py](mrx/nullspace.py):

1. **Closed-form** — `compute_nullspaces(seq)`: uses the fact that the harmonic 
    spaces are easy to characterise when the domain has no holes.
    - $k=0$, no DBC: the constant function $\mathbf{1}/\|\mathbf{1}\|_{M^0}$.
    - $k=3$, DBC: $\mathbf{1}$ lifted via $(\mathbb{M}^3)^{-1}$ and normalised.
    - $k=1,2$ (plain / DBC): start from $\mathbf{1}$, take a Leray
      projection to get a divergence-free / curl-free seed $v$, then
      subtract its exact curl contribution by solving one auxiliary
      Hodge Laplacian; the residual $v - \mathrm{curl} a$ is the
      harmonic representative. **This relies that the domain has no holes**.
      We will probably remove this soon.

2. **Iterative** — `compute_nullspaces_iterative(seq, betti_numbers, eps)`:
   inverse power iteration with a small shift $\epsilon$ against the
   Hodge Laplacian for each $(k, \text{BC})$ pair. Handles arbitrary
   topology: you pass $(b_0, b_1, b_2, b_3)$ (with the convention
   $b_0 = 1$, $b_3 = 0$), and it finds $b_k$ null vectors for each
   $k$. `eps` is a regulariser that keeps the shifted system SPD.

Both functions store results as attributes `seq.null_k` and
`seq.null_k_dbc` (lists of vectors). The singular-CG wrapper
`solve_singular_cg` consults these via `seq._get_nullspace(k, dirichlet)`
to deflate each Krylov iterate.

**Order requirement.** Nullspace computation calls
`apply_inverse_mass_matrix`, `apply_leray_projection`, and (for
$k=1,2$) `apply_inverse_hodge_laplacian`. All of those require a
fully-assembled `SequenceOperators`, so compute nullspaces **after**
`assemble_all_operators` (or after every relevant `assemble_*` call
for your chosen `ks`).

Saddle-point solves (`solve_saddle_point_minres`) need a second set of
null vectors on the lower block. These are derived lazily from the
primary ones in `get_saddle_point_nullspaces(seq, k, dirichlet)` using
the identity: 

$$v \in \ker(S_k + \mathbb{D}^{k-1}(\mathbb{M}^{k-1})^{-1}(\mathbb{D}^{k-1})^\top) \quad \Rightarrow \quad [v,\; (\mathbb{M}^{k-1})^{-1}(\mathbb{D}^{k-1})^\top v] \in \ker \begin{bmatrix} S_k & \mathbb{D}^{k-1} \\ (\mathbb{D}^{k-1})^\top & 0 \end{bmatrix}.
$$
