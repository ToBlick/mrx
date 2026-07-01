# Matrix-free scalar operators in MRX — avoiding assembly, handling changing geometry

**Scope.** This note covers the **scalar** forms only: the 0-form mass `M0` (and,
by duality, the 3-form mass `M3`). Both are single-component, single-weight
operators, so they expose the matrix-free machinery in its cleanest form without
the 3×3 metric block structure of the 1- and 2-form (vector) cases. Everything
here is grounded in the production code:

- `mrx/local_assembly.py` — the sum-factorized matvec (`build_matrixfree_mass_apply`).
- `mrx/geometry.py` — geometry term computation (`compute_geometry_terms`).
- `mrx/preconditioners.py` — the Greville-collocation inverse (`_build_greville_mass_block_factors`).

---

## 1. The two problems

A scalar mass matrix on a tensor-product spline space of size
`n = n_r · n_θ · n_ζ` is

$$
(M_0)_{ij} \;=\; \int_{\hat\Omega} \Lambda_i(\xi)\,\Lambda_j(\xi)\,\det(DF(\xi))\;d\xi,
\qquad \Lambda_i = \Lambda^r_{i_r}\!\otimes \Lambda^\theta_{i_\theta}\!\otimes \Lambda^\zeta_{i_\zeta}.
$$

Two things make the naive approach untenable on the grids we care about
(`12³`–`16³` with degree `p` up to 5, on a GPU):

1. **Assembly cost / storage.** Even though `M0` is sparse (bandwidth `(p+1)`
   per axis), forming and storing it, and especially the *vector* blocks, runs
   into an `O(n² )`-ish memory wall and an `O(n³ p⁶)` global-einsum assembly path
   (see the module docstring in `mrx/local_assembly.py:1`). We want to *apply*
   `M0` to a vector inside a Krylov solve without ever building it.

2. **Changing geometry.** The map `F: \hat\Omega \to \Omega` changes — between
   problems (cylinder / toroid / W7-X), and within a single run when the geometry
   itself is a spline whose coefficients move (equilibrium iterations,
   `SplineMap`). The discretization (bases, quadrature, element topology) is
   *fixed*; only `F` changes. We want the cost of "new geometry" to be one cheap
   pointwise re-evaluation, not a re-assembly.

The key structural fact that makes both solvable: **geometry enters only as a
pointwise scalar weight** $W(\xi) = \det(DF(\xi))$ evaluated at quadrature points.
It multiplies the basis product; it never couples basis functions to each other.
That multiplicativity is what keeps the operator separable.

---

## 2. Discretization scaffolding (geometry-independent, built once)

These pieces depend only on the spline space and the mesh, never on `F`:

- **1D bases at quadrature points.** For each axis, evaluate the active
  B-splines on each element at that element's Gauss points
  (`evaluate_basis_local`, `mrx/local_assembly.py:42`):

  ```
  B_loc : (n_elem, q_per_elem, p+1)   # basis values
  gdof  : (n_elem, p+1)               # global DOF id of each local basis
  ```

  Periodic axes have `n_elem = n` with wrap-around DOF ids; clamped axes have
  `n_elem = n − p`. This is computed once per axis and cached.

- **Element counts and Gauss weights.** `_elem_counts` (`:97`) derives
  `(ne_x, ne_y, ne_z, qx, qy, qz)` from the k=0 basis; `_quad_gauss_weight`
  (`:365`) forms the per-element outer product
  `w_x[q]·w_y[r]·w_z[s]` of the 1D Gauss weights.

- **Gather / scatter index plans.** `_flat_dof_plan` (`:397`) precomputes, *on
  the host*, the flat indices that map each element's local DOF cube to the
  global flattened DOF grid. One gather array per (column) component and one
  segment-id array per (row) component. The matvec then does exactly **one
  gather** and **one `segment_sum`** with no index arithmetic at run time.

The only geometry-dependent input to the whole apply is a single array of length
`N_q` (the quadrature-point count): the weight `W`.

---

## 3. How geometry enters: the pointwise weight

For the scalar forms the weight is a plain scalar field at quadrature points
(`build_matrixfree_mass_apply`, `mrx/local_assembly.py:463`):

| form | weight `W(ξ)` | basis on each axis |
|------|---------------|--------------------|
| k=0  | `det(DF)`     | primal `Λ` on all three axes |
| k=3  | `1 / det(DF)` | derivative `dΛ` on all three axes |

`det(DF)` lives on the geometry object as `geometry.jacobian_j`, shape `(N_q,)`,
computed by `compute_geometry_terms` (`mrx/geometry.py:56`):

```python
def G(x):
    DF = jax.jacfwd(map)(x)      # 3×3 Jacobian by autodiff
    return DF.T @ DF
...
jacobian_j = jax.lax.map(jacobian_determinant(map), quad_x,
                         batch_size=mrx.MAP_BATCH_SIZE_INNER)
```

The Jacobian is obtained by **automatic differentiation of the map** (`jacfwd`),
so any differentiable `F` works with no hand-coded derivatives, and the
`lax.map` batching keeps the W7-X case (large `N_q`) inside GPU memory.

Inside the apply builder the scalar weight is reshaped to per-element blocks and
the Gauss weights are folded in once (`:500`–`505`):

```python
Wf = _split_field(weight_of[(0,0)], nx, ny, nz, ne_x, ne_y, ne_z, qx, qy, qz)
W_split[(0,0)] = Wf * gw     # geometry weight × Gauss weights, per element
```

So `W_split` is a `(ne_x, ne_y, ne_z, qx, qy, qz)` array carrying
`det(DF)(ξ)·w_q w_r w_s`. This is the **only** place geometry touches the kernel.

---

## 4. The matrix-free matvec (sum factorization)

The operator `y = M0 · x` is applied element-by-element by *folding the
contraction against the input vector* instead of forming the dense element block.
For the scalar case there is a single component pair `(cr, cc) = (0, 0)`, and the
core is `_element_apply` (`mrx/local_assembly.py:416`):

```python
def _element_apply(Bvals_r, Bvals_c, W, x_flat_c, gather_idx_c):
    Bxr, Byr, Bzr = Bvals_r          # row bases (= col bases for scalar)
    Bxc, Byc, Bzc = Bvals_c

    # GATHER element-local input (one precomputed gather, no index math)
    x_local = x_flat_c[gather_idx_c]            # (ne_x,ne_y,ne_z, p+1,p+1,p+1)

    # COLUMN bases: coefficients -> quadrature values, one axis at a time
    t1 = jnp.einsum('xqb,xyzbdf->xyzqdf', Bxc, x_local)   # r-axis
    t2 = jnp.einsum('yrd,xyzqdf->xyzqrf', Byc, t1)        # θ-axis
    u  = jnp.einsum('zsf,xyzqrf->xyzqrs', Bzc, t2)        # ζ-axis

    # WEIGHT at quadrature points (det(DF)·Gauss weights, already folded)
    u = u * W

    # ROW bases: quadrature values -> coefficients, one axis at a time
    s1      = jnp.einsum('xqa,xyzqrs->xyzars', Bxr, u)
    s2      = jnp.einsum('yrc,xyzars->xyzacs', Byr, s1)
    y_local = jnp.einsum('zse,xyzacs->xyzace', Bzr, s2)
    return y_local
```

The structure is **forward transform → multiply by weight → adjoint transform**:

1. **Forward (col bases).** Three sequential 1D contractions take the
   element-local spline coefficients to values at the element's quadrature grid.
   Doing one axis at a time is *sum factorization*: it costs
   `O(q·(p+1) + q²·(p+1) + q³)` per element instead of the `O(q³(p+1)³)` of a
   single dense contraction.

2. **Weight multiply.** A pointwise multiply by `W` — this is the entire
   geometric content of the operator.

3. **Adjoint (row bases).** The mirror-image three contractions take quadrature
   values back to coefficient space. Because row and column bases are identical
   for a mass matrix, this is the transpose of step 1, which makes the applied
   operator symmetric by construction.

The driver loops over component pairs (trivial for scalar) and scatters with a
single `segment_sum` (`:526`–`543`):

```python
@jax.jit
def _impl(x, Bvals, W_split, gather_idx, seg_idx):
    Xc = [x[starts_t[c]:starts_t[c+1]] for c in range(n_comp)]   # n_comp == 1
    out_parts = []
    for cr in range(n_comp):
        acc = jnp.zeros((nseg[cr],), dtype=x.dtype)
        for cc in range(n_comp):
            y_local = _element_apply(Bvals[cr], Bvals[cc], W_split[(cr,cc)],
                                     Xc[cc], gather_idx[cc])
            acc = acc + jax.ops.segment_sum(
                y_local.reshape(-1), seg_idx[cr], num_segments=nseg[cr])
        out_parts.append(acc)
    return jnp.concatenate(out_parts)
```

`segment_sum` accumulates the overlapping element-local contributions of shared
DOFs into the global output vector — this is the matrix-free analogue of
"scatter into the global matrix", done against the result vector instead.

**Why this avoids assembly.** Nothing of size `O(n²)` or `O(n p⁶)` is ever
formed. The transient working set is `O(n·(p+1)·q)` (the partially-transformed
element fields), and the persistent state is just the cached 1D bases, the index
plans, and the length-`N_q` weight. The returned `apply` is a closure over those;
it is the operator. (The same module can also emit the sparse `BCOO` matrix via
`assemble_m0_local`/`_assemble_scalar_local`, `:167`/`:305` — same sum
factorization, but materialized — for the rare paths that genuinely need a stored
matrix.)

**Raw vs. extracted space.** `build_matrixfree_mass_apply` acts on the *raw*
tensor-product DOF layout. Boundary / polar-axis extraction `E (·) Eᵀ` is applied
by the caller exactly as for the stored matrix (see
`_apply_extracted_mass_operator`, `mrx/preconditioners.py:1884`), so the
matrix-free apply is a drop-in replacement for `M0 @ x`.

---

## 5. Dealing with changing geometry

The whole design hinges on the separation established above: **bases + topology
are fixed; geometry is one pointwise weight array.** Consequences:

### A new map = one re-evaluation, no re-assembly

To switch geometry you rebuild only `geometry.jacobian_j` by evaluating
`compute_geometry_terms(new_map, quad_x)` (`mrx/geometry.py:56`). The 1D bases,
the element plan, the gather/scatter indices, and the JIT-compiled kernel are all
unchanged. The new weight flows into `W_split` and the *same compiled matvec*
runs. Cost of "new geometry" = `N_q` Jacobian-determinant evaluations, nothing
more.

### Geometry as a differentiable pytree

`SequenceGeometry` is an `eqx.Module` (`mrx/geometry.py:87`) whose
`metric_jkl / metric_inv_jkl / jacobian_j` are dynamic pytree leaves. When the
map is itself a `SplineMap`, its coefficients are tracked leaves too. This means
the whole apply is differentiable and JIT-able with respect to the geometry —
the geometry can change *under autodiff* (e.g. shape derivatives, equilibrium
solves) without re-tracing structure.

### Spline-map fast path

When `F` is a tensor-product spline (the W7-X case), evaluating it at every
quadrature point naively is `O(N_q · n_r n_θ n_ζ)`. `compute_geometry_terms_from_spline`
(`mrx/geometry.py:284`) and `_tp_evaluate` (`:174`) instead evaluate `F` and `DF`
by the *same* sum factorization used for the operator — three sequential 1D
contractions, `O(N_q (n_r + n_θ + n_ζ))`. So even a moving spline geometry is
cheap to re-collocate.

### Robustness to degenerate Jacobians

A clamped spline map's `evaluate()` has a constant branch at the parameter
endpoint, so `jacfwd` gives `det = 0` over the entire `r = 1` layer (≈288/2880
points on W7-X; analytic maps are fine). The operator weight tolerates this
(those rows are handled by the surgery-Schur split), but the *preconditioner*
must not divide by it — see §6.

---

## 6. The preconditioner: Greville collocation (scalar case)

Applying `M0` matrix-free is half the story; a Krylov solve needs `M0⁻¹` applied
cheaply, and that inverse must also follow the geometry without assembly. The
production atom is **Greville collocation** (`docs/preconditioner_plan.md`).

The idea: factor the geometry out of the spline connectivity. Collocate the
geometric weight at the **Greville abscissae** (the natural spline nodes) to get a
diagonal `D`, and approximate

$$
P \;=\; D^{1/2}\,M_0^{\text{unweighted}}\,D^{1/2},
\qquad
P^{-1} \;=\; D^{-1/2}\,\big(M_{0r}^{-1}\!\otimes M_{0\theta}^{-1}\!\otimes M_{0\zeta}^{-1}\big)\,D^{-1/2},
$$

where `M_{0a}` are the small **unweighted** 1D mass matrices (each `n_a × n_a`),
whose dense inverses are precomputed once. `D` is the diagonal of `det(DF)`
sampled at the 3D Greville points.

This is implemented in `_build_greville_mass_block_factors`
(`mrx/preconditioners.py:2417`). The scalar k=0 path (`wkind == "J"`):

```python
M0_r = _assemble_weighted_1d_mass(bases[0], quad_w[0])   # unweighted 1D masses
M0_t = _assemble_weighted_1d_mass(bases[1], quad_w[1])
M0_z = _assemble_weighted_1d_mass(bases[2], quad_w[2])
inv_r, inv_t, inv_z = map(jnp.linalg.inv, (M0_r, M0_t, M0_z))   # cached inverses

# Greville points per axis, geometry sampled there:
metric, minv, jac = compute_geometry_terms(seq.map, pts)
weight = jac                      # k=0: D = det(DF)  (k=3 dual: 1/jac)
D = weight.reshape(nr, ntc, nzc)
inv_sqrt_D = 1.0 / jnp.sqrt(D)
```

and the apply (`mrx/preconditioners.py:795`–`802`) is exactly the sandwich —
scale by `D^{-1/2}`, three 1D inverse contractions, scale by `D^{-1/2}`:

```python
f = inv_sqrt_D * x
f = jnp.einsum("ij,jkl->ikl", inv_r, f)     # M0_r^{-1} along r
f = jnp.einsum("ij,kjl->kil", inv_t, f)     # M0_t^{-1} along θ
f = jnp.einsum("ij,klj->kli", inv_z, f)     # M0_z^{-1} along ζ
out = inv_sqrt_D * f
```

Why this is the right tool for *changing* geometry:

- **SPD by construction.** `D = det(DF) > 0` and the sandwich `D^{-1/2}(…)D^{-1/2}`
  keeps `P` symmetric positive-definite, so PCG stays stable on any geometry.
- **Geometry is one diagonal.** A new map changes only `D` (one Greville-point
  re-evaluation). The 1D inverses are geometry-independent and reused verbatim.
- **No inner Krylov, no stored 3D matrix.** The apply is pure dense 1D matvecs +
  two diagonal scalings: `O(n^{4/3})` work, `O(n_a²)` storage for three small
  matrices.
- **Exact for the scalar mass.** Because k=0 has a *single* channel, the weight
  is exactly of the form `c·D(ξ)` and the Greville sandwich conditions `M0` to
  `κ ≈ 1.3` on all geometries including W7-X (`docs/preconditioner_plan.md` §C/§D).
  (This is special to the scalar case — the vector Laplacian has three channels
  with non-constant ratios, which is the open W7-X difficulty.)

### Two Greville gotchas (both handled in the code)

1. **Differentiated-axis double point** (relevant to k=3 / the `diff` axes). Do
   *not* read Greville points from `dΛ[axis].s` — that inner basis carries the
   parent's degree-`p` knots while declaring degree `p−1`, producing a spurious
   double Greville point at a clamped endpoint. Build a fresh
   `SplineBasis(dΛ.n, dΛ.p, dΛ.type)` instead (`mrx/preconditioners.py:2455`–2459).

2. **`det = 0` at the clamped boundary.** A spline map's clamped `evaluate()` is
   constant at the endpoint → `jacfwd` det is zero over the whole `r=1` Greville
   layer. Fix: clip clamped-axis Greville coords to `[1e-7, 1−1e-7]`
   (`:2460`–2462), and **median-floor** any residual non-positive `D` rather than
   tiny-flooring it (`:2478`–2484) — a tiny floor would spike `1/√D` into a
   spurious near-null mode; that region is surgery-corrected anyway.

---

## 7. Cost & memory summary (scalar k=0)

| quantity | naive dense | sum-factorized matrix-free |
|----------|-------------|-----------------------------|
| storage of operator | `O(n²)` | **none** (closure over 1D bases + `N_q` weight) |
| one matvec | `O(n²)` | `O(n·(p+1)·q)` transient, `O(n)` persistent |
| assembly to switch geometry | full re-assembly | **one** `N_q`-length re-evaluation of `det(DF)` |
| preconditioner storage | `O(n²)` (factorized `M0⁻¹`) | three `n_a × n_a` inverses + one `n`-diagonal `D` |
| preconditioner apply | — | `O(n^{4/3})`, no inner Krylov |

The unifying principle: **geometry is a pointwise weight, not a coupling.** That
single fact lets the operator stay separable (so sum factorization applies and no
matrix is stored) and lets a geometry change cost exactly one cheap pointwise
re-collocation — both in the forward apply (`det(DF)` at quadrature points) and
in the preconditioner (`det(DF)` at Greville points).
</content>
</invoke>
