# What MRX actually needs: the core building blocks

At the end of the day the whole MRX discretization runs on a small set of
sparse building blocks. Everything else -- mass solves, weak derivatives,
Hodge Laplacians, preconditioners -- is a *composition of matvecs* against
these. Nothing larger than the mass matrices is ever materialized; the operator
compositions are applied factor by factor.

This note collects the blocks, their sizes, and where they live, then explains
how a matrix-free apply replaces the largest of them.

## The blocks

| symbol | name | what it is | geometry-dependent? | stored as |
| ------ | ---- | ---------- | ------------------- | --------- |
| `g`    | 1D bases (+ derivatives) | per-axis spline values at quad points | no (map-independent) | small dense `(n_elem, q, p+1)` |
| `E_k`  | extraction | maps reduced (polar/boundary) DOFs ↔ raw tensor-product DOFs | no (topological) | sparse `BCSR` (`e{k}`, `e{k}_dbc`, …) |
| `G_k`  | incidence / exterior derivative | topological `{-1,0,+1}` grad/curl/div on the raw grid | no (topological) | sparse `BCSR` (`g0,g1,g2`) |
| `M_k`  | mass matrix | `∫ Λ^k_i · W_k · Λ^k_j` with metric weight `W_k` | **yes** | sparse `BCSR` (`m0,m1,m2,m3`) |
| `P`    | L²-projections | inter-degree projection matrices | yes | sparse `BCSR` (`p03,p30,p12,p21`) |

That is essentially the entire vocabulary. Three of these (`g`, `E`, `G`) are
**geometry-independent** -- they depend only on the spline degrees, the mesh,
and the de Rham topology, so they are computed once and never change when the
mapping changes. Only the mass matrices `M_k` (and the projections `P`, which
are built from masses) carry the geometry.

### `g` — the 1D bases

`evaluate_basis_local` returns, per element and per axis, the `p+1` locally
active 1D basis values at the local Gauss points, plus the global DOF id of
each. The derivative bases (used by the vector forms) report one fewer local
function. These tiny arrays are the seed of *all* assembly via sum
factorization.

### `E_k` — extraction

The raw tensor-product space has clean separable structure but the wrong DOFs
at the polar axis and the domain boundary. `E_k` is the sparse map that turns
the raw space into the reduced, conforming space:

```
M_k^reduced = E_k M_k E_k^T
```

Variants exist for free (`e{k}`), Dirichlet (`e{k}_dbc`), and boundary
(`e{k}_bc`) DOFs, each with its transpose. `E_k` is topological -- it does not
move when the geometry changes -- so it is built once.

### `G_k` — incidence (strong exterior derivative)

`g0, g1, g2` are the discrete grad / curl / div on the raw grid, with entries in
`{-1, 0, +1}`. They encode the de Rham complex itself and are
geometry-independent. The *weak* derivatives compose them with a mass matrix
(below); the *strong* derivatives apply `G_k` directly.

### `M_k` — mass matrices (the only big, geometry-dependent blocks)

The four mass matrices are the heart of the geometry coupling:

| form | weight `W_k`   | basis on axis `c` |
| ---- | -------------- | ----------------- |
| k=0  | `J`            | primal (all axes) |
| k=1  | `G^{-1} J`     | derivative on `c` |
| k=2  | `G (1/J)`      | primal on `c`     |
| k=3  | `1/J`          | derivative (all)  |

with `J = det DF` and `G` the metric of the map. The k=1/k=2 forms carry the
full 3×3 metric, so they are the largest objects in the whole code. See
[local_assembly.md](local_assembly.md) for the assembly and its memory
profile. `M1` in particular reaches tens to hundreds of GB at the high-(n, p)
corners; it is the practical bottleneck.

## How everything is composed from the blocks

Nothing beyond `M_k` is assembled. The compound operators are applied as
chained matvecs (in reduced space, with the extraction `E` on the outside):

* **Mass apply** `M_k^reduced v = E_k ( M_k ( E_k^T v ) )`.
* **Weak derivative** `D_k = M_{k+1} G_k`, applied as
  `E_out ( M_{k+1} ( G_k ( E_in^T v ) ) )` — the dense `D_k` is never formed.
* **Strong derivative** = `G_k` alone (no mass solve).
* **Hodge / stiffness** `K_k = G_k^T M_{k+1} G_k`, never materialized; applied
  as `G_k^T ( M_{k+1} ( G_k v ) )` (with extraction `E_k (...) E_k^T` around
  it). For `k=0` this is the scalar Laplacian used by the Poisson solve.
* **Preconditioners** are built from the *same* blocks: the tensor mass /
  Laplacian preconditioners use low-rank CP fits of the separable metric
  factors; the Jacobi diagonal is `diag(E_k M_k E_k^T)` (or
  `diag(E_k G^T M G E_k^T)` for stiffness), computed directly from the block
  nonzeros without probing.

So the dependency graph is shallow: build `g` → assemble `M_k` (the only
geometry work) → compose with the static `E_k`, `G_k` for every operator the
solver needs.

## Matrix-free apply: removing the `M_k` storage

Because every compound operator only ever *applies* `M_k` to a vector, `M_k`
does not actually need to be stored — it can be applied straight from the same
element-local data that assembly uses. This removes the single largest
allocation (e.g. ~83 GB for `M1` at `n=32, p=4`) at the cost of recomputing the
element contraction on each apply.

### The idea

Assembly forms, per element, the dense block

```
C[a,b,c,d,e,f] = Σ_q  B^row(q) B^col(q) W(q)
```

and scatters it into `M_k`. A matvec `y = M_k x` only needs
`y_i = Σ_j C_ij x_j`, so we can fold the input vector *into* the contraction and
never build `C`:

```
1. gather    x_local = x[ global DOF ids of each element's columns ]
2. push to quad:   x_q = Σ_cols B^col · x_local          (sum factorization)
3. weight:         x_q ← W · x_q                          (metric at quad pts)
4. pull back:      y_local = Σ_quad B^row · x_q           (sum factorization)
5. scatter-add:    y[ global row DOF ids ] += y_local     (segment_sum)
```

Steps 2 and 4 are done axis by axis (three small einsums each), exactly the same
sum factorization used in assembly. For the vector forms (`k=1, 2`) the loop
runs over the nine metric component pairs `(cr, cc)`.

### Why it is cheaper in memory and the cost trade

* **Memory.** The largest transient is one push-to-quad field,
  `O(n^3 (p+1)^2 q)`, versus the stored matrix `O(n^3 (p+1)^6)`. For `M1` this
  is the difference between a few hundred MB of working set and tens of GB of
  resident matrix.
* **Compute.** Per element the matvec costs `O((p+1)^4 q)` flops, while a stored
  sparse matvec costs `O((p+1)^6)` (one multiply per nonzero). So matrix-free
  has *fewer* flops as `p` grows, but it pays for repeated basis contractions
  and a scatter on every apply, which a single fused `BCSR` kernel avoids. At
  small/moderate sizes the stored matvec is usually faster in wall-clock; the
  matrix-free version wins when the matrix would not fit at all, and closes the
  gap as `p` increases.

`scripts/benchmark_matrixfree_m1.py` measures exactly this trade for `M1`
(correctness + per-call exec time + the M1 storage size) across `n` and `p`.

### When to use which

* Use the **stored `BCSR`** `M_k` whenever the matrix fits — it is the simplest
  and usually the fastest per apply, and it is reused across many CG iterations.
* Use the **matrix-free** apply for the `(n, p)` corners where `M_k` exceeds
  device memory; it trades recompute for never holding the matrix, which is the
  only way those cases run on a single device.
