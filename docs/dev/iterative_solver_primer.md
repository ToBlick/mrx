# Iterative Solver Primer

This note records the current production picture for iterative solves in
`mrx`. It is meant to be read together with
[docs/mrx_primer.md](mrx_primer.md), which covers the spaces, extraction
operators, and Laplacian constructions.

The focus here is narrower: which Krylov solve is used for which operator,
what the default preconditioners are, and how the mass and Laplacian
benchmark scripts fit into that picture.

## 1. Top-Level Solver Split

All inverse operators are applied matrix-free: the code passes callable
matvecs into Krylov solvers rather than assembling dense inverses.

The current split is:

| Problem | Main solver | Default preconditioner idea |
|---|---|---|
| `M_k u = f` | preconditioned CG | mass preconditioner, `auto` => tensor if assembled, else Jacobi |
| `L_0 u = f` | singular CG | scalar Hodge preconditioner, `auto` => tensor if assembled, else Jacobi |
| `(L_0 + eps M_0) u = f` | CG on the shifted scalar operator | tensor-or-Jacobi complement, plus harmonic coarse correction when appropriate |
| `L_k u = f`, `k = 1, 2, 3` | saddle-point MINRES | structured block preconditioner |
| `(L_k + eps M_k) u = f`, `k = 1, 2, 3` | saddle-point MINRES | same structured block preconditioner |
| `M_k + eps L_k` solves | scalar CG for `k = 0`, saddle MINRES for `k = 1, 2, 3` | simple diagonal block scalings |

So the main production distinction is scalar versus mixed/saddle, not just
unshifted versus shifted.

At the API level there are two layers to keep separate:

- user-facing preconditioner specifications,
- internal callable preconditioner applies.

The user-facing specification objects are:

- `MassPreconditionerSpec` for mass-like and scalar-preconditioner slots,
- `SchurPreconditionerSpec` for the inner/outer Schur choices,
- `SaddlePointPreconditionerSpec` for the full mixed block preconditioner.

The assembled data itself lives in `SequenceOperators`: sparse mass and
derivative blocks, diagonal inverses, stored nullspaces, and the tensor-factor
payload in `operators.mass_preconds`.

The solver wrappers on `DeRhamSequence` accept either strings such as `"auto"`,
`"jacobi"`, `"tensor"` or the full spec objects above. In `mrx/operators.py`
those inputs are materialized into concrete callable applies such as
`_build_mass_preconditioner_apply(...)` or the Schur/coupled saddle builders,
and those callables are what finally get passed into CG or MINRES.

## 2. Mass Solves

Mass solves are SPD and use preconditioned CG. The production preconditioner
interface is `MassPreconditionerSpec`.

The practical options are:

- `jacobi`: inverse diagonal of the extracted mass matrix.
- `tensor`: the structured tensor/block preconditioner.
- `auto`: prefer `tensor` when the relevant factors have been assembled,
  otherwise fall back to `jacobi`.

The tensor path is now the intended production route for all four form degrees
`k = 0, 1, 2, 3`. The structure is degree-dependent:

- `k = 0`: small dense Schur block plus one scalar tensor bulk block.
- `k = 1`: outer surgery Schur, inner coupled Schur, and tensor bulk blocks.
- `k = 2`: smaller outer Schur plus three tensor bulk blocks.
- `k = 3`: direct scalar tensor block.

The interactive mass benchmark script
[scripts/interactive/k0_mass_preconditioner_choices.py](../scripts/interactive/k0_mass_preconditioner_choices.py)
compares the production tensor route against Jacobi and polynomial baselines
such as Richardson and Chebyshev.

## 3. Laplacian Solves

### 3.1 Scalar `k = 0`

The scalar Hodge solve stays in the scalar formulation.

- Unshifted `L_0`: use `solve_singular_cg` and deflate the harmonic mode.
- Shifted `L_0 + eps M_0`: do not deflate. The harmonic mode moves from the
  kernel to an eigenmode of size `eps`.

For shifted free-boundary solves, the preferred structure is:

- a complement preconditioner on the non-harmonic subspace,
- plus an explicit `(1 / eps)` coarse correction on the stored harmonic mode,
  once that mode is available.

If a valid harmonic vector is not yet known, the code stays conservative and
uses the robust complement path without the coarse correction. In particular,
inverse iteration disables the harmonic coarse term while it is still
constructing the nullspace vector.

The scalar preconditioner options follow the same naming as the mass case:

- `jacobi` is the minimal fallback,
- `tensor` is the preferred structured path,
- `auto` means tensor when assembled and applicable, else Jacobi.

### 3.2 Mixed degrees `k = 1, 2, 3`

For `k = 1, 2, 3`, the library currently uses the mixed saddle-point form and
solves it with `solve_saddle_point_minres`, both for unshifted and shifted
problems.

The exposed preconditioner is `SaddlePointPreconditionerSpec`, with four
pieces:

- `mass`: lower-block mass preconditioner.
- `schur.inner`: approximation used inside the Schur complement.
- `schur.outer`: preconditioner for the resulting Schur operator.
- `coupled`: optional coupled completion of the block preconditioner.

The current defaults are:

- `mass = tensor`,
- `schur.inner = tensor`,
- `schur.outer = jacobi`,
- `coupled = False`.

That means the current default outer preconditioner for saddle problems is
Jacobi.

The interactive Laplacian benchmark script
[scripts/interactive/k0_laplacian_preconditioner_choices.py](../scripts/interactive/k0_laplacian_preconditioner_choices.py)
holds the lower mass block and the Schur-inner block fixed at `tensor`, and
then compares several Schur-outer choices:

- `jacobi`,
- `richardson-N`,
- `chebyshev-N`.

So those polynomial outers are benchmark candidates, not the production
default.

## 4. How The Preconditioners Fit Together

The mass and saddle preconditioners use the same basic building blocks.

`jacobi` is the cheap fallback: it needs only diagonal data and is always the
safe baseline.

`tensor` is the production structured route: it keeps the extracted-space block
structure exact where needed and approximates only the diagonal tensor blocks
through low-rank/tensor data.

`richardson` and `chebyshev` are polynomial operator preconditioners. In the
current code they are mainly used as outer layers for benchmark studies, with a
Jacobi smoother underneath.

For saddle problems, the picture is:

- the lower block is a mass solve in degree `k - 1`,
- the Schur-inner approximation supplies the inverse mass action that appears
  inside the Schur complement,
- the Schur-outer preconditioner then acts on the Schur operator itself.

This last point is important for the tensor route. The tensor preconditioner is
valid for mass blocks and for the inverse mass action used in `schur.inner`,
but not for `schur.outer`: after the sandwich `D M_{tensor}^{-1} D^T`, the
result is no longer a tensor mass block on the Schur space.

This is why the same mass-preconditioner family shows up both in plain mass
benchmarks and inside the Laplacian saddle benchmarks.

## 5. Nullspaces And Shifted Problems

The rule is simple but important:

- unshifted singular problems use nullspace deflation,
- shifted problems do not deflate the harmonic space.

For shifted scalar solves, the right replacement is an explicit harmonic coarse
correction rather than projection. For saddle solves, the unshifted case passes
the nullspaces into MINRES; the shifted case does not.

The operator bundle stores nullspace arrays even before the real harmonic
vectors have been computed. Those zero placeholders simplify the control flow,
but they do not change the rule above.

## 6. Practical Reading Guide

- Read [docs/mrx_primer.md](mrx_primer.md) for the finite-element and operator
  background.
- Read [docs/mass_preconditioners.md](mass_preconditioners.md) for the
  degree-by-degree tensor mass structure.
- Read [docs/tensor_preconditioner_primer.md](tensor_preconditioner_primer.md)
  for the tensor-preconditioner mechanism and why it does not extend directly
  to `schur.outer`.
- Read [docs/laplacian_preconditioner_notes.md](laplacian_preconditioner_notes.md)
  for the settled scalar shifted-Laplacian policy.

This note is the status-quo summary. The two detailed notes above hold the more
specialized mass-side and scalar-Laplacian-side design decisions.