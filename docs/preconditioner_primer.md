# Preconditioner Primer

This note describes the **current production** preconditioning strategy in
`mrx`. It is intentionally a snapshot of what the code does today, not a record
of how we got here. For background on operators, extraction, and the de Rham
sequence, read [`docs/mrx_primer.md`](../mrx_primer.md) first. For the parallel
solver-side picture, see [`docs/dev/iterative_solver_primer.md`](dev/iterative_solver_primer.md).

The production-facing companion notes are
([`mass_preconditioners.md`](../mass_preconditioners.md),
[`preconditioner_cleanup_todo.md`](preconditioner_cleanup_todo.md)).
More experimental or historical notes live under [`docs/dev/`](dev/).

## 1. Three Preconditioner Families

Three families of preconditioner are exposed in the production code, all
applied matrix-free:

1. **Jacobi** — inverse of the assembled extracted-space diagonal. Built by
   probing the sparse matrix with unit vectors. The minimal fallback.
2. **Chebyshev** — polynomial wrapper around a cheaper inner smoother
   (typically Jacobi or a tensor block). Self-tuning: `λ_min`, `λ_max` come
   from a Lanczos iteration on the preconditioned operator
   (`_estimate_chebyshev_lanczos_bounds_apply`).
3. **Tensor** — the hand-built structured preconditioner. The production
   default for mass and the scalar `k = 0` Hodge.

Jacobi and Chebyshev are baselines; the tensor route is what new work targets.

The user-facing handles live in [`mrx/preconditioners.py`](../../mrx/preconditioners.py):
`MassPreconditionerSpec`, `SchurPreconditionerSpec`, and
`SaddlePointPreconditionerSpec`. Solver wrappers on `DeRhamSequence` accept
either string shorthands (`"auto"`, `"jacobi"`, `"tensor"`,
`"chebyshev"`, `"richardson"`) or these spec objects, and
[`mrx/operators.py`](../../mrx/operators.py) materializes them into concrete
matvec applies via `_build_mass_preconditioner_apply`,
`_build_scalar_hodge_preconditioner_apply`,
`_build_diffusion_preconditioner_apply`, and the saddle-point builders.

## 2. The Tensor Route: Schur + Tensor Block Inverses

The tensor route never tries to invert the full extracted operator directly.
The structure is:

- An exact dense **Schur complement** strips out the small extracted-space
  surgery rows.
- The remaining **bulk** lives on tensor-product index space and is inverted
  with cheap one-dimensional factors built from a low-rank fit of the diagonal
  mapped coefficient field.

The mapped diagonal coefficient fields the bulk fits target are:

- `k = 0` mass: `J`
- `k = 1` mass: `J g^{rr}`, `J g^{θθ}`, `J g^{ζζ}`
- `k = 2` mass: `g_rr / J`, `g_θθ / J`, `g_ζζ / J`
- `k = 3` mass: `1 / J`
- `k = 0` scalar Hodge / stiffness: `J g^{rr}`, `J g^{θθ}`, `J g^{ζζ}`

### 2.1 Degree-by-degree mass shape

| `k` | Schur | Bulk blocks | Inner Schur option |
|----:|-------|-------------|--------------------|
| 0 | scalar core | one scalar bulk | n/a |
| 1 | outer surgery (θ, ζ) | `r`, `θ_bulk`, `ζ_bulk` | `cp_kwargs["k1_inner_schur"]` |
| 2 | outer surgery (`r`) | `r_bulk`, `θ`, `ζ` | `cp_kwargs["k2_inner_schur"]` |
| 3 | none | one scalar bulk | n/a |

The optional inner coupled bulk Schur for `k = 1, 2` exists but is **not the
production default**: it lowers iteration counts modestly and costs more in
wall-clock time on the tested geometry family.

### 2.2 The bulk block inverse

`TensorDiagonalBlockInverseFactors` carries the data for one tensor bulk
block. The hot apply is `_apply_tensor_diagonal_block` in
[`mrx/preconditioners.py`](../../mrx/preconditioners.py).

There are three active inverse modes packed into the same struct:

- **Rank 1.** A single separable term `M_r ⊗ M_t ⊗ M_z`. Inversion is exact on
  three dense 1-D factors via `direct_inv_{r,t,z}`.
- **Rank 2.** Two separable Kronecker terms. The code uses exact Lynch
  fast-diagonalization per axis, storing `fd_V_{r,t,z}` and the reciprocal of
  the diagonal denominator `1 + λ_r ⊗ λ_t ⊗ λ_z`.
- **Rank ≥ 3.** The leading two terms define the same FD basis. Additional
  terms are projected into that basis and only their per-axis diagonals are
  added to the modal denominator. This is no longer exact for the assembled
  rank-`r` tensor model, but the apply cost stays at the same six einsums as
  the rank-2 case.

The active bulk builder treats the diagonal mapped coefficient fields as black
boxes. Analytic priors, radial baselines, and `fit_strategy` are still accepted
by some APIs for compatibility, but the assembled production bulk factors do not
currently use them.

The dense Schur complement on the surgery rows is built once at assembly time
and stored as `schur_inv`. At apply time it is just a small `nₛ × nₛ` matvec.

### 2.3 The CP fit

`_greedy_cp_terms` in [`mrx/preconditioners.py`](../../mrx/preconditioners.py)
is the active low-rank fitter. It builds rank `r` sequentially by repeatedly
fitting a rank-1 CP term to the current residual and subtracting it.

This has two practical effects:

- rank `(r + 1)` extends rank `r` monotonically, rather than refitting all
  components from scratch,
- and the resulting bulk builder can use the leading term (or leading two
  terms) as the inversion backbone while treating the rest as corrections.

The production assembled bulk path no longer divides by analytic priors before
fitting.

## 3. Saddle-Point Solves For `k = 1, 2, 3`

For `k ≥ 1` the Hodge-Laplacian solve is structurally a saddle-point system,
solved by `solve_saddle_point_minres`. The exposed handle is
`SaddlePointPreconditionerSpec` with four pieces:

- `mass`: lower-block mass preconditioner
- `schur.inner`: mass inverse used inside the Schur complement
- `schur.outer`: preconditioner on the resulting Schur operator
- `coupled`: optional full block completion

Production defaults:

- `mass = tensor`
- `schur.inner = tensor`
- `schur.outer = jacobi` (because the sandwich `D M_tensor⁻¹ D^T` is no
  longer a tensor mass block, so the tensor route is **not valid** as the
  outer; the code rejects `schur.outer = "tensor"`)
- `coupled = False`

`schur.outer` accepts polynomial wrappers (`richardson`, `chebyshev`,
`exact_jacobi`) for benchmarking; their spectral bounds are estimated via
Lanczos and cached in the `IterativeRuntimeTuning` payload on
`SequenceOperators` so they do not have to be re-estimated on every solve.

## 4. Scalar `k = 0` Hodge

The scalar `k = 0` Laplacian solve stays in scalar form with
`solve_singular_cg` (or shifted CG). The active tensor route assembles a
core-plus-bulk Schur structure built from the three diagonal stiffness channels

- `α_rr = J g^{rr}` for `K_r ⊗ M_t ⊗ M_z`,
- `α_θθ = J g^{θθ}` for `M_r ⊗ K_t ⊗ M_z`,
- `α_ζζ = J g^{ζζ}` for `M_r ⊗ M_t ⊗ K_z`.

Each channel is fit independently. The leading rank-1 terms define the per-axis
modal basis used by the bulk inverse; higher-rank terms reuse that basis and
contribute only projected diagonal corrections to the additive denominator.
The production data lives on `operators.k0_tensor_hodge_precond`. Legacy
fast-diagonal payloads (`fd_V_p_*`, `dd0_fd_scale_K`) still exist for older
debug/compatibility paths but are not the active solve route.

Nullspace handling:

- `eps == 0` (singular): deflate the harmonic mode in CG.
- `eps > 0` (shifted): do **not** deflate; if a harmonic vector is available,
  add an explicit `1/ε` coarse correction. Until that vector exists, the code
  stays on the conservative complement-only path
  (`_wrap_shifted_harmonic_coarse_correction` is gated by
  `_shifted_harmonic_coarse_ready`).

For free-boundary `k = 0`, the tensor preconditioner also projects and
regularizes its small core Schur block, but the singular solve remains
well-posed because `solve_singular_cg` explicitly projects the nullspace in the
outer Krylov iteration.

## 5. Eager Production Defaults

The production assembly path is `assemble_all_operators` in
[`mrx/operators.py`](../../mrx/operators.py), which by default eagerly builds:

- the per-degree mass tensor preconditioner with `rank = 1` for all four
  degrees,
- the legacy scalar `k = 0` FD Hodge payload via
  `assemble_tensor_hodge_preconditioner`,
- and, when the Hodge operators are updated, the production scalar `k = 0`
  tensor Hodge/stiffness preconditioner.

Those eager defaults are why `"auto"` resolves to the tensor route in normal
use: the data is already there.

Even though multirank mass assembly is active, `assemble_all_operators`
currently chooses the conservative rank-1 eager default.

## 6. Validation Posture

The current validation status of the production tensor applies
([`docs/dev/tensor_debug_findings.md`](dev/tensor_debug_findings.md)):

- The routed algebra is correct: each tensor apply matches the inverse of its
  own assembled tensor model to machine precision. Remaining error is
  model-quality, not routing.
- Mass-side multirank tensor models are active again: rank 2 is exact FD, and
  rank ≥ 3 uses diagonal-truncated corrections in the fixed FD basis.
- Scalar `k = 0` stiffness now has an assembled FD-based tensor inverse. On the
  tested rotating-ellipse family it is decisively better than Jacobi, while the
  effect of increasing rank beyond 1 is presently modest.

## 7. Quick Reference

| Solve | Default preconditioner |
|---|---|
| `M_k u = f`, all `k` | `tensor` (eager default currently rank 1 per degree; explicit assembly can request higher rank) |
| `L_0 u = f` (singular) | scalar `k = 0` tensor Hodge + nullspace deflation |
| `(L_0 + ε M_0) u = f` | scalar `k = 0` tensor Hodge + harmonic coarse correction when available |
| `L_k u = f`, `k = 1, 2, 3` | saddle MINRES, `mass = tensor`, `schur.inner = tensor`, `schur.outer = jacobi` |
| `(M_k + ε L_k) u = f`, `k = 0` | scalar CG with diffusion preconditioner (mass-tensor when assembled) |
| `(M_k + ε L_k) u = f`, `k ≥ 1` | saddle MINRES with the same diffusion building blocks |
