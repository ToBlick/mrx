# Preconditioner Primer

This note describes the **current production** preconditioning strategy in
`mrx`. It is intentionally a snapshot of what the code does today, not a record
of how we got here. For background on operators, extraction, and the de Rham
sequence, read [`docs/mrx_primer.md`](../mrx_primer.md) first. For the parallel
solver-side picture, see [`iterative_solver_primer.md`](iterative_solver_primer.md).

The companion notes in this folder
([`mass_preconditioners.md`](../mass_preconditioners.md),
[`tensor_preconditioner_primer.md`](tensor_preconditioner_primer.md),
[`laplacian_preconditioner_notes.md`](laplacian_preconditioner_notes.md),
[`tensor_debug_findings.md`](tensor_debug_findings.md),
[`surgery_schur_refactor_plan.md`](surgery_schur_refactor_plan.md))
contain more detail on individual decisions and benchmarks.

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
- `k = 0` scalar Hodge: shared surrogate field plus operator-aware bulk model

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

There are two inverse modes packed into the same struct:

- **Rank 1 (default).** A single separable term `M_r ⊗ M_t ⊗ M_z`. Inversion
  is exact on three banded 1-D mass factors via `direct_inv_{r,t,z}`. This is
  cheap and is the *backbone* in the new strategy.
- **Multi-rank shared modal basis.** When the CP fit produces several terms,
  the code builds a single shared `M`-orthonormal eigenbasis per axis and
  inverts the modal denominator `Σ_t λ_r ⊗ λ_t ⊗ λ_z` directly
  (`modal_basis_*` and `modal_denom`). This is dense in the modal coordinates
  and was the former multi-rank inverse path.

The **new "rank 1 + correction" path** is selected by
`fit_strategy="split"` together with `richardson_steps > 0`:

- The CP fit is split into `B_0 ⊙ C_0 + B_1 ⊙ C_1_tilde`, where the leading
  geometric backbone term is constrained to rank 1 and inverted exactly via
  `split_backbone_inv_{r,t,z}`.
- That backbone inverse becomes the **smoother** for a fixed number of
  Richardson iterations against the true rank-`r` forward apply
  (`_apply_tensor_diagonal_block_forward`):

  ```text
  x = backbone_inv @ rhs
  for _ in range(richardson_steps):
      r = rhs - A_full @ x
      x = x + omega * backbone_inv @ r
  ```

This is the production direction for higher-rank work: the cheap rank-1
geometry-aware backbone is preserved for inversion, and the residual is
absorbed into a small number of Richardson sweeps, instead of paying for the
shared modal basis.

The dense Schur complement on the surgery rows is built once at assembly time
and stored as `schur_inv`. At apply time it is just a small `nₛ × nₛ` matvec.

### 2.3 The CP fit

`_cp_als_3tensor` in [`mrx/preconditioners.py`](../../mrx/preconditioners.py)
is the underlying CP-ALS routine. Two input regimes feed it:

- **Plain multiplicative prior** (default). The diagonal coefficient field is
  divided by an analytic prior (e.g. major-radius factors via
  `_major_radius_prior_terms`); CP-ALS fits the residual; the modeled field is
  `P · C_fit`.
- **Split prior** (`fit_strategy="split"`). A rank-1 backbone is fit against a
  geometric channel set built by `_build_split_geometry_prior_channels`, and
  any remaining rank budget goes into a free correction channel fit on the
  residual.

The split path is what the rank-1 + Richardson approach is built on.

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
core-plus-bulk Schur structure built from operator-aware stiffness factors
(`_assemble_k0_tensor_hodge_preconditioner`). The data lives on
`operators.k0_tensor_hodge_precond`, separately from the legacy fast-diagonal
data. Stored bulk inverses are dense `core_size × core_size` Schur inverses
plus one `TensorDiagonalBlockInverseFactors` block.

Nullspace handling:

- `eps == 0` (singular): deflate the harmonic mode in CG.
- `eps > 0` (shifted): do **not** deflate; if a harmonic vector is available,
  add an explicit `1/ε` coarse correction. Until that vector exists, the code
  stays on the conservative complement-only path
  (`_wrap_shifted_harmonic_coarse_correction` is gated by
  `_shifted_harmonic_coarse_ready`).

## 5. Eager Production Defaults

The production assembly path is `assemble_all_operators` in
[`mrx/operators.py`](../../mrx/operators.py), which by default eagerly builds:

- the per-degree mass tensor preconditioner with `rank = 2` for all four
  degrees (`k0_rank = k1_rank = k2_rank = k3_rank = 2`),
- the surgery preconditioner data for `k = 0, 1, 2`,
- the scalar `k = 0` Hodge tensor preconditioner.

Those eager defaults are why `"auto"` resolves to the tensor route in normal
use: the data is already there.

The default `richardson_steps` on `TensorMassPreconditioner` is `0`. So the
**rank-1 + Richardson smoother is currently opt-in** through
`cp_kwargs={"fit_strategy": "split", "richardson_steps": N, "richardson_omega": ω}`,
not the default that ships from `assemble_all_operators`.

## 6. Validation Posture

The current validation status of the production tensor applies
([`tensor_debug_findings.md`](tensor_debug_findings.md)):

- The routed algebra is correct: each tensor apply matches the inverse of its
  own assembled tensor model to machine precision. Remaining error is
  model-quality, not routing.
- Mass-side rank-2 tensor models are mature for all four degrees.
- Scalar `k = 0` stiffness is the open inverse-construction problem: better
  fits do not yet imply better solves at higher rank.

## 7. Quick Reference

| Solve | Default preconditioner |
|---|---|
| `M_k u = f`, all `k` | `tensor` (rank 2 per degree, Schur + tensor bulk) |
| `L_0 u = f` (singular) | scalar `k = 0` tensor Hodge + nullspace deflation |
| `(L_0 + ε M_0) u = f` | scalar `k = 0` tensor Hodge + harmonic coarse correction when available |
| `L_k u = f`, `k = 1, 2, 3` | saddle MINRES, `mass = tensor`, `schur.inner = tensor`, `schur.outer = jacobi` |
| `(M_k + ε L_k) u = f`, `k = 0` | scalar CG with diffusion preconditioner (mass-tensor when assembled) |
| `(M_k + ε L_k) u = f`, `k ≥ 1` | saddle MINRES with the same diffusion building blocks |
