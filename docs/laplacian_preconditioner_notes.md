# Laplacian Preconditioner Notes

This note records the settled production policy for the Hodge-Laplacian
preconditioners after the recent `k = 0` shifted/nullspace cleanup.

The target operators are

$$
(L_k + \varepsilon M_k) u = f
$$

for shifted solves, and the Moore-Penrose inverse action of `L_k` for the
singular `\varepsilon = 0` case.

Legacy code and scripts may still refer to `hx` or `fd` for the scalar helper.
In the current implementation and in this note, the preferred term is
`tensor` for the structured scalar `k = 0` Laplacian-side preconditioner.

## 1. General Rule

The harmonic space must be treated differently in the shifted and unshifted
problems.

For `\varepsilon = 0`, the correct operation is nullspace deflation:

$$
L_k^{+} = P_{\perp} L_k^{-1} P_{\perp}
$$

on the range of `L_k`, interpreted as the pseudoinverse action.

For `\varepsilon > 0`, deflation is wrong. The harmonic mode is no longer in
the kernel; it becomes an eigenmode with eigenvalue `\varepsilon`. The correct
structure is

$$
(L_k + \varepsilon M_k)^{-1}
\approx
\left(I - Z Z^T M_k\right) B_{\perp}
\left(I - M_k Z Z^T\right)
+ \frac{1}{\varepsilon} Z Z^T,
$$

where the columns of `Z` form an `M_k`-orthonormal basis of the harmonic
subspace.

So the policy is:

- `\varepsilon = 0`: use nullspace deflation.
- `\varepsilon > 0`: use a complement preconditioner plus an explicit harmonic
	coarse correction.

## 2. Settled `k = 0` Policy

### 2.1 When the harmonic mode is known

For free `k = 0`, the preferred shifted preconditioner is:

- the structured tensor preconditioner on the complement,
- plus the exact harmonic coarse term `(1 / \varepsilon) z_0 z_0^T`.

In formula form,

$$
P_{0,\mathrm{shift}}^{-1}
=
\left(I - z_0 z_0^T M_0\right) B_{0,\perp}
\left(I - M_0 z_0 z_0^T\right)
+ \frac{1}{\varepsilon} z_0 z_0^T.
$$

This is the intended end state for ordinary shifted scalar solves.

### 2.2 When the harmonic mode is not yet known

During inverse iteration, or in any early shifted solve before a valid stored
coarse vector exists, the solver must stay robust without a nullspace guess.

The current implementation therefore uses this rule:

- inverse iteration explicitly disables the harmonic coarse correction;
- inverse iteration for `k = 0` uses `richardson-2` with a tensor inner
  smoother;
- inverse iteration for `k >= 1` uses the saddle route with
  `schur.outer=richardson-2` and terminal `schur.inner=tensor`;
- the shifted free `k = 0` tensor branch falls back to shifted Jacobi until a
	real coarse vector is available;
- once the coarse vector is available, ordinary shifted solves switch back to
	the tensor-plus-coarse path.

This is the key control-flow split in the codebase now.

### 2.3 What not to do

An `L^{-1}`-based expansion such as

$$
L^{+} - \varepsilon L^{+} M L^{+}
$$

is not a complete shifted preconditioner on the full space. It is only a
complement approximation. On the harmonic mode it misses the required
`1 / \varepsilon` scaling entirely.

The same applies to variants like

$$
L^{+} - \varepsilon L^{+} M L^{+} + \varepsilon M^{-1}.
$$

Those expansions may be useful as complement refinements, but they are not a
replacement for the explicit harmonic coarse term.

## 3. Settled `k = 3` Policy

For shifted `k = 3`, the practical production default remains conservative:

- use shifted Jacobi on the complement,
- add the explicit harmonic coarse correction when the harmonic mode is known.

The scalar-duality / tensor-duality round trip is still available as an
experimental upper-block ingredient, but it is not the default practical
recommendation at this point.

So the current reading is:

- `k = 0`: tensor complement preconditioner is a production-quality building
	block.
- `k = 3`: explicit coarse handling is production-quality; the transferred
	scalar-duality correction is still experimental.

## 4. Nullspace Construction Policy

The nullspace arrays are now always initialized on the operator bundle, with
zero vectors as placeholders until real harmonic vectors are computed.

That simplifies the control flow, but it does **not** change the solver rules:

- unshifted solves still use nullspace deflation;
- shifted solves still avoid deflation;
- inverse iteration still disables harmonic coarse correction while it is
	discovering the harmonic vectors.

There is now one additional explicit restriction:

- nullspace inverse iteration must not use Chebyshev preconditioning.

The reason is not that Chebyshev is inherently invalid. The issue is that the
current Chebyshev interval generation uses guarded spectral bounds, and the
reliable lower-end estimate should come only after the harmonic space has been
deflated. Nullspace construction is exactly the process that discovers that
space, so it stays on the Richardson path instead.

Once the nullspace is available, ordinary Chebyshev Laplace preconditioners may
use the deflated guarded-Lanczos bounds from the production operator builder.

So the zero placeholders are a storage convention, not a license to treat the
shifted problem as if its harmonic space should be projected out.

## 5. Practical Summary

- Unshifted `k = 0`: deflate the constant mode and use the tensor complement
	preconditioner.
- Shifted `k = 0`, harmonic mode known: tensor complement plus exact harmonic
	coarse correction.
- Shifted `k = 0`, harmonic mode unknown: no harmonic coarse correction; use
	the robust shifted fallback until the nullspace vector is available.
- Shifted `k = 3`: Jacobi plus explicit harmonic coarse correction remains the
	practical default.

## 6. Planned Runtime-Tuning Storage

The current code builds Richardson and Chebyshev parameters inside the
preconditioner-builder closures. That is workable for one-shot setup, but it is
the wrong shape for the intended online-update workflow where the operator
payload may change under JIT and the tuning should be replaceable without
recompiling the solve.

So the planned design is:

- keep assembled reusable factors in `SequenceOperators.mass_preconds` and the
	existing Hodge-side operator fields;
- add a dedicated runtime-tuning subtree on `SequenceOperators` for tuned
	spectral data such as `omega`, `lambda_min`, and `lambda_max`;
- treat those tuned values as dynamic payload leaves, not as Python locals
	captured by the apply closures.

The intended ownership rule is:

- tensor, Jacobi, and surgery data stay on the assembly/cache side;
- Richardson/Chebyshev spectral data live on the runtime-tuning side.

That runtime-tuning subtree is expected to separate at least the following
operator families:

- plain mass operators,
- surgery-bulk mass operators,
- scalar Hodge/Laplacian operators,
- Schur-outer saddle operators.

So the tuning storage is meant to be attached to the operator bundle, but not
embedded inside the tensor-fit payloads themselves.

### 6.1 Shifted-Epsilon Simplification

The awkward case is the shifted operator

$$
L_k + \varepsilon M_k.
$$

For now, the accepted simplification is to avoid a separate cache for every
`\varepsilon` value.

Instead, the planned structure distinguishes only between:

- unshifted data,
- shifted data.

For the shifted branch, the plan is:

- tune one baseline shifted case at a small reference `\varepsilon_0`
	(for example `10^{-6}`),
- store that baseline shifted lower-bound information in the tuning payload,
- store an `eps_scale = \varepsilon / \varepsilon_0` on the operator side,
- and scale the shifted lower-end estimate online from that baseline.

So the shifted path should behave as if it carries:

- a boolean `shifted` mode,
- a baseline shifted lower-bound datum,
- and a dynamic `eps_scale` that can be changed online without rebuilding the
	tuning structure itself.

The practical intent is:

- reuse the unshifted/Laplace-style upper-spectrum information for small
	`\varepsilon`,
- treat the lower end as a small-shift correction derived from the baseline
	shifted tuning,
- and avoid per-`\varepsilon` tuning caches until they are really needed.

This is explicitly a small-`\varepsilon` policy. If future use cases require
widely varying shifts, then a more explicit `\varepsilon`-dependent tuning model
may become necessary.

### 6.2 Why This Plan

This runtime-tuning layout is meant to support the following workflow:

- exchange the matvec payload online,
- update `omega` or the Chebyshev interval online,
- keep the same compiled solve structure,
- and avoid recompiling only because the tuned scalar parameters changed.

So the long-term goal is not just better diagnostics. The goal is to make the
tuned iterative parameters first-class dynamic operator data.

### 6.3 Current Storage Audit

At the moment, the solver and preconditioner controls are split across four
different storage modes.

#### Persistently stored on the sequence or operator bundle

- `seq.tol`, `seq.maxiter`: default Krylov tolerance and iteration cap.
- `SequenceOperators`: assembled sparse operators, extraction operators,
  nullspaces, Jacobi diagonals, tensor factors, surgery factors, and the
  existing tensor-Hodge payload.
- `operators.mass_preconds.tensor.rank`
- `operators.mass_preconds.tensor.cp_maxiter`
- `operators.mass_preconds.tensor.cp_tol`
- `operators.mass_preconds.tensor.cp_ridge`

The current scalar `k = 0` tensor-Hodge builder reuses those same tensor-fit
controls, so there is not yet a separate scalar-Hodge tensor-configuration
payload.

#### Persistently stored in the public preconditioner specs

- `MassPreconditionerSpec.kind`
- `MassPreconditionerSpec.surgery_schur`
- `MassPreconditionerSpec.steps`
- `MassPreconditionerSpec.power_iterations`
- `MassPreconditionerSpec.damping_safety`
- `MassPreconditionerSpec.min_eig_fraction`
- `MassPreconditionerSpec.lanczos_iterations`
- `MassPreconditionerSpec.lanczos_max_eig_inflation`
- `MassPreconditionerSpec.lanczos_min_eig_deflation`
- `MassPreconditionerSpec.lanczos_min_eig_floor_fraction`
- `MassPreconditionerSpec.smoother`
- `SchurPreconditionerSpec.inner`
- `SchurPreconditionerSpec.outer`
- `SaddlePointPreconditionerSpec.mass`
- `SaddlePointPreconditionerSpec.schur`
- `SaddlePointPreconditionerSpec.coupled`

So the algorithmic tuning policy is already part of persistent spec state.

#### Passed per solve call

- `rhs`
- `k`
- `dirichlet`
- `guess`
- `eps`
- `preconditioner`
- `tol` and `maxiter` overrides
- `use_harmonic_coarse`
- `return_info`

These values are not stored back onto the operator bundle. They are just the
active call-time controls.

#### Computed into closures instead of stored persistently

These are the parameters that motivate the planned runtime-tuning refactor:

- Richardson `omega`
- runtime `lambda_max` estimates
- Chebyshev `lambda_min`
- Chebyshev `lambda_max`
- shifted-Jacobi diagonals such as the effective `L + eps M` diagonal inverse
- exact probed Schur diagonals for `schur.outer = exact_jacobi`
- the actual assembled preconditioner applies for `precond_lower`,
  `precond_upper`, Schur applies, and coupled saddle applies

So today the preconditioner *policy* is persistent, but the tuned spectral
numbers and the actual iterative apply operators are still transient builder
locals.

#### Hard-coded policy values

There are also a few values that are not really stored as configuration at
all, but are embedded as explicit policy choices in the current code:

- nullspace inverse iteration forces Richardson rather than Chebyshev,
- the nullspace shifted path currently uses `steps = 2`,
- the nullspace shifted path forces a tensor inner smoother,
- shifted nullspace construction disables harmonic coarse correction.

Those are not part of the runtime-tuning payload yet; they are fixed control-
flow decisions in the current implementation.

### 6.4 Current Implementation Status

The first small-scope JIT-table runtime-tuning slice is now implemented.

- `SequenceOperators` now carries a dedicated `runtime_tuning` subtree.
- The currently populated families are `runtime_tuning.mass`,
	`runtime_tuning.scalar_hodge`, `runtime_tuning.schur`, and
	`runtime_tuning.diffusion`.
- The public update helpers are `update_mass_runtime_tuning(...)` and
	`update_scalar_hodge_runtime_tuning(...)` and
	`update_schur_runtime_tuning(...)` and
	`update_diffusion_runtime_tuning(...)`.
- Richardson still does not store `omega`; the apply path derives it online as
	`damping_safety / lambda_max` from the stored spectral payload.

The currently validated slices are:

- `k = 3` free mass polynomial tuning, with stored `lambda_max` used by the
	Richardson mass preconditioner path,
- shifted `k = 0` scalar Hodge tuning, with stored `lambda_min` and
	`lambda_max` used by the Chebyshev path,
- shifted `k = 1` Dirichlet Schur-outer tuning, with stored `lambda_min` and
	`lambda_max` used by the Chebyshev Schur preconditioner path,
- shifted `k = 1` Dirichlet diffusion tuning, with stored `lambda_min` and
	`lambda_max` used by the Chebyshev diffusion preconditioner path.

These slices were validated with the staged interactive harness
`scripts/interactive/runtime_tuning_debug.py`, where both the mass solve and
the shifted scalar-Hodge solve converged and returned finite solutions while
consuming the stored runtime-tuning payload. The same harness now also
validates the shifted `k = 1` saddle solve with Schur-outer Chebyshev tuning;
in the current implementation that path converged with finite output and 51
MINRES iterations on the small interactive case. The shifted `k = 1`
diffusion solve is also now validated there; it converged with finite output
and 70 MINRES iterations on the same small interactive case.

One practical detail from that validation is that the `k = 3` mass polynomial
path should be configured directly as an outer polynomial spec, for example
`MassPreconditionerSpec(kind='richardson', steps=...)`. Supplying an explicit
inner smoother for `k = 3` is misleading here because the degree-normalization
logic collapses that case to the scalar inner leaf.

For the current Schur implementation, the runtime-tuning payload does not
record which `schur.inner` route was used when the outer spectral data was
estimated. Instead, one Schur combination is assumed to remain fixed for the
lifetime of a run, so stored Schur tuning is not reused across mismatched
inner operators.

The following parts are still outside this first implemented slice:

- surgery-Schur mass polynomial tuning,
- the larger shifted-`eps_scale` reuse model described above.