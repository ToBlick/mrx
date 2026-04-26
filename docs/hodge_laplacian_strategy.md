# Hodge Laplacian Strategy

This note is the forward-looking plan for Hodge-Laplacian preconditioners.
Unlike the mass preconditioners, the main `k >= 1` Hodge solves are naturally
saddle-point / Schur-complement problems, so the design should be organized
around two separate ingredients:

- an upper-block preconditioner for the `k`-form block,
- a lower-block preconditioner for the `(k-1)`-form mass block.

The goal of this note is not to re-summarize past experiments. That already
exists in [laplacian_preconditioner_notes.md](laplacian_preconditioner_notes.md).
The goal here is to define the next implementation strategy.

The first degree-specific strategy note is now [k3_hodge_strategy.md](k3_hodge_strategy.md).
That file should be treated as the active plan for `k = 3`, since `k = 3`
already has a concrete HX / Hodge-duality implementation path and should be
analyzed through that lens before the more generic `k = 1, 2` saddle routes.

## 1. Problem Split

For `k = 0`, the Hodge-Laplacian solve is a scalar SPD problem.

For `k >= 1`, the code already solves the inverse Hodge-Laplacian and shifted
variants through the symmetric saddle system

$$
\begin{bmatrix}
S_k + \varepsilon M_k & D_{k-1} \\
D_{k-1}^T & -M_{k-1}
\end{bmatrix}
\begin{bmatrix}
u \\
\sigma
\end{bmatrix}
=
\begin{bmatrix}
f \\
0
\end{bmatrix},
$$

and for diffusion-type solves through

$$
\begin{bmatrix}
M_k + \alpha S_k & \alpha D_{k-1} \\
\alpha D_{k-1}^T & -\alpha M_{k-1}
\end{bmatrix}
\begin{bmatrix}
u \\
\sigma
\end{bmatrix}
=
\begin{bmatrix}
f \\
0
\end{bmatrix}.
$$

So the reusable abstraction is:

- upper block: `S_k + eps * M_k` or `M_k + alpha * S_k`,
- lower block: `M_{k-1}`.

That means the next strategy should not start from "invent one monolithic Hodge
preconditioner". It should start from "build strong upper and lower block
preconditioners, then decide whether extra coupling is worth it".

## 2. Design Principles

- Keep the lower block simple whenever possible. It is a mass matrix, so it
  should reuse the mass-preconditioner work instead of creating a separate
  Hodge-specific approximation family.
- Treat the upper block as the hard part. This is where the Schur-complement
  structure and the geometry-dependent stiffness/Hodge terms live.
- Do not tune upper and lower approximations simultaneously at first. Fix the
  lower block to the strongest available mass preconditioner for degree `k-1`,
  then measure the upper block in isolation.
- Keep nullspace handling explicit and mass-orthogonal. For `eps = 0`, use
  deflation / pseudoinverse logic, and remember that for extracted block
  preconditioners the free-case nullspace may move into the Schur complement
  even when the bulk block is invertible. For shifted problems, keep the
  explicit harmonic coarse correction separate from the ordinary block
  preconditioners.
- Start with block-diagonal preconditioning. Only move to block-triangular or
  coupled block preconditioners if the uncoupled upper/lower pair stalls.

## 3. Lower-Block Strategy

The lower block is `M_{k-1}`. The current default strategy should therefore be:

- `k = 1`: lower block uses the `k = 0` scalar tensor mass preconditioner,
- `k = 2`: lower block uses the `k = 1` tensor mass preconditioner,
- `k = 3`: lower block uses the `k = 2` tensor mass preconditioner.

Practical rule:

- first benchmark with `lower_precond_kind='auto'`,
- then repeat with `jacobi` only if we need to separate lower-block quality
  from upper-block quality.

This gives a clean diagnostic split:

- if changing `lower_precond_kind` barely moves MINRES, the upper block is the
  bottleneck,
- if it moves a lot, then the saddle solve is still limited by the mass solve
  under the Schur term.

## 4. Upper-Block Strategy

The upper block is the place to use Schur-complement reasoning.

### 4.1 Scalar `k = 0`

`k = 0` is no longer just the scalar reference case. The extracted-space
preconditioner is now effectively done in the debug harness.

- The extracted stiffness `E G^T M_1 G E^T` keeps the same core-plus-bulk split
  as the scalar mass problem.
- A pure-diagonal bulk surrogate built from the three scalar fields
  `J g^{rr}`, `J g^{theta theta}`, and `J g^{zeta zeta}` already gives a strong
  dense debug preconditioner.
- In the free case, the nullspace is carried by the Schur complement, so the
  Schur solve must be deflated / pseudoinverted rather than inverted naively.

So the scalar Hodge ingredient should now be treated as available. The
remaining work is to promote it into the source implementation and reuse it as
the auxiliary scalar block for higher-degree strategies.

### 4.2 `k = 1`

The upper block is

$$
S_1 + D_0 M_0^{-1} D_0^T
$$

or its shifted / diffusion variant.

Strategy:

- freeze the lower block to the current strongest `k = 0` mass preconditioner,
- test upper-block approximations against the exact small-grid dense truth,
- keep the Schur term explicit at the operator level,
- only approximate `M_0^{-1}` through the already validated scalar mass path.

The first question is not whether HX is elegant. It is whether the resulting
upper correction is spectrally aligned with the `k = 1` complement once the
lower mass solve is already good.

### 4.3 `k = 2`

This is the direct analogue of `k = 1`:

$$
S_2 + D_1 M_1^{-1} D_1^T.
$$

Strategy:

- reuse the production `k = 1` tensor mass preconditioner as the lower block,
- study the upper block as its own Schur-complement problem,
- expect the geometry structure to resemble the `k = 1` case more than the
  scalar cases,
- only add new block surgery once dense small-grid inspections show that the
  upper block itself has a stable extracted structure worth exploiting.

### 4.4 `k = 3`

For `k = 3`, the Hodge Laplacian is already a pure Schur-type operator:

$$
L_3 = D_2 M_2^{-1} D_2^T.
$$

So `k = 3` should be treated as the clearest stress test of the whole strategy.

The shifted notes already suggest caution here:

- the scalar helper can be strong while the transferred correction is still
  weak,
- harmonic coarse correction and complement preconditioning must be diagnosed
  separately.

So the default plan for `k = 3` should be conservative:

- keep nullspace/harmonic handling explicit,
- benchmark `jacobi + coarse` style baselines first,
- only then test whether a stronger `k = 2` lower-block solve actually improves
  the `3 -> 2 -> 3` Schur correction on the complement.

## 5. Coupled Saddle Preconditioners

The first implementation target should be the block-diagonal preconditioner

$$
P_{\mathrm{diag}}^{-1}
=
\begin{bmatrix}
P_{\mathrm{upper}}^{-1} & 0 \\
0 & P_{\mathrm{lower}}^{-1}
\end{bmatrix}.
$$

Only if this stalls should we move to block-coupled variants.

The next escalation path is a triangular or approximate block-factorization
preconditioner, for example

$$
P_{\mathrm{tri}}^{-1}
\approx
\begin{bmatrix}
I & -P_{\mathrm{upper}}^{-1} D_{k-1} \\
0 & I
\end{bmatrix}
\begin{bmatrix}
P_{\mathrm{upper}}^{-1} & 0 \\
0 & P_{\mathrm{lower}}^{-1}
\end{bmatrix}
\begin{bmatrix}
I & 0 \\
-D_{k-1}^T P_{\mathrm{upper}}^{-1} & I
\end{bmatrix},
$$

or the lower/upper ordering analogue.

This should be a second phase, not the starting point. Otherwise it becomes
too hard to tell whether the real issue is:

- the upper block,
- the lower block,
- or the block coupling itself.

## 6. Experimental Stages

### Stage 1: Upper/Lower Separation

- [ ] For `k = 1, 2, 3`, benchmark the saddle solve with `lower_precond_kind='auto'` and with lower Jacobi only.
- [ ] Record how much the stronger lower mass preconditioner changes MINRES iterations.
- [ ] Use that comparison to identify whether each degree is upper-limited or lower-limited.

### Stage 2: Small-Grid Upper Truth Models

- [ ] On small grids, build dense upper-block truth operators for `k = 1, 2, 3`.
- [ ] Compare `apply_hodge_laplacian_approx` against the exact upper truth on random vectors.
- [ ] Separate nullspace/harmonic directions from complement directions in those comparisons.

### Stage 3: Degree-Specific Schur Structure

- [ ] For `k = 1` and `k = 2`, inspect whether the upper block has a stable extracted-space surgery/core split analogous to the mass problems.
- [ ] For `k = 3`, inspect whether the dominant failure is still the transferred `2 -> 3` correction rather than the coarse mode.
- [ ] Only introduce new block reorderings or surgery models if the dense truth shows a repeatable structure.

### Stage 4: Upper-Block Preconditioner Prototypes

- [ ] Start from the current upper Jacobi/HX approximations as the baseline.
- [ ] Replace only the lower mass inverse inside the Schur term first, using the best available tensor mass preconditioner.
- [ ] If that is insufficient, test upper-block Schur approximations that preserve the exact derivative couplings and only compress the diagonal tensor blocks.

### Stage 5: Saddle Coupling Upgrades

- [ ] Keep the default saddle preconditioner block-diagonal until the previous stages are understood.
- [ ] If block-diagonal stalls, test one triangular/block-factorized variant at a time.
- [ ] Compare upper-triangular and lower-triangular orderings separately instead of combining changes.

### Stage 6: Production Gate

- [ ] Promote a new Hodge-Laplacian preconditioner only after the upper/lower split is understood for that degree.
- [ ] Keep the interactive debug harness for each degree as the regression benchmark after promotion.
- [ ] Require complement-focused evidence, not just harmonic-mode improvements.

## 7. Acceptance Criteria

- Lower-block improvements should be visible in standalone mass benchmarks first.
- Upper-block approximations should agree with dense small-grid truth well enough to explain the saddle solve behavior.
- For `eps = 0`, results should be judged on the deflated complement, not on the nullspace direction.
- For shifted problems, the explicit harmonic coarse term should be tested separately from the ordinary upper/lower preconditioners.
- Production promotion should require iteration-count wins on random right-hand sides, not just favorable operator-fit plots.

## 8. Near-Term Plan

The sensible near-term order is:

1. Use the now-stronger tensor mass preconditioners as the default lower block.
2. Re-run the Hodge experiments with that lower block frozen.
3. Identify which degree is still limited by the upper block.
4. Start a degree-specific upper-block checklist from the first case that is clearly upper-limited.

That keeps the next Hodge work aligned with what already worked on the mass side:
preserve the correct Schur structure, improve only the expensive inverse-like
pieces, and validate each abstraction layer before coupling everything back
into the full solve.