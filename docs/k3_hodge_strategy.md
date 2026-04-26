# k=3 Hodge-Laplacian Strategy

This note is the degree-specific plan for the `k = 3` Hodge Laplacian.
The goal is to stop treating `k = 3` as a generic upper/lower saddle problem
and instead organize the work around the two structural views that are
actually available in this codebase:

- the Hodge-dual scalar view, where `L_3` is spectrally equivalent to a
  scalar `k = 0` Laplacian on the derivative spline space with the dual
  boundary condition,
- the mixed / saddle view, where `L_3 = D_2 M_2^{-1} D_2^T` and the only
  genuinely hard non-topological block is the `k = 2` mass inverse.

The immediate implementation goal is not to invent a final preconditioner.
It is to turn these two views into a clean sequence of experiments, so we can
see whether the right route is:

- a direct HX / scalar-duality preconditioner,
- a `k = 3` saddle solve driven by the now-strong `k = 2` mass
  preconditioner,
- or a hybrid of the two.

## 1. Structural Identity

For `k = 3`, the Hodge Laplacian is

$$
L_3 = D_2 M_2^{-1} D_2^T.
$$

So there are two equivalent ways to look at the problem.

### 1.1 Dual scalar view

The current FD/HX infrastructure already uses the fact that `L_3` is the
Hodge-dual scalar Laplacian:

- the 1-D derivative-space stiffness is assembled as the Hodge dual of the
  regular scalar stiffness,
- the `k = 3` HX path maps into a `k = 0` auxiliary space with the flipped
  boundary condition,
- the same metric coefficients `J g^{ii}` drive the FD approximation for both
  `k = 0` and `k = 3`.

So the right scalar comparison is not the primal `k = 0` problem on the same
boundary condition. It is the dual scalar problem with BCs flipped.

That is the first reason `k = 3` should be studied before `k = 1` and `k = 2`:
the scalar side is already structurally understood.

### 1.2 Saddle / Schur view

At the same time,

$$
L_3 = D_2 M_2^{-1} D_2^T
$$

is exactly the Schur complement of the symmetric saddle system

$$
\begin{bmatrix}
0 & D_2 \\
D_2^T & -M_2
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

or, for shifted problems,

$$
\begin{bmatrix}
\varepsilon M_3 & D_2 \\
D_2^T & -M_2
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

This matters because we now know how to precondition `M_2` rather well. So
unlike the `k = 0` mixed idea, the lower block here is not hypothetical: it is
exactly one of the production mass-preconditioner targets we already improved.

That makes `k = 3` the first degree where an HX-like or Schur-like strategy may
actually reduce to ingredients we can already solve.

## 2. Current Debug Target

The current goal is narrower and more concrete than the earlier HX wording.
We are not trying to solve `k = 3` directly with a monolithic `3 -> 0 -> 3`
operator. We are trying to solve the symmetric saddle system with MINRES and
use a block-diagonal preconditioner whose upper block is an HX-like round trip.

The first case to study is the nullspace-free pair

- `k = 3`, `dirichlet = False`,
- dual scalar auxiliary problem `k = 0`, `dirichlet = True`.

This removes the harmonic complication from the `3`-form side and lets the
auxiliary scalar block stay strictly SPD.

### 2.1 Saddle system and Schur complement

For `eps = 0`, the `k = 3` solve is written as the saddle problem

$$
\begin{bmatrix}
0 & D_2 \\
D_2^T & -M_2
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

The second row gives

$$
\sigma = M_2^{-1} D_2^T u.
$$

Substituting into the first row gives the `k = 3` Hodge Laplacian

$$
D_2 M_2^{-1} D_2^T u = f,
$$

so the Schur complement seen by `u` is exactly

$$
S_3 = D_2 M_2^{-1} D_2^T = L_3.
$$

For shifted problems the saddle matrix becomes

$$
\begin{bmatrix}
\varepsilon M_3 & D_2 \\
D_2^T & -M_2
\end{bmatrix},
$$

and the corresponding Schur complement is

$$
S_{3,\varepsilon} = \varepsilon M_3 + D_2 M_2^{-1} D_2^T.
$$

### 2.2 What is being preconditioned

The current debug solve uses MINRES on the saddle matrix itself, not CG on a
materialized Schur complement. So the actual preconditioner is block diagonal:

$$
P^{-1}_{\mathrm{diag}} =
\begin{bmatrix}
\widetilde S_3^{-1} & 0 \\
0 & \widetilde M_2^{-1}
\end{bmatrix}.
$$

The lower block is straightforward:

$$
\widetilde M_2^{-1} = \text{tensor mass preconditioner for } M_2.
$$

The upper block is the actual experiment. We do not precondition `u` with a
direct `3`-form approximation. Instead we approximate the Schur-complement
inverse by transporting the `k = 3` residual into the dual scalar problem,
applying a `k = 0` preconditioner there, and transporting back.

### 2.3 HX-like upper block

Write the transfer operators as

$$
T_{3 \to 0} \approx \widetilde M_0^{-1} P_{03},
\qquad
T_{0 \to 3} \approx \widetilde M_3^{-1} P_{30},
$$

where

- `P03` maps `3`-form primal DoFs to `0`-form dual variables,
- `P30 = P03^T` maps `0`-form primal DoFs to `3`-form dual variables,
- `\widetilde M_0^{-1}` and `\widetilde M_3^{-1}` are mass preconditioners that
  convert those dual variables back to primal vectors.

The upper-block approximation is then

$$
\widetilde S_3^{-1}
=
T_{0 \to 3}
\, \widetilde S_0^{-1} \,
T_{3 \to 0}.
$$

This is the precise meaning of the current “HX-like round trip” language.
The goal is to test whether a good scalar preconditioner, wrapped by these
transfer operators, is already good enough as a Schur-complement preconditioner
for `k = 3`.

### 2.4 What `\widetilde S_0^{-1}` is in the current debug script

In the current debug harness, `\widetilde S_0^{-1}` is not the old FD scalar
inverse. It is a `k = 0`, `dirichlet = True` extracted scalar preconditioner.

At the moment the script uses the simplest strong version of that object:

- form the extracted scalar matrix

  $$
  S_0 = E_0 G_0^T M_1 G_0 E_0^T,
  $$

- reorder it into the standard scalar core-plus-bulk structure,
- keep the exact bulk inverse,
- and form the corresponding extracted block preconditioner.

So the current middle operator is a `k = 0` preconditioner in the same scalar
family as the `k0` debug work, not a direct dense inverse inserted for its own
sake.

## 3. What The Script Is Trying To Measure

The current `scripts/k3_preconditioning.py` experiment answers one question:

> If we solve the `k = 3` saddle problem with MINRES, precondition the lower
> block by tensor mass on `M_2`, and precondition the upper Schur block by
> `T_{0 \to 3} \widetilde S_0^{-1} T_{3 \to 0}`, does that beat a trivial upper
> Jacobi baseline?

So the comparison in the script is intentionally narrow:

- lower block is fixed to the strong `k = 2` tensor mass preconditioner,
- upper block is either plain `3`-form Jacobi or the HX-like round trip,
- the solve itself is always MINRES on the saddle matrix.

This is not yet a claim that the transfer is final or that the auxiliary block
is final. It is the first controlled test of the exact Schur-complement idea
we want for `k = 3`.

## 4. What We Found About The `k = 2` Stiffness Block

The relevant auxiliary operator on the `k = 2` side is the div-div stiffness
matrix

$$
K_2 = G_2^T M_3 G_2,
$$

which in weak form is

$$
(K_2)_{ij} = \int (\operatorname{div} \Lambda_i^2)
                (\operatorname{div} \Lambda_j^2)
                \frac{1}{J}\,dx.
$$

For mapped polar geometry the weight is the scalar field `1 / J`. That does
not mean `K_2` splits into three independent component blocks. The discrete
divergence has three component contributions, so the assembled operator is one
matrix on the full extracted `k = 2` coefficient space. But because the weight
is scalar, the tensor terms are much simpler than they would be for a generic
vector-valued operator: each CP term of `1 / J` produces the same scalar-style
threefold Kronecker stiffness structure, built from 1-D grad-grad overlaps in
the three coordinate directions.

Equivalently: the bulk tensor structure is still one operator on one extracted
`k = 2` space, but its tensor representation is closer to the scalar `k = 0`
stiffness case than to a generic coupled vector-valued matrix. The important
distinction is:

- not three uncoupled component blocks,
- but also not a generic vector-valued tensor operator with unrelated block
  terms.

### 4.1 How `K_2` shows up in the `k = 3` upper block

The inverse-side div branch used in the `k = 3` saddle solve is

$$
G_2 \, B_2 \, G_2^T,
$$

where `B_2` should approximate `K_2^{-1}` on the div-active quotient.
Using the weak derivative `D_2 = M_3 G_2`, the same object can be written as

$$
M_3^{-1} D_2 \, B_2 \, D_2^T M_3^{-1}.
$$

So the role of the `k = 2` auxiliary operator is completely local: it appears
only inside this inverse-side Schur correction for the upper `k = 3` block.

### 4.2 What the dense experiments say

The current dense debug comparisons established three points.

- The old scalar weak-gradient branch is not the relevant missing ingredient
  for the top-form problem and does not improve the `k = 3` saddle solve.
- The div-side correction `G_2 K_2^{+} G_2^T` is the first branch that gives a
  real improvement over pure `3`-form Jacobi.
- Replacing `K_2^{+}` by the dense full `k = 2` Hodge inverse,

  $$
  G_2 L_2^{-1} G_2^T,
  $$

  changes very little in practice.

The last point is expected: once the outer `G_2` maps are applied, the curl
part of the full `k = 2` Hodge inverse is invisible to the `k = 3` Schur
correction. So the useful information is already concentrated in the div-div
operator `K_2` itself.

### 4.3 Immediate consequence

The remaining missing ingredient is not another dense surrogate. It is a real
approximate inverse for the `k = 2` div-div stiffness matrix,

$$
B_2 \approx K_2^{-1},
$$

built in the same tensor / extracted spirit as the successful `k = 0`
stiffness debug preconditioner.

That is now the active path for `k = 3`: keep the upper correction in the form

$$
G_2 \, B_2 \, G_2^T,
$$

and make `B_2` into a genuine tensor preconditioner for the extracted
`k = 2` div-div stiffness matrix.

## 5. Immediate Experimental Program

### Stage 1: Nullspace-free saddle case

- Work first with `k = 3`, `dirichlet = False`, which is dual to scalar
  `k = 0`, `dirichlet = True`.
- Solve the saddle system with MINRES instead of forming `L_3` explicitly.
- Fix the lower block to the `k = 2` tensor mass preconditioner.
- Compare two upper blocks: Jacobi on the `3`-form space versus the HX-like
  round trip `T_{0 \to 3} \widetilde S_0^{-1} T_{3 \to 0}`.

This is the cleanest place to check whether the Schur-complement idea itself is
working before any nullspace handling is added back.

### Stage 2: Transfer audit

- Hold `\widetilde S_0^{-1}` fixed.
- Change only the transfer pieces.
- Test whether the observed quality is limited by the scalar block or by the
  `3 <-> 0` transport.

### Stage 3: Replace the auxiliary scalar block

- Once the round-trip structure is behaving sensibly, replace the current
  scalar block by the intended `k = 0` production/debug preconditioner rather
  than the current strongest script-local surrogate.

### Stage 4: Reintroduce the nullspace case

- After the nullspace-free case is understood, move to the free singular case
  and add the harmonic/nullspace treatment explicitly at the saddle level.
### Stage 5: BC-flipped scalar equivalence check

- On small grids, compare spectra / Rayleigh quotients of the deflated `k = 3`
  operator against the dual-BC scalar derivative-space model.
- Use that to calibrate how literally the "same as scalar with BCs flipped"
  slogan should be taken in the extracted discrete setting.

This is a verification stage, not a production path. The point is to know
exactly how much equivalence survives after extraction, boundary conditions,
and mapped geometry.

## 6. Production Gate

Promote a `k = 3` preconditioner only if it clears all three of these bars:

- it beats nullspace-deflated `jacobi` on complement-focused tests,
- it stays robust for both shifted and unshifted solves,
- it explains clearly which ingredient is doing the work: the scalar dual
  solve, the strong `M_2` block, or the transport.

If a candidate does not identify its own source of improvement, it is not ready
for production.

## 7. Working Hypothesis

The current best working hypothesis is:

- `k = 3` should be analyzed through Hodge duality first,
- the direct saddle route is now newly plausible because `M_2` is no longer a
  weak ingredient,
- the useful recursive correction is the `k = 2` div branch

  $$
  G_2 \, B_2 \, G_2^T,
  $$

  with `B_2` targeting the extracted div-div stiffness matrix,
- and therefore the next useful experiment is not another scalar transfer
  variant but a real tensor inverse for `K_2` itself.

If that experiment wins, then the `k = 3` story becomes very simple: Jacobi on
top plus a `k = 2` div-div tensor correction. If it loses, then the next task
is to understand which non-div ingredient is still missing and why the Schur
view alone is not yet enough.

## 8. Current Supported Preconditioners

The experimental surface is now much narrower than the earlier HX wording.
For the current `k = 3` work, the preconditioner families we expect to keep
supporting are:

- `none`, included only as a completeness baseline;
- Jacobi-Richardson, where `m = 1` is exactly Jacobi and larger fixed `m`
  gives a linear stationary approximate inverse;
- tensor preconditioners for mass matrices and for the scalar `k = 0`
  Laplacian-side building blocks.

For the top-form saddle solve, this means the active upper-block comparisons
are built from these pieces only:

- `upper-none`;
- `upper-jacobi`;
- Schur-side Jacobi-Richardson on

  $$
  	\tilde S_3 = D_2 \, \widetilde M_2^{-1} \, D_2^T;
  $$

- `k = 2` middle-block Jacobi-Richardson inside the div sandwich,

  $$
  D_2 \, B_{K_2} \, D_2^T;
  $$

- and the additive sum of the Schur-side and `k = 2` Richardson pieces.

The lower saddle block is fixed to the rank-3 tensor mass preconditioner for
`M_2`.

## 9. Hyperparameters

The supported families above need only a short list of tuning parameters.

### 9.1 None

There are no hyperparameters.

### 9.2 Jacobi-Richardson

For a fixed linear Richardson preconditioner

$$
x_{j+1} = x_j + \omega B (b - A x_j),
$$

the relevant hyperparameters are:

- the iteration count `m`;
- the base smoother `B`, currently Jacobi;
- the number of power iterations used to estimate the stability scale for
  `\omega`;
- the safety factor used to turn the estimated spectral radius into the final
  damping.

For the current `k = 3` saddle experiments, the same Richardson family is used
in two places:

- outside, on the Schur-side operator `D_2 \widetilde M_2^{-1} D_2^T`;
- inside the div sandwich, on the extracted `k = 2` stiffness matrix `K_2`.

### 9.3 Tensor Mass Preconditioners

For the tensor mass preconditioners, the main hyperparameters are:

- the retained tensor rank;
- the CP-ALS tolerance;
- the CP-ALS iteration cap;
- any regularization / ridge used in the low-rank fit.

In the current `k = 3` saddle experiments, the lower block is not tuned: it is
fixed to the rank-3 tensor mass preconditioner for `M_2`.

### 9.4 Scalar `k = 0` Laplacian Block

The scalar `k = 0` Laplacian-side preconditioner remains a structured tensor /
scalar building block rather than a generic sparse inverse. Its tunable
parameters should stay aligned with the same general categories:

- retained tensor or modal ranks, when a low-rank factorization is used;
- fit tolerances and iteration caps;
- any explicit nullspace or coarse-space treatment needed by the scalar free
  case.

The important simplification is that we are no longer treating the `k = 3`
story as an open-ended zoo of unrelated coarse corrections. The supported
building blocks are now `none`, Jacobi-Richardson, and tensor-structured
inverse surrogates.

## 10. Current TODO

- Drop the `k = 2` Richardson sandwich from the active `k = 3` benchmark
  surface if the higher-resolution runs keep confirming that it does not help.
- Keep `upper-none`, `upper-jacobi`, and the Schur-side Jacobi-Richardson
  family as the active block-diagonal references.
- Add a symmetric coupled saddle preconditioner as the next structural test,
  built from the current Schur-side Richardson upper approximation, the fixed
  rank-3 tensor lower block, and the explicit `D_2` coupling.
- Keep the coupling experiment compatible with outer `MINRES`, so the coupled
  preconditioner should stay symmetric positive definite rather than use an
  ad hoc triangular or indefinite block factorization.
- Revisit the `k = 2` surgery split with a hybrid inverse: keep the surgery
  Schur complement exact, but replace the explicit bulk inverse by a fixed-step
  Richardson solve on the bulk block rather than trying to build a closed-form
  bulk inverse.
- Check carefully which inner product the current `MINRES` implementation is
  actually using in the coupled experiments. At the moment the Krylov
  recurrence itself is based on Euclidean dot products with preconditioned
  norms `r^T M r`; the FEM mass matrices only enter through the optional
  nullspace projections.
- If the coupled block keeps showing one rogue mode, try a solver-level
  reformulation with the natural FEM/Riesz metric
  `\operatorname{diag}(M_3, M_2)` rather than the current Euclidean-dot-product
  `MINRES` recurrence.
- Diagnose the remaining rogue coupled mode in the actual matrix-free
  implementation, without attributing it to a dense inverse artifact. The
  current coupled prototype applies `\widetilde S^{-1}` and
  `\widetilde M_2^{-1}` as operator applies; only the tiny surgery Schur blocks
  are dense.
- The current coupled / surgery-style Richardson fallbacks are not yet
  competitive enough to treat as active methods. Keep them around as debug or
  fallback experiments, but for now prioritize the plain Schur-side
  Jacobi-Richardson family and revisit the coupled/surgery variants later.

One natural SPD coupled ansatz is the factorized form

$$
P_{\mathrm{cpl}}^{-1}
=
L^T
\begin{bmatrix}
\widetilde S^{-1} & 0 \\
0 & \widetilde M_2^{-1}
\end{bmatrix}
L,
\qquad
L=
\begin{bmatrix}
I & 0 \\
-\widetilde M_2^{-1} D_2^T & I
\end{bmatrix},
$$

with `\widetilde S^{-1}` given by the current Schur-side Richardson
approximation and `\widetilde M_2^{-1}` given by the fixed rank-3 tensor mass
preconditioner.

Implementation note: the first coupled prototype applied the lower inverse one
extra time in the `L^T` lift, i.e. it used `D_2 \widetilde M_2^{-1} z_s` with
`z_s = \widetilde M_2^{-1}(s - \widetilde M_2^{-1} D_2^T u)`, which turns the
intended coupling into a spurious `\widetilde M_2^{-2}` term. The fix is to
apply the transpose lift exactly once,

$$
z_s = \widetilde M_2^{-1}(s - \widetilde M_2^{-1} D_2^T u),
\qquad
u_{\mathrm{out}} = \widetilde S^{-1} u - D_2 z_s,
$$

so the code now matches the factorization `P_{\mathrm{cpl}}^{-1} = L^T
\operatorname{diag}(\widetilde S^{-1}, \widetilde M_2^{-1}) L` instead of
inserting an extra lower solve into the lift.