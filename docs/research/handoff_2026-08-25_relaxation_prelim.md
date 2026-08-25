# Relaxation preliminary: logical-profile IC, min-B^2 descent, and what is wrong with L-BFGS

2026-08-25. Branch `relaxation-prelim` off `greville-prod` @ 76bf5f3.
Preliminary and exploratory: a report on what works, not a feature.

All numbers below are from GPU runs on `quasr44970` (nfp=3, geometry loaded
from `data/quasr_0044970.h5`), `ns=(8,16,8)`, `p=3`, `n2_dbc=2192`, H100,
float64. Jobs 16769367 (pilot), 16769981 / 16769984 / 16769990 (250-step
arms), 16770040 / 16770416 / 16770741 (helicity diagnosis), 16771114 /
16771115 (helicity-corrected repeat and hyperregularisation).

---

## 0. Summary

* **Relaxation toward min B^2 works.** `||F||` falls 7.06e-01 -> 2.87e-01 in
  250 steps, energy decreases monotonically on 250/250 steps in every healthy
  arm, and `div B` stays pinned at its initial 3.68e-04.
* **L-BFGS as shipped is a pessimisation**, not merely unhelpful: it removes
  *less* energy than plain steepest descent (0.2089% vs 0.2497%). There are
  **two independent defects**, and **fixing only the obvious one makes it
  worse.** Fixed, L-BFGS removes 0.5571%, 2.2x steepest descent.
* **CG has a third, milder defect** (the Polak-Ribiere formula uses the
  previous search DIRECTION where the previous GRADIENT belongs), worth ~25%.
* **Two bugs found outside the brief's scope**, both by identity tests:
  `compute_helicity` solves with a right-hand side in the wrong space and is
  wrong by a factor ~1150 (FIXED here); and `minres`'s documented `info`
  convention was inverted relative to the code (fixed on greville-prod as
  9bf044b by the coordination session after I hit it).
* **Not verified:** one geometry, one resolution, one IC, 250 steps, no
  convergence-to-equilibrium, no p- or h-refinement. See section 8.

### AND THE BIGGEST FINDING, which is in PART II -- read section 19 first

* **The maximal linesearch step destroys field-line topology.** Several runs
  relaxed from cleanly nested surfaces into full chaos. The mechanism is
  measured, not guessed: `ANALYTIC_LINESEARCH` takes the largest step that
  still lowers E, and the O(dt^2) frozen-flux error of explicit Euler then acts
  as numerical reconnection. **The clean experiment is LR3 vs W5** -- same
  geometry, IC, optimizer and step count, differing only in the step -- where
  the small step loses **58x less helicity per unit energy removed** and keeps
  its surfaces. The linesearch buys ~2x energy per step and pays 58x in
  topology.
* **Almost nothing the scheme enforces noticed.** Energy monotone 3000/3000,
  `||div B||` 6.7e-14, the linesearch identity at 6.5e-17, the harmonic
  amplitude at 1e-16 -- all satisfied while the physics was destroyed.
* **But one diagnostic does detect it**, and it was in the traces all along:
  the ABSOLUTE helicity change `|dH|` at `||B||_M = 1`, monotone with a blind
  visual classification of every Poincare pair, survivors and casualties
  separated ~30x. Normalising by H -- which is what I had been doing --
  destroys the signal, since H spans three orders of magnitude here.

**Sections 12 and 17 are SUPERSEDED.** 12 called the chaos universal and
fundamental; 17 blamed `beta_max = 13%`. Both are wrong and section 19 says why
with the numbers.

**What Tobias must decide** is in section 9, as revised by 19.5.

---

## 1. Which IC route, and why

**Used: the logical-profile IC.** `B_hat = (0, Phi'(iota - lam_z), Phi'(1 + lam_c))`
in the reference 2-form frame, built exactly as `scripts/debug/logical_profile_ic.py`
builds it (imported from it, not copied), with `iota` 0.4 -> 0.9, `iota_exp` 2,
`dPhi = rho^1`, `lambda = 0`, and the GEOMETRY loaded from
`data/quasr_0044970.h5` via `build_gvec_map`.

Structural gates at the IC, all as `logical_profile_ic.py` predicts:

    ||div B||_L2                 3.676e-04
    ||P_Leray B - B||            2.719e-06
    max|B^rho| / max|B^zeta|     2.965e-04
    E                            5.000000e-01   (B normalised to ||B||_M = 1)
    ||F||_M                      7.0552e-01     (large, and expected to be)

**The GVEC Clebsch route is blocked, and not by anything in the code.**
`scripts/debug/gvec_clebsch_ic.py` reads
`data/gvec_nfp3_hegna_80cubed_clebsch.h5`. That file does not exist on this
host: a search for `*clebsch*` and `gvec_nfp3_hegna*` across `/scratch` and
`/kfs3/scratch` returns only the script itself. It is the only export carrying
the `clebsch/dPhi_dr`, `clebsch/dchi_dr`, `clebsch/LA` groups the route needs.
Every other file in `data/` has `B`; three (`quasr_0009983.h5`,
`quasr_0044970.h5`, `w7x_vacuum_co_contra.h5`) additionally have
`B_contra`/`B_cov`; none has a Clebsch group. **Route 2b is blocked on data,
not code.** Independently confirmed by the coordination session.

B interpolation (the fallback) was not used.

The `frame='ref'` trap was avoided as instructed: `omega_ref` is pushed forward
explicitly (`DF @ omega / det DF`) and handed to `load(..., 2, dirichlet=True)`
in the physical frame, because `load(frame='ref')` wants `g omega / J` and
accepts `omega` silently.

---

## 2. The identity everything rests on, and the trap inside it

`ANALYTIC_LINESEARCH` is not a heuristic step size. Along the ray `B + t dB`
with `dB` FROZEN the energy is exactly quadratic,

    E(B + t dB) = E(B) + t <B, dB>_M + (t^2/2) ||dB||_M^2

so the minimiser is `t* = -<B, dB>_M / ||dB||_M^2`. The code uses
`dt = (F, u)_M / ||dB||_M^2`. These agree iff

    <B, dB>_M = -(F, u)_M                                              (*)

which is provable in steps that each name a different piece of the code:

    <B, curl E>_M2 = (weak_curl B, E)_M1     [curl adjointness]
                   = (J, u x H)               [E = M1^-1 load(u x H)]
                   = -(u, J x H)_L2           [triple product]
                   = -(u, F)_M2               [Leray M-orthogonality, div u = 0]

The first step is exact at the matrix level, not merely formally:
`apply_strong_curl(E) = M2^-1 D1 E` and `apply_weak_curl(B) = M1^-1 D1^T B`, so
both sides are `B^T D1 E`.

So the per-step energy drop has a closed form, `dE = -dt (F,u)_M / 2`, and
comparing it against the MEASURED drop tests curl adjointness,
`cross_product_load`'s sign and argument order, and Leray M-orthogonality
simultaneously, against zero. Measured (healthy arms): median 1.3e-11 relative,
**5.6e-17 against the energy scale**.

### THE TRAP

Because `dt` is the exact minimiser of an exactly-quadratic function along the
ray, and `t` is free to go NEGATIVE, **the energy decreases monotonically for
ANY direction `u`** — an ascent direction, a random vector, anything.

**"Energy decreased monotonically" therefore does not validate a descent
method here.** It is a claim that cannot fail. Do not report it as evidence
that an optimizer works; it is evidence that the linesearch is a linesearch.

What discriminates is the SIZE of the drop, i.e.

    cos_M(F, u) = (F,u)_M / (|F|_M |u|_M)

**but read it as a breakage detector, not a quality ranking.** cos = 1 means
"behaving exactly like steepest descent", which is what an unaccelerated method
does; a good CG or L-BFGS direction is SUPPOSED to differ from the gradient.
In the table below `cg-legacy` has the *higher* cos (0.79) and performs
*worse* than `cg` (0.15), precisely because it stays closer to plain gradient
descent. Do not optimise for cos. Breakage looks like cos ~ 0 or negative
together with an exploding `dt` and a SPIKING `gain`.

(`gain = ||dB||_M / ||u||_M`, the amplification of `C_B: u -> curl(u x H)` on
the chosen direction, and free to compute since `||dB||^2 = (F,u)/dt` exactly.
An earlier draft of this document said breakage shows up as a *collapsed*
gain. That is wrong and the measured numbers say so: the broken L-BFGS arm's
gain ROSE, median 6.96e+01 and max 9.53e+03 against 2.72e+01 for healthy CG.
What collapsed was the MAGNITUDE of `u` and of `dB` -- to ~3e-14 and ~3e-11
once the clamp annihilated the gradient term -- not the ratio between them.)

### AND A SECOND TRAP, in the other direction: ||F|| IS NOT GUARANTEED TO FALL

`F` **is** the gradient of the objective being minimised. A descent method
promises the OBJECTIVE decreases; it promises nothing at all about the norm of
its own gradient along the path. `||grad||` rises routinely on a perfectly
correct descent trajectory — through a narrowing valley, past a region of
higher curvature, any time the landscape steepens on the way down.

So an early rise in `||F||` is not an anomaly, not a discretisation artefact,
and not evidence about the IC. It requires no explanation, and "I could not
determine why it rises" would be just as wrong, because it implies a gap where
there is none.

    ENERGY monotonicity is the guarantee.  ||F|| is a diagnostic.

Report `||F||`, plot it, use it to rank arms at the end. Never gate on it,
never assert it is monotone, and do not explain excursions in it. Every
comparison below therefore leads with ENERGY REMOVED and carries `||F||` as
supporting evidence.

**Method note, earned the hard way:** two of us independently started
inventing a physical cause for this non-anomaly before Tobias pointed out that
the quantity was never guaranteed to trend that way. *Before explaining a
trend, check whether the thing was ever guaranteed to trend.*

---

## 3. Operator health on this base

Run before any IC and any optimizer, on random inputs, because the base was
known to be red at the time (six failures, all in `test/test_projectors.py`,
none on this path):

    curl_adjoint_rel   4.022e-13     <B, strong_curl E>_M2 == <weak_curl B, E>_M1
    leray_div_rel      4.708e-12     strong_div(P_Leray v) == 0
    leray_orth_rel     1.576e-14     P_Leray is M2-orthogonal
    div_curl_rel       1.261e-10     strong_div . strong_curl == 0

`seq.tol` is 1e-12 and each of these routes through mass solves, so these ARE
the round-off floor.

**`div_curl_rel` was 1.3e-10 because of an operator choice I got wrong; it is
now 8.6e-16.** I used `apply_strong_curl = M_2^-1 D_1` and then described its
residual as a property of the discretisation — "the floor on how well `div B`
can be held, by anyone". **There is no such floor.** Measured, same geometry:

    div_curl_rel_massproj        1.261e-10
    div_curl_rel_incidence       8.641e-16     <- machine zero
    curl_massproj_vs_incidence   1.025e-12     <- so the swap is free

`dB = curl E` now uses `apply_incidence_matrix`. Three consequences, all
measured: `div B` is conserved EXACTLY rather than to the mass solver's
tolerance, one Krylov solve leaves the hot path, and the trajectory does not
move (the two curls agree to 1e-12).

**Why the wrong one got picked — the same trap as §6.2.**
`DeRhamSequence.apply_incidence_matrix`'s docstring still says the raw
incidence breaks `d.d` under polar extraction and that the mass-projected
`strong_*` forms "should be preferred when exact `d.d = 0` on extracted DoFs
is required". `mrx/operators.py:2039`, the code that actually runs, says its
Gram correction at the polar axis makes the incidence form exact everywhere. I
read the docstring on the method I was calling and believed it. That is two
stale docstrings in `mrx/` that each cost real work in a single day; a docs
pass should grep for claims about `d.d` and about solver return conventions.

---

## 4. Does relaxation toward min B^2 work?

Yes. 250 steps, `history_size=1`, `gamma=0`, `ANALYTIC_LINESEARCH`, EXPLICIT:

| arm | energy removed | \|\|F\|\| final | cos median | sy<0 | dt<0 | E up | s/step |
|---|---|---|---|---|---|---|---|
| gradient | 0.2497% | 5.210e-01 | +1.0000 | 0/250 | 0/250 | 0/250 | 1.24 |
| cg (corrected beta) | 0.5553% | **2.883e-01** | +0.1489 | 0/250 | 0/250 | 0/250 | 1.13 |
| cg-legacy beta | 0.4239% | 3.455e-01 | +0.7918 | 0/250 | 0/250 | 0/250 | 1.08 |
| lbfgs-legacy (shipped) | 0.2089% | 4.576e-01 | +0.2462 | 31/250 | 31/250 | 1/250 | 1.28 |
| lbfgs-paired (lag fixed only) | 0.1044% | 2.032e+00 | -0.0000 | 135/250 | 135/250 | 0/250 | 1.15 |
| lbfgs (both fixed) | **0.5571%** | 3.273e-01 | +0.1404 | 0/250 | 0/250 | 0/250 | 1.08 |

`||F||` 7.0552e-01 -> 2.883e-01 for the best arm. `div B` max 3.676e-04 in
every arm, i.e. unchanged from the IC. The linesearch identity holds at
1e-11 relative / 1e-16 absolute in every healthy arm.

**a. ANALYTIC_LINESEARCH works and energy decreases monotonically** — 0/250
increases in every arm except `lbfgs-legacy`, which manages 1/250. But see
section 2: monotonicity here is nearly vacuous, and the single legacy increase
is interesting only because it takes a broken direction to achieve it (see
section 5).

**b. CONJUGATE_GRADIENT works** and is the best arm on final force
(2.883e-01). The corrected beta beats the shipped one by ~25% on energy
removed.

**c. L-BFGS: see section 5.**

Convergence is SLOW in absolute terms — 0.56% of the initial energy in 250
steps — and no arm reached an equilibrium. This study did not attempt to run
to convergence.

---

## 5. What is wrong with L-BFGS

### The root cause: the pair is not in the space the direction is used in

The iteration is

    B_{k+1} = B_k + dt * C(B_k) u_k,      C(B) u = curl(u x H(B))

and the direction the two-loop recursion returns is consumed as a **velocity**
`u`. The gradient identity in the docstring, `grad_M E = -F`, is a derivative
with respect to `u`, not with respect to `B`: the admissible variation is
`dB = C u` and `dE[dB] = -(F, u)_M`. So the step in the descent variable is
`dt * u`.

The shipped code stores `s_k = B_{k+1} - B_k`, which is `dt * C u` — a
different vector in a different space. The implied secant operator for the
velocity parameterisation is `A = -DF . C`, mapping velocity to velocity, and
`H_k F ~ A^{-1} F = dt * u`; that is the self-consistent choice. Storing the
B-increment instead means the recursion adds B-increments to a velocity.

**Defect 1 (space).** `s = B_{k+1} - B_k` instead of `s = dt * u`.

**Defect 2 (lag).** `y_k = F_prev - F` was pushed at the END of the step,
alongside the brand-new `s_k`. But `y` is a difference over the step that
already happened, so it belongs with `s_{k-1}`. As shipped, `y` lagged its
paired `s` by exactly one step.

### THE FACTORIAL, AND THE TRAP IN IT

| variant | s | y | energy removed | \|\|F\|\| final | sy<0 |
|---|---|---|---|---|---|
| `legacy` (shipped) | B-increment | lagging | 0.2089% | 4.576e-01 | 31/250 |
| `paired` | B-increment | aligned | **0.1044%** | **2.032e+00** | **135/250** |
| `velocity` (fix) | `dt*u` | aligned | **0.5571%** | 3.273e-01 | **0/250** |

**Fixing ONLY the obvious off-by-one makes L-BFGS WORSE THAN LEAVING IT
ALONE.** Energy removed halves, `||F||` *grows* from 7.06e-01 to 2.03e+00, and
negative curvature goes from 31/250 steps to 135/250.

The mechanism: correcting the pairing while `s` remains in the wrong space
produces a *more consistently* wrong secant. The lag was partially CANCELLING
the space error.

**This is a live trap.** Anyone who notices the off-by-one, fixes it in
isolation, measures a regression and backs it out will conclude the off-by-one
was correct. It is not. **Both defects have to go together**, and the space
one is the load-bearing half.

### The full causal chain, with a number at every link

From a single legacy trace:

    space mismatch + lag
      -> sy = -1.330e-17            negative curvature; H_k not positive definite
      -> gamma = sy/yy < 0          clamped to 1e-30 by jnp.maximum(gamma, 1e-30)
      -> the gradient's contribution to the direction is annihilated, leaving
         only the stored-s combination from the second loop
      -> cos_M(F,u) = -0.0000       the "descent direction" is orthogonal to F
      -> ||dB|| collapses to solver-noise level
      -> dt explodes to -2.639e+07  to compensate
      -> the quadratic line model stops holding, and energy RISES:
         dE_meas = +7.855e-08 against dE_pred = -3.796e-07

The energy rise is NOT a violation of the exact-line-minimiser identity. It is
that `dB` had become numerical noise, so there was no meaningful ray to
minimise along. The identity holds at 1e-12 in both healthy arms and degrades
only in the arm where `dB` has collapsed — which is why the summary reports the
discrepancy against the energy scale as well as relatively (a direction
generating no `dB` makes `dE_pred ~ 0`, and the relative measure then divides
by nothing, which is a statement about the direction, not the operators).

### The clamp

`_lbfgs_direction` computes `gamma = sy/yy` and then
`gamma = jnp.maximum(gamma, 1e-30)`. This is the masked-invariant pattern the
house rules forbid. When curvature is negative the recursion is not merely
inaccurate — `H_k` is not positive definite and what it returns need not be a
descent direction at all; flooring the scaling does not repair that, it
silently annihilates the gradient term. The standard remedy is to SKIP or damp
the update when the curvature condition fails, loudly.

Measured firing rate (`sy <= 0`): **31/250 legacy, 135/250 paired, 0/250
fixed.** It fires, it fires a lot, and it stops entirely once the pairs are
right.

It was left in place while the `legacy` arm still existed, so the A/B
reproduced shipped behaviour faithfully. **It has since been DELETED** (see
section 9), and `sy` is surfaced every step as a new `State` field instead. No
`skip on sy <= 0` guard was added in its place: with the pairing fixed, a
correct run keeps `sy > 0` (0/250 measured), so a guard there would be exactly
the kind of defensive code that turns a future regression into "slow
convergence". The `1e-30` terms that remain in `_lbfgs_direction` are a
different thing and are kept — they cover the first step, where the history is
exactly zero by construction and the recursion is meant to fall back to
steepest descent.

### history_size

Asked in the brief. At m=5: fixed L-BFGS is unchanged (0.5582% vs 0.5571%),
CG is bit-identical (it only ever reads `state.v`, so `history_size` is inert
for CG by construction), and legacy L-BFGS gets **worse** (0.1273%, `||F||`
grows to 7.95e-01). `history_size` does not interact badly by itself; it
amplifies whatever the pairs already are. The fixed version gaining nothing
from m=5 suggests the pairs go stale fast, which is expected since `A = -DF.C`
changes as `B` moves.

---

## 6. Two bugs found outside the brief's scope

### 6.1 `compute_helicity` solved with a right-hand side in the wrong space

`apply_inverse_hodge_laplacian` solves the saddle form

    | S    D   | | A |   | f |
    | D^T  -M  | | s | = | 0 |

in which `f` is a **dual** k-form. `compute_helicity` passed
`seq.apply_weak_curl(B)` = `M1^-1 D1^T B`, a **primal** 1-form — one mass
inverse too many. `apply_leray_projection`, solving the same kind of system,
gets it right: it passes `apply_derivative_matrix(...)`, not
`apply_strong_div(...)`. Measured on the same field with the same solver:

    primal rhs (shipped):  ||A||=3.4486e+00  ||B - curl A||/||B|| = 8.5569e+01  H = +1.991417e+01
    dual rhs   (correct):  ||A||=1.0214e-02  ||B - curl A||/||B|| = 9.7380e-01  H = +1.726811e-02

**H was wrong by a factor of ~1150.** FIXED in `mrx/relaxation.py`.

It failed *silently and reproducibly*: three different initial guesses on the
identical field agreed to 8.4e-13. A reproducibility check passes it cleanly.
What catches it is an **identity**: in the Dirichlet complex `B_harm` is a
genuine harmonic remainder and MUST satisfy `||B_harm|| <= ||B||`. **85.6 is
not a fraction of anything.**

#### The absolute-vs-relative Betti trap, which I walked into

I first read `betti_numbers=(1,1,0,0)` from `build_sequence` as `b2 = 0`,
concluded there are no harmonic 2-forms at all, and therefore that the
corrected 0.974 was *also* wrong. That was my error. Those are the **absolute**
Betti numbers of a solid torus. `compute_helicity` works in the **Dirichlet**
complex, whose relative cohomology satisfies

    b_k^rel = b_{3-k}^abs

so `b2^rel = b1^abs = 1`. There IS exactly one harmonic 2-form, our IC is
dominated by net toroidal flux which IS that mode, and **97.4% harmonic is the
physically right answer.** The next reader will make the same misreading; the
relation is not visible from the call site.

Note also that `compute_helicity` = +1.73e-02 against the IC's own closed-form
natural-gauge value of +2.54e-03 is a legitimate gauge difference (the harmonic
1-form ambiguity on a torus, as `logical_profile_ic.py` documents). A factor
~6.8 is a gauge difference; a factor 7800, which is what the shipped code gave,
is not.

### 6.2 `minres`'s documented `info` convention was inverted

`mrx/solvers.py:401` said `"info: 0 if converged, >0 = number of iterations if
not converged"` while line 546 returns
`jnp.where(converged_final, -k_final, k_final)`. So `info = -468` means
**converged in 468 iterations**, and a reader following the docstring concludes
the solve failed. I nearly did.

The coordination session found on checking that a sibling function in the same
file had had this exact docstring corrected on 2026-08-24, with a note that the
stale version "caused converged solves to be read as failures" — but `minres`
itself, the one actually called here, was left live. The same trap caught two
people on two consecutive days. Fixed on greville-prod as **9bf044b** (not by
me; `mrx/solvers.py` was outside my claim).

---

## 7. Implicit midpoint: an opinion, and a defect nobody has hit yet

Not attempted, as instructed. Three things for whoever picks it up.

**There is a fourth L-BFGS/CG defect waiting on that path.**
`midpoint_picard_step` runs `_relaxation_step` as its Picard body, and
`_relaxation_step` pushes an `(s, y)` pair and overwrites `state.v` on **every
Picard iteration**. So with `IMPLICIT_MIDPOINT` plus CG or L-BFGS, the descent
history fills with inner Picard iterates rather than time-step differences;
with the default `history_size=1` and `picard_k_restart=20`, the surviving pair
is the last Picard iterate's. This is independent of the two defects in section
5 and is not fixed here. **It materially raises the cost estimate for implicit
stepping**, which is information worth having before scheduling it.

**`ANALYTIC_LINESEARCH` and IMPLICIT_MIDPOINT are structurally incompatible as
written.** `dt` is computed from the current `dB` inside the step, but under
midpoint `dB` depends on `B_{n+1}`, which depends on `dt` — a nonlinear scalar
equation coupled into the Picard loop. One of the two has to give.

**The value proposition is conservation, not stability.** With an exact line
minimiser there is no stability limit being fought, so the reason to want
implicit midpoint is its exact preservation of quadratic invariants. Whether
that is worth the cost should be decided by first measuring how badly EXPLICIT
actually drifts — which needs the corrected `compute_helicity` from section
6.1, since every helicity number produced before that fix was meaningless.

---

## 8. What I did NOT verify

* **One geometry, one resolution, one IC.** `quasr44970` at `ns=(8,16,8)`,
  `p=3`, `iota` 0.4->0.9, `lambda = 0`. No W7-X, no cylinder, no `p` or `h`
  refinement, no `lambda != 0` arm. None of the optimizer conclusions has been
  shown to survive a change of geometry or resolution.
* **No run to convergence.** 250 steps removes 0.56% of the initial energy and
  reaches no equilibrium. The ranking of CG vs fixed L-BFGS (0.5553% vs
  0.5571%, and 2.883e-01 vs 3.273e-01 on final force) is **well inside what a
  longer run could reorder** — I would not call a winner between those two.
  The separation from `gradient` and from the broken L-BFGS variants is large
  and is not in doubt.
* **The GVEC Clebsch IC** was never run (missing data file, section 1), so
  nothing here says whether it works.
* **Helicity conservation** is verified only on this one configuration and
  only over 250 steps (section 10.1). Every helicity number produced BEFORE
  the section 6.1 fix is void, including those in earlier logs of this study.
* **Hyperregularisation** was tested at exactly one setting, `gamma=1`,
  `mu=1e-3`, on one geometry. No `mu` sweep, no `gamma=2`, and no check that
  the `M + eps L` preconditioner behaves at larger `mu` or higher `p`.
* **The reversal in 10.3** (gradient beating the accelerated methods once
  hyperregularised) rests on a single arm on a single geometry and should be
  confirmed before it is acted on.
* **The `1e-30` guards** in `_lbfgs_direction` other than the `gamma` clamp
  (the `rho_i` denominators) were not individually audited; they are reachable
  only when a history slot is exactly zero, which happens on the first step.
* **`dirichlet_H=True`** was never exercised; all runs used the default `False`.
* **Stochastic mode, resistivity, and the noise schedule** were not exercised
  (`eta = 0`, `stochastic=False`) .

---

## 9. What Tobias must decide

1. **The missing Clebsch export.** `data/gvec_nfp3_hegna_80cubed_clebsch.h5`
   is not on this host and is the only file carrying the groups
   `gvec_clebsch_ic.py` needs. Route 2b cannot be tested until it is restored.
   This is the one item nobody here can unblock.
2. **Whether to keep chasing L-BFGS or settle on CG.** They are level and
   cannot be separated by this study (section 10.2). CG is simpler, has no
   curvature condition to violate, and its `history_size` is inert by
   construction. Fixed L-BFGS gained nothing from m=5. If something has to be
   dropped, CG is the cheaper thing to keep.
3. **Hyperregularisation vs acceleration** (section 10.3). At `gamma = 1`
   plain gradient becomes the best arm and the acceleration advantage
   disappears. If that holds up on other geometries, the two are partly
   alternatives rather than complements, and that changes where effort should
   go. Worth one confirming run on a second geometry before believing it.
4. **`gamma > 0` combined with a quasi-Newton direction** silently produces
   ascent directions (section 10.4). The linesearch hides it. Needs a decision
   if that combination is ever used in production.

Two things that were open when this study started and are now settled, so they
do NOT need a decision: the `gamma` clamp in `_lbfgs_direction` has been
deleted, and the `lbfgs_pairing` / `cg_beta` knobs I added to make the
factorial decisive in a single job have been deleted too, per the standing
rule that production stays clean and known-bad paths are not kept as knobs.
**Commit `ecfa3ef` is the one that still carries them** if the factorial ever
needs re-running; everything after it has only the corrected path.

---

## 10. Corrected helicity, and hyperregularisation

### 10.1 Helicity is conserved, once the diagnostic is fixed

Repeat of the `gamma=0` arms with the section 6.1 fix in place (job 16771114).
Energy reproduces section 4 to four digits, confirming the helicity fix is
diagnostic-only and does not touch the descent path:

| arm | energy removed | \|\|F\|\| final | helicity drift |
|---|---|---|---|
| gradient | 0.2497% | 5.2105e-01 | **-1.095e-05** |
| cg | 0.5554% | 2.8692e-01 | -4.216e-04 |
| lbfgs | 0.5572% | 2.9315e-01 | -4.809e-04 |

**Helicity is conserved to 1.1e-05 relative over 250 gradient steps**, against
a spurious +1.558e-01 for the same arm before the fix. The scheme does
preserve the ideal invariant; the 15-30% "drift" seen in every run before
section 6.1 was entirely the broken diagnostic. The accelerated arms drift
~40x more than `gradient`, which is consistent with their taking larger and
less physical steps, but 5e-04 over 250 steps is still small.

### 10.2 RUN-TO-RUN REPRODUCIBILITY, and why no winner is called

The `lbfgs` arm's final `||F||` is **2.9315e-01 here against 3.2734e-01 in
section 4** — the same computation, same m=1, same IC, different job. CG by
contrast reproduces across three jobs to 0.5% (2.8833 / 2.8693 / 2.8692e-01).

So the final `||F||` is reproducible only to ~10% for L-BFGS run-to-run (XLA
reduction-order differences amplified through 250 steps of a stiff nonlinear
iteration), while the integral quantity — energy removed — is stable to four
digits. **Neither metric separates CG from fixed L-BFGS**: 0.5553/0.5554% vs
0.5571/0.5572% on energy is a 0.3% gap, and the `||F||` gap is inside the
reproducibility noise. Do not read a winner out of these two. The separation
from `gradient` and from the broken L-BFGS variants is 2x or more and is not
in doubt.

### 10.3 Hyperregularisation looks NEEDED

The brief says to keep `gamma = 0` and report on whether it looks needed, and
sections 4-5 are all `gamma = 0`. One arm at `gamma = 1`, `mu = 1e-3` was run
to answer the question with data (job 16771115):

| arm | gamma | energy removed | \|\|F\|\| final | s/step | helicity drift |
|---|---|---|---|---|---|
| gradient | 0 | 0.2497% | 5.210e-01 | 1.22 | -1.1e-05 |
| **gradient** | **1** | **0.5228%** | **1.016e-01** | 4.26 | -2.8e-05 |
| cg | 0 | 0.5554% | 2.869e-01 | 1.12 | -4.2e-04 |
| **cg** | **1** | 0.5013% | **1.247e-01** | 4.07 | +7.5e-04 |
| lbfgs | 0 | 0.5572% | 2.932e-01 | 1.09 | -4.8e-04 |
| **lbfgs** | **1** | 0.4675% | **2.445e-01** | 3.94 | +6.7e-04 |

The force residual — the quantity that actually measures approach to
equilibrium — improves **5.1x for gradient** and 2.3x for CG. It also wins per
unit wall-clock, not just per step: `gamma=1` CG passes `||F|| ~ 1.8e-01` by
step 100 (~407 s) whereas `gamma=0` CG needs all 250 steps (~280 s) to reach
only 2.87e-01. Helicity conservation is unaffected.

**The `M + eps L` solve gave no trouble at `mu = 1e-3`**, costing ~3.1 s/step
on top of the ~1.1 s/step baseline. Whether that holds at larger `mu`, higher
`p`, or on W7-X was not tested.

**Note the ordering REVERSES.** At `gamma = 0` the accelerated methods beat
gradient 2.2x; at `gamma = 1` plain gradient is the BEST arm on final force
(1.016e-01), ahead of CG (1.247e-01) and L-BFGS (2.445e-01). Hyperregularising
appears to condition the problem enough that acceleration stops paying — which
would make "fix L-BFGS" and "add hyperregularisation" partly ALTERNATIVE routes
to the same end rather than complementary ones. On one geometry at one `mu`,
this is a hint and not a result.

### 10.4 A real interaction: gamma > 0 voids the descent guarantee for L-BFGS

At `gamma = 1` the `lbfgs` arm took a step with `dt < 0` — i.e. an ASCENT
direction — on 5/250 steps, while `sy > 0` on 250/250. That is not a curvature
failure.

`apply_regularization` is applied to `u` AFTER the direction is formed, so the
direction actually used is `u = R H_k F` with `R = (I - mu*Delta)^{-1}`. Both
`R` and `H_k` are SPD, but **the product of two SPD operators need not be SPD**
in the relevant inner product, so `(F, R H_k F)_M` can be negative. Plain
gradient is safe because there `u = R F` and `R` alone is SPD.

The exact linesearch absorbs it (a negative `dt` still minimises along the
ray, so energy still fell on 250/250 steps), which is precisely why this would
never show up as a visible failure. If `gamma > 0` and a quasi-Newton
direction are ever used together in production, this needs a decision rather
than silence.

---

## 11. Reproduction

    sbatch slurm/job_relax_prelim.sh --geometry quasr44970 --ns 8,16,8 --p 3 \
        --steps 250 --arms gradient,cg,lbfgs --out out/relax_prelim/main.json

    # hyperregularisation (section 10.3)
    sbatch slurm/job_relax_prelim.sh --geometry quasr44970 --ns 8,16,8 --p 3 \
        --steps 250 --gamma 1 --mu 1e-3 --arms gradient,cg,lbfgs \
        --out out/relax_prelim/gamma1.json

    # re-render a saved field at high resolution, NO relaxation.  MEASURED:
    # 100 seeds x 400 periods x 8 zeta planes, both fields, ~10 min TOTAL --
    # ~6 min of that is sequence setup and ~1 min per field is the trace.
    # I twice estimated this at 1-2.5 h by scaling from a small trace whose
    # 300 s was almost entirely JIT COMPILATION rather than integration.
    # Do not budget a replot from a single small trace; the compile dominates
    # it.  Extra zeta planes are free -- they reuse the trace.
    sbatch slurm/job_relax_prelim.sh --geometry w7x-fmm002 --ns 8,16,8 --p 3 \
        --poincare-from out/relax_prelim/W1/B.h5 --pc-seeds 100 \
        --pc-periods 400 --pc-saves 16 --pc-steps 48 \
        --pc-zeta 0,0.125,0.25,0.375,0.5,0.625,0.75,0.875 --out .../hi.json

    # operator identities / IC gates only, no descent (cheap, ~6 min)
    sbatch slurm/job_relax_prelim.sh --geometry quasr44970 --ic-only \
        --out out/relax_prelim/ic.json

**The factorial in section 5 needs commit `ecfa3ef`**, which still carries the
`--arms cg-legacy,lbfgs-legacy,lbfgs-paired` variants and the
`lbfgs_pairing` / `cg_beta` knobs behind them. They were deleted afterwards
(section 9); on the current HEAD those arm names do not exist.

`slurm/job_relax_prelim.sh` is NOT committed — `.gitignore` carries
`slurm/job_*`. It is a stock `gpu-h100` job with one non-obvious line that
matters for reproducibility:

    export PYTHONPATH="$WORKTREE:$PYTHONPATH"

The venv carries an EDITABLE install of `mrx` whose finder points at
`/kfs3/scratch/tblickhan/mrx/mrx`, i.e. the MAIN checkout. That finder is
appended to `sys.meta_path`, so it resolves AFTER the `sys.path` finder —
putting the worktree on `PYTHONPATH` is what makes `import mrx` pick up the
code under test. **Without it a worktree job silently runs the main line's
`mrx` and every A/B in it is void.** `relax_prelim.py` prints `mrx.__file__` as
its first line so the log always says which copy ran.

`data/` is gitignored; the worktree has it as a symlink to
`/scratch/tblickhan/mrx/data`.

---
---

# PART II — findings after the first write-up

Sections 1-11 above were written before Tobias supplied the finite-beta Clebsch
exports and before the Poincare sections were rendered. Everything below is
later and, where it conflicts with the above, supersedes it.

## 12. THE FIELD RELAXES INTO CHAOS — the most important result here

`w7x-ini-clebsch`, Clebsch IC, 3000 CG steps, `gamma=0`. Poincare of the
initial and final fields, 40 seeds x 150 periods:

    quantity              IC          final
    lines lost           1/40        15/40
    axis offset       1.433e-04    1.027e-01
    h/2 drift         1.800e-03    1.600e+00
    B^zeta/|B| min      +0.520       +0.137

Every field line fills the volume; the logical chart is uniformly filled, so
`B^rho` is large everywhere. **All nested surfaces are destroyed.** The h/2
drift at 1.6 turns means the field-line integration is not even converged.

### NOT ONE GUARANTEED INVARIANT NOTICED

    energy monotone            3000/3000 steps
    ||div B|| max              6.685e-14
    G1 linesearch identity     6.5e-17 against the energy scale
    helicity drift             -5.8e-03

Everything this scheme enforces was satisfied while the physics was destroyed,
because **nested surfaces are protected by none of them**. A diagnostic suite
can be complete with respect to what a scheme conserves and still be blind to
whether the answer is any good. That lesson generalises past this study.

### Working hypothesis — testable, NOT asserted

The scheme is an energy **descent**, not an ideal **flow**. Explicit Euler on
`dB/dt = curl(v x B)` preserves frozen-in flux only to O(dt^2) per step, and
`ANALYTIC_LINESEARCH` deliberately takes the LARGEST step that still lowers E.
That error acts as **numerical reconnection**, applied 3000 times.

What makes this more than a story: the survivors are exactly those Taylor's
argument says survive reconnection — helicity (0.58% drift), `div B` and the
harmonic flux — while the ideal Casimirs that hold surfaces together are not
conserved at all. A Taylor state in stellarator geometry generically has islands
and chaos. **So the chaos may be the correct answer to the variational problem
the code is solving, rather than the one we want solved.** If so,
`ANALYTIC_LINESEARCH` is fundamentally at odds with ideal relaxation: it
maximises progress per step and pays in topology.

Controls run to separate this (results in the run directories): a much smaller
fixed step (`--dt-mode fixed --dt0 1e-3`), which should retain surfaces if
numerical reconnection is the mechanism; and `gamma=1`, testing whether
smoothing the velocity suffices. A roughness diagnostic `||J||/||B||` separates
"physically chaotic but smooth" from "numerically shredded at the grid scale".
**Resolution has NOT been tested and is the first thing to check.**

### The counter-example: `dzeta` behaves exactly as predicted

`B_hat = (0,0,1)` has an exactly known target AND an exactly known energy floor,
because the harmonic amplitude is conserved by construction:

    dB = curl E is an EXACT form, and a harmonic 2-form of the Dirichlet
    complex satisfies D_1^T h = 0, so with D_1 = M_2 G_1
        <h, dB>_M2 = h^T M_2 G_1 E = h^T D_1 E = (D_1^T h)^T E = 0

identically, every step, for any E. That component IS the net toroidal flux, so
by the M-orthogonal split `||B||^2 = ||B_harm||^2 + ||B_exact||^2` the descent
can only remove the exact part:

    E_min = (1/2) cos^2 ||B||_M^2 = (1/2)(0.682018)^2 = 0.2325746

Measured: `E` fell 0.5 -> 0.2438 by step 1560, still descending — 87% of the
removable energy gone, 4.8% above the predicted floor.

**This is why the field cannot relax to B = 0, and it is NOT helicity.** `dzeta`
has zero shear, so its helicity is zero analytically (measured -2.48e-03) and a
helicity-based argument would permit collapse. The flux invariant explains the
case where the obvious answer fails.

Corroboration from two independent solve chains: at the `dzeta` IC the harmonic
remainder from the k=1 Hodge decomposition (6.8202e-01) and the alignment
against `compute_nullspaces`' harmonic vector (0.682018) agree to five digits.

## 13. The Clebsch IC validated end to end on finite beta

Route 2b was blocked on a missing file at the start of this study; Tobias
supplied two finite-beta W7-X exports and it became the best-validated route
here. `gvec_clebsch_ic.py` rebuilds B from three scalars, pushes it through OUR
force operator, and compares the recovered pressure with GVEC's stored profile:

| file | beta_mean | iota ratio vs file | \|\|F\|\|/\|\|B\|\| | pressure shape residual |
|---|---|---|---|---|
| `w7x_ini_00000000` | 5.8% | +0.2000 (= 1/nfp) | 4.76e-02 | **8.93e-02** |
| `w7x_fmm002` | 1.8% | +0.2000 (= 1/nfp) | 1.88e-02 | **4.51e-02** |

The value is in the **conjunction** — map, representation and force operator
validated together, on a finite-beta target, which is much harder than a vacuum
field. A reader skimming "4.5%" will not see that unless told.

**`1/nfp` measured three independent ways**, being the factor most likely to be
wrong by inheritance: (1) finite differences of `LA` against the stored
derivatives, 6.2602 vs `2pi` and 1.2535 vs `2pi/nfp`; (2) `dchi_dr/dPhi_dr` runs
+0.86 -> +1.00, W7-X's published full-turn iota; (3) the reconstructed-to-file
ratio is +0.2000 at every radius. On `fmm002` the file's `dchi/dPhi` is
**negative** (-0.93 -> -1.05) with the reconstruction tracking it sign and all,
ratio still +0.2000 — closing the mirror-flip question `gvec_clebsch_ic.py`'s
own docstring flagged as open. `dchi_dr/dPhi_dr` is a flux function to 6.7e-16.

**`load_clebsch` had a silent convention bug.** It dropped the last point on
every periodic axis unconditionally — right for hegna (closed sample, duplicated
endpoint), wrong for these (half-open, like quasr), where it discards real data
and mis-registers the wrap. Now decided from the data with the same 1e-8 cut
`_periodic_axis` already used for the map: `|LA(0) - LA(1)| = 6.9e-02` here
against ~1e-16 for a genuine duplicate.

**The IC must be Leray-cleaned.** `dB = curl E` preserves `div B` exactly, so
the IC's divergence is carried for the whole run. The Clebsch IC is div-free in
the reference frame but the L2 projection through `M_2` reintroduces
`||div B|| = 2.7e-02` at `ns=(8,16,8)` — seventy times the logical IC's 3.7e-04.
Cleaning once up front takes it to 6.5e-14 and it stays ~1e-12 throughout.

**A prediction of mine failed.** I said the multiplier's shape would stay
consistent with the file; over 3000 steps it drifted 8.94e-02 -> 3.16e-01. The
prediction was invalid, not the run: convergence to `J x B = grad p` does not
pin down WHICH equilibrium. GVEC prescribes `p(s)` and `iota(s)` and solves for
shapes; this scheme fixes topology, minimises energy, and lets `p` be whatever
the multiplier turns out to be. GVEC's state being a fixed point says nothing
about it being a minimum. Candidates not distinguished: (a) it is a saddle of
our functional, (b) discretisation error, (c) the run went chaotic (section 12)
and the multiplier followed.

## 14. Hyperregularisation, resistivity, warm starts

### gamma > 0 helps the force residual a lot, and reverses the ranking

`gamma=1`, `mu=1e-3`, 250 steps, `quasr44970`:

| arm | gamma | energy removed | \|\|F\|\| final | s/step |
|---|---|---|---|---|
| gradient | 0 | 0.2497% | 5.210e-01 | 1.22 |
| **gradient** | **1** | **0.5228%** | **1.016e-01** | 4.26 |
| cg | 0 | 0.5554% | 2.869e-01 | 1.12 |
| **cg** | **1** | 0.5013% | **1.247e-01** | 4.07 |
| lbfgs | 1 | 0.4675% | 2.445e-01 | 3.94 |

5.1x better on force for gradient, 2.3x for CG, at 3.5x cost per step — and it
wins per unit wall-clock too. The `M + eps L` solve gave no preconditioner
trouble at `mu = 1e-3`. **The ordering REVERSES**: at `gamma=0` the accelerated
methods beat gradient 2.2x; at `gamma=1` plain gradient is best. That would make
"fix L-BFGS" and "add hyperregularisation" partly ALTERNATIVE routes. One
geometry, one `mu`: a hint, not a result.

### gamma > 0 voids the descent guarantee for quasi-Newton

At `gamma=1` the `lbfgs` arm took an ASCENT step on 5/250 while `sy > 0` on
250/250. Not a curvature failure: the direction is `u = R H_k F` with
`R = (I + mu L)^-1`, and **the product of two SPD operators need not be SPD**,
so `(F, R H_k F)_M` can go negative. Plain gradient is safe (`u = R F`). The
exact linesearch absorbs it silently, so it would never surface as a failure.

### Warm starts are worth about 1%

Zeroing `p, p_v, H, JxH, E` between steps changes only each Krylov solve's
starting vector, so the trajectory is unchanged and only cost differs:

    warm  1.16 s/step   energy removed 0.5682%   ||F|| 3.111e-01
    cold  1.17 s/step   energy removed 0.5682%   ||F|| 3.075e-01

**0.9%.** The solves are well enough preconditioned that the starting vector
barely matters — a real simplification opportunity, within the envelope
`quasr44970`, `ns=(8,16,8)`, `p=3`, `gamma=0`.

### Resistivity breaks the linesearch, predictably

With `eta > 0` the step is `dB = curl(u x H - eta J)`, so
`<B,dB>_M = -(F,u)_M - eta ||J||^2_M1`, but `ANALYTIC_LINESEARCH` computes
`dt = (F,u)/||dB||^2` and **omits the resistive term**. `dt` then under-steps.
Energy still falls (the omitted term is sign-definite), so this is a loss of
optimality, not correctness; the G1 identity breaks by exactly
`dt*eta*||J||^2`.

## 15. Decisions, revised

**Section 12 comes first.** Is the chaos acceptable physics for this variational
problem, or must the scheme become an actual ideal flow? If the latter,
`ANALYTIC_LINESEARCH` is the thing to give up — a design decision, not tuning.
Test resolution before anything else.

Then: CG vs fixed L-BFGS (level, unresolvable here); hyperregularisation vs
acceleration; `gamma > 0` with quasi-Newton silently producing ascent
directions; and whether the warm-start plumbing can go.

## 16. Also not verified

Everything in section 10, plus: the chaos has not been tested against
resolution; no `mu` sweep; the W7-X warm/cold repeat and the `gamma>0`
warm-start case were still running; and a mislabeled diagnostic shipped for
several hours (the residual printed volume-average brackets while dividing by
the L2 norm, which is the correct pairing — numbers were always right). Worth a
line because a mislabeled diagnostic is one step from someone recomputing a
correct quantity to "fix" it.

## 17. THE CHAOS IS NOT UNIVERSAL — section 12 substantially revised

Section 12 reported the `w7x-ini-clebsch` run relaxing into chaos and called it
the headline. More cases have since finished and the picture is different.

### `w7x-fmm002` keeps its surfaces, and looks like real physics

Same IC route, same 3000 CG steps, same everything except the file:

| run | file | beta_mean | \|\|F\|\|/\|\|B\|\| at IC | h/2 drift final | axis offset IC -> final | iota final |
|---|---|---|---|---|---|---|
| **W1** | `w7x_fmm002` | 1.8% | 1.88e-02 | **2.6e-04** | 2.548e-04 -> 3.071e-03 | 0.833 -> 1.050 |
| LR3 | `w7x_ini` | 5.8% | 4.76e-02 | 1.6e+00 | 1.433e-04 -> 1.027e-01 | 0.008 -> 1.031 |

W1's final section has **nested surfaces across the whole volume**, an
integrator that is CONVERGED (h/2 drift 2.6e-04, against 1.6 for LR3), and
clean **island chains at the 5/6, 10/11 and 5/5 resonances** — physical
magnetic islands at rational surfaces, which is what a genuine 3-D equilibrium
has. This is the relaxation working.

**So "the scheme is fundamentally broken" is not supported.** Whatever happens
in LR3 is specific to that case.

### What distinguishes them, and what that suggests

`w7x_ini` has `beta_mean` 5.8% and `beta_max` **13%**, and its IC sits 2.5x
further from equilibrium than `fmm002`'s. A high-beta equilibrium can be
ideally UNSTABLE, and an energy-minimising relaxation would find the
instability and destroy the surfaces — which would be physics, not a bug.
That is now the leading hypothesis for LR3. It is NOT established: the
numerical alternatives (step size, unregularised high-k content, resolution)
are being tested on that same case by W4 (`gamma=1`) and W5 (small fixed dt),
and resolution remains untested.

### Two corrections to section 12's reasoning

* **h/2 drift does not discriminate.** I offered `h/2 drift = 1.6` as evidence
  the integration was broken. For a genuinely chaotic field, O(1) drift under
  step-halving is EXPECTED — Lyapunov divergence — so it separates "chaotic"
  from "not chaotic" and says nothing about whether the chaos is numerical.
  The `||J||/||B||` roughness diagnostic is the one that discriminates, and it
  postdates the runs in section 12.
* **Unconverged is not broken.** The `dzeta` run converged to
  `cos = 0.991348` against the harmonic field — it IS going to the right
  answer — and its section is still chaotic, because 13% of its norm is not yet
  harmonic and a 13% perturbation on a vacuum field overlaps islands easily. A
  chaotic section on an unconverged intermediate state is not evidence of a
  broken scheme. None of these runs is converged.

The section-12 lesson that **no guaranteed invariant detects a bad answer**
still stands, and is still the most transferable thing in this document. What
does not stand is the inference from it that the scheme was fundamentally
wrong.

## 18. Resistivity, and three more geometries

### 18.1 eta works, and breaks the linesearch by exactly the predicted term

`quasr44970`, logical IC, `eta_max = 1e-4`, tanh schedule tapering to 3.5e-10:

    energy removed     0.6798% in 4000 steps   (0.6493% in 6266 steps at eta=0)
    ||F||              7.055e-01 -> 6.492e-02
    helicity drift     -6.8e-03                (-3.4e-03 at eta=0)
    G1 identity        median 1.830e-01        (1.3e-11 at eta=0)

Resistivity does what it is supposed to: it relaxes the topological
constraint, so helicity drifts about twice as much and roughly 1.6x more
energy becomes accessible per step.

**The G1 break was predicted before the run and confirmed in size and sign.**
With `eta > 0` the step is `dB = curl(u x H - eta J)`, so
`<B,dB>_M = -(F,u)_M - eta||J||^2_M1`, while `ANALYTIC_LINESEARCH` computes
`dt = (F,u)/||dB||^2` and omits the resistive term. The measured drop is
therefore MORE negative than the predicted one -- `dE_meas = -3.898e-04`
against `dE_pred = -3.798e-04` on step 1 -- and `dt` UNDER-steps. Energy still
falls, since the omitted term is sign-definite, so this is a loss of
optimality rather than of correctness. Anyone using `eta > 0` with
`ANALYTIC_LINESEARCH` should know the step is not the line minimiser.

### 18.2 Three geometries, ranked by how well the surfaces survived

All Clebsch or logical ICs, 3000 CG steps, `gamma = 0`, `ns=(8,16,8)`, p=3.

| run | geometry / IC | axis offset IC -> final | h/2 drift final | verdict |
|---|---|---|---|---|
| **W1** | `w7x_fmm002` clebsch, beta 1.8% | 2.548e-04 -> 3.071e-03 | **2.6e-04** | nested, islands at 5/6, 10/11, 5/5 |
| W2 | `w7x` (W7-X.h5) logical, invented | 1.195e-04 -> 5.828e-02 | -- | degraded, iota 0.848-0.991 -> 0.299-1.059 |
| LR3 | `w7x_ini` clebsch, beta 5.8% | 1.433e-04 -> 1.027e-01 | 1.6e+00 | chaotic |

W2 is worth noting separately: it is an INVENTED field (`iota` 0.17 -> 0.20
prescribed) on a real W7-X map, so it starts far from equilibrium and its
degradation is much less surprising than LR3's, which started from GVEC's own
solution.

`W3` (`dzeta` on W7-X) is not tabulated because its IC section is already
degenerate -- `iota = 0` by construction makes the poloidal angle about the
magnetic axis nearly undefined, and the IC Poincare is correspondingly poor.
It remains useful for the harmonic-amplitude test in section 12 and useless as
a picture.

## 19. WHAT DESTROYS THE SURFACES: step size, measured

Tobias read all the Poincare pairs and classified them **BLIND** -- he gave the
verdicts (good / mostly nice / mostly chaos / pure chaos) from the plots alone,
before seeing any of the traces and before I had computed `|dH|` for any run.
Cross-referencing that classification against the traces afterwards is what
makes what follows evidence rather than pattern-matching: the ordering below
could not have been contaminated by the numbers, because the numbers did not
exist when the ordering was made.

It settles the question sections 12 and 17 left open, and overturns two of my
own claims.

### 19.1 The classification, against the numbers

`||B||_M = 1` in every run, so absolute helicity changes are comparable.

| run | geometry / IC | dt | dE | \|dH\| | \|dH\|/dE | Poincare |
|---|---|---|---|---|---|---|
| W1 | `fmm002` clebsch | linesearch | 1.22e-04 | 1.54e-06 | 1.3e-02 | good -> good |
| **W5** | `w7x_ini` clebsch | **fixed 1e-3** | 2.09e-03 | 2.00e-06 | **9.6e-04** | nice -> mostly nice |
| LR4 | `quasr` logical | linesearch | 3.24e-03 | 5.93e-05 | 1.8e-02 | mostly chaos |
| W3 | `w7x` dzeta | linesearch | 2.67e-01 | 6.42e-05 | 2.4e-04 | chaos -> more chaos |
| LR3 | `w7x_ini` clebsch | linesearch | 4.22e-03 | 2.33e-04 | 5.5e-02 | **pure chaos** |
| W2 | `w7x` logical | linesearch | 4.13e-03 | 3.00e-04 | 7.3e-02 | **pure chaos** |

### 19.2 ABSOLUTE helicity change predicts surface loss; RELATIVE drift does not

Ordered by `|dH|`, the classification is monotone, with the survivors and the
casualties separated by a factor ~30:

    1.54e-06  good        6.42e-05  chaos
    2.00e-06  mostly nice 2.33e-04  pure chaos
    5.93e-05  mostly chaos 3.00e-04  pure chaos

**Relative drift is actively misleading here.** W1 has the LARGEST relative
drift of any run (-8.66e-03) and perfect surfaces, because its helicity is
itself tiny (-1.78e-04). Helicity spans three orders of magnitude across these
cases, so dividing by it destroys the signal.

This partly closes the gap section 12 opened. There IS a scalar diagnostic that
detects surface destruction — it is `|dH|` at fixed `||B||_M`, and it was in
the traces the whole time, misread because it was normalised.

### 19.3 STEP SIZE IS THE CONTROL VARIABLE — the clean experiment

**LR3 vs W5**: same geometry, same IC, same optimizer, same step count. The
only difference is that W5 takes a fixed `dt = 1e-3` instead of the linesearch
step. Everything moves together:

    quantity                       LR3 (linesearch)   W5 (dt = 1e-3)
    energy removed                    0.8448%            0.4179%
    ||F|| reduction                     5.6x               2.7x
    |dH| per unit energy removed      5.5e-02            9.6e-04     (58x less)
    pressure shape vs GVEC        8.9e-02 -> 3.2e-01  8.9e-02 -> 2.8e-02
    roughness ||J||/||B||               --            1.62 -> 1.37 (smoother)
    Poincare                        pure chaos         mostly nested

The maximal linesearch step buys about **2x more energy per step and pays 58x
in topology**. That is a bad trade, and it is the mechanism behind everything
in sections 12 and 17.

**It also rescues the prediction section 13 recorded as failed.** With small
steps the Leray multiplier converges TOWARD GVEC's pressure (3.2x closer)
instead of away (3.5x further). The prediction was right; LR3's large steps
were numerically reconnecting the field away from the equilibrium, and the
"failure" was a symptom of that, not a flaw in the reasoning.

### 19.4 Two of my earlier claims, withdrawn

**Beta is not the discriminator.** Section 17 proposed that `w7x_ini`'s
`beta_max = 13%` made it ideally unstable and that the relaxation was finding
the instability. W5 is the same file at the same beta and keeps its surfaces.

The refutation is by MECHANISM, not by measurement, which is why it is
decisive: an ideal instability is a property of the equilibrium and its
spectrum. It cannot depend on the integrator's time step. If beta-driven
instability were destroying the surfaces, shrinking `dt` would change how fast
they were destroyed and not whether. It changed whether. **Refuted.**

**W1 is much weaker evidence than I presented it as.** Section 17 said W1 shows
"the relaxation working". W1 removed **0.0244%** of the initial energy -- it
began near equilibrium (`||F||/||B|| = 1.9e-02`) and barely moved. Its
reconnection per unit energy (1.3e-02) is no better than the chaotic LR4's
(1.8e-02). Its surfaces survived because it did not go anywhere. The nested
section and the clean island chains are still worth having as evidence that the
*discretisation* represents such a field faithfully, but they are not evidence
that the *relaxation* preserves topology.

### 19.5 What this means for the scheme

`ANALYTIC_LINESEARCH` maximises energy removed per step, and the quantity it
spends to get there is exactly the topology. Options, in the order I would try
them:

1. **Cap the step.** A safety factor on the linesearch `dt`, chosen so `|dH|`
   per step stays under a threshold. Cheap, and `|dH|` is already measured.
2. **Gate on `|dH|` directly** -- reject or halve a step that costs more than a
   budget. This is a control loop on a measured invariant, not a fudge.
3. **Give up the linesearch** for a genuinely small fixed or adaptive step, and
   accept the ~2x cost in energy per step.

Not yet known: whether the trade improves with resolution (S01-S03, S13-S14) or
with `gamma > 0` (S04-S06, W4), both of which were running when this was
written.

## 20. WHICH W7-X FILE EACH RUN USED, AND WHY IT MATTERS

The two finite-beta exports are not the same kind of object, and I had been
treating them as though they were. From their own attributes:

| | LR3 and W5 | W1 |
|---|---|---|
| file | `w7x_ini_00000000_clebsch_mrx.h5` | `w7x_fmm002_clebsch_mrx.h5` |
| `gvec_source` | `W7X_ini_State_0000_00000000.dat` | `GVEC_State_final.dat` |
| axis_R | **5.5 exactly** | 5.5359528014861 |
| beta mean / max | 5.8% / 13.0% | 1.85% / 3.69% |
| IC `\|\|F\|\|/\|\|B\|\|` | 4.76e-02 | 1.88e-02 |

**`w7x_ini` is GVEC's INITIAL GUESS, not a converged equilibrium.** "ini",
iteration `0000_00000000`, and an axis at exactly `R = 5.5` with no Shafranov
shift -- that is an input number, not a solution. `fmm002` is a converged final
state, and its axis carries fourteen significant figures because it was solved
for.

Three things follow, and they correct earlier sections:

* **The "GVEC's state is a fixed point of our flow" argument does not apply to
  LR3.** Section 13 leaned on `J x B = grad p` holding at GVEC's state. It does
  not hold at an unconverged guess, so the pressure-drift test was never
  meaningful on that file. Section 13's "failed prediction" is void as stated.
* **Distance from equilibrium is a second control variable** alongside step
  size. LR3 travels far *because* its IC is not an equilibrium, and travelling
  far under maximal steps is exactly what maximises numerical reconnection.
  Section 19 identified the step; this identifies what makes the step
  expensive.
* **Section 19.4's demotion of W1 was too harsh in one respect.** W1 removes
  only 0.0244% of the energy, which is still the reason its `|dH|` is small --
  that part stands. But it is not "barely doing anything": see below.

### 20.1 W1 is the strongest result in this study

Starting from GVEC's CONVERGED equilibrium, 3000 CG steps:

    ||F||                  1.8768e-02 -> 9.9498e-05      189x
    residual               1.0214e-01 -> 6.1451e-04      166x
    PRESSURE vs GVEC       4.503e-02  -> 2.133e-02       2.1x CLOSER
    energy removed         0.0244%
    surfaces               preserved, with islands at 5/6, 10/11, 5/5

The energy removed is tiny because there was almost none to remove; what the
run actually did was drive the FORCE RESIDUAL down by 166x while holding the
topology and moving the pressure profile toward the file. That is our scheme
refining GVEC's own solution, and it is the cleanest evidence here that the
discretisation, the force operator and the descent are all correct together.

**It also rescues section 13's prediction properly.** The multiplier converges
toward GVEC's pressure on BOTH cases where the field stays coherent -- W1 (2.1x
closer, converged IC, big steps) and W5 (3.2x closer, unconverged IC, small
steps). It diverges only on LR3, where the field was being shredded and the
target was not an equilibrium anyway. The prediction was right; it needed a
coherent field to be tested on.

### 20.2 The three-way picture

| start from | step | force | surfaces | pressure vs GVEC |
|---|---|---|---|---|
| converged equilibrium (W1) | linesearch | **166x down** | kept | 2.1x closer |
| unconverged guess (W5) | fixed 1e-3 | 2.1x down | mostly kept | 3.2x closer |
| unconverged guess (LR3) | linesearch | 2.4x down | destroyed | 3.5x further |

Read as: the scheme is sound. It costs topology in proportion to how far it has
to move and how greedily it moves, and both of those are controllable.

## 21. lambda = 0 relaxes to the PURELY HARMONIC field

`S12`, `w7x_fmm002`, Clebsch IC with `--no-lambda`, 3000 CG steps:

    energy removed        2.9040%          (against 0.0244% with lambda on)
    ||F||                 4.587e-02 -> 6.271e-04        73x
    residual              6.006e-02 -> 4.001e-03        15.0x
    HARMONIC cos          0.985353 -> 0.999980
    residual off span(h)  1.705e-01 -> 6.355e-03
    harmonic amplitude    drift 1.13e-16   (the exact invariant, holding)

**The field converges to the harmonic field to five nines.** That is the
cleanest convergence result in this study, and it is consistent rather than
coincidental: this IC's helicity is ~0 (-1.8e-04), and the minimum-energy state
at zero helicity IS the harmonic (vacuum) field -- the same argument that
governs the `dzeta` case in section 12, arrived at from a completely different
starting field.

It also shows what lambda is FOR. lambda carries the within-surface
redistribution, i.e. the Pfirsch-Schlueter current, i.e. the part of the field
that supports pressure. Delete it and there is nothing left to hold the field
off the vacuum state, so it sheds 2.9% of its energy -- a hundred and twenty
times what the lambda-on case had to shed -- and lands on the harmonic mode.

### 21.1 A correction: lambda does NOT leave `compute_helicity` invariant

I claimed in discussion that lambda preserves helicity, and cited it as a gate
the `--no-lambda` arm should satisfy. **That is wrong, and the gate as I framed
it was testing a quantity with no reason to be invariant.**

The lambda-invariance belongs to the NATURAL-GAUGE analytic helicity, eq. (1)
of `logical_profile_ic.py`:

    A = Phi(rho) dchi - X(rho) dzeta
    H = int_0^1 (Phi X' - X Phi') drho

lambda does not appear in that formula at all -- it is built from the flux
functions alone, which lambda preserves -- so it is invariant by construction,
trivially.

`compute_helicity` computes a DIFFERENT functional, and
`logical_profile_ic.py`'s own docstring says so: it solves for the CO-EXACT `A`
via the Hodge Laplacian and adds the harmonic remainder back, giving the
relative/generalised helicity `<A, B + B_harm>`. That is a functional of `B`
alone with its own gauge fixing, so "lambda is a pure gauge transformation of
A" does not apply to it. lambda genuinely changes `B` -- it redistributes field
within each surface, which is exact-form content -- and the co-exact `A`
changes with it. Only `B_harm` is lambda-invariant, since lambda preserves the
fluxes.

Measured at the IC: **-1.780589e-04 with lambda on, -1.843373e-04 with it off**,
a 3.5% difference. I had attributed that to discretisation. It is not
discretisation; it is two different functionals, and the docstring warned that
they "do NOT have to agree".

## 22. THE dt BRACKET: the greedy step is pure loss when the field is far from equilibrium

`w7x_ini` clebsch IC, 3000 CG steps, only the step size differing:

| run | dt | dE | \|dH\| | \|\|F\|\| reduction | h/2 drift | surfaces |
|---|---|---|---|---|---|---|
| LR3 | linesearch (~5e-3 .. 1.4e-2) | 4.22e-03 | 2.33e-04 | 5.63x | 1.57e+00 | destroyed |
| **D1** | **fixed 3e-3** | 3.03e-03 | **3.17e-06** | **5.75x** | **2.3e-03** | **preserved** |
| W5 | fixed 1e-3 | 2.09e-03 | 2.00e-06 | 2.67x | 1.06e-04 | preserved |

**D1 gets the SAME force reduction as the linesearch while destroying 73x less
helicity, and keeps its surfaces.** On this case the greedy step buys no extra
progress at all -- it is pure loss. Its section has 1 line lost of 40, clean
5/6 and 10/11 island chains, and a smooth iota 0.78 -> 0.975.

**The threshold is sharp, not graded.** `dt = 3e-3` is only two to four times
below the linesearch's own step, and the outcome is categorical.

### 22.1 But the opposite holds near equilibrium

On `w7x_fmm002` -- a CONVERGED equilibrium -- the comparison inverts:

| run | dt | \|\|F\|\| | \|dH\| | surfaces |
|---|---|---|---|---|
| **W1** | **linesearch** | 1.877e-02 -> **9.950e-05 (189x)** | 1.54e-06 | preserved |
| D4 | fixed 1e-3 | 1.877e-02 -> 2.913e-03 (6.4x) | 1.17e-07 | preserved |

The linesearch is **thirty times more productive** here and perfectly safe. So
the rule is not "the linesearch is bad":

    The greedy step is worth taking when the field is NEAR equilibrium,
    and catastrophic when it is FAR.

Which is the same statement as section 20's: distance from equilibrium is what
makes a large step expensive. A step-size policy should therefore be adaptive
in the residual, not fixed -- and `|dH|` per step is the quantity to gate on,
since it is already measured and it is what actually goes wrong.

## 23. THE eta SWEEP: converging by forgetting the problem

`w7x_fmm002`, 4000 CG steps, tanh schedule tapering to ~0:

| eta_max | \|\|F\|\| final | helicity drift | G1 identity break (median) |
|---|---|---|---|
| 0 (W1) | 9.950e-05 | -0.9% | 1.3e-11 |
| 1e-4 | 9.628e-05 | +25.6% | 5.25 |
| 1e-3 | 1.405e-05 | +82.0% | 49.5 |
| **1e-2** | **2.203e-09** | **+99.98%** | 7.41 |

At `eta = 1e-2` the force residual reaches **2.2e-09** -- a numerically exact
equilibrium -- and the section is perfectly nested, zero lines lost, h/2 drift
2.8e-05, iota smooth and monotone 0.873 -> 1.040, **with the islands gone**.
W1's 5/6, 10/11 and 5/5 chains have vanished and the profile crosses those
resonances with no island structure.

That is exactly what a vacuum field looks like: no current, so no resonant
islands, so perfect nesting. The run is a complete and physically correct
demonstration of Taylor relaxation -- destroy the topological constraint and
the field falls to the lowest state available, which at H = 0 is the vacuum
field.

**And it is a demonstration of why strong eta is useless in an equilibrium
solver: it converges beautifully by forgetting the equilibrium it was given.**
99.98% of the helicity is gone; the answer is no longer related to the input.

`eta = 1e-4` is simply a bad trade -- 25.6% of the helicity spent for no force
gain over `eta = 0` (9.63e-05 against 9.95e-05).

The G1 identity breaks by 5-50x at every nonzero eta, as predicted in section
18.1: `dt = (F,u)/||dB||^2` omits the `-eta||J||^2` term in `<B,dB>`.

### 23.1 S17 was a badly designed arm -- recorded as such, not as a result

`S17` put `eta = 1e-3` on `w7x_ini` UNDER THE LINESEARCH, i.e. on the case
already being destroyed by step size (section 22). It stacks two destructive
mechanisms and the output is uninterpretable:

    iota          0.855 - 0.975 (IC)  ->  0.0000 - 0.6933
    axis offset   1.433e-04           ->  1.536e-01
    helicity      -50.3%

iota collapsing to zero means the field lines have stopped winding poloidally.
That is a wrecked field, not a measurement.

My stated rationale was "if numerical reconnection is what destroys the
surfaces, adding real resistivity should not make it notably worse". That test
cannot work when the baseline is already destroyed -- there is nothing to
compare against. The eta question was answered cleanly by S08-S10 on the
HEALTHY file, where the baseline is intact and the trend is monotone.

**The correct version, if anyone wants it,** holds `dt` fixed at 3e-3 (D1's
setting, which preserves surfaces on this file) and varies only eta. Not run:
S08-S10 already answer the question.

*Lesson, and it is the same one as the W1 confound in section 19.4: an arm that
varies one thing on top of an already-broken baseline measures nothing. Check
that the control is healthy before adding a variable to it.*

## 24. THE OPTIMIZER RANKING, on a case that behaves

Section 4's factorial ran on `quasr44970` with an INVENTED far-from-equilibrium
IC, and could not separate CG from fixed L-BFGS. `S11` repeats the comparison
on `w7x_fmm002` from GVEC's converged equilibrium -- a well-conditioned case --
3000 CG steps each:

| arm | \|\|F\|\| final | residual reduction | cos median | sy<0 | \|dH\| |
|---|---|---|---|---|---|
| gradient | 4.616e-04 | 36.1x | +1.0000 | 0/3000 | **1.96e-07** |
| lbfgs (fixed) | 1.570e-04 | 105.3x | +0.0365 | **0/3000** | 1.43e-06 |
| **cg** | **9.950e-05** | **166.2x** | -- | -- | 1.54e-06 |

**CG > L-BFGS > gradient**, and 166 / 105 / 36 is well outside the ~10%
run-to-run reproducibility measured in section 10.2 -- so unlike section 4,
this comparison DOES separate them. Section 4's "no winner can be called"
stands for that configuration and is superseded here for this one.

The fixed L-BFGS is healthy throughout (`sy < 0` on 0/3000), which confirms the
section-5 repair on a second geometry and a second IC route.

**And the same trade appears in miniature:** plain gradient loses ~7x less
helicity (1.96e-07 against ~1.5e-06) for 3-5x less force reduction.
Acceleration costs topology here too, just mildly enough that all three keep
their surfaces (axis offsets <= 3e-03).

Caveat: one run per arm, not replicated.

### 22.2 The G1 identity does NOT apply under a fixed dt -- a reading trap

`dE_pred = -dt (F,u)_M / 2` is derived assuming `dt` is the EXACT line
minimiser. Under `--dt-mode fixed` the true drop is

    dE = -dt (F,u)_M + (dt^2 / 2) ||dB||_M^2

so for a `dt` well below the minimiser the quadratic term is negligible and
`dE_meas -> 2 * dE_pred`. **Every fixed-dt run therefore reports the G1
identity as "broken by a factor 2", by construction.** Measured on D2
(`dt = 3e-4`): `dE_meas = -1.447e-07` against `dE_pred = -7.233e-08`, i.e.
2.0000.

A ratio near 2 is the HEALTHY signature for a small fixed step, not a fault.
The script now prints that warning next to the number. The identity remains a
genuine operator test only for `ANALYTIC_LINESEARCH`, where it holds at 1e-11
relative / 1e-16 against the energy scale.

## 25. THREE LEVERS ON NUMERICAL RECONNECTION, ranked

Everything in sections 19-24 points at one quantity -- how much topology the
descent destroys per unit progress. Three independent controls have now been
measured on real cases.

| lever | helicity loss reduced by | cost |
|---|---|---|
| **step size** (LR3 -> D1, linesearch -> fixed 3e-3) | **73x** | **none** -- same force reduction |
| hyperregularisation gamma=1 (LR1 / S04) | 15x / 2.4x | ~3x per step, ~12% less force reduction |
| resolution 8^3 -> 12^3 (S01) | 3.4x relative | ~3x per step |

**Step size is by far the largest lever, and the only free one.** On the case
where it matters, capping the step cost nothing at all in force reduction
(5.75x against the linesearch's 5.63x) while keeping the surfaces.

### 25.1 gamma = 1 on the healthy case (S04)

| | gamma=0 (W1) | gamma=1 (S04) |
|---|---|---|
| energy removed | 0.0244% | 0.0241% |
| \\|\\|F\\|\\| | 9.950e-05 (189x) | 1.139e-04 (165x) |
| \\|dH\\| | 1.54e-06 | **6.47e-07** |
| axis offset | 3.071e-03 | **3.386e-04** |
| roughness | -- | 0.281 -> 0.0756 (0.27x) |

Same energy removed, ~12% less force reduction, 2.4x less helicity lost and a
**9x tighter axis**. Consistent with LR1, where gamma=1 cut helicity loss 15x
on `quasr44970` while giving the best force reduction of any arm there. So
hyperregularisation is a genuine topology-preserving lever and not a
coarse-grid artefact -- it now holds on two geometries and two IC routes.

### 25.2 Resolution: the safe regime survives refinement (S01, S03)

`w7x_fmm002`, 3000 CG steps:

| | 8^3 p=3 (W1) | 12^3 p=3 (S01) | 8^3 p=4 (S03) |
|---|---|---|---|
| n2_dbc | 2192 | 8376 | 2192 |
| energy removed | 0.0244% | 0.0063% | 0.0276% |
| \\|\\|F\\|\\| reduction | 189x | 77x | 230x |
| \\|dH\\|/H | 0.87% | **0.25%** | 0.29% |
| axis offset | 3.071e-03 | **9.483e-04** | 1.568e-03 |
| roughness | -- | 0.29x | **0.12x** |

Both refinements preserve the surfaces and IMPROVE the topology metrics. The
finer grid removes **3.9x less energy** in the same number of steps, which is
consistent with part of the coarse grid's energy release having been numerical
rather than physical -- though that is an inference, not a measurement.

### 25.3 A CAVEAT ON MY OWN CRITERION: helicity is not resolution-converged

Section 19.2 established that ABSOLUTE `|dH|` predicts surface destruction
where relative drift does not. That stands WITHIN a resolution. It does not
transfer across one:

    IC helicity, same field, same file:   -1.780e-04 at 8^3
                                          -3.162e-05 at 12^3

a factor 5.6. The helicity of this field is simply not converged between those
grids -- unsurprising, since it is a small number arising from cancellation and
the field is ~99.99% harmonic (`||B - curl A||/||B||` = 0.9999 at 8^3, 1.000 at
12^3). **So compare `|dH|` absolutely within a resolution and relatively across
resolutions**, and do not read a cross-resolution `|dH|` ratio as a physical
statement. The table in 25.2 uses the relative form for exactly this reason.
