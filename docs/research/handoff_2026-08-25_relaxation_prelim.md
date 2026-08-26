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
| ~~**p-refinement** p=3 -> p=4 (S03)~~ | ~~**3.4x**~~ **WITHDRAWN, s33.2** | 1.7x per step, no extra DoFs |
| h-refinement 8^3 -> 12^3 (S01) | ~~**none**~~ **QUALIFIED, s31/s32** | 3x per step |

**READ THE TWO STRUCK ROWS BEFORE THE TABLE.** Both refinement rows are
withdrawn or qualified, for two SEPARATE reasons that a reader should not
merge:

* **s32 -- no arm in this campaign floored.** Every run is still descending at
  its last step, so neither refinement row was ever a floor result. The point
  of refining is to reach a LOWER floor, not to get there faster; a finer arm
  being slower is expected and is not a finding.
* **s33.2 -- the p sweep alternates operators.** p=2 and p=4 are the only
  even-p runs here and both predate the even-p quadrature parity fix, so the
  sweep's shape is not readable and `3.4x` divides a post-fix point by a
  pre-fix one.

**The p axis is not gone.** What survives, plainly: p=1, p=3 and p=5 are
internally comparable to each other and to every other run in this document.
`n2_dbc` is 2192 at BOTH p=3 and p=4 -- raising the order there adds no DoFs
and does not shrink the step -- which is a fact about the SPACES and stands on
its own; it simply no longer has a measurement attached to it. Re-measuring P2
and S03 post-merge (~2 GPU-hours total) restores the axis outright.

The three unstruck rows are unaffected by both: none is a refinement claim.

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

Both refinements preserve the surfaces. **But the apparent improvement from
h-refinement is not real, and my first reading of it was wrong.**

I inferred that the finer grid removing 3.9x less energy meant part of the
coarse grid's release had been numerical. Tobias proposed the simpler
explanation -- a finer grid is STIFFER, so `dt` shrinks and ALL the dynamics
slow together, which accounts for less energy removed, larger `||F||` and less
drift with one mechanism instead of three. Normalising helicity loss by energy
actually removed settles it:

| run | dE | \|dH\|/H | **per unit dE** |
|---|---|---|---|
| 8^3 p=3 (W1) | 1.220e-04 | 8.659e-03 | **70.9** |
| 12^3 p=3 (S01) | 3.146e-05 | 2.548e-03 | **81.0** |
| 8^3 p=4 (S03) | 1.380e-04 | 2.884e-03 | **20.9** |

**h-refinement buys nothing** -- per unit progress the 12^3 grid destroys as
much helicity as the 8^3 one, marginally more. My inference is refuted: had the
coarse grid's release been numerical, the fine grid would show LESS loss per
unit progress.

**WITHDRAWN 2026-08-25 -- see s33.2. The p=4 arm was measured on the pre-fix
even-p quadrature operator, so this comparison divides a post-fix point by a
pre-fix one.** The paragraph below is left as written for the record.

**p-refinement is genuinely 3.4x better**, and consistently with the same
mechanism rather than as an exception to it: `n2_dbc` is 2192 at BOTH p=3 and
p=4, so raising the order does not shrink the grid scale or the step. It
resolves the curl better at the same h and the same dt. So h-refinement changes
the RATE; p-refinement changes the FIDELITY PER STEP. p is also the cheaper of
the two -- 1.47 s/step against 2.71 s/step.

*Lesson: "everything got better" across a refinement usually means "everything
got slower". Normalise by progress before reading a refinement as an
improvement.*

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

## 26. WHAT THE w7x_ini RUNS ARE AND ARE NOT FOR

Every run on `w7x_ini_00000000_clebsch_mrx.h5` -- LR3, W4, W5, D1, D2, D3,
S13, S14, S17 -- starts from GVEC's INITIAL GUESS, not a converged equilibrium
(section 20). They are a **stress test**, and should be read only as one.

**What they are good for, and it is a lot:** a field with far to travel
exercises the reconnection failure mode hard, which is what made the
step-size mechanism isolable at all. LR3 vs W5 vs D1 is a clean controlled
experiment precisely because the case is stressed -- on `fmm002` the effect is
too small to separate from noise in 3000 steps.

**What they are NOT good for:** any statement about W7-X equilibria, about
what this scheme converges to physically, or about pressure profiles. The
target was never an equilibrium, so `J x B = grad p` does not hold at the
start and the pressure comparison has no reference (section 13).

**All physics conclusions in this document rest on `w7x_fmm002`**, the
converged state: the 166x residual reduction with surfaces intact (section
20.1), the optimizer ranking (24), the eta sweep (23), the gamma and p levers
(25), and the lambda = 0 convergence to the harmonic field (21).

S13 is the one place the distinction nearly misleads: at 12^3 the `_ini` case
improves 100x on every scalar and still does not recover nested surfaces, which
is a real statement about the SCHEME's step-size sensitivity, not about the
device.

## 27. THE gamma / mu STUDY SATURATES IMMEDIATELY

`w7x_fmm002`, Clebsch IC, cg. S05 and S06 hit their 9000 s cap at 2410 and
1961 steps, so `||F||` is not comparable across the row -- but all four removed
~1.2e-04 of energy, so helicity lost PER UNIT ENERGY REMOVED is:

| arm | steps | wall s | dE | \|\|F\|\| | resid x | \|dH\|/H | **/dE** | axis |
|---|---|---|---|---|---|---|---|---|
| W1 g=0 | 3000 | **2624** | 1.220e-04 | **9.950e-05** | **166.2** | 8.659e-03 | 70.9 | 3.071e-03 |
| **S04 g=1 mu=1e-3** | 3000 | 8212 | 1.207e-04 | 1.139e-04 | 145.9 | 3.633e-03 | **30.1** | **3.386e-04** |
| S05 g=1 mu=1e-2 | 2410 | 9002 | 1.182e-04 | 3.159e-04 | 50.3 | 5.187e-03 | 43.9 | 1.945e-03 |
| S06 g=2 mu=1e-3 | 1961 | 9004 | 1.189e-04 | 2.950e-04 | 50.0 | 3.742e-03 | 31.5 | 1.659e-03 |

All four removed the SAME energy (1.18-1.22e-04), so the comparison is fair
despite S05/S06 hitting their 9000 s cap at fewer steps.

**`gamma = 1, mu = 1e-3` is the sweet spot and both ways of pushing further are
dead ends:**

* **More smoothing HURTS.** `mu = 1e-2` is markedly worse than `mu = 1e-3`
  (43.9 against 30.1), presumably because it smooths away real structure along
  with the noise -- the roughness ratio is the same 0.36x in both, so it is not
  buying extra smoothness for the cost either.
* **`gamma = 2` buys nothing** over `gamma = 1` (31.5 against 30.1) at 1.7x the
  per-step cost.

So the lever saturates at its cheapest useful setting, which is a convenient
place for it to saturate. All four keep their surfaces (axis offsets
<= 3.4e-03).

### 27.1 But on THIS case gamma is a trade, not a win -- and I said otherwise

**`gamma = 0` wins on force outright here**: 9.950e-05, the lowest of the four,
in 2624 s -- **3.1x faster** than any `gamma > 0` arm. `gamma = 1, mu = 1e-3`
is 14% WORSE on force at 3.1x the cost, and buys 2.4x on topology and a 9x
tighter axis.

Reporting LR1 I wrote that `gamma = 1` gave "the best force reduction of any
arm". That was true THERE -- `quasr44970`, logical IC, far from equilibrium --
and is false here. The benefit is conditional, and stating it unconditionally
was wrong.

**Both levers turn out to have the same conditional**, which is the useful
generalisation:

| | far from equilibrium | near equilibrium |
|---|---|---|
| small step | FREE -- same force, 73x topology (D1 vs LR3) | costs 30x force (D4 vs W1) |
| gamma = 1 | improves force AND topology (LR1) | costs 14% force + 3.1x time, buys 2.4x topology |

Both pay when the field has far to travel and charge when it does not -- the
same conditional as section 22.1. A policy that switches on the residual would
get both right; a fixed choice cannot.

Note this makes `gamma = 1, mu = 1e-3` -- the value used throughout sections 9
and 25 because it was the first thing tried -- the right choice by measurement
rather than by luck.

## 28. RUNNING LONGER IS NOT FREE (S07)

`w7x_fmm002`, W1's exact settings, 13018 steps instead of 3000:

    quantity          3000 steps    13018 steps
    energy removed      0.0244%       0.0253%
    ||F||             9.950e-05     1.812e-04     <- 1.8x WORSE
    |dH|              1.542e-06     3.343e-06     <- 2.2x more
    roughness             --          0.26x       <- SMOOTHER
    h/2 drift         2.6e-04       3.1e-03       <- 12x worse, still small
    axis offset       3.071e-03     2.087e-03
    iota range        0.907-1.048   0.857-1.624   <- greatly expanded

**The extra 10000 steps bought 0.0009% more energy** and cost a 1.8x worse
force residual, twice the helicity, and the core. The section shows the outer
surfaces still cleanly nested but the core restructured: large island chains at
mid-radius, scattered points near the axis, and an iota profile that now rises
to 1.62 near the axis with a shear reversal down to 0.85 at `a_eff ~ 0.18`.

**The energy is converged by ~3000 steps on this case.** Everything after that
is the field rearranging structure at no benefit.

### 28.1 The energy episode: neither hypothesis won

Section 22 noted the energy re-accelerating after a plateau -- decrement per 500
steps 5.6e-08 (plateau) -> 6.53e-07 (burst) -> 1.13e-07 (settled). I read that
as topology tearing releasing locked energy; Tobias pointed out it could equally
be the field escaping a shallow basin toward a smoother minimum, and that
nothing in the energy curve distinguishes them.

The three discriminators SPLIT:

* roughness FELL (0.26x) -- supports the smoother-minimum reading
* `|dH|` DOUBLED -- supports the tearing reading
* the section shows PARTIAL core restructuring -- neither

So it is recorded as undetermined. What is NOT undetermined is the practical
consequence: on this case the extra steps are not worth taking, whichever
mechanism is responsible.

*Method note: my earlier "||F|| looks like it is flooring out" was reaching for
the wrong quantity -- but the instinct was right and the ENERGY shows it
cleanly. Third instance of the same error in this study (see the trap in
section 2).*

## 29. THE COMPLETE dt BRACKET: a cliff, not a trade

`w7x_ini`, Clebsch IC, cg. Step counts scaled so the small-dt arms are not
confounded with "barely moved" (section 19.4's error):

| run | dt | steps | dE | \|dH\|/H | **per unit dE** | \|\|F\|\| | axis |
|---|---|---|---|---|---|---|---|
| LR3 | linesearch | 3000 | 4.224e-03 | 5.808e-03 | **1.3750** | 8.452e-03 | 1.027e-01 |
| **D1** | **3e-3** | 3000 | 3.032e-03 | 7.897e-05 | **0.0260** | 8.265e-03 | 1.640e-02 |
| W5 | 1e-3 | 3000 | 2.089e-03 | 4.994e-05 | 0.0239 | 1.789e-02 | 1.157e-02 |
| D2 | 3e-4 | 11000 | 2.181e-03 | 5.304e-05 | 0.0243 | 1.709e-02 | 1.204e-02 |
| D3 | 1e-4 | 12000 | 1.282e-03 | 2.769e-05 | 0.0216 | 2.538e-02 | 6.440e-03 |

**Reconnection efficiency is FLAT across two decades of dt** -- 0.0216 to
0.0260 from 1e-4 to 3e-3 -- and then jumps **53x** at the linesearch. This is a
threshold, not a gradual trade: there is a cliff somewhere between 3e-3 and the
linesearch's own step (~5e-3 to 1.4e-2 on this case).

**dt = 3e-3 is therefore optimal**: it sits just under the cliff, so it removes
the most energy per step of any safe choice. Going smaller is pure waste --
D3 at 1e-4 spent 4x the steps to remove 2.4x LESS energy, for a 17% better
efficiency nobody needs.

### 29.1 The operating point, for this problem

Combining with section 28 (energy converged by ~3000 steps on the healthy
case):

    dt ~ 3e-3, ~3000 steps, gamma = 1 / mu = 1e-3 when topology matters
    more than wall-clock.

**But 3e-3 is problem-specific and should not be hardcoded.** The cliff is in
`|dH|` per step, which is already traced. A step policy that caps the
linesearch by a `|dH|` budget would find this point automatically and would
transfer to problems where the number differs -- which is the recommendation in
section 19.5, now with a measured threshold behind it rather than an
intuition.

## 30. WHAT THE PRECONDITIONER FIX BOUGHT, AND WHERE IT DID NOT

`w7x_fmm002`, gamma=1, 300 steps, before and after swapping the diffusion
solve's `diag(M)^-1` for the production `block_jacobi` (a93bec5):

| mu | eps * lambda_max | before | after | change |
|---|---|---|---|---|
| 1e-3 | ~0.26 | 2.74 s/step | **2.10 s/step** | **-23%** |
| 1e-2 | ~2.6 | 3.74 s/step | 3.85 s/step | +3% (noise) |

**Exactly what the theory predicts, including the null result.**
`block_jacobi` approximates `M` and knows nothing about `eps L`, so it is a
large win while the operator is mass-dominated and does nothing once it is not.
At `mu = 1e-2` the Laplacian term is what hurts, and improving the
M-approximation cannot touch it. Per the standing single-digit rule the +3% is
noise and was not investigated.

This also bounds what the Neumann correction of section 25's note could buy: it
is the term that WOULD address `eps L`, and `mu = 1e-2` is where it would pay
-- but that is also where it goes indefinite, which is why it was not taken.

### 30.1 I retracted section 27 too hastily

I wrote that section 27's "more smoothing hurts" conclusion was "partly a
preconditioner artefact and should be re-measured". That was over-stated.

**A preconditioner changes wall-clock, not the converged answer.** Section 27
ranked mu on `|dH|` per unit energy removed (43.9 at 1e-2 against 30.1 at
1e-3), computed from solves that converged to tolerance either way. That
number is contaminated only if those solves were FAILING to converge, which I
never checked before announcing the retraction.

So the correct statement is narrower: section 27's TIMING column was
preconditioner-limited; its QUALITY conclusion probably stands. The re-run
settles it -- if `mu = 1e-2` comes back at 43.9 again, over-smoothing is real
and the retraction was wrong.

*Third time in this study that I have attached a mechanism to an observation
before checking whether the mechanism could produce it. The check here was one
question: does this quantity depend on the preconditioner at all?*

## 31. h-REFINEMENT: it buys nothing, or it buys 260x, depending on the case

Section 25.2 concluded "h-refinement buys nothing" from `fmm002` alone. With
`w7x_ini` refined at the same linesearch, that is only half the story.

    |dH|/H per unit energy removed, LINESEARCH throughout

    grid    fmm002        w7x_ini
    8^3      70.9          1.375
    12^3     81.0          0.01346     (102x better)
    16^3    469.2*         0.005286    (260x better)

    * see the caveat below -- fmm002's H is not converged and this number
      is not physical.

**The three `w7x_ini` arms removed COMPARABLE energy** -- 4.224e-03, 3.823e-03,
3.714e-03 -- so this is emphatically NOT the stiffness slowdown that explained
the `fmm002` column. The dynamics did not slow; the reconnection genuinely fell
260x.

**The conditional is the same one that runs through this whole study:**
refinement fixes reconnection where reconnection is the limiting factor, and
does nothing where it is not. On `fmm002` the run was never
reconnection-limited, so there was nothing to buy.

### 31.1 Refinement and step-capping are ALTERNATIVE routes to the same floor

    w7x_ini, |dH|/H per unit dE

    8^3  linesearch        1.375        chaotic
    8^3  dt = 3e-3         0.02605      nested
    12^3 linesearch        0.01346
    16^3 linesearch        0.005286

The refined arms at the greedy step reach a LOWER reconnection rate than the
coarse arm with a capped step. So the two levers substitute for one another,
and the cheaper one wins on cost: D1 is 0.91 s/step against S14's ~7 s/step for
the same job.

That said, section 22 already showed 12^3-at-linesearch does not recover nested
SURFACES even though its scalars are good, so "same reconnection rate" is not
"same outcome" -- the accumulated damage over 3000 greedy steps still matters.
Capping the step remains the recommendation; refinement is the expensive
substitute, not the cheap one.

### 31.2 fmm002's numbers in that column are not physical

`fmm002`'s H collapses under refinement (-1.780e-04, -3.162e-05, -5.756e-06)
because `compute_helicity` returns exactly zero for a purely harmonic field
(`delta B = 0` gives `A = 0`) and this field is almost entirely harmonic. So
the `fmm002` column is the drift of a vanishing residue divided by a vanishing
denominator, and its apparent 469 at 16^3 means nothing.

**CORRECTED 2026-08-25 (s35).** This paragraph originally put the
current-driven fraction at "1.7%, falling to 0.9%" for `fmm002` and "17%" for
`w7x_ini`. Both numbers were wrong, by two orders of magnitude and by one
respectively. Measured `1 - B_harm_rel`:

    fmm002    8^3  0.0139%   12^3  0.0048%   16^3  0.0037%   <- COLLAPSES
    w7x_ini   8^3  1.4162%   12^3  1.4227%   16^3  1.4236%   <- STABLE

The conclusion survives and is strengthened: what separates the two cases is
not the size of the fraction but that `fmm002`'s **collapses** under
refinement while `w7x_ini`'s is **stable to three digits**. A denominator that
moves with the grid cannot normalise a comparison across grids. `w7x_ini` is
where this metric is sound, and that is where every step-size conclusion was
measured.

## 32. RETRACTION: every resolution conclusion here is a RATE claim, and rate was never the question

Tobias, 2026-08-25: **"The goal of h-refinement is not to converge faster, it
is to converge to a lower -- or more physically accurate -- floor."**

A finer arm taking more steps, more wall clock, or more iterations per step is
EXPECTED. It is not a regression and not a finding. The measurable claim of an
h-refinement study is WHERE THE RUN FLOORS: does ||F|| bottom out lower, does
the energy settle nearer the true minimum. That is the number and it is the
only number. And if refining does NOT lower the floor, that is a real finding,
because it says something other than the discretisation is limiting the
result.

This is the same family as the monotonicity correction in s2 TRAP 2 -- both are
cases of judging a run by a quantity that was never the point.

### 32.1 Measured: NOT ONE ARM FLOORED

A floor claim requires the run to have flattened. Two tests: is ||F|| still
moving over the last 20%, and is the dissipation rate still above round-off?

                     steps   |F| end   |F| last20%  -dE/dt end  rate/rate@50%
    fmm002   8^3      3000  9.95e-05      x0.74      9.7e-09        1.02
            12^3      3000  1.62e-04      x0.64      1.8e-08        0.68
            16^3      1567  7.93e-04      x1.05      2.8e-07        0.23
    w7x_ini  8^3 ls   3000  8.45e-03      x1.87      4.2e-05        1.21
             8^3 cap  3000  8.27e-03      x0.81      6.9e-05        0.34
            12^3      3000  1.71e-03      x0.69      1.3e-06        0.026
            16^3      1330  4.57e-02      x0.72      1.2e-03        0.85

Every arm is still descending at its last step. The rates run 1e-9 to 1e-3
against the ~1e-16 round-off floor that S10 demonstrated is reachable (s31
note). **There is no floor measurement anywhere in this campaign.**

### 32.2 Worse: the truncation biases every comparison AGAINST refinement

S02 (16^3) stopped at 1567 steps and S14 (16^3) at 1330, both on a wall-clock
budget set from the coarse case's 3000. So the finest arms are the ones
FURTHEST from their floors, and every resolution comparison in this document
is biased in the direction of the conclusion it reached. Matching step counts
across resolutions is the wrong design; the dt bracket got this right (s20,
step counts scaled inversely so each arm removed comparable energy) and the
resolution arms did not.

### 32.3 RETRACTED

**"Capping dt at 8^3 gets 53x of the quality for 1.3x the time, so the cap
stays the recommendation and refinement is the expensive substitute"**
(commit c01c2ce). Two errors:

1. It rejects refinement on COST, which is the error above.
2. More fundamentally, **capping dt and refining h are not substitutes.**
   Capping reduces the time-integration error at a FIXED discretisation.
   Refining changes what the discretisation can represent -- that is, where
   the floor is. They answer different questions and s31.1 put them on one
   axis. The step-size finding (s20-s22) stands on its own; it is a statement
   about dt at fixed h and nothing in it was ever a resolution claim.

**QUALIFIED, not withdrawn**: s25.2's "h-refinement buys nothing" and s31's
"buys nothing or 260x". ``|dH|/H per unit energy removed`` is a real
measurement of reconnection efficiency and those numbers stand as that. What
does not stand is presenting it as the verdict on refinement, which is a floor
question this campaign never asked.

### 32.4 The one hint, offered as a hint

S13 (w7x_ini, 12^3) is the closest thing to a floored arm here: its
dissipation rate fell to **2.6% of its mid-run value**, by far the flattest of
the set, and it reached ``||F|| = 1.71e-03`` -- **5x lower than either 8^3 arm**
(8.45e-03, 8.27e-03) at the same 3000 steps and comparable energy removed.

That is consistent with refinement lowering the floor. It is NOT evidence of
it: S13 has not floored either, and one arm nearer flat than the others is a
hint about direction, not a measurement of where either lands.

### 32.5 What it would take, and why it was not launched

Establishing the floor needs each resolution run to FLAT -- rate at round-off,
||F|| stationary -- not to a step count. I have not estimated how long that is
and will not invent a number: S13 needed 3000 steps to reach 2.6% of mid-run
rate and the remaining distance to 1e-16 is not something to extrapolate from
one arm. The honest way to find out is to run ONE arm to flat and measure.

Not launched because the user's standing instruction as of 2026-08-25 is to
stop launching and drain the queue. This is the user's call on their own GPU
budget, and it is recorded here rather than acted on.

**Until such a run exists, this document contains no claim about whether
h-refinement lowers the relaxation floor.** The same caveat applies to the
p-sweep in s25.1: "p buys 3.4x" is the identical category of claim about a
different refinement axis, measured the same way, and it is equally not a
floor result.

## 33. PROVENANCE: which mrx produced these numbers

Checked 2026-08-25 after a cross-session warning that the venv's editable
install pins `mrx` to the MAIN checkout regardless of cwd, so a slurm job
without the `PYTHONPATH` shim silently runs the main line's library and
nothing errors.

**Every job in this campaign is clear.** All launches went through
`slurm/job_relax_prelim.sh` or `slurm/job_clebsch_ic.sh`, both of which export
`PYTHONPATH="$WORKTREE:$PYTHONPATH"`; all six `scripts/sweeps/*.sh` reference
the former and nothing else. The two wraps in `slurm/` that do NOT export it
(`job_poincare.sh`, `job_relax_from_nfs.sh`) are tracked files from main and
were never used here.

The evidence is in the logs, not in the wrap: `relax_prelim.py` prints
`[env] mrx from <path>` as its first line, and **all 45 archived run logs carry
the worktree path**, with zero deviations -- including the jobs in flight at
the time of the check.

### 33.1 The flip side: these numbers describe a STALE mrx

The shim pins the library, which is the point -- but it pins it to
**76bf5f3 plus this branch's own commits**, and `greville-prod` moved roughly
twenty commits on 2026-08-25 without any of it reaching this worktree. Missing
here: the `raw_kron` deletion, the atom's rename to `metric_lumping`,
histopolation, the Poincare work, the eqx payload change.

So the results are provenance-KNOWN, not provenance-clean. One concrete
consequence is already identified: **P2 (`fmm002` p=2) was measured before the
even-p quadrature parity fix**, which Tobias confirms is solved on main. Read
that arm as pre-fix. Any other even-p or `raw_kron`-dependent number here
carries the same caveat.

Merging `greville-prod` in is the fix and it is deliberately deferred until
the queue drains: merging mid-flight would change the library under running
jobs, which is the exact failure the shim exists to prevent.

## 33.2 WITHDRAWN: the p-refinement headline rests on a pre-fix operator

The Coordinator asked me to flag every even-p result, not just P2. There are
exactly two in the whole campaign, and their identity is the problem:

    P2    p=2   fmm002 8^3     <- pre-fix
    S03   p=4   fmm002 8^3     <- pre-fix
    (41 other runs are p=1, 3 or 5 and are unaffected)

The fix: periodic Greville spans cross x=1 at even p, and the basis was
evaluated unwrapped while the moments wrapped. One line, 32 passed / 0 failed
on `greville-prod`. So at p=2 and p=4 -- and only there -- these runs used an
operator that has since changed.

**The p-sweep is therefore alternating pre- and post-fix operators**, and the
headline claim sits exactly on the seam:

    |dH|/H per unit dE, fmm002 8^3

    p=1   13.74     post-fix
    p=2   11.38     PRE-FIX
    p=3   70.94     post-fix
    p=4   20.90     PRE-FIX
    p=5   (not landed)

"p-refinement buys 3.4x" is `70.94 / 20.90` -- **one post-fix point divided by
one pre-fix point.** It is not a measurement of p-refinement; it is a
measurement of p-refinement confounded with an operator change, and the two
cannot be separated from this data.

### The direction makes it worse, not better

The two pre-fix arms are the two BEST-LOOKING points in the sweep. p=3, the
only interior post-fix point, is the worst at 70.94, and it is flanked by
11.38 and 20.90. If the parity bug made the even-p operators wrong, then the
"improvement" at p=4 may be an artefact of the bug rather than a benefit of
raising the order -- a wrong quadrature can easily look like less reconnection
per unit energy while simply mis-measuring both.

I cannot tell which it is from here, and I am not going to argue it either
way. What I can say is that the sweep's SHAPE -- the non-monotonicity, with
even points sitting below odd ones -- is exactly the pattern an even/odd
operator split would produce, and that alone is enough to stop quoting the
number.

### What stands

Nothing in the p-sweep is a floor result (s32.5 already applies). Beyond that,
p=1, p=3 and p=5 are internally comparable to each other and to every other
run here; p=2 and p=4 are comparable to nothing until re-measured. The
observation that `n2_dbc` is 2192 at both p=3 and p=4 -- so raising the order
adds no DoFs and does not shrink the step -- is a fact about the spaces and
survives; it just no longer has a measurement attached to it.

**Re-measure P2 and S03 after merging `greville-prod`.** Two 8^3 arms at ~0.5
and ~1.5 GPU-hours; this is the cheapest open item in the study.

## 34. FUTURE SWEEPS: a shelf of experiments

Tobias, 2026-08-25: *"We are not launching anything more, but we can collect
ideas and open questions for future sweeps."* Nothing below was run. Each entry
is meant to be pickable off the shelf without re-deriving anything, so it
carries the question, what it decides, the cheapest experiment that decides it,
a cost ESTIMATE, and what a null result means.

Costs are estimates from measured s/step in s33 and are flagged as such. Class
A restores something already half-known; class B opens new ground. Within B,
some are bounded and some are not, and that is stated per item.

**Read P0 first.** It is a code change rather than a sweep, it costs no GPU
time, and it is a hard prerequisite for B1 and B2.

### PREREQUISITE (P0) -- a CODE change, not a sweep. Different owner.

**`relax_prelim.py` cannot express the floor experiment.** It stops on
`--steps` or `--seconds-per-arm` and has no convergence criterion of any kind.
**That is why nothing in this campaign ever floored** (s32) -- not "we did not
get around to the floor run", but "the tool has no way to say *run until
flat*".

This is a missing VOCABULARY, not a missing feature, and the distinction is
the whole reason P0 outranks every sweep below it: **every arm in this
campaign stopped somewhere arbitrary because arbitrary was the only thing the
interface could say.** `--steps` and `--seconds-per-arm` are both BUDGETS, and
a budget cannot express a convergence criterion no matter how large you make
it. Running longer does not fix this; it just moves where the arbitrary stop
lands.

Anyone reading B1 as a sweep will queue a job and get another arm that stops
where the budget stopped it, which is the exact failure s32 documents.

1. Can `relax_prelim.py` terminate on convergence rather than on a budget?
2. Unblocks **B1 and B2, and with them any h- or p-refinement claim at all.**
   Nothing in this document currently supports one.
3. Add `--stop-rate`: track `-dE_meas/dt`, maintain its running minimum, and
   stop once the rate has stayed within a factor of the round-off floor for N
   consecutive steps. S10 gives the calibration -- its energy is constant to 16
   digits from step ~500 with `dE_meas` at 1.11e-16, so the floor is
   observable and the criterion is testable against an arm already on disk.
   Keep `--seconds-per-arm` as the outer guard.
4. **0 GPU-h.** It is a code change. Validate it by REPLAYING S10's archived
   trace, not by running anything.
5. NULL is not applicable -- this is not an experiment. It either lands or it
   does not, and until it lands B1 and B2 cannot be run as written.

### CLASS A -- restores a withdrawn or confounded claim

**A1. Re-measure P2 and S03 on the post-fix operator.**
1. Does p-refinement reduce helicity loss per unit energy, once every point in
   the sweep is measured on the same operator?
2. Restores or kills the withdrawn `3.4x` (s33.2) and makes the whole p axis
   readable. Currently p=2 and p=4 are comparable to nothing.
3. After merging `greville-prod`, re-run P2 and S03 exactly as launched:
   `--geometry w7x-fmm002 --ic clebsch --ns 8,16,8 --p {2,4} --steps 3000
   --arms cg`. Nothing else needs re-running -- the other 41 runs are odd-p.
4. **~1.6 GPU-h** (estimate: 0.46 and 1.47 s/step measured x 3000).
5. NULL = the even-p numbers come back unchanged, the parity fix does not
   touch this quantity, the sweep's shape was real all along and `3.4x` is
   reinstated as a measurement. That is a useful answer, not a wasted run.

**A2. Bracket the mu minimum.**
1. Where is the minimum of `|dH|/H per unit dE` in mu, and is 1e-4 it?
2. Whether `gamma=1, mu~1e-4` is a recommendable default -- M1 gives 1.719
   against 70.94 at mu=0, a 41x reduction for 1.4x the cost per step -- or
   whether one interior sample is being over-read.
3. **Wait for M2-M5 first; they may already bracket it.** If they do not: two
   more `gamma=1` 8^3 fmm002 arms at `--mu 3e-5` and `--mu 3e-4`, 3000 steps,
   otherwise identical to the M-series.
4. **~2 GPU-h** (estimate, 1.24 s/step measured).
5. NULL = 3e-5 and 3e-4 also land near 1.7, so the minimum is a broad basin
   rather than a point, mu is insensitive, and no tuning is needed. That is
   the better outcome for anyone using this.

**A3. Isolate polish from the preconditioner.**
1. Does harmonic-form inverse-iteration polish help or hurt at k=1,2 once its
   shifted solve uses the atom instead of `schur.outer='jacobi'`?
2. Whether polish stays gated off. My run did not separate the two, so "the
   atom fixes polish" is consistent-with, not demonstrated-by, this data.
3. `--ic-only`, both geometries, polish on/off x preconditioner old/new = 4
   short arms, judged on `relL2_direct`. No relaxation loop needed.
4. **~0.5 GPU-h** (estimate; IC only).
5. NULL = polish is neutral in all four, and the machinery can be deleted
   rather than left gated.

**A4. S15's cost anomaly.**
1. Is fmm002 12^3 `gamma=1` really 22.6 s/step -- ~2.5x slower than the h- and
   gamma-scalings predict together -- or was that the pre-fix `M + eps L`
   preconditioner?
2. Only whether a cost anomaly exists. Nothing physical rides on it, and if
   the answer is "the preconditioner", it is already fixed.
3. **Probably no experiment at all**: H1/H2 are 12^3 `gamma=1` on the new
   preconditioner and are in flight -- just read their s/step when they land.
   Only if BOTH truncate: one 12^3 `gamma=1 mu=1e-3` arm, 500 steps, timing
   only.
4. **0 GPU-h** if H1/H2 land, else ~0.5.
5. NULL = H1/H2 also show ~22 s/step, so the anomaly is real and predates the
   preconditioner. Then it belongs in the solver, not in this study.

### CLASS B -- opens new territory

**B1. Run ONE arm to FLAT. (NOT bounded)**
1. Where does a relaxation arm actually floor -- what `||F||` and what
   dissipation rate does it reach when run to flat rather than to a step count?
2. **This is a prerequisite for EVERY refinement claim in this document.** No
   arm in this campaign has ever floored (s32), which is why s31 and s33.2 are
   withdrawn or qualified rather than merely uncertain.
3. One arm, fmm002 8^3 p=3 gamma=0 -- cheapest per step (0.87 s) and the
   best-behaved case. No step cap. Stop when `-dE/dt` has stayed within 2x of
   its round-off floor for 500 consecutive steps -- i.e. **P0 must land
   first**. Do NOT queue this as a sweep before it does: without P0 the arm
   stops where its budget stops it and you get another unfloored run.
4. **NOT ESTIMABLE.** S13 reached 2.6% of its mid-run rate in 3000 steps and
   the remaining distance to ~1e-16 cannot be extrapolated from one arm. Budget
   it as open-ended with a wall-clock cap and accept it may not finish.
5. NULL = it never flattens within a large budget. That IS the finding: the
   scheme has no reachable floor at this discretisation, and the floor framing
   itself needs rethinking before any refinement study is worth running.

**B2. Resolution ladder on a floor criterion. (gated on B1)**
1. Does finer h floor LOWER?
2. The actual h-refinement question, which this campaign never asked -- it
   measured rate and efficiency instead.
3. 8^3 / 12^3 / 16^3, each run to B1's criterion (so **P0 first, then B1**),
   **NOT to a common step count**. Matching step budgets is the design flaw that invalidated s31: the
   finest arms truncated hardest, so every comparison was biased toward the
   conclusion it reached.
4. Unknown, gated on B1's timescale; at least 3x B1, dominated by the finest
   arm.
5. NULL = the same floor at every h, which is a REAL and more interesting
   result than a lower one: it says something other than the discretisation is
   limiting the answer, and that is worth chasing.

**B3. Find the dt knee. (bounded)**
1. Where between `dt=3e-3` and the linesearch's ~3e-2 does reconnection go
   over the cliff?
2. How much of the linesearch's speed can be kept without losing surfaces.
   Today there is a flat shelf (0.0216-0.0261 across dt=1e-4..3e-3) and a jump
   to 1.375 at the linesearch, with **a full decade unsampled between them**.
   58-73x is one comparison, not a curve.
3. Two w7x_ini 8^3 arms at fixed `dt=1e-2` and `dt=3e-2`, 3000 steps, Poincare
   on, otherwise as D1.
4. **~1.5 GPU-h** (estimate, 0.91 s/step measured).
5. NULL = 1e-2 is already over the cliff, so the shelf simply ends at 3e-3 and
   the cap is the recommendation with no tuning available.

**B4. L-BFGS at m > 1. (bounded)**
1. Does history `m > 1` help now that the velocity-space secant bug is fixed?
2. Whether the repaired L-BFGS has more to give. Every number in the L-BFGS
   factorial (s(lbfgs)) was measured at `m=1`, and an earlier m=5 probe
   suggested pairs go stale fast -- but that probe predates the fix.
3. fmm002 8^3 p=3, `--arms lbfgs --history {2,5,10}`, 3000 steps each.
4. **~2.5 GPU-h** (estimate).
5. NULL = m>1 is flat or worse, `m=1` is the setting, the staleness
   observation is confirmed post-fix, and nobody needs to look again.

**B5. The pressure-shape TURNAROUND. (bounded)**
1. Why does the pressure-shape residual move TOWARD the reference profile and
   then AWAY as a run lengthens?
2. Whether the scheme's fixed point IS the file's equilibrium or merely passes
   near it -- which is the substance of "does relaxation recover the
   equilibrium".
3. One fmm002 8^3 arm, 14000 steps, `--helicity-every 100` (S07 sampled at
   500, too coarse to locate the turn), to find the turnaround step and check
   it against the helicity trace.
4. **~3.4 GPU-h** (estimate, 0.87 s/step x 14000).
5. NULL = the turnaround tracks accumulated `|dH|`, in which case it is the
   same reconnection mechanism as everything else here and needs no separate
   explanation.

**B6. IMPLICIT_MIDPOINT. (NOT bounded)**
The brief explicitly forbade attempting it and asked for an opinion instead;
that opinion is recorded earlier in this document and is unchanged. It stays
on the shelf as design work, not as a sweep -- there is nothing to launch
until someone decides the nonlinear solve is worth building.

### 34.1 CONSIDERED AND DROPPED, because the data already decides them

**"Does the pressure profile drift only on cases that are NOT equilibria?"**
This was the proposed framing, on the reasoning that `w7x_ini` is GVEC's
initial guess while `fmm002` is a real equilibrium at beta 1.8%. **DROPPED --
the data refutes the premise.** `p_resid` first -> final:

    fmm002  8^3    3000 steps    0.0450 -> 0.0213   TOWARD  x0.47
    fmm002  8^3   13018 steps    0.0450 -> 0.0789   AWAY    x1.75
    fmm002 12^3    3000 steps    0.0253 -> 0.0158   TOWARD  x0.62
    w7x_ini 8^3    linesearch    0.0894 -> 0.3157   AWAY    x3.53
    w7x_ini 8^3    capped dt     0.0894 -> 0.0122   TOWARD  x0.14
    w7x_ini 12^3   linesearch    0.0928 -> 0.0846   TOWARD  x0.91

The split is **not** equilibrium-versus-guess. D1 -- `w7x_ini`, GVEC's initial
guess, capped step -- shows the STRONGEST convergence in the whole set
(x0.14), while S07 -- `fmm002`, a genuine equilibrium, run long -- moves AWAY
(x1.75). S07 and W1 are the same case with **identical geometry, ns, p, gamma
and mu**, differing only in length (14000 vs 3000 steps), so the residual
turns around within a single configuration.

What the table is consistent with is the mechanism that runs through this
whole study: greedy-and-short goes toward, greedy-and-long goes away, capped
goes strongly toward. That is accumulated reconnection, not a property of the
target. Stated as consistent-with, not demonstrated -- which is exactly why B5
replaces this item rather than deleting it.

## 35. THE p-SWEEP REVERSES SIGN ON THE HONEST METRIC -- and P5 shows why

P5 landed and completes the odd-p series, which is the part of the sweep that
is readable at all (s33.2: p=2 and p=4 are pre-fix).

    fmm002 8^3, gamma=0, 3000 steps

    p    H(0)          |dH|/H per dE      |dH| per dE
    1    -3.792e-03        13.74           5.209e-02
    2*   -5.380e-04        11.38           6.123e-03
    3    -1.781e-04        70.94           1.263e-02
    4*   -1.692e-04        20.90           3.535e-03
    5    +6.878e-06       161.80           1.113e-03
                                  * pre-fix even p, s33.2

**H(0) CHANGES SIGN BETWEEN p=4 AND p=5.** It is negative at every other
degree and lands at +6.88e-06 at p=5 -- 550x smaller in magnitude than p=1 and
straddling zero. So on the readable points the RELATIVE metric is dividing by a
quantity that is passing through its own zero, and `161.8` is not a
measurement of anything.

### 35.1 The absolute metric says the OPPOSITE, and monotonically

`|dH|` normalised by energy removed, which is the form s19.2 established
correlates with surface destruction under blind classification of the Poincare
pairs:

    p=1  5.209e-02
    p=3  1.263e-02
    p=5  1.113e-03      <- 47x better than p=1, monotone

The relative metric said p-refinement makes reconnection 12x WORSE from p=1 to
p=5. The absolute metric says it makes it **47x better, monotonically**. The
difference is entirely the denominator: `H` falls 550x across that range while
`|dH|` falls 790x, so their ratio rises even as the thing anyone cares about
improves.

The even points, pre-fix as they are, sit below their odd neighbours on this
metric too (6.12e-03 and 3.54e-03) -- so the even/odd split noted in s33.2 is
still visible and A1 is still worth running. But it is now a second-order
effect on top of a clean monotone trend, rather than the whole signal.

### 35.2 What this costs the rest of the document

**The `|dH|/H per dE` column is unusable for every `fmm002` row**, not just the
p-sweep ones. Its denominator is a vanishing, sign-changing residue of a field
that is 99.99% harmonic, and it collapses with BOTH h and p (s31.2 as
corrected). That column is most of the table.

It remains sound on `w7x_ini`, whose current-driven fraction is stable to three
digits under refinement -- and every step-size conclusion, which is the
study's main result, was measured there. So the finding that survives
untouched is the one that mattered most.

**What this does NOT do:** it does not reinstate the withdrawn "p buys 3.4x"
(s33.2), which compared p=3 to p=4 and therefore still straddles the operator
fix. And s32 applies here as everywhere -- no arm floored, so none of this is a
floor result. It is a statement about helicity conservation per unit energy
removed at a fixed step budget, which is a real quantity and not the same
thing.

## 36. The preconditioner A/B, cleanly: same answer, 1.54x faster

M2 landed and is the controlled comparison the mu sweep was missing -- S04 and
M2 are `gamma=1, mu=1e-3` on `fmm002` 8^3, differing ONLY in the `M + eps L`
preconditioner (s(precond)).

              dE          |dH|/H per dE    s/step
    S04   1.207e-04          30.09          2.74     old (diag(M), eps ignored)
    M2    1.207e-04          28.98          1.78     new (metric-lumping)

Energy removed agrees **to four digits**. The helicity ratio differs by 3.7%,
which is below the two-digit threshold this project treats as worth
investigating. Cost falls **1.54x**.

This is exactly what the theory predicts and what I claimed before measuring
it: a preconditioner changes the path to the answer, not the answer. It is
worth having measured rather than asserted, because the claim was load-bearing
-- M1's 41x result (s(mu)) is attributed to `mu`, and that attribution is only
valid if the preconditioner change riding along with it is inert. It is.

`||F||` final differs more (1.139e-04 vs 2.017e-04). That is not a
counterexample: `||F||` is the gradient's norm, is not monotone, and is a
pointwise reading at an arbitrary stopping step. Energy and helicity -- the
integrated quantities -- agree.

### 35.3 An independent check of 35.1 that came back FLAT

The p-refinement result in s35.1 rests on `|dH|`, and the whole reason s35
exists is that the helicity denominator on `fmm002` is untrustworthy. So it is
worth testing the same claim with a diagnostic that has no such denominator:
roughness `||J||/||B||`, where growth means grid-scale structure and decay
means the field is getting smoother.

    fmm002 8^3, ||J||/||B|| first -> last        (* pre-fix even p)

    p=1    7.034e-01 -> 1.298e-01    x0.185
    p=2*   2.850e-01 -> 5.339e-02    x0.187
    p=3    -- NOT RECORDED, see below
    p=4*   4.441e-01 -> 5.150e-02    x0.116
    p=5    3.198e-01 -> 6.062e-02    x0.190

**Flat.** Every degree lands between 0.116 and 0.190, with no trend in p. The
h-sweep is the same story (12^3 x0.287, 16^3 x0.308).

**This does NOT corroborate s35.1, and it does not contradict it either.** The
two diagnostics measure different things: `|dH|` is an integral, topological
quantity, and `||J||/||B||` is a gradient scale. p-refinement improving
absolute helicity conservation 47x while leaving the final smoothness
unchanged is coherent -- it just means the independent confirmation I went
looking for is not available from this diagnostic. Recorded because a check
that comes back null is worth as much as one that lands, and burying it would
leave the next person to run it again.

### 35.4 The roughness diagnostic is missing from the two arms that need it most

`JoverB` was added to the trace partway through the campaign, so it is absent
from:

* **W1** -- the p=3 baseline, which puts a hole in the MIDDLE of the p series
  above, and W1 is the reference every other `fmm002` arm is compared against.
* **LR3** -- the chaotic linesearch arm. This is the sharpest gap: the
  diagnostic exists precisely to separate "physically chaotic but smooth" from
  "numerically shredded at the grid scale", and it is missing from the one arm
  in the campaign that is unambiguously chaotic (s(poincare): nested -> pure
  chaos).

So the question the diagnostic was added to answer -- **was LR3's chaos
physical or numerical?** -- is still open, and cannot be closed from what is on
disk. Both fields are saved (`B.h5`), so this needs a short re-run rather than
a full arm, but it does need GPU time and nothing is being launched.

**Shelf item A5** (class A, restores a confounded claim): re-measure
`||J||/||B||` at the IC and final state for W1 and LR3 from their saved
fields. ~0.2 GPU-h estimated, both are 8^3. NULL = LR3's roughness is flat
like every other arm, which would say its chaos is physical rather than
grid-scale, and would materially change how s(poincare) should be read.

## 37. The mu sweep completes: a SHARP minimum at 1e-4, and it is also the cheapest

M1-M5 all landed. `fmm002` 8^3 p=3, 3000 steps, every arm removing the same
energy to three digits (1.17-1.22e-04), so these are directly comparable -- and
because they share a geometry, resolution and p, they share an `H(0)` too, so
s35's collapsing-denominator objection does NOT apply within this group.

    gamma  mu       |dH|/H per dE     s/step
    0      --          70.94           0.87     W1
    1      1e-4         1.719          1.24     M1   <- minimum
    1      1e-3        28.98           1.78     M2
    1      1e-2        41.71           3.49     M3
    1      1e-1        43.76           7.99     M4   (truncated, 2004 steps)
    2      1e-3        28.36           2.64     M5

**The minimum is sharp, not a shelf.** M1 is 17x better than its nearest
neighbour one decade away and 41x better than no hyperregularisation at all.
Above 1e-4 the metric climbs back toward the gamma=0 value and flattens near
it -- so strong smoothing buys nothing, which matches the mechanism: at
mu*lambda_max >> 1 the smoother is damping resolved modes, i.e. physics, not
grid noise.

**And 1e-4 is the CHEAPEST of the hyperregularised arms**, at 1.24 s/step
against 1.78, 3.49 and 7.99. Cost rises steeply with mu because the shifted
solve gets harder, so there is no trade to make here: the best setting is also
the fastest. That is unusual in this study and worth stating plainly.

**gamma barely matters; mu is the lever.** M5 (gamma=2) and M2 (gamma=1) at the
same mu land at 28.36 and 28.98 -- a 2% difference, well inside the noise floor
this project treats as uninteresting. Raising the hyperregularisation ORDER is
not what produced M1's result; the SCALE is.

### 37.1 Two things this does not settle

**The dip could be deeper.** Nothing has sampled below 1e-4, so 1e-4 may be the
minimum or merely the best of four decades. That is shelf item A2, and it is
the arm I trimmed from the `w7x-ini-conv` sweep to stay inside 20 GPU-h --
which now looks like the wrong trim.

**It may not transfer.** M1-M5 are one geometry. H1/H2 (12^3, mu = 4.4e-4 and
1e-3) test whether the optimum moves with resolution as `mu ~ h^2` predicts,
and C5-C7 re-test the bracket on `w7x-ini-conv`. Both were in flight when this
was written.

s32 applies as everywhere: no arm floored, so this is helicity conservation per
unit energy removed at a fixed step budget, not a floor result.
