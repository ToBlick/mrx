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

**What Tobias must decide** is in section 9.

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
*together with* a collapsed `gain` and an exploding `dt`.

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

**Worth keeping: `div_curl_rel` is 1.3e-10, not machine zero.** The strong
operators are M-projected (`M2^-1 D1`) rather than raw incidence — which is
deliberate and correct under polar extraction, where the extraction is
non-unitary and only the mass-projected form preserves `d.d = 0` on extracted
DoFs. The consequence is that **1e-10 is the best any trajectory can hold
`div B` on this discretisation**, by any method. Observed `div B` never moved
off its IC value of 3.676e-04 in any run, so this floor was never the binding
constraint here — but it is the number people otherwise rediscover as a bug.

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

**I left it in place** so the `legacy` arm reproduces shipped behaviour
faithfully for the A/B, and instead surfaced `sy` every step as a new `State`
field. **Recommendation: delete the clamp and skip the update on
`sy <= 0`.** With the pairing fixed it never fires on this problem, so it is
dead weight that can only hide a future regression.

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
* **Helicity conservation.** Every helicity number produced before the section
  6.1 fix is void. Corrected numbers are in section 10 if job 16771114 landed
  before this was written; if that section says otherwise, they are not
  available and no claim about helicity conservation should be read into this
  document.
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
2. **Delete the `gamma` clamp in `_lbfgs_direction`** and skip the update on
   `sy <= 0` instead. With the pairing fixed it never fires, so it can now only
   hide a regression.
3. **Whether `lbfgs_pairing` / `cg_beta` should stay as knobs at all.** I added
   them to make the A/B decisive in a single job. The `legacy` and `paired`
   values are now known-bad and exist only to reproduce the bug. If they are
   not wanted in production, delete them and keep only the fixed path — the
   defaults already select it.
4. **Whether to keep chasing L-BFGS or settle on CG.** They are level here
   (section 8), and CG is simpler, has no curvature condition to violate, and
   its `history_size` is inert. Fixed L-BFGS gained nothing from m=5.

---

## 10. Hyperregularisation, and corrected helicity

`gamma = 0` throughout sections 4 and 5, as the brief instructs.

RESULTS PENDING — jobs 16771114 (helicity-corrected repeat) and 16771115
(`gamma=1`, `mu=1e-3`) were still running when this section was written.

---

## 11. Reproduction

    sbatch slurm/job_relax_prelim.sh --geometry quasr44970 --ns 8,16,8 --p 3 \
        --steps 250 --arms gradient,cg,cg-legacy,lbfgs-legacy,lbfgs-paired,lbfgs \
        --out out/relax_prelim/mainA.json

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
