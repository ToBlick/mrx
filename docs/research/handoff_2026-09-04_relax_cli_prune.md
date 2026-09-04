# Handoff 2026-09-04 — pruning the `scripts/relax.py` hyperparameters

Purpose: decide which relaxation-driver CLI options are stale and cut them, and
name the survivors better. Companion to `docs/research/simplify_findings_2026-09-04.md`
(item 6) and `docs/research/implicit_midpoint_2026-09-04.md`. No code changed yet.

## Current inventory (`scripts/relax.py:214-295`)

~40 flags. Grouped:

- geometry/discretisation: `--geometry --nfp --ns --p --precision`
- IC: `--ic {clebsch,analytic,dzeta}`; analytic-only `--iota --iota-exp --flux-exp --lam`;
  clebsch-only seed `--seed --seed-eps`
- descent: `--method {gradient,lbfgs} --history --velocity-smoothing-order
  --velocity-smoothing-scale --stepper {h,bonly}`
- pre-smoothing: `--presmooth --presmooth-eps --presmooth-jb`
- time step: `--dt-mode {linesearch,fixed} --dt0 --cfl`
- resistivity: `--eta-max --eta-schedule {tanh,constant,linear,pulse} --eta-pulse --eta-every`
- adaptive pulses: `--pulse-adaptive --pulse-stall --pulse-eps0 --pulse-grow
  --pulse-gain --pulse-helicity --pulse-spacing`
- termination: `--maxiter --tol --steps --seconds --floor-tol --floor-steps`
- I/O: `--save-every --qoi-every --out --checkpoint --restart`

## Factual checks (answering the questions raised)

- **`--ic`**: we do NOT run only from files. `clebsch` = VMEC/GVEC file (production
  default). `analytic` = synthetic logical-grid B, still live (the file-free smoke
  path in `slurm/README.md` and `docs/source/relaxation.md`; only route to exact
  cylinder equilibria). `dzeta` = constant 2-form -> harmonic field, appears only in
  August research tables, no current script/test. => keep clebsch+analytic, drop dzeta.
- **`--method`**: "LBFGS(0) == gradient" is false two ways. `history_size < 1` is
  rejected (`mrx/relaxation.py:436`), so there is no m=0. m=1 (default) is memoryless
  BFGS == nonlinear CG (Polak-Ribiere), NOT steepest descent (`relaxation.py:373-377`).
  `--method gradient` is a separate genuine steepest-descent branch that nothing runs.
  => drop `--method`, hardcode LBFGS; `--history` can also be fixed at 1 ("larger
  history_size measured to add nothing", `relaxation.py:377`) unless still swept.
- **`--floor-tol` vs `--tol`**: different layers, both needed. `--tol` -> `seq.tol`,
  the inner MINRES/CG solve tolerance of every linear solve. `--floor-tol` -> the
  outer stop test on the windowed relative force residual. Coupled (the force residual
  cannot floor below the solve tol, `relax.py:114-116`) but not redundant. Suggest
  renaming `--tol` -> `--solve-tol`.

## Answers from the implicit-midpoint branch (`origin/implicit-midpoint`, head 2549514)

- **`--dt-mode`**: drop `fixed`, keep the line search as the only dt rule. Midpoint
  does NOT replace step selection; it is a separate `--scheme {explicit,midpoint}`
  flag (IntegrationScheme) that takes the explicit predictor's line-search dt + CFL
  cap as given -- only the induction `B_{n+1} = B_n + dt curl(u x H_mid)` is implicit
  (Picard on the increment, internal dt-halving that never triggered). Caveat for docs:
  under midpoint the line-search identity `dE = -dt<F,u>/2` is no longer exact (true
  change `-dt<u,F_mid>`); their driver labels it.
- **`--dirichlet-H`**: already exposed on their branch, threaded through the driver's
  compute_force/weak_pressure and TimeStepper. Never set true before. It changes the
  descent (H_t=0 at the wall: 1.7x more energy removed, 5x higher force floor on the
  smoke mesh); acceptability is Tobias's call (their doc section 5).
- **bonly cleanup**: their branch removes the ~110-line duplication --
  `BOnlyTimeStepper` overrides `_ideal_increment`/`_midpoint_solve` instead of copying
  `relaxation_step`. So H-off becomes a clean first-class branch, and
  `--stepper bonly --scheme midpoint` runs.
- **pulses**: no interaction. The resistive half (backward Euler on B_ideal) is
  untouched by the scheme; pulse-as-%-of-H is purely the resistive step's business.

## Physics call for Tobias: H off by default drops exact helicity conservation

Measured on the midpoint branch:

- H off (bonly): conserves nothing exactly -- helicity drifts ~1e-6 (projection error
  of the pairing; same for explicit and midpoint).
- H on + `--dirichlet-H`: exact discrete helicity ~5e-12 (li383 (8,16,16) p2 f64;
  2e-12 vs 9e-8 on (12,24,24) p3), because `E^T P B = E^T M_1 H` needs E and H in the
  same space.

So "H off by default" = the default scheme no longer conserves helicity to machine
precision; it leaks at the ~1e-6 projection-error level. May be the intended honest
default (exact conservation as an opt-in publication mode), but confirm before baking
it in -- helicity conservation is the framework's headline invariant.

## Decisions taken (Tobias) and status

| Change | Status |
|---|---|
| drop `--ic dzeta` | decided |
| drop `--method` (hardcode LBFGS) | decided (see history note) |
| drop `--presmooth{,-eps,-jb}` (kept for the deleted h5 route) | decided |
| hardcode `--dt0` (=1.0) | decided |
| drop `--dt-mode fixed`, keep line search | decided, confirmed by midpoint |
| drop resistivity schedule `--eta-schedule --eta-max --eta-every --eta-pulse` | decided |
| pulses: keep spacing+strength only, strength as % of H; kill controller (`--pulse-adaptive --pulse-stall --pulse-eps0 --pulse-grow --pulse-gain`) | decided |
| `--stepper` -> `--auxiliary-magnetic-field` (H on/off), default OFF | decided; confirm helicity trade-off above |

## Proposed final CLI

Removed (~16): `--ic dzeta`, `--method`, `--presmooth{,-eps,-jb}`, `--dt0`,
`--dt-mode fixed`, `--eta-schedule`, `--eta-max`, `--eta-every`, `--eta-pulse`,
`--pulse-adaptive`, `--pulse-stall`, `--pulse-eps0`, `--pulse-grow`, `--pulse-gain`.

Renamed:
- `--stepper` -> `--auxiliary-magnetic-field` (H on/off); keep `--dirichlet-H` as its
  sub-option and have the driver REJECT H-off + `--dirichlet-H` (meaningless combo)
  rather than ignore it.
- `--pulse-helicity` -> `--pulse-strength` (target dH/H per pulse). Dose estimator:
  a resistive dose spends `dH ~ -2 (eta dt) int J.B`, so `eta dt = (dH/H) H / (2 int J.B)`
  -- one force eval per pulse, which the qoi sampler already computes. NOTE: this is the
  analytic leading-order relation (assumes J.B ~ const over the pulse); NOT checked
  against data. Confirm the sign and the factor of 2 against one real pulse before
  relying on it.
- (optional) `--tol` -> `--solve-tol`.

Added by the midpoint merge: `--scheme {explicit,midpoint}`, `--dirichlet-H`.

## Sequencing

The midpoint branch rewrites `relaxation_step` and `relax.py` in the same regions the
prune touches, and there is a known conflict: their `relaxation_step` collides with the
resistive-clock change on `li383-followups` (clock advances only while eta > 0).
Resolution: keep the clock rule inside the shared base `relaxation_step` (both steppers
now share it).

Recommendation: land `implicit-midpoint` first, then do the whole prune as one pass on
top of it -- `--scheme`/`--dirichlet-H` present, bonly refactor in, clock conflict
resolved once. Pruning now on a parallel branch doubles the merge pain.

## Script dependencies on the cut flags (verified 2026-09-04)

`scripts/li383_pub.sh` is the heavy dependency -- it exercises nearly every cut flag,
so the prune breaks its arms and they must be migrated or deleted in lockstep (this IS
the "which experiments are closed" call):

- resistive-ramp arms (~L124-137): `--eta-schedule tanh --eta-max --eta-every`
- scheduled-pulse arms (~L179-184): `--eta-schedule pulse --eta-pulse --eta-max --eta-every`
- adaptive-controller arms (~L188-194): `--pulse-adaptive --pulse-spacing`
- twin H/bonly arms (L153,155,164,165): `--stepper bonly` and `--stepper $st` (loops h,bonly)

Migration: pulse arms -> `--pulse-spacing` + `--pulse-strength` (dH/H); H arms ->
`--auxiliary-magnetic-field` on/off (the bonly arms just drop the flag once H-off is
default); the tanh/linear ramp arms have no successor and are the ones to delete if that
experiment is closed.

Other:
- `scripts/li383_sweep.sh:21` passes `--method cg` -- already invalid today (choices are
  gradient|lbfgs; the cg arm was deleted 2026-08-28), so it is stale/broken now, not
  just by the prune. Drop it with the `--method` removal.
- The `--tol` -> `--solve-tol` rename touches NO script: every `--tol` in `slurm/` is a
  different (diag/solve) script, and li383_pub/sweep use `--floor-tol`, not `--tol`.
- `scripts/tutorials/li383_resistive.py` has its OWN `--eta-max`/`--eta-every` argparse
  (separate script, own tanh schedule) -- not a relax.py dependency, but it still teaches
  the tanh ramp if the concept is retired framework-wide.

## Open decisions

1. Confirm H-off-as-default is intended (accepting the ~1e-6 helicity leak in the
   default; exact conservation via H-on + `--dirichlet-H`).
2. Wait for the midpoint merge before pruning (recommended), or cut the
   merge-independent bits (dzeta, method, presmooth, eta-schedule) now.
3. Keep `--history` as a swept knob, or fix it at 1?
4. Rename `--tol` -> `--solve-tol` (touches existing scripts), or leave it?

## Executed 2026-09-04 (li383-followups, after the implicit-midpoint merge)

Tobias's answers to the open decisions, and what landed:

- `--history 0` IS steepest descent inside L-BFGS (the two-loop recursion with
  no pair returns `F`; one static guard); `DescentMethod`, `--method` gone.
- `--stepper h|bonly` -> `--auxiliary-B-field {false,true}`, default **false**
  (the 2-form `B` in both cross products). `true` is the H route, and H is
  ALWAYS the Dirichlet 1-form now ("if we use H, we should use it with
  Dirichlet, else it loses its purpose"), so `--dirichlet-H` is gone with it.
  `BOnlyTimeStepper` folded into `TimeStepper` as static branches
  (`compute_force(B, seq, auxiliary_B_field)` returns `(F, p, J, X, JxX)`);
  `mrx/experimental/bonly_relaxation.py` deleted. Consequence: no run of the
  publication sweeps (natural H) is reproducible bit for bit; `aux pairs` in
  `scripts/li383_pub.sh` is the bridge.
- `--ic` gone: the geometry file decides (`.nc` vmec, `.dat` gvec, `.json`
  analytic map + profile parameters; `data/torus.json`, `cylinder.json`,
  `rot_ellipse.json`); bare `toroid`/`cylinder`/`rot-ellipse` names gone,
  `--iota/--iota-exp/--flux-exp/--lam` moved into the JSON `profile` block;
  `dzeta_form`, `parse_lambda` deleted.
- dt: line search only (`TimeStepChoice`, `--dt-mode`, `--dt0` gone); the
  midpoint solve keeps its halving, the three `--picard-*` flags became
  `PICARD_TOL_FACTOR = 10` (times `seq.tol`), `PICARD_MAX = 20`,
  `PICARD_RESTARTS = 4`.
- Resistivity: the in-loop half is deleted entirely (`State.eta`,
  `resistive_*`, `eta_every`, `resistive=`, the clock rule, `eta_of_step`,
  `relaxation_loop(resistivity_schedule=)`); `resistive_step` stays as the
  standalone reconnection solve. `--reconnect-eps c` -> `--reconnect-helicity X`
  with `eps = X |H| / (2 |J^T P B|)` (checked on GPU, see the smoke log).
- `--tol`/`--maxiter` -> `--solve-tol`/`--solve-maxiter`; `--presmooth*` gone.
- Consumers: `scripts/li383_pub.sh` (eta and pulse groups deleted, `bonly` ->
  `aux`, `--history 0` in the anchor group, reconnect doses in helicity),
  `scripts/midpoint_sweep.sh` (natural-H arms dropped), `scripts/li383_sweep.sh`
  deleted (`--method cg`, dead since 2026-08-28), plotters no longer read
  `maxiter` from the h5 attributes, test updated, docs updated.
  `scripts/tutorials/` and `docs/source/tutorials.md` belong to the tutorials
  session (branch tutorial-cell-markers), which has the final signatures.
