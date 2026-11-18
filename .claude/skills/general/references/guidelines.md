# MRX guidelines

General guidelines for code development -- these are up for discussion, reach out on github!

## Matrix-free and solver strategy

- **Matrix-free.** Nothing should have a memory footprint larger than constant times number of DoFs (n^3).
  This rules out factorizations of magnetic-surface or poloidal-plane operators, which grow like n^4. The 
  slab model is cut to a theta band for that reason.
- **No inner Krylov in the preconditioner of MINRES or CG.** These solvers assume linear operators. Krylov-in-
  Krylov is only ok for flexible solvers like FGMRES or flexible GCR. In some cases, `M^{-1}` inside a 
  preconditioner can be replaced by the mass atom.
- **Polynomial inverses** (Chebyshev and other fixed polynomials) only on merit and at low degree, on
  well-conditioned operators or as smoothers. Prefer one atom apply, exact small dense blocks, or
  restructuring at the solver level (sequential SPD solves as in the Hodge split).
- **Singular `S_k` only with a consistent right-hand side** (in the range of `G_k^T`, formed from a lifted
  vector in float32). Genuine Krylov breakdowns (`beta == 0`) terminate, they do not divide.
- **Metric factors explicit.** Never fold a metric weight into a matrix that gets inverted or into an outer
  diagonal sandwich. Write each factor as a term, return weight and mesh part separately. Hidden factors get
  "corrected" by fitted constants, and the fit then validates the bug. This also keeps shape optimization simple.

## Code

- **No zombie code.** Removing a function, kind, flag or mode removes it fully in one change: accept-lists and
  their dispatch, defaults naming it, environment variables, calling scripts. No aliases, no back-compat shims,
  no "deprecated but accepted". Prefer a fresh name over recycling a retired one. If a number can never be
  regenerated after a deletion, measure it and put it in `docs/research/` first.
- **No defensive code.** No `abs()` inside `sqrt()`, no fallback on a denominator, no clamps over quantities the
  algorithm guarantees. Let it NaN and state the invariant in a comment. A masked invariant turns a bug into
  slow convergence that gets blamed on physics: MINRES `sqrt(abs(.))` hid an indefinite preconditioner in the past.
- **No map sign.** Maps are always positive-Jacobian, angles right-handed.
- **jit/host rules** in `code-map.md`. Use JAX linear algebra for anything on device.
- **Precision:** new solve code works in both precisions. Its accurate part (iterate, residual, the applies
  that form them) goes through `lift` and the `DS` operator applies, everything else stays `LOW_DTYPE`.
- **Shipped surface is minimal:** library, tests, `scripts/relax.py`, `poincare_trace.py`, `poincare_plot.py`,
  `plot_mesh.py`, tutorials. Study, sweep and figure code lives next to its outputs under `$MRX_DATA/runs/mrx/`
  or on branch `paper`, and is not polished or documented as if shipped. Code used only by paper scripts
  counts as dead on `main`.
- Validation that needs a GPU: package it as a job through `slurm/run.sh`, cheapest focused check first.
- MRX is a relatively small repo at <20k lines of code and it should stay that way. Try to avoid bloat, check if
  something already exists or can be extended before re-writing it from scratch. The same goes for additional 
  library dependencies.

## Tests (`docs/source/concepts/testing_strategy.md`)

- A test states a mathematical or functional claim at a tolerance class: exact identities at a multiple of
  `mrx.eps()`, solver quantities at a multiple of `seq.tol`, measured bands with the measured value stated next
  to them. No "is finite" tests, no tests of bad arguments or array shapes.
- Use the li383 fixtures of `test/conftest.py`. A new test gets the production configuration plus at most one
  contrasting case. It must pass in both precisions.
- The suite is lean and compile-bound: a test costs the distinct solves it compiles. No expensive or
  parametrised tests.

## Docs and the measurement record

- One documentation tree, `docs/source/` (Sphinx).
- `docs/research/` is the measurement record, nothing that duplicates the docs. An entry is terse and dated:
  setting, number, meaning. Timings come with their iteration counts and their conditions (case, mesh,
  degree, precision, device). Failed attempts are recorded too, in the "decided against / removed"
  section of `preconditioners.md`, `relaxation.md` and `solvers_precision.md`. No plans, handoffs or code 
  walkthroughs. This is really a vault for coding agents to read.
- Compare methods by total time, not by iteration count.

## Outputs

- Runs, figures, records and scratch files go under `$MRX_DATA/runs/mrx/<line>/`, never into the checkout or a
  worktree. Figures use the house style (`mrx/diagnostics/plotstyle.py`, `mrx/diagnostics/mrx.mplstyle`).

## Cluster use

- On a cluster, nothing that imports `mrx` and builds a sequence, solves, traces or runs pytest runs on a login
  node, not even with `JAX_PLATFORMS=cpu`. Submit a job through `slurm/run.sh`. Only matplotlib re-renders and
  pure-numpy checks run inline (`running.md`).

## Git

- `main` is protected on GitHub. Changes reach it through pull requests from a branch.
