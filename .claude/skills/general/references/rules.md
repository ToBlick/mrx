# MRX-specific rules

The matrix-free and solver rules below bind every contribution. The general working rules (git, compute
budget, code hygiene, tests, performance, docs, drafts) are in the user-level skill `working-with-tobias`,
the rest of this file holds their MRX instances.

## Matrix-free and solver strategy

Every contribution respects these.

- **Matrix-free.** No assembled or dense shortcuts in production unless the task needs it. Dense
  references only on tiny sequences in tests. Preconditioner memory grows like the DoFs (the slab model
  is cut to a theta band for that reason).
- **No inner Krylov in the preconditioner of MINRES or CG.** A preconditioner whose action is an unresolved
  Krylov solve breaks a non-flexible outer method. Use a well-defined approximate apply instead: where
  `M^{-1}` sits inside a preconditioner, the mass atom replaces it. An inner solve inside the operator is
  allowed only at a tolerance tight enough that the outer method sees a fixed operator (the Newton mass solves
  at 1e-6). An inner CG in a preconditioner needs a flexible outer method (FGMRES, flexible GCR), with its
  tolerance derived from the outer solve rather than a fixed step count.
- **Polynomial inverses** (Chebyshev and other fixed polynomials) only on merit and at low degree, on
  well-conditioned operators or as smoothers. Prefer one atom apply, exact small dense blocks, or
  restructuring at the solver level (sequential SPD solves as in the Hodge split).
- **Singular `S_k` only with a consistent right-hand side** (in the range of `G_k^T`, formed from a lifted
  vector in float32). Genuine Krylov breakdowns (`beta == 0`) terminate, they do not divide.
- **Metric factors explicit.** Never fold a metric weight into a matrix that gets inverted or into an outer
  diagonal sandwich. Write each factor as a term, return weight and mesh part separately. Hidden factors get
  "corrected" by fitted constants, and the fit then validates the bug.

## Code

- No defensive code over solver invariants: the MINRES `sqrt(abs(.))` hid an indefinite preconditioner for
  months.
- **No map sign, no `frame='ref'` loads.** Maps are always positive-Jacobian, angles right-handed.
- **jit/host rules** in `code-map.md`. Use JAX linear algebra for anything on device.
- **Precision:** new solve code works in both precisions. Its accurate part (iterate, residual, the applies
  that form them) goes through `lift` and the `DS` operator applies, everything else stays `LOW_DTYPE`.
- **Shipped surface:** library, tests, `scripts/relax.py`, `poincare_trace.py`, `poincare_plot.py`,
  `plot_mesh.py`, tutorials. Study, sweep and figure code lives next to its outputs under `$MRX_DATA/runs/mrx/`
  or on branch `paper`. Code used only by paper scripts counts as dead on the dev branch.
- Validation that needs a GPU: package it as a job through `slurm/run.sh`, cheapest focused check first.

## Tests

- Use the li383 fixtures of `test/conftest.py`. Exact identities at a multiple of `mrx.eps()`, solver
  quantities at a multiple of `seq.tol`, measured bands stated next to the value. Run in both precisions.

## Docs

- The tree is `docs/source/` (Sphinx). New concept pages go into `docs/source/concepts/`. `docs/research/` is
  the measurement record, nothing that duplicates the docs.

## Git

- `main` is write-protected (since v0.1.0, 2026-09-30).
- Commit-message and PR drafts go to files under the data root, never into the checkout.

## Figures, tables and the paper

House style (`mrx/diagnostics/plotstyle.py`, `mrx.mplstyle`), look at the PNG before delivering, write it under `$MRX_DATA/runs/mrx/<line>/`, no paper text
ahead of verified results.
