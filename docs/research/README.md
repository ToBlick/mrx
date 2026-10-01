# Research record

**Info dump for your coding agent/LLM to go through.** 

These notes are terse on purpose: dense lists of settings and numbers. 
The Sphinx docs (`docs/source/`, the hosted documentation) should be easier to digest.

A record of the measurements behind MRX's design: spectra, iteration counts, timings, convergence
rates, and the numbers that ruled alternatives out. It does not describe the code. The Sphinx docs under `docs/source/` do. Nothing here is duplicated there.

Every entry is setting + number + what it established, dated (YYYY-MM-DD). Where a finding was
superseded only the newest is kept. Removed methods appear as one-liners under "Decided against /
removed" when their measurement justified a decision.

- `preconditioners.md`: metric-lumped Laplacian and mass atoms, the natural-BC term and
  `PRODUCTION_BC_SCALE = 3.0`, h-scaling, the shifted-stiffness atom, the Newton preconditioner,
  refuted variants.
- `solvers_precision.md`: Hodge split, shifted split, saddle MINRES, harmonic forms, operator
  identities, precision (refinement, pollution law, floor study, Newton inner solve).
- `performance.md`: per-step rates and scaling, Newton cost, symmetry models, setup and compile,
  memory, tracing, TPU vs GPU.
- `relaxation.md`: descent, step size and topology, Newton spectrum and parallel penalty, helicity,
  resolution sweeps, seeded islands and resistivity.
- `geometry_vacuum.md`: GVEC/VMEC conventions, the spline map, QA and W7-X vacuum convergence, initial
  conditions, discretisation checks, stellarator-symmetry reductions.
- `poincare_islands.md`: tracer calibration, island-width measure, Cary-Hanson residue, sections of relaxed states.
- `open.md`: unexplained effects and measurements never taken.
