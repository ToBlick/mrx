# Research record

Dated measurements behind MRX: timings with their iteration counts, the numbers that ruled alternatives out,
checks against external codes, precision limits and the basis of each tuned constant. How the code works is in the
Sphinx docs under `docs/source/`.

- `preconditioners.md`: the Laplacian and mass atoms, the slab mass model, the S_2 multigrid, the boundary-penalty scale, the Newton preconditioner, and the preconditioners that lost.
- `solvers_precision.md`: the Hodge split, L_3 and Leray, the shifted split, stopping rules and part solves, harmonic forms, precision limits.
- `performance.md`: quadrature, Newton step cost, per-apply profiles, symmetry models, setup, memory, tracing, TPU.
- `relaxation.md`: descent smoothing, the parallel-flow penalty, the Gauss-Newton direction against GCR and Newton-MR, and the relaxation methods that lost.
- `geometry_vacuum.md`: GVEC and VMEC conventions, vacuum convergence, the map, initial conditions, symmetry reductions, graded spaces.
- `poincare_islands.md`: tracer calibration, island width, Cary-Hanson residue.
