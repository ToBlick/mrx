# Whole-repo cleanup review, 2026-09-04 (findings only, nothing applied)

Twelve review passes over the worktree at 1af3c80 (li383-followups): three slices (solver core: operators, metric_lumping_laplacian, derham_sequence, solvers, preconditioners, mass, nullspace, projectors, extraction_operators; the rest of the library; scripts, tutorials, tests, slurm) by four angles (reuse, simplification, efficiency, altitude). Each agent read its slice in full and grepped the rest for callers. Line numbers are those of 1af3c80. The per-slice reports follow the summary verbatim apart from formatting; nothing here has been checked beyond what the reports state, and nothing has been changed. Tobias decides.

## Status at 3420b39 (same day, after the midpoint merge and the relaxation prune)

Checked item by item against the tree (symbols grepped; the note's line
numbers have drifted, the symbols have not).

**Done since the note.** The B-only stepper duplication (library reuse 1,
altitude 1, summary B): folded into `TimeStepper` as static branches on
`auxiliary_B_field`, `mrx/experimental/bonly_relaxation.py` deleted. The
`relax.py` options nothing runs, the `dirichlet_H` threading and the
driver's eta schedule (summary B, scripts simplification 6, altitude 5):
gone with the prune (`docs/research/handoff_2026-09-04_relax_cli_prune.md`,
"Executed"). `scripts/li383_sweep.sh` deleted. `parse_lambda` and
`dzeta_form` deleted with `--ic`. The `_PROJECTION_SPACES` lookup written
twice (simplification 9): replaced by the pair rule, and the scalar pairs
turned out to be transposed, not just duplicated (OPEN 3.12, resolved
306d174). The li383 tutorials' copies of the driver mechanics (altitude 2,
reuse 5/6, simplification 9): the tutorials session redesigned tutorials 4
and 5 on `tutorial-cell-markers` and rewired 3 and 5 to the pruned API;
judge those items after that branch lands, not from this note.

**Done in 872e87c (the driver/library split, summary B "Driver vs
library", altitude 6 and 13, scripts altitude 1).** `mrx.relaxation.relax`
is the production loop (chunks, sampler, floor, wall budget, reconnection,
`on_chunk`), `force_scale` the residual normaliser through
`dot_product_load`, `make_sampler` the QoI sampler, `initial_field` the
initial condition from the geometry file, `write_checkpoint` /
`read_checkpoint` the run files (named State leaves in HDF5; `state.eqx`
and `B.h5` gone, the weak pressure computed on demand). `relaxation_loop`
deleted; the tests drive the production loop, including a reconnection and
a checkpoint round-trip; the tutorials' hand-written `B.h5` writers gone.
`scripts/relax.py` is the CLI plus the JSON writer. The `evaluate_at_xq`
reach-in in `pressure_diagnostics` (altitude 5) still stands.

**Still valid, unchanged.** Every section-A deletion in the solver core:
the Jacobi family (the contradiction with the memory and the release
review stands: recorded deleted 2026-08-27, present in the code), the
Schur-probe and `coupled` machinery, the boundary-DoF extraction family
with `apply_bc_mass_correction` reading an attribute that no longer
exists, the dead constructor knobs, `_assemble_weighted_1d_mass` twice,
the deflation projector three times, `apply_laplacian` /
`apply_laplacian_approx`, the dense `differential_forms` evaluators,
`stellarator_map` / `extend_map_nfp`, `clebsch_form`, `update_field`,
`relaxation_loop(callback=)`, `compute_geometry_terms`,
`_bootstrap_nullspace_guesses`, `compute_divergence_norm` next to
`divergence_norm`. Every efficiency quick win: deflation applying `M` to
`x` with `mass_vs` in hand, `eps * M` with a Python zero, `eigh` per
shifted solve inside the scan, `vs_lower` and `sigma` recomputed per step,
`gap_sweeps=5` by default, `build_sequence` building all eight atoms for
plotters (`scripts/plot_mesh.py` already sidesteps it by building the
sequence and map directly: the precedent). Section B: preconditioner
selection over nine functions, the four owners of the derivative-axes
rule, the double `set_operators(compute_nullspaces(...))` install, `sp`
threaded through five reader signatures, `mu = 0.064 h^2` with the
launcher still carrying five hand-computed scales, the solve-count
experiments (velocity Leray at smoothing order 0, warm-started `J`),
`from .plotting import *`. Scripts: `quad_order_equivalence.py` and
`li383_summary.py` still exist, `slurm/regen_poincare.sh` is still its own
sbatch wrapper with the TeX path in the repo, `MRX_MAP_BATCH_SIZE_INNER`
still read in two scripts, `poincare_relax.py`'s `--from-npz` branch still
a second render path.

**Changed in kind.** `make_force_normaliser` in the driver and the
`evaluate_at_xq` reach-in in `pressure_diagnostics` are still there, but
the public surface now covers both: `dot_product_load(B, B, 0, 2, 2)` (=
`magnitude_squared_load`) and `evaluate_at_quadrature`.

## Status after the cleanup commits of 2026-09-04 (evening)

**Done: the solver-core deletions** (Jacobi family, Schur probe,
`coupled`, boundary-DoF extraction, dead knobs, `_bootstrap_nullspace_guesses`,
the spec dataclasses; one preconditioner per solve, hardcoded), **the
duplicates** (`deflation_projectors` once, `_laplacian_apply`, one
divergence norm, one Clebsch loader through `read_equilibrium`, `inv33`,
`_winding`/`_slope`, `_pack_blocks`, `spline_map_DF_at_quad`) and **the
library rest** (dense evaluators, `div`/`curl`/`grad`, `stellarator_map`,
`extend_map_nfp` as `plot_torus(nfp=)`, `clebsch_form`, `update_field`,
`compute_geometry_terms`, `trace(adaptive=)`, `SectionLimits.coerce`, the
style parameters, the `__all__` shim deleted; `Pullback` kept with
`test_forms.py`, `jacobian_determinant` used by the det DF check).

**Done: the efficiency quick wins.** The shifted atom's core pair
diagonalised at build (`core_V`, `core_mu`; the per-solve `eigh` gone);
`vs_lower` gone with the lower-block deflation (a harmonic form has `D^T v
= 0`, so the saddle nullspace is `(v, 0)`: the deflation was a no-op that
cost a mass solve per Leray call and two mass applies per MINRES
iteration); `sigma` returned by `apply_inverse_laplacian_saddle` and used
as the Leray projection's gradient part (the second `M_2` solve gone);
`J` carried in `State` and warm-started, `sigma` warm-started from the
previous `JxX - F`; `eps * M` with a Python zero not applied;
`compute_nullspaces(gap_sweeps=0)` by default. Measured in
`outputs/prune_smoke/probe_batchD.py`. Not done: `build_sequence` for the
plotters (they solve: the weak pressure and the force need the k=1/2 mass
atoms and the k=0/3 Laplacian atoms; only the k=1/2 Laplacian atoms are
spare), the drift-check recompiles, the double parse of the equilibrium.

**Done: the two small ones.** `pressure_diagnostics` reads the two
pressures through `evaluate_at_quadrature`; `cross_product_load_values`
forms the cross product in the representation that pairs metric-free
with the output basis and hands it to `_vector_load_values`, the eight
explicit cases gone. Per apply against the old kernel
(`outputs/prune_smoke/probe_batchE.py`, li383 (8,16,16) p=2): the force
kernel bit-identical, the induction to 2e-10, the other cases at float32
round-off (1e-7, the `1/J` associated with the product instead of the
weight).

**Done 2026-09-05: mixed precision and its consequences.** Every solve
is float32 Krylov refined against a float64 residual on the float64 view
of the sequence (`concepts/precision.md`); the velocity Leray projection
left the step; the harmonic forms and the gap are built on the view; the
mass and projection weights ride on the geometry pytree with the plans
on the sequence (level 1 of "the applies take the geometry as an
argument"). Level 2, the step reading geometry and bundle from its
arguments with the preconditioner atoms as pytrees, is what would remove
the geometry literals from the jitted step and allow `vmap` over an
ensemble; not done.

**Done 2026-09-05: one stopping criterion.** Every solve runs under
`refine` in every configuration, the true residual of the outer
operator measured in the mass-atom norm of its space; the Hodge split
and the shifted split run under one pair loop on `(x, w)` with the
saddle residual (two applies, no nested inverse; their own solves are
inner solves at the square root of the tolerance; the Laplacian's
harmonic forms deflated, nothing for the shifted operator). Before,
three criteria coexisted (preconditioned norms inside the Krylov
iterations, a 2-norm in the refinement, nothing for the composites),
and the k=1 Hodge split reported convergence at 1e-8 with a true
residual of 2e-6, which the k=2 harmonic form inherited squared. The
criterion then exposed the root cause of that residual: the atoms'
dense cores were inverted with a cut-off of 4096 float32 epsilons,
5e-4, which zeroed real modes on p=3 meshes and left the preconditioner
singular, so CG's own criterion was blind there. The cores are probed
and inverted on the float64 view at 1e-12 now; in a float32 process the
k=1 Dirichlet builder solve converges in 936 / 1082 iterations to 1e-13
on li383 (12,24,24) / QA (16,32,32) p=3 (was 3e5 iterations, 2e-7), the
k=2 form's weak half is 1e-24. One tolerance per solve (defaults 1e-8
refined float32, 1e-10 float64, 1e-6 plain float32; the inner tolerance
its square root); `relax` refuses a tolerance above `floor_tol^2`. Cost:
the Poisson counts are 2.5-3x the 2026-09-02 baselines (one float32
pass to 3.5e-4 then, two passes to 1e-8 now), the step 0.68 s against
0.50 at (12,24,24) p=3 float32.

**The "div f = 0" question, measured** (`probe_batchD.py`, li383
(12,24,24) p=3 float32, 200 steps in). The shifted split solves `(M_2 +
eps S_2) x = M_2 u` and then the k=1 curl-curl level. For a
divergence-free `u` the UPPER solve would be the trivial one (`S_2 u = 0`
gives `x = u`) and the lower solve the whole cost; nothing about the
lower solve is skippable. But the descent direction is divergence-free
only to the Leray solve's tolerance relative to `J x B`, and at a relaxed
state `|F| << |J x B|`, so in float32 the remnant is O(1) relative to
`F`: `|div F| / |F| = 12` (the scale of a generic field), the upper
solve moves `u` by 7.8% and takes 47 iterations from `x0 = u` against
60 cold. The fact is not exploitable in float32; the earlier claim that
a full solve was wasted on a near-zero right-hand side was wrong. The
same remnant is what the velocity Leray removes each step (340-376
MINRES iterations, the most expensive solve of the step, insensitive to
warm starts because the remnant is new every step); in float64 that
solve would be nearly free, which is the case for the "skip the velocity
Leray" experiment.

**Warm starts, measured** (same probe, iteration counts of the next
step's solves): the force Leray 249 cold, 198 with `p` alone, 8-16 with
`p` and `sigma` (the previous `JxX - F`); `J` 23 cold, 0-4 warm; `JxB`
0-6 warm. The returned `sigma` differs from the recomputed weak gradient
by 1.2e-3 in the M norm (3.5 tol, float32). The diagonalised shifted
core differs from the per-solve inverse by 4e-5 (k=1) and 2e-6 (k=2),
round-off of the two routes. Step time, same mesh, 300 steps in chunks of
100, the previous commit aa939e1 against b8701a1
(`outputs/prune_smoke/timing_ab.py`): 0.551 -> 0.247 s/step on the last
chunk, 0.760 -> 0.370 over the last two, a 2.2x faster step (the per-solve
`eigh`, the lower-block mass solve and its two mass applies per MINRES
iteration, the sigma recompute and the cold `J` together; the two
trajectories differ at round-off, F_300/F_0 8.6e-2 vs 9.8e-2).
