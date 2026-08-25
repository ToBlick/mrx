# Handoff 2026-08-24 — preconditioner defaults, the boundary penalty, harmonic forms

Everything below is committed except where marked. Six commits:

```
6ab7117 nullspace: terminate inverse iteration on the Rayleigh quotient
a59facb docs: the bc-alpha sweep, the harmonic form, and what has to be re-measured
cff1034 tools: real-solve benchmark, GVEC geometries, nullspace and scale probes
1ed22ae bc: ship only the boundary penalty, drop the convention machinery
f285977 solvers: stop masking violated invariants in minres and CG
58c623d fix: the k>=1 saddle solves were running without a preconditioner
```

`fecaa04` between `a59facb` and `6ab7117` is from the OTHER session (the fork),
as is everything currently uncommitted in the tree (a dead-code audit touching
`mrx/assembly.py`, `circulation.py`, `io.py`, `plotting.py`, `projectors.py`,
`relaxation.py`, `utils.py`, `preconditioners.py`, and further edits to
`nullspace.py`). **Nothing of mine is uncommitted.**

---

## 1. The big one: k>=1 solves ran with no preconditioner

`_materialize_default_mass_preconditioner` gated the saddle solve's LOWER (mass)
block on `_tensor_available` and fell back to a per-DoF jacobi diagonal. That
gate was right when `default_mass_preconditioner()` meant `'tensor'`; it has
meant `'block_jacobi'` since 2026-08-22, which is always buildable. And
`schur.outer` was `'jacobi'` unconditionally, with `'block'` not even a legal
kind — the production Laplacian preconditioner was unreachable from the
production solve.

**`apply_inverse_hodge_laplacian` went from 2/18 to 18/18 cells converged.**
toroid k=2 free 9612 -> 314; quasr9983 k=1 free 6317 -> 857; w7x k=2 dbc
10000! -> 1522. Block outer is worth a further 2.51x on top.

The jacobi fallback now warns. `compute_nullspaces` assembles the atom for its
own solves.

## 2. What that fixed downstream

The k=1 free harmonic form degraded with p (w7x relL2 8.4e-13 / 3.0e-04 /
1.7e-01 at p=2/3/5) because its inner `L_2` free solve needed 20k-38k MINRES
iterations against a 10k budget. At default settings now: **w7x p=5
6.1e-04 -> 2.0e-11**, quasr44970 7.4e-04 -> 1.3e-10. hegna p=5 is still
3.6e-05 — its inner solve does not fit in 10k even with the atom, and that is
the one open case.

## 3. Inverse iteration — solved, and it is not fickle

Trajectory (`scripts/debug/invit_trajectory.py`, w7x p=5):

| | seed | after 1 sweep | inner solve residual |
| --- | --- | --- | --- |
| k=1 free | 1.81e-11 | **1.36e-04** | 1.09e-04 |
| k=3 dbc | 7.33e-12 | 6.15e-13 | 2.39e-08 |

**One sweep replaces the vector with the shifted solve's output, so it can never
beat that solve's accuracy.** At k=1 free the shifted solve reaches only ~1e-04
and the output lands exactly there; the seed was seven orders better, so one
sweep destroys it, then it plateaus. `eps` = 1e-2/1e-4/1e-6 are identical, so
the shift is irrelevant.

**Rule: inverse iteration helps only when the direct vector is worse than its
own inner solve.** With the direct route at 2e-11 it should never be used at
k=1 free. Do not gate on a fixed tolerance — gate against the inner solve's
achievable accuracy.

Also fixed: `find_nullspace_vectors` terminated on `||L_k v||`, a dual vector in
the primal mass norm, which carries `||L|| ~ h^-2` so a fixed `abs_tol` moved
the stopping point with h and p. Now terminates on the Rayleigh quotient
(`abs_tol` bounds `sqrt(rq)`); the stall guard and the seeded early-exit use the
same criterion.

## 4. The boundary penalty ships alone

One expression, one code path: `alpha_k = <m_k sqrt(g^rr)> / <m_k/J> / h_last`.
All other conventions and their probes are deleted (-152 lines).

**The scale is settled — see `s_scale_2026-08-25.md`.** Measured on the real
solve path, free rows, p=3 — the optimum spans **0.5 (hegna) to 16
(quasr9983)**, but the basin is flat and `s = 3` is within 27% of every
geometry's own optimum. `s=0` is the worst value in 17 of 18 rows, so the term
always earns its place. hegna k=2 free does not converge at *any* `s`, so it is
a geometry problem, not a scale one.

Four predictors tested against the six k=3 optima (2.37 / 5.37 / 9.64 / 9.05 /
19.60 / 22.63):

| predictor | residual spread |
| --- | --- |
| null (constant s) | 9.55x |
| **`alpha_exact/alpha_5`** (ratio of face means) | **1.85x** |
| variance of `J sqrt(g^rr)` | 43.78x |
| J spread / g^rr spread | 18.65x / 2360x |

Only `alpha_exact/alpha_5 = <J g^rr>/h` over the penalty works, giving
`s ~ 2.1 * (alpha_exact/alpha_5)` — but there is no mechanism for it and it
fails on k=1,2 (flat basins, ratios 0.43-7.54). Note `alpha_exact` is NOT in
the code any more; it lives in `scripts/debug/face_coefficient_probe.py`.

**Refuted, all cleanly:**
- the scale is not a Nitsche trace constant. `e` is one-hot at the boundary and
  27% of it lies outside `range(K_local)`, so the local Rayleigh quotient is
  UNBOUNDED on every geometry — there is no finite discrete trace constant. Our
  term is a rank-one correction to a SEPARABLE approximation, and the energy
  available to bound it (`Ktilde`) does not control the last derivative-spline
  coefficient. Worth one sentence in the paper.
- the scale does not track face variance; if anything it is anti-correlated
  (quasr9983 has the smallest variance and the largest s).
- CGS vs MGS in the minres Lanczos step: identical counts, `max|v.r1| ~ 1e-18`.
  `solvers.py` needs no change there.

## 5. Everything pre-2026-08-24 is superseded

Two independent reasons: the harness solved a surrogate operator
(`apply_hodge_laplacian_approx`), and the production path was unpreconditioned.
`verify_block_jacobi.py` now has an explicit `--operator {approx,exact}` flag.
See `sweep_plan_2026-08-24.md` for what to re-measure. Conclusions largely
survive as relative comparisons; tables do not.

One correction worth knowing: the quasr "A5 is worse out of sample" result was a
grid artefact. The matched-scale grids used a toroid conversion factor of 28.3;
the true factor is 108 on quasr9983.

## 6. Trajectory runs — complete

`outputs/invit_traj/2026-08-24/16-47-18/`. All four finished; the k=1 runs are
flat from sweep 1 to sweep 100 (w7x k=1, eps=1e-4: relL2 1.36e-04 at 1 sweep,
1.32e-04 at 100). Two things this pins down beyond the conclusion above:

**The stall guard now works.** `n_iters = 2` at every requested maxiter from 1
to 100, and the wall time is flat at 151 s. Before the Rayleigh-quotient
termination change the same case burned 176 s grinding toward maxiter. So the
criterion fix did have a real effect — it just cannot rescue a vector the inner
solve is unable to represent.

**A smaller shift makes it worse, not better.** eps=1e-6 gives an inner residual
of 5.8e-03 (against 1.09e-04 at eps=1e-4) and a correspondingly worse output,
4.9e-04 against 1.36e-04. That is the expected direction — a smaller shift is
more ill-conditioned — and it confirms the inner solve, not the iteration, is
the binding constraint.

## 7. NOT committed and cannot be

`papers/block_jacobi/main.tex` — everything after `\newpage` was rewritten:
sections the author already covered deleted, tone matched, the boundary section
rebuilt with the penalty as the primary object and the `E^T M^-1 E` reduction
demoted to a remark explaining where the degree dependence comes from. Builds
clean, 10 pages, no undefined references. `papers/` is gitignored, so this lives
only on disk.

Results section is a marked `[XXX]` placeholder — every number that would have
gone in it predates the fixes above.

## 8. Open

1. ~~**The scale.**~~ **CLOSED 2026-08-25** — `s_scale_2026-08-25.md`. Merging
   the s-zero, cheap-S2 and round-2 grids gives `s` from 0 to 512 at ratio
   `sqrt(2)`, and all 18 free rows bracket on both sides. Larger `s` was not the
   answer: everything turns around by 32 and `s >= 64` stalls cells. The
   per-geometry optima do span 0.5-16, but the basin is flat enough that the
   worst geometry pays only 1.27x at `s = 3` (1.21x at the minimax point
   `s = 4`) — so a factor-32 spread in `argmin` is worth at most 27%.
   Per-geometry calibration and `alpha_exact` are dropped; `PRODUCTION_BC_SCALE
   = 3.0` stands, now measured. The paper reports the basin, not the argmin.
2. **hegna** is the standing anomaly: smallest `1/ratio`, k=2 free diverges at
   s>=4, the only S1 cell where the block atom LOST to jacobi, and the only
   geometry whose harmonic form is still weak at p=5.
3. **The harmonic-form gate** — measure the Rayleigh quotient after the direct
   construction and escalate if bad. Designed, not implemented. Escalation is
   "give the inner solve more budget", NOT "switch to inverse iteration".
4. **Regenerate the paper's numbers** — `sweep_plan_2026-08-24.md`, S1 first.
