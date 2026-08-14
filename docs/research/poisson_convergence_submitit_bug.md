# Poisson convergence "floor" turned out to be a submitit launcher bug

Bug-hunting log for the apparent stagnation of the torus Poisson convergence
test around `~2-3e-4` for `p>=3` in
[scripts/config_scripts/test_torus_poisson_sparse.py](../scripts/config_scripts/test_torus_poisson_sparse.py).

## Symptom

Sweep `p=1,2,3 × n=8,12,16` at `q=2p` launched via
`bash slurm/job_poisson_convergence.sh "p=1,2,3" "n=[8,12,16]"` produces:

| p | n=8 | n=12 | n=16 | observed rate (n=12→16) |
|---|---|---|---|---|
| 1 | 4.60e-2 | 1.85e-2 | 9.97e-3 | 2.14 (expected ~2) |
| 2 | 4.13e-3 | 1.18e-3 | 7.61e-4 | 1.52 (expected ~3) |
| 3 | 6.85e-4 | 3.12e-4 | 3.20e-4 | -0.08 (expected ~4) |

Same "floor near `3e-4` for `p=3`" had been seen in earlier sweeps
(`multirun/2026-05-09/19-22-05`, `multirun/2026-05-18/20-31-10`) and was the
working hypothesis we had been trying to localize.

## What we ruled out

- **Manufactured pair.** `f = -Δ_LB u` analytically and to roundoff at sample
  points for `u = (r²-r⁴)/4 · cos(2πz)` on `toroid_map(ε=1/3, κ=1)`. `u`
  vanishes at `r=0,1` (DBC).
- **Quadrature.** Earlier sweeps with `q = 2p+4` and `q = 2p+6` give the same
  floor.
- **Tolerance.** `cg_tol=1e-12` already; tightening doesn't help.
- **Projection floor.** Cell G in
  [scripts/interactive/debug_poisson_convergence.py](../scripts/interactive/debug_poisson_convergence.py)
  shows `||u - Π_{L2} u||` decays at expected `p+1` order through `p=3` (e.g.
  `2.35e-5` at `p=3, n=16`) — i.e. the discrete space can represent `u` far
  better than the production solve does.
- **Production vs assembled K0 paths agree at small n.** Cell S compared the
  matrix-free production path against an assembled scalar grad-grad path on
  the same problem; at `p=3, n=6,8,10` they agree bit-for-bit (`op_gap ~ 4e-16`)
  and the solve hits the projection floor exactly.

## The diagnostic that broke things open

Cell S at `p=3, n=8` reported `solve_err = 4.77e-4`. The slurm sweep at the
same `(p, n, q)` with apparently identical configuration reported `6.85e-4`.
Same script, same package, same H100 partition, but different answer.

A fresh `compute_error(...)` call in an interactive `srun` shell reproduced
**4.77e-4** with full `p+1 ≈ 4` convergence (`1.74e-4` at n=10, `7.88e-5` at
n=12). So the "plateau" only exists on the slurm path.

Then narrowed further with three single-point runs of `(p=3, n=8)`:

| Path | error | `m1_nnz_actual` |
|---|---|---|
| Direct call to `compute_error` (no Hydra) | 4.77e-4 | 576128 |
| `python test_torus_poisson_sparse.py p=3 n=[8]` (Hydra, no `-m`) | 4.77e-4 | 576128 |
| `python test_torus_poisson_sparse.py -m p=3 n=[8]` (Hydra + submitit) | 6.85e-4 | **788634** |

The `M1` mass matrix itself is different on the submitit path (37% more
entries above the `1e-12` threshold). Hydra alone is fine; only **submitit**
breaks it.

## Patching `compute_error` to capture solver telemetry

Added to
[scripts/config_scripts/test_torus_poisson_sparse.py](../scripts/config_scripts/test_torus_poisson_sparse.py):

- `apply_inverse_laplacian(..., return_info=True)` to capture CG iter
  count and convergence flag.
- Explicit Euclidean residual `||K0 u_hat - rhs|| / ||rhs||` recomputed via
  `seq.apply_laplacian` after the solve.
- All three new fields written to `result.json` and printed in the per-run
  summary.

Single-point `(p=3, n=8)` outcome with patch:

| Path | iters | M-norm rel res (CG criterion) | Euclidean rel res | error |
|---|---|---|---|---|
| direct interactive | 11 | < 1e-12 | **8.17e-14** | 4.77e-4 |
| submitit | 11 | < 1e-12 | **4.93e-7** | 6.85e-4 |

Both paths take **exactly 11 iterations**; both pass CG's M-norm convergence
check. But on submitit the iterate does not actually satisfy the equation we
re-check it against — the Euclidean residual is 7 orders of magnitude larger
than interactive's.

That gap implies one of:

1. The matrix CG uses (`apply_stiffness`) and the matrix we recheck with
   (`apply_laplacian`) differ on the submitit path.
2. Either of those apply paths is nondeterministic across calls on submitit,
   so CG's recursive residual decouples from the true residual within a
   handful of iterations.

Both possibilities are downstream of the launcher, not the algorithm.

## Confirmed cause

Adding `print`s at the top of `main` for `jax.config.jax_enable_x64` and the
types/values of `cfg.epsilon`, `cfg.n`, `cfg.cg_tol`, then running the same
single point twice:

```
# python ... p=3 n=[8]    (Hydra, no -m)
x64 enabled: True
epsilon type: float value: 0.3333333333333333
n type: ListConfig value: [8]
cg_tol type: float value: 1e-12
...
Relative L2 error: 4.771335e-04
CG iters: 11  converged: True  final ||K0 u - b||/||b||: 8.172e-14
m1_nnz_actual................. 576128

# python ... -m p=3 n=[8]   (Hydra + submitit)
x64 enabled: False
epsilon type: float value: 0.3333333333333333
n type: ListConfig value: [8]
cg_tol type: float value: 1e-12
...
Relative L2 error: 6.842729e-04
CG iters: 11  converged: True  final ||K0 u - b||/||b||: 4.623e-07
m1_nnz_actual................. 788634
```

The only difference between the two configurations is the first line:
`x64 enabled: False` on the submitit worker. Everything else (cfg types,
values, JAX device) is identical. The whole pipeline is running fp32 on the
submitit worker. `5e-7` Euclidean residual and the `m1_nnz_actual` jump are
both fp32-roundoff fingerprints.

The module-level `jax.config.update("jax_enable_x64", True)` in
`test_torus_poisson_sparse.py` is not honoured on the submitit worker.
Most likely submitit's worker bootstrap imports JAX (or unpickles
JAX-touching state) before the script's top-level code runs, and after JAX
has been initialised the `jax.config.update` call has no effect on already-
traced state. The robust fix is to set `JAX_ENABLE_X64=1` as an environment
variable before the worker process starts.

## Fix

One line added to `conf/config_poisson_test.yaml`:

```yaml
hydra:
  launcher:
    setup:
      - "source /scratch/tblickhan/mrx/.venv/bin/activate"
      - "export JAX_ENABLE_X64=1"   # NEW
```

Verification after the fix is the same one-point sweep with `-m p=3 n=[8]`:
should reproduce `error = 4.77e-4`, `m1_nnz_actual = 576128`, and
`||K0 u - b||/||b|| < 1e-13`.

## Other affected configs

The following Hydra configs also use `submitit_slurm` and may need the same
fix if their scripts rely on `jax.config.update("jax_enable_x64", True)` at
module top:

- [conf/config_mass_preconditioner.yaml](../conf/config_mass_preconditioner.yaml)
- [conf/config_mc_poisson.yaml](../conf/config_mc_poisson.yaml)

Any prior multirun results from those configs should be re-examined to see
whether fp32 corrupted the numbers.

## Implications for prior analyses

Every Poisson "floor" we measured via the SLURM launcher
(`multirun/2026-05-09/...`, `multirun/2026-05-18/...`,
`multirun/2026-05-28/10-37-04`) is invalidated: the operator being solved was
the fp32 operator. Those convergence tables should be regenerated with the
launcher fix in place.

The K0 stiffness-preconditioner phase-2 results (recorded in
[docs/preconditioner_primer.md](preconditioner_primer.md) and
`/memories/repo/mrx-preconditioning.md`) were CG-iteration-count studies, not
absolute-error studies, and were run on the same Hydra+submitit path. The
iteration counts in this debug (`11` iters at `p=3, n=8`) match between the
x64 and the (broken) fp32 runs, which suggests CG iter counts at small `n`
are not materially fp32-perturbed — but the larger-`n` numbers in those
sweeps should still be sanity-checked once we have an x64 baseline.
