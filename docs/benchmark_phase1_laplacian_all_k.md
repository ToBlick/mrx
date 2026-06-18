# Phase1 Laplacian Benchmark (`scripts/benchmark_phase1_laplacian_all_k.py`)

> Last updated: 2026-06-10 (sparse branch)

This benchmark compares Laplacian and mass-matrix solve strategies across
form degrees `k=0,1,2,3` for one `(n,p)` cell (phase1 mode) or a sweep over
`n,p,kappa` (phase2-style mode).

## What It Measures

- Laplace solves (`apply_inverse_laplacian`) for selected `k`.
- Mass solves (`apply_inverse_mass_matrix`) for selected `k`.
- Methods:
  - `jacobi` (skip all jacobi rows with `--no-jacobi`)
  - `tensor(rank=...)`
  - `chebyshev(steps=...)` (optional; disabled when `--cheb-steps` is empty)

For Laplace `k>=1`, the saddle-point solve uses:
- **lower M block**: tensor preconditioner at the current rank.
- **Schur outer**: Jacobi or Chebyshev, with diagonal estimated per `--schur-diag-modes`.
- **Schur inner** (tensor_probe mode only): tensor mass preconditioner.

The script reports per-method:

- iteration statistics
- solve wall-times
- final relative residual
- setup/warmup overhead

## Current Defaults

- `--cg-tol 1e-9`
- `--cg-maxiter 2000`
- `--cheb-steps` defaults to empty (Chebyshev off)
- `--schur-diag-modes` defaults to `diag,tensor_probe,exact_probe`
- `--no-jacobi`: off by default (jacobi rows emitted)
- `--solve-jit`: on by default (per-method solve wrappers are JIT-compiled)
- `--preassemble-schur-diags`: off by default

## Nullspace-Free Laplace BC Mode

Use:

```bash
--nullspace-free-laplace-bcs
```

This overrides Laplace boundary conditions by form degree:

- `k=0`: DBC
- `k=1`: DBC
- `k=2`: NBC
- `k=3`: NBC

This is useful when comparing preconditioners without singular/nullspace cases
in Laplace solves.

Mass solves continue to use the global `--dirichlet` selection.

## Empirical Observations (ns=(6,12,6) p=3, rotating ellipse)

- **Tensor rank saturates quickly for mass**: k=0 at rank 3, k=3 at rank 2.
  k=1 and k=2 mass tensor reaches iteration floor at rank 1 with 15 iters.
- **Tensor rank has no effect on Laplace k>=1 iteration counts** with `schur-diag-modes=diag`.
  The Schur outer diagonal (diag mode) does not change with rank; the lower M block
  quality improvement at higher rank is negligible against the Schur outer bottleneck.
- **k=2 NBC Laplace is the hardest case**: ~470 iterations vs ~185 for k=3 NBC.

## Recommended Run Patterns

### Jacobi + tensor sweep (baseline)

Jacobi vs tensor for all degrees, nullspace-free BCs, preassembled Schur diagonals:

```bash
python scripts/benchmark_phase1_laplacian_all_k.py \
  --ns 6,12,6 \
  --p 3 \
  --nullspace-free-laplace-bcs \
  --tensor-ranks 1,2,3,4,5 \
  --schur-diag-modes diag \
  --preassemble-schur-diags \
  --laplace-ks 0,1,2,3 \
  --mass-ks 0,1,2,3 \
  --cg-tol 1e-9 \
  --cg-maxiter 2000 \
  --solve-jit
```

### Chebyshev sweep (no jacobi)

Tensor + chebyshev outer, skip all jacobi rows:

```bash
python scripts/benchmark_phase1_laplacian_all_k.py \
  --ns 6,12,6 \
  --p 3 \
  --nullspace-free-laplace-bcs \
  --tensor-ranks 1,2,3,4,5 \
  --cheb-steps 1,2,4 \
  --schur-diag-modes diag \
  --preassemble-schur-diags \
  --no-jacobi \
  --laplace-ks 0,1,2,3 \
  --mass-ks 0,1,2,3 \
  --cg-tol 1e-9 \
  --cg-maxiter 2000 \
  --solve-jit
```

## SLURM Submission

Submit directly using `--wrap` (the helper `slurm/job_phase1_laplacian_all_k.sh`
does not expose `--no-jacobi` or `--nullspace-free-laplace-bcs`):

```bash
STAMP=$(date +%Y-%m-%d/%H-%M-%S)
OUTDIR=outputs/phase1_laplacian/$STAMP
mkdir -p "$OUTDIR/slurm_logs"

sbatch \
  --partition=gpu-h100 \
  --account=extremedata \
  --gpus-per-node=1 \
  --cpus-per-task=32 \
  --time=360 \
  --mem=80G \
  --job-name=phase1_lap_ns6x12x6_p3 \
  --output="$OUTDIR/slurm_logs/phase1_lap_ns6x12x6_p3.log" \
  --wrap="set -euo pipefail; cd /scratch/tblickhan/mrx; source .venv/bin/activate; \
    python scripts/benchmark_phase1_laplacian_all_k.py \
      --ns 6,12,6 --p 3 \
      --nullspace-free-laplace-bcs \
      --tensor-ranks 1,2,3,4,5 \
      --cheb-steps 1,2,4 \
      --schur-diag-modes diag \
      --preassemble-schur-diags \
      --no-jacobi \
      --laplace-ks 0,1,2,3 --mass-ks 0,1,2,3 \
      --cg-tol 1e-9 --cg-maxiter 2000 --solve-jit \
      --out $OUTDIR/phase1_laplacian_ns6x12x6_p3.csv \
      --json-out $OUTDIR/phase1_laplacian_ns6x12x6_p3.json"
```

## Output

- Console table summary.
- Optional JSON via `--json-out`.
- Optional CSV via `--out`.

In sweep mode, JSON also contains per-cell summary entries.
