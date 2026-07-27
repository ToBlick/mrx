#!/usr/bin/env bash
# Fixed-aperture poloidal ladder: 8x20x8, 8x28x8, 8x36x8 (solve only).
# Resume known 8-series records; run sequentially.
set -euo pipefail
cd /Users/aak572/mrx
export PATH="/Users/aak572/.conda/envs/mrx/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONPATH="/Users/aak572/mrx${PYTHONPATH:+:$PYTHONPATH}"

MVP=/Users/aak572/mrx/scripts/minimal_vacuum_problem
LOGDIR="$MVP/fem_convergence_robust"
DRIVER="$MVP/fem_convergence_highres/nt18_logs/nt18_lean_driver.py"
PY=/Users/aak572/.conda/envs/mrx/bin/python

mkdir -p "$LOGDIR"

RESUME_ARGS=(
  --resume-json "$MVP/fem_convergence_high_order_refined.json"
  --resume-json "$MVP/fem_convergence_run_c.json"
  --resume-json "$MVP/fem_convergence_highres_8x32x8.json"
)

run_solve() {
  local grid="$1"
  local out="$LOGDIR/robust_solve_${grid}.json"
  local log="$LOGDIR/robust_solve_${grid}.log"
  if [[ -s "$out" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date)] SKIP_COMPLETE $grid"
    RESUME_ARGS+=(--resume-json "$out")
    return
  fi
  echo "[$(date)] START_SOLVE $grid"
  "$PY" "$DRIVER" \
    --target "$grid" \
    --mode solve \
    "${RESUME_ARGS[@]}" \
    --output-json "$out" \
    --log "$log" \
    --dense-max-dofs 12000 \
    --time-budget 36000
  RESUME_ARGS+=(--resume-json "$out")
  echo "[$(date)] DONE_SOLVE $grid"
}

run_solve 8x20x8
run_solve 8x28x8
run_solve 8x36x8

echo "[$(date)] ALL_8X_POLOIDAL_LADDER_SOLVES_COMPLETE"
