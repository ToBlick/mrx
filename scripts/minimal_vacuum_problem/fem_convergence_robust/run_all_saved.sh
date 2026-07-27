#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/aak572/mrx"
PY="/Users/aak572/.conda/envs/mrx/bin/python"
DRIVER="$ROOT/scripts/minimal_vacuum_problem/fem_convergence_highres/nt18_logs/robust_island_driver.py"
OUT="$ROOT/scripts/minimal_vacuum_problem/fem_convergence_robust"

mkdir -p "$OUT"

run_one() {
  local grid="$1"
  local dof="$2"
  local output="$OUT/robust_${grid}.json"
  if [[ -s "$output" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP_COMPLETE $grid"
    return
  fi
  echo "START $grid"
  "$PY" "$DRIVER" \
    --grid "$grid" \
    --dof "$ROOT/scripts/minimal_vacuum_problem/$dof" \
    --output-dir "$OUT" \
    --output-json "$output" \
    --log "$OUT/robust_${grid}.log" \
    --zoom-nrho 16 \
    --zoom-phases 6 \
    --turns 2000 \
    --fourier-n 48 \
    --bootstrap-samples 2000
  echo "DONE $grid"
}

dof_for() {
  case "$1" in
    4x8x4) echo fem_convergence_final/fem_sweep_4x8x4_dof.npy ;;
    5x10x5) echo fem_convergence_final/fem_sweep_5x10x5_dof.npy ;;
    6x12x6) echo fem_convergence_final/fem_sweep_6x12x6_dof.npy ;;
    6x14x6) echo fem_convergence_run_b/fem_sweep_6x14x6_dof.npy ;;
    6x16x6) echo fem_convergence_run_b/fem_sweep_6x16x6_dof.npy ;;
    7x14x7) echo fem_convergence_high_order/fem_sweep_7x14x7_dof.npy ;;
    8x16x8) echo fem_convergence_high_order_refined/fem_sweep_8x16x8_dof.npy ;;
    8x20x8) echo fem_convergence_highres/fem_sweep_8x20x8_dof.npy ;;
    8x24x8) echo fem_convergence_run_c/fem_sweep_8x24x8_dof.npy ;;
    8x28x8) echo fem_convergence_highres/fem_sweep_8x28x8_dof.npy ;;
    8x32x8) echo fem_convergence_highres/fem_sweep_8x32x8_dof.npy ;;
    8x36x8) echo fem_convergence_highres/fem_sweep_8x36x8_dof.npy ;;
    9x18x9) echo fem_convergence_highres/fem_sweep_9x18x9_dof.npy ;;
    10x20x10) echo fem_convergence_highres/fem_sweep_10x20x10_dof.npy ;;
    10x24x10) echo fem_convergence_highres/fem_sweep_10x24x10_dof.npy ;;
    10x30x10) echo fem_convergence_highres/fem_sweep_10x30x10_dof.npy ;;
    11x18x11) echo fem_convergence_highres/fem_sweep_11x18x11_dof.npy ;;
    11x22x11) echo fem_convergence_highres/fem_sweep_11x22x11_dof.npy ;;
    12x24x12) echo fem_convergence_highres/fem_sweep_12x24x12_dof.npy ;;
    13x18x13) echo fem_convergence_highres/fem_sweep_13x18x13_dof.npy ;;
    *) echo "unknown grid $1" >&2; return 1 ;;
  esac
}

if (( "$#" > 0 )); then
  for grid in "$@"; do
    run_one "$grid" "$(dof_for "$grid")"
  done
  echo "REQUESTED_ROBUST_DIAGNOSTICS_COMPLETE"
  exit 0
fi

run_one 4x8x4 fem_convergence_final/fem_sweep_4x8x4_dof.npy
run_one 5x10x5 fem_convergence_final/fem_sweep_5x10x5_dof.npy
run_one 6x12x6 fem_convergence_final/fem_sweep_6x12x6_dof.npy
run_one 6x14x6 fem_convergence_run_b/fem_sweep_6x14x6_dof.npy
run_one 6x16x6 fem_convergence_run_b/fem_sweep_6x16x6_dof.npy
run_one 7x14x7 fem_convergence_high_order/fem_sweep_7x14x7_dof.npy
run_one 8x16x8 fem_convergence_high_order_refined/fem_sweep_8x16x8_dof.npy
run_one 8x24x8 fem_convergence_run_c/fem_sweep_8x24x8_dof.npy
run_one 8x32x8 fem_convergence_highres/fem_sweep_8x32x8_dof.npy
run_one 9x18x9 fem_convergence_highres/fem_sweep_9x18x9_dof.npy
run_one 10x20x10 fem_convergence_highres/fem_sweep_10x20x10_dof.npy
run_one 10x24x10 fem_convergence_highres/fem_sweep_10x24x10_dof.npy
run_one 10x30x10 fem_convergence_highres/fem_sweep_10x30x10_dof.npy
run_one 11x18x11 fem_convergence_highres/fem_sweep_11x18x11_dof.npy
run_one 11x22x11 fem_convergence_highres/fem_sweep_11x22x11_dof.npy
run_one 12x24x12 fem_convergence_highres/fem_sweep_12x24x12_dof.npy
run_one 13x18x13 fem_convergence_highres/fem_sweep_13x18x13_dof.npy

echo "ALL_ROBUST_DIAGNOSTICS_COMPLETE"
