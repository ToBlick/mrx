#!/usr/bin/env bash
# Local CPU sweep: spectrum-diag (axis vs shaping decomposition of the atom
# spread) + fat-core A/B (C2-surgery emulation) on the four CPU-friendly
# geometries. Validated auto window rule (--cheb-lo 0.85 --auto-m), both BCs,
# fd + fdax, r_scale 0.5, (12,24,12).
#
#   bash scripts/debug/run_mg_k0_local_fatcore.sh [outdir]
set -uo pipefail
cd "$(dirname "$0")/../.."

OUT=${1:-outputs/laplacian_mg_k0/local_fatcore_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
CSV="$OUT/results.csv"
# (8,16,8): laptop-budget size (~2.5 min/run, ~15 min total). Cylinder is
# covered by the standalone sanity run + the plan doc's plain-run numbers.
NS=${NS_OVERRIDE:-"8 16 8"}
COMMON="--ns $NS --smoothers fd,fdax --cheb-lo 0.85 --auto-m --two-level-check --spectrum-diag --csv $CSV"

run() {
  local name=$1; shift
  echo "====================================================================="
  echo ">>> $name  ($(date +%H:%M:%S))"
  python scripts/debug/laplacian_mg_k0.py "$@" 2>&1 | tee "$OUT/$name.log" | \
    grep -E "^\[|^===|spectrum-diag|lam=|window kappa" || true
}

for GEO_ARGS in \
  "toroid --geometry toroid" \
  "cerfon --geometry cerfon --kappa 1.7 --alpha 0.4" \
  "rotellipse --geometry rotating_ellipse --kappa 1.5 --nfp 3" \
  ; do
  set -- $GEO_ARGS
  name=$1; shift
  run "$name"      $COMMON "$@"
  run "$name-fat"  $COMMON "$@" --fat-core
done

echo "done -> $CSV"
