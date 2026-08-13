#!/bin/bash
# Polar-order A/B (handoff priority-1): does the C^2-on-axis surgery buy the
# same atom-spread reduction as the fat-core R=1 emulation -- at a 6nz core
# instead of 3nz + nt*nz -- and does C^0 (control) show that the spread tracks
# the exact-region extent rather than pole smoothness per se?
#
# Arms (all dbc, 8^3, fd+fdbund, cheb-lo 0.85, auto-m, two-level):
#   c1      C^1 polar (production layout) + production baseline
#   c1fat   C^1 + --fat-core 1  (same bulk window as C^2, bigger core)
#   c2      C^2 polar extraction (6nz core, window start ring 3)
#   c0      C^0 polar extraction (nz core, window start ring 1) -- control
set -e
cd "$(dirname "$0")/../.."
OUT=outputs/laplacian_mg_k0/polar_order_ab_20260805
mkdir -p "$OUT"
CSV=$OUT/results.csv
COMMON="--ns 8 16 8 --bc dbc --smoothers fd,fdbund --cheb-lo 0.85 --auto-m --two-level-check --csv $CSV"

run() {
  local tag=$1; shift
  echo "=== $tag : $* ==="
  python scripts/debug/laplacian_mg_k0.py $COMMON "$@" 2>&1 | tee "$OUT/$tag.log"
}

for geo in toroid cerfon rotating_ellipse; do
  case $geo in
    toroid)           GARGS="--geometry toroid" ;;
    cerfon)           GARGS="--geometry cerfon --kappa 1.7 --alpha 0.4" ;;
    rotating_ellipse) GARGS="--geometry rotating_ellipse --kappa 1.5 --nfp 2" ;;
  esac
  run "$geo-c1"    $GARGS
  run "$geo-c1fat" $GARGS --fat-core 1 --no-baseline
  run "$geo-c2"    $GARGS --polar-order 2
  run "$geo-c0"    $GARGS --polar-order 0
done
echo "ALL DONE"
