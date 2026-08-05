#!/bin/bash
# Point-Jacobi vs fd/fdbund Chebyshev smoother A/B in the two-level MG
# envelope (8^3, fat-core R=1, anchored-xi1, auto-m). The million-dollar
# question: does the cheap exact-diagonal atom (sees all 9 metric blocks
# pointwise ON the diagonal, ~free apply) match the FD approximate-inverse
# atoms once the V-cycle's coarse correction handles the low end?
#
# Windows: fd-family keeps its tuned absolute floor (--cheb-lo 0.85, spectrum
# of S*A clusters at 1); jacobi gets the classic RELATIVE window (S*A spectrum
# is not clustered) at kappa=4 (m=3) and kappa=9 (m=4).
set -e
cd "$(dirname "$0")/../.."
OUT=outputs/laplacian_mg_k0/jacobi_ab_20260724
mkdir -p "$OUT"
CSV=$OUT/results.csv
COMMON="--ns 8 16 8 --auto-m --two-level-check --fat-core 1 --anchor-xi1 --csv $CSV"

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
  run "$geo-fd"      $GARGS --smoothers fd,fdbund --cheb-lo 0.85 --no-baseline
  run "$geo-jac4"    $GARGS --smoothers jacobi
  run "$geo-jac9"    $GARGS --smoothers jacobi --cheb-window 9 --no-baseline
done
echo "ALL DONE"
