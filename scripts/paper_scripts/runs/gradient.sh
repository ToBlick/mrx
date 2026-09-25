#!/bin/bash
# Fig. 8, Tabs. C.6, C.7 and the smoothing constant of Sec. 5.3: gradient descent at (16, 32, 32). The gamma = 1,
# potential-velocity, c = 0.0737 arm of all four is newton_convergence/gradient_16.
#   bash scripts/paper_scripts/runs/gradient.sh
. "$(dirname "$0")/common.sh"
O=$RECORDS/gradient
G="$LI383 --method gradient --ns 16,32,32"

# Fig. 8: no smoothing, as long in wall time as the smoothed run
sub gd_gamma0 70 "" scripts/relax.py $G --velocity-smoothing-order 0 --steps 7000 --chunk 500 --out $O/gamma0
# Tab. C.7: the Leray projection instead of the potential velocity, the smoothing of newton_convergence/gradient_16
sub gd_leray 80 "" scripts/relax.py $G --velocity-smoothing-order 1 --velocity-smoothing-scale 7.8125e-05 \
  --potential-velocity false --steps 5000 --chunk 500 --out $O/leray
# Tab. C.6: the plain and the corrected explicit step, no smoothing
sub gd_hel_mp 30 "" scripts/relax.py $G --velocity-smoothing-order 0 --steps 1000 --chunk 100 \
  --helicity-correction false --out $O/helicity_mixed_plain
sub gd_hel_mc 30 "" scripts/relax.py $G --velocity-smoothing-order 0 --steps 1000 --chunk 100 \
  --helicity-correction true --out $O/helicity_mixed_corrected
sub gd_hel_dp 40 "" scripts/relax.py $G --precision float64 --velocity-smoothing-order 0 --steps 1000 --chunk 100 \
  --helicity-correction false --out $O/helicity_float64_plain
sub gd_hel_dc 40 "" scripts/relax.py $G --precision float64 --velocity-smoothing-order 0 --steps 1000 --chunk 100 \
  --helicity-correction true --out $O/helicity_float64_corrected
# Sec. 5.3: eps = c <g_rr> h_r^2 with <g_rr> h_r^2 = 1.0606e-3 at (16, 32, 32), p = 2
C="$G --velocity-smoothing-order 1 --steps 2000 --chunk 500"
sub gd_c0.007 50 "" scripts/relax.py $C --velocity-smoothing-scale 7.424e-06 --out $O/smoothing_c0.007
sub gd_c0.02 50 "" scripts/relax.py $C --velocity-smoothing-scale 2.121e-05 --out $O/smoothing_c0.02
sub gd_c0.2 60 "" scripts/relax.py $C --velocity-smoothing-scale 2.121e-04 --out $O/smoothing_c0.2
sub gd_c0.7 70 "" scripts/relax.py $C --velocity-smoothing-scale 7.424e-04 --out $O/smoothing_c0.7
