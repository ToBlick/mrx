#!/bin/bash
# Tabs. C.8, C.9 and Figs. C.16, C.17: Newton at (16, 32, 32) away from the defaults (kappa 3, one pass of 200
# MINRES iterations, forcing term 0.1, p = 2, mixed precision, solver tolerance 1e-8). The kappa = 3 arm is also the
# 200-iteration row of Tab. C.9; the reference arm of Figs. C.16, C.17 is newton_convergence/newton_16.
#   bash scripts/paper_scripts/runs/newton_sweeps.sh
. "$(dirname "$0")/common.sh"
O=$RECORDS/newton_sweeps
T="$LI383 --method newton --resolution 16,32,32 --steps 200 --chunk 20"
F="$LI383 --method newton --resolution 16,32,32 --steps 40 --chunk 20"

for k in 0 1 3 10; do
  sub ns_kappa$k 90 "" scripts/relax.py $T --newton-penalty $k --out $O/kappa$k
done
for it in 50 100 400; do
  sub ns_minres$it 150 "" scripts/relax.py $T --newton-maxiter $it --out $O/minres$it
done
sub ns_p1 40 "" scripts/relax.py $F --spline-degree 1 --out $O/p1
sub ns_p3 90 "" scripts/relax.py $F --spline-degree 3 --out $O/p3
sub ns_p4 180 "" scripts/relax.py $F --spline-degree 4 --out $O/p4
sub ns_float64 60 "" scripts/relax.py $F --precision float64 --out $O/float64
sub ns_float32 40 "" scripts/relax.py $F --precision float32 --out $O/float32
sub ns_tol6 40 "" scripts/relax.py $F --solve-tol 1e-6 --out $O/tol1e-6
sub ns_tol10 40 "" scripts/relax.py $F --solve-tol 1e-10 --out $O/tol1e-10
