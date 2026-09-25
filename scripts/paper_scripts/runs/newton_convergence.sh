#!/bin/bash
# Tab. 3, Figs. 9, 10, 11, C.18: gradient descent at n_r = 16 and Newton at n_r = 16 ... 64, (n, 2n, 2n) splines;
# the n_r = 32 run is the paper's reference run. The Newton arms at 16, 24 and 32 continue from their last checkpoint
# to about two hours each (cont/, at their measured s/step). Traces: the n_r = 48 state (Fig. 11), the reference
# run's state at step 150 (Fig. 14 top, the width baseline of Tabs. 4 and 5), the best states of the three
# continuations (Fig. C.18).
#   bash scripts/paper_scripts/runs/newton_convergence.sh
. "$(dirname "$0")/common.sh"
O=$RECORDS/newton_convergence
N="$LI383 --method newton"
TRACE="--lines 320 --periods 600"

# the smoothing scale of the paper's run, its code's default then: 0.02 / n_r^2 = 0.0737 <g_rr> h_r^2 (today 0.075)
sub nc_gd16 60 "" scripts/relax.py $LI383 --method gradient --ns 16,32,32 --steps 5000 --chunk 500 \
  --velocity-smoothing-scale 7.8125e-05 --out $O/gradient_16
J16=$(sub nc_nw16 60 "" scripts/relax.py $N --ns 16,32,32 --steps 100 --chunk 20 --out $O/newton_16)
J24=$(sub nc_nw24 120 "" scripts/relax.py $N --ns 24,48,48 --steps 100 --chunk 20 --out $O/newton_24)
J32=$(sub nc_nw32 240 "" scripts/relax.py $N --ns 32,64,64 --steps 150 --chunk 30 --out $O/newton_32)
J48=$(sub nc_nw48 720 "" scripts/relax.py $N --ns 48,96,96 --steps 200 --chunk 20 --out $O/newton_48)
# the initial field's histopolation needs --map-batch at this size; the paper's record ends at the 10 h limit, step 120
sub nc_nw64 600 "" scripts/relax.py $N --ns 64,128,128 --steps 180 --chunk 20 --map-batch 8192 --out $O/newton_64

C16=$(sub nc_nw16c 240 afterok:$J16 scripts/relax.py $N --ns 16,32,32 --steps 720 --chunk 20 \
  --restart $O/newton_16/checkpoints/state_000100.h5 --out $O/newton_16/cont)
C24=$(sub nc_nw24c 240 afterok:$J24 scripts/relax.py $N --ns 24,48,48 --steps 400 --chunk 20 \
  --restart $O/newton_24/checkpoints/state_000100.h5 --out $O/newton_24/cont)
C32=$(sub nc_nw32c 240 afterok:$J32 scripts/relax.py $N --ns 32,64,64 --steps 150 --chunk 30 \
  --restart $O/newton_32/checkpoints/state_000150.h5 --out $O/newton_32/cont)

sub nc_tr48 120 afterok:$J48 scripts/poincare_trace.py --run $O/newton_48 --fields final $TRACE
sub nc_tr32 60 afterok:$J32 scripts/poincare_trace.py --run $O/newton_32 --fields final $TRACE
sub nc_tr16c 40 afterok:$C16 scripts/poincare_trace.py --run $O/newton_16/cont --fields best --lines 160 --periods 400
sub nc_tr24c 60 afterok:$C24 scripts/poincare_trace.py --run $O/newton_24/cont --fields best --lines 160 --periods 400
sub nc_tr32c 90 afterok:$C32 scripts/poincare_trace.py --run $O/newton_32/cont --fields best --lines 160 --periods 400
