#!/bin/bash
# Tab. 5, Figs. 14, 15: three converged states at (32, 64, 64), the reference run's state at step 150 unseeded, with
# the (5, 1) chain at a*/4 and with the (6, 1) chain at a* (each its part of the joint optimum of paper/seed.py,
# relaxed 200 Newton steps); each gets 200 resistive steps, eta dt = 0.064 <g_rr> h_r^2 against ONE reference, the
# nested state smoothed by c = 0.03, then 100 ideal steps. Needs newton_convergence/newton_32 (whose trace is the
# unseeded state's before).
#   bash paper/runs/reconnection.sh
. "$(dirname "$0")/common.sh"
O=$RECORDS/reconnection
N32=$RECORDS/newton_convergence/newton_32
REF=$N32/checkpoints/state_000150.h5
N="$LI383 --method newton --ns 32,64,64"
TRACE="--fields final --lines 320 --periods 600"

phases() {  # arm, start checkpoint, its step, dependency: the resistive and the final ideal phase, traced
  local arm=$1 start=$2 step=$3 dependency=$4 res after
  res=$(sub rc_${arm}_res 400 "$dependency" scripts/relax.py $N --restart $start --steps 200 --chunk 20 \
    --resistivity 0.064 --reference-smoothing 0.03 --reference $REF --out $O/$arm/resistive)
  after=$(sub rc_${arm}_after 150 afterok:$res scripts/relax.py $N \
    --restart $O/$arm/resistive/checkpoints/state_$(printf %06d $((step + 200))).h5 --steps 100 --chunk 20 \
    --out $O/$arm/ideal_after)
  sub rc_${arm}_tr_res 60 afterok:$res scripts/poincare_trace.py --run $O/$arm/resistive $TRACE
  sub rc_${arm}_tr_after 60 afterok:$after scripts/poincare_trace.py --run $O/$arm/ideal_after $TRACE
}

seeded() {  # arm, paper/seed.py's selection: seed, relax, trace, then the two phases
  local arm=$1 seed ideal
  shift
  seed=$(sub rc_${arm}_seed 120 "" paper/seed.py --run $N32 --step 150 "$@" --out $O/$arm/seeded.h5)
  ideal=$(sub rc_${arm}_ideal 240 afterok:$seed scripts/relax.py $N --restart $O/$arm/seeded.h5 --steps 200 \
    --chunk 20 --out $O/$arm/ideal)
  sub rc_${arm}_tr_ideal 60 afterok:$ideal scripts/poincare_trace.py --run $O/$arm/ideal $TRACE
  phases $arm $O/$arm/ideal/checkpoints/state_000350.h5 350 afterok:$ideal
}

phases unseeded $REF 150 ""
seeded s61 --chain 6,1
seeded s51q --chain 5,1 --scale 0.25
