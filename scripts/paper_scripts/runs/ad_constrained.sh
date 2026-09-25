#!/bin/bash
# Sec. 3.4, App. D, Figs. 6, 7: the constrained QA design at (24, 48, 24), p = 3, float64 (scripts/paper_scripts/ad_recovery.py): the
# pilot from the device for 3000 iterations, and four starts with random 10 mm RMS boundary displacements (draw 3
# fails the fold guard) stopped at LP's criterion. The draws measure their end against the pilot's, so they wait for
# it. Records in <RECORDS>/shape_optimization.
#   bash scripts/paper_scripts/runs/ad_constrained.sh
. "$(dirname "$0")/common.sh"
export EXTRA_ENV="MRX_DTYPE=float64" CPUS=4 MEM_GB=16
A="--records $RECORDS --stage constrained"

M0=$(sub ad_M0 480 "" scripts/paper_scripts/ad_recovery.py $A --maxiter 3000 --tag _M0)
for k in 0 1 2 4; do
  sub ad_M10s$k 480 afterok:$M0 scripts/paper_scripts/ad_recovery.py $A --stop-qa 1.0 --perturb-mm 10 --perturb-seed $k \
    --reference-end qa_constrained_M0.npz --tag _M10s$k
done
