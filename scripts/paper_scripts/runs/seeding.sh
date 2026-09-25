#!/bin/bash
# Tab. 4, Figs. 12, 13: every resonance of the reference run's state at step 150 seeded at its energy-optimal
# amplitude (scripts/paper_scripts/seed.py), relaxed for 200 Newton steps, traced. Needs newton_convergence/newton_32.
#   bash scripts/paper_scripts/runs/seeding.sh
. "$(dirname "$0")/common.sh"
O=$RECORDS/seeding

S=$(sub sd_seed 120 "" scripts/paper_scripts/seed.py --run $RECORDS/newton_convergence/newton_32 --step 150 --out $O/seeded.h5)
R=$(sub sd_relax 150 afterok:$S scripts/relax.py $LI383 --method newton --resolution 32,64,64 --restart $O/seeded.h5 \
  --steps 200 --chunk 20 --out $O/relax32)
sub sd_trace 90 afterok:$R scripts/poincare_trace.py --geometry $GEOMETRY $O/relax32/checkpoints/state_000350.h5 --lines 320 --periods 600
