#!/bin/bash
# Fig. 6, Sec. 3.4: LP's criterion <Q_QA^2>_{r >= h_r} on its interpolated map at (n, 2n, n), n = 16, 24, 32, p = 3,
# float64 (scripts/paper_scripts/ad_recovery.py --stage baseline, setup only): the grey lines of Fig. 6, n = 24 the runs' unit.
# Writes <RECORDS>/shape_optimization/qa_baseline_<n>.json.
#   bash scripts/paper_scripts/runs/ad_baseline.sh
. "$(dirname "$0")/common.sh"
export EXTRA_ENV="MRX_DTYPE=float64" CPUS=4 MEM_GB=16

for n in 16 24 32; do
  sub ad_baseline$n 60 "" scripts/paper_scripts/ad_recovery.py --records $RECORDS --stage baseline --ns $n,$((2 * n)),$n --tag _$n
done
