#!/bin/bash
# Fig. 8: Poincare sections of LP and of the end field of the 10 mm draw 0 on LP's interior (--remesh), 160 lines x
# 400 periods, float64 (paper/ad_recovery.py --stage trace). Needs ad_constrained.sh's draw 0 (qa_recover_M10s0.npz).
# Writes <RECORDS>/shape_optimization/qa_trace_M10s0_remesh.npz, which paper/ad_poincare_page.py renders.
#   bash paper/runs/ad_trace.sh
. "$(dirname "$0")/common.sh"
export EXTRA_ENV="MRX_DTYPE=float64" CPUS=4 MEM_GB=16

sub ad_trace 60 "" paper/ad_recovery.py --records $RECORDS --stage trace --remesh --tag _M10s0
