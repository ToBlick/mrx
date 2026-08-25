#!/bin/bash
# dt sweep on the case that went chaotic (w7x_ini clebsch), now that LR3 vs W5
# has identified step size as the control variable.
#
# Step counts scale INVERSELY with dt so each arm removes comparable energy --
# otherwise "small dt keeps its surfaces" is confounded with "small dt barely
# moved", which is exactly the confound that made me over-read W1.
set -u
# RUN FROM THE REPO ROOT: the paths below (slurm/job_relax_prelim.sh)
# are relative to it, not to this file's directory.
G="--geometry w7x-ini-clebsch --ic clebsch --ns 8,16,8 --p 3"
PC="--poincare --pc-seeds 40 --pc-periods 150"
O=/scratch/tblickhan/mrx/out/relax_prelim
S=slurm/job_relax_prelim.sh

sub () {
  tag=$1; shift
  mkdir -p "$O/$tag"
  # shellcheck disable=SC2086
  jid=$(sbatch --time=4:00:00 "$S" "$@" --save-b "$O/$tag/B.h5" \
        --out "$O/$tag/$tag.json" | awk '{print $4}')
  echo "$tag -> $jid"
  ln -sfn "/scratch/tblickhan/mrx/logs/relaxprelim_$jid.out" \
     "$O/logs/live_$tag.out"
}

# W5 already covers dt = 1e-3 at 3000 steps.  Bracket it either side, and give
# the smaller steps proportionally more of them.
sub D1_dt3e3  $G --dt-mode fixed --dt0 3e-3 --steps 3000  --helicity-every 250 \
    --seconds-per-arm 11000 --arms cg $PC
sub D2_dt3e4  $G --dt-mode fixed --dt0 3e-4 --steps 11000 --helicity-every 500 \
    --seconds-per-arm 11000 --arms cg $PC
sub D3_dt1e4  $G --dt-mode fixed --dt0 1e-4 --steps 12000 --helicity-every 500 \
    --seconds-per-arm 11000 --arms cg $PC

# And the same bracket on the case that survived, to check the trade is a
# property of the SCHEME rather than of w7x_ini.
sub D4_fmm_dt --geometry w7x-fmm002 --ic clebsch --ns 8,16,8 --p 3 \
    --dt-mode fixed --dt0 1e-3 --steps 3000 --helicity-every 250 \
    --seconds-per-arm 11000 --arms cg $PC
echo submitted
