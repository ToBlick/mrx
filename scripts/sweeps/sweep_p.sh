#!/bin/bash
# RUN FROM THE REPO ROOT.
#
# p-refinement at fixed h on the W1 case.  S03 showed p=3 -> p=4 buys 3.4x
# less helicity lost per unit energy removed while h-refinement buys nothing,
# because n2_dbc is unchanged by p (2192 at both) so the step scale does not
# move -- p raises the FIDELITY PER STEP where h only lowers the RATE.
# These fill in the curve either side.
#
# Two known risks, both worth measuring rather than avoiding:
#   p=1  the degree-0 / unit-integral D-spline subtleties, and polar
#        extraction at the lowest order.
#   p=5  memory records k=2 vector mass breaking down above p=4 on W7-X
#        (k1 works to p5, k2 <= p4, k2 p5 open).  If it fails, that is a
#        finding about the preconditioner, not about relaxation.
set -u
G="--geometry w7x-fmm002 --ic clebsch --ns 8,16,8"
PC="--poincare --pc-seeds 40 --pc-periods 150"
O=/scratch/tblickhan/mrx/out/relax_prelim
S=slurm/job_relax_prelim.sh

sub () {
  tag=$1; shift
  mkdir -p "$O/$tag"
  # shellcheck disable=SC2086
  jid=$(sbatch --time=6:00:00 "$S" "$@" --save-b "$O/$tag/B.h5" \
        --out "$O/$tag/$tag.json" | awk '{print $4}')
  echo "$tag -> $jid"
  ln -sfn "/scratch/tblickhan/mrx/logs/relaxprelim_$jid.out" \
     "$O/logs/live_$tag.out"
}

sub P1 $G --p 1 --steps 3000 --helicity-every 250 --seconds-per-arm 12000 \
    --arms cg $PC
sub P2 $G --p 2 --steps 3000 --helicity-every 250 --seconds-per-arm 12000 \
    --arms cg $PC
sub P5 $G --p 5 --steps 3000 --helicity-every 250 --seconds-per-arm 12000 \
    --arms cg $PC
echo submitted
