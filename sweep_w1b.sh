#!/bin/bash
# Second batch.  The first covered W1's own axes; this one covers the
# COMBINATIONS that matter and the one control the chaotic case still lacks.
set -u
PC="--poincare --pc-seeds 40 --pc-periods 150"
O=/scratch/tblickhan/mrx/out/relax_prelim
S=slurm/job_relax_prelim.sh

sub () {
  tag=$1; shift
  sbatch_args=""
  while [ "$1" != "--" ]; do sbatch_args="$sbatch_args $1"; shift; done
  shift
  mkdir -p "$O/$tag"
  # shellcheck disable=SC2086
  jid=$(sbatch $sbatch_args "$S" "$@" --save-b "$O/$tag/B.h5" \
        --out "$O/$tag/$tag.json" | awk '{print $4}')
  echo "$tag -> $jid"
  ln -sfn "/scratch/tblickhan/mrx/logs/relaxprelim_$jid.out" \
     "$O/logs/live_$tag.out"
}

# THE MISSING CONTROL.  W4 (gamma) and W5 (small dt) test the numerical
# explanations for w7x_ini's chaos; RESOLUTION is untested and is gap #1 in
# the handoff.  If the chaos is discretisation it should ease at 12,24,12;
# if it is an ideal instability at beta_max 13% it should not.
sub S13_ini_res12 --time=6:00:00 -- --geometry w7x-ini-clebsch --ic clebsch \
    --ns 12,24,12 --p 3 --steps 3000 --helicity-every 250 \
    --seconds-per-arm 12000 --arms cg $PC

# Same question one step further, if 12,24,12 is ambiguous.
sub S14_ini_res16 --time=6:00:00 -- --geometry w7x-ini-clebsch --ic clebsch \
    --ns 16,32,16 --p 3 --steps 1500 --helicity-every 250 \
    --seconds-per-arm 12000 --arms cg $PC

# Does the gamma benefit SURVIVE refinement, or is it a coarse-grid artefact?
sub S15_res12_g1 --time=6:00:00 -- --geometry w7x-fmm002 --ic clebsch \
    --ns 12,24,12 --p 3 --steps 3000 --helicity-every 250 \
    --gamma 1 --mu 1e-3 --seconds-per-arm 12000 --arms cg $PC

# Longest run on the best-conditioned setting: how far does the residual go?
sub S16_g1_long --time=4:00:00 -- --geometry w7x-fmm002 --ic clebsch \
    --ns 8,16,8 --p 3 --steps 12000 --helicity-every 500 \
    --gamma 1 --mu 1e-3 --seconds-per-arm 11000 --arms cg $PC

# eta on the CHAOTIC case: if reconnection is what destroys the surfaces,
# adding real resistivity should not make it notably worse.
sub S17_ini_eta3 --time=4:00:00 -- --geometry w7x-ini-clebsch --ic clebsch \
    --ns 8,16,8 --p 3 --steps 4000 --helicity-every 250 --eta-max 1e-3 \
    --seconds-per-arm 9000 --arms cg $PC

echo "submitted"
