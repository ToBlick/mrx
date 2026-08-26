#!/bin/bash
# RUN FROM THE REPO ROOT.
#
# The mu sweep, re-run on the block_jacobi diffusion preconditioner (a93bec5).
#
# The first sweep (S04/S05/S06) ran on diag(M)^-1 with eps discarded, so its
# conclusion -- "gamma/mu saturates at mu=1e-3, more smoothing hurts" -- was
# partly a preconditioner artefact: mu=1e-2 was penalised BOTH by
# over-smoothing AND by its solves being less converged per unit work as
# eps*lambda_max passed 1.  This separates the two.
#
# lambda_max(M^-1 L) ~ n_theta^2 ~ 256 at ns=(8,16,8), so eps*lambda_max is
# roughly 0.026 / 0.26 / 2.6 / 26 across the four mu below.  block_jacobi
# approximates M and ignores eps L, so it should hold to mu=1e-3, strain at
# 1e-2 and fail at 1e-1 -- which is the point of including 1e-1.
#
# Budgets are generous on purpose: the first sweep truncated S05 and S06 at
# their --seconds-per-arm cap while sitting inside a 4 h allocation, which
# made their ||F|| incomparable.
set -u
G="--geometry w7x-fmm002 --ic clebsch --ns 8,16,8 --p 3"
PC="--poincare --pc-seeds 40 --pc-periods 150"
O=/scratch/tblickhan/mrx/out/relax_prelim
S=slurm/job_relax_prelim.sh

sub () {
  tag=$1; shift
  mkdir -p "$O/$tag"
  # shellcheck disable=SC2086
  jid=$(sbatch --time=6:00:00 "$S" "$@" --steps 3000 --helicity-every 250 \
        --seconds-per-arm 16000 --arms cg $PC \
        --save-b "$O/$tag/B.h5" --out "$O/$tag/$tag.json" | awk '{print $4}')
  echo "$tag -> $jid"
  ln -sfn "/scratch/tblickhan/mrx/logs/relaxprelim_$jid.out" \
     "$O/logs/live_$tag.out"
}

sub M1_mu1e4 $G --gamma 1 --mu 1e-4
sub M2_mu1e3 $G --gamma 1 --mu 1e-3
sub M3_mu1e2 $G --gamma 1 --mu 1e-2
sub M4_mu1e1 $G --gamma 1 --mu 1e-1
sub M5_g2mu3 $G --gamma 2 --mu 1e-3
echo submitted
