#!/bin/bash
# RUN FROM THE REPO ROOT.
#
# Does the optimal mu scale like h^2?
#
# The smoother (I + mu L)^-1 damps a mode of eigenvalue lambda by
# 1/(1 + mu lambda), and lambda_max(M^-1 L) ~ 1/h^2, so the natural rule is
# mu ~ h^2 -- touch the top of the spectrum and leave resolved modes alone.
# At ns=(8,16,8) the finest direction is poloidal (n_theta = 16), lambda_max
# ~ 256, and the measured optimum mu = 1e-3 sits at mu*lambda_max ~ 0.26,
# i.e. mu ~ h^2/4.  mu = 1e-2 (mu*lambda_max ~ 2.6) was worse because at that
# strength it damps a mode at lambda_max/4 by 39% -- smoothing physics, not
# noise.
#
# That is theory plus two points at ONE resolution.  This tests the scaling.
# At 12^3, n_theta = 24, so h^2 falls by (16/24)^2 = 0.444 and the rule
# predicts mu_opt ~ 4.4e-4.
#
# FALSIFIABLE EITHER WAY:
#   ranking FLIPS (4.4e-4 beats 1e-3)  -> mu ~ h^2 holds
#   1e-3 still wins                    -> mu is resolution-independent and the
#                                         h^2 reasoning is wrong
#
# Judged on |dH|/H per unit ENERGY REMOVED, which is step-count independent
# and does not care whether the arms truncate at the same wall-clock.
set -u
G="--geometry w7x-fmm002 --ic clebsch --ns 12,24,12 --p 3 --gamma 1"
PC="--poincare --pc-seeds 40 --pc-periods 150"
O=/scratch/tblickhan/mrx/out/relax_prelim
S=slurm/job_relax_prelim.sh

sub () {
  tag=$1; shift
  mkdir -p "$O/$tag"
  # shellcheck disable=SC2086
  jid=$(sbatch --time=6:00:00 "$S" "$@" --steps 3000 --helicity-every 250 \
        --seconds-per-arm 18000 --arms cg $PC \
        --save-b "$O/$tag/B.h5" --out "$O/$tag/$tag.json" | awk '{print $4}')
  echo "$tag -> $jid"
  ln -sfn "/scratch/tblickhan/mrx/logs/relaxprelim_$jid.out" \
     "$O/logs/live_$tag.out"
}

sub H1_r12_mu4e4 $G --mu 4.4e-4
sub H2_r12_mu1e3 $G --mu 1e-3
echo submitted
