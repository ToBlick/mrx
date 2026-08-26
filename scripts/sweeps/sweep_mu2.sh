#!/bin/bash
# The mu sweep on the metric-lumping diffusion preconditioner.
#
# lambda_max(M^-1 L) ~ n_theta^2 ~ 256 at ns=(8,16,8), so mu*lambda_max is
# roughly 0.026 / 0.26 / 2.6 / 26 across the four mu below. The
# preconditioner approximates M and ignores mu L, so it should hold to
# mu=1e-3, strain at 1e-2 and fail at 1e-1. Budgets are generous: the first
# sweep truncated S05/S06 at their arm budget and made ||F|| incomparable.
#
# Usage: run from anywhere; MRX_ROOT defaults to the repository containing
# this file. Site settings come from slurm/site.env or the environment
# (slurm/README.md). Each arm is one single-GPU job through slurm/run.sh
# running scripts/relax.py; logs land under outputs/sweep_mu2/<date>/<time>/,
# results (relax.json, B.h5) under $OUT/<tag>/.
#
# The arms stop on the energy floor (relax.py --floor-tol) or on --steps /
# --seconds, whichever comes first. relax.py does not render Poincare
# sections; B.h5 holds the initial and final fields for that.
set -u
MRX_ROOT=${MRX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
OUT=${OUT:-outputs/sweep_mu2/results}

# sub <tag> <walltime minutes> <relax.py args...>
sub () {
  tag=$1; minutes=$2; shift 2
  mkdir -p "$MRX_ROOT/$OUT/$tag"
  SCRIPT=scripts/relax.py JOB_NAME="$tag" OUTSUB="sweep_mu2" TIMEOUT_MIN="$minutes" \
    ARGS="$* --out $OUT/$tag" MRX_ROOT="$MRX_ROOT" bash "$MRX_ROOT/slurm/run.sh"
}

G="--geometry w7x-fmm002 --ic clebsch --ns 8,16,8 --p 3 --method cg --steps 3000 --diag-every 250 --seconds 16000"

sub M1_mu1e4 360 $G --gamma 1 --mu 1e-4
sub M2_mu1e3 360 $G --gamma 1 --mu 1e-3
sub M3_mu1e2 360 $G --gamma 1 --mu 1e-2
sub M4_mu1e1 360 $G --gamma 1 --mu 1e-1
sub M5_g2mu3 360 $G --gamma 2 --mu 1e-3

echo "submitted 5 arms"
