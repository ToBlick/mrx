#!/bin/bash
# Does the optimal mu scale like h^2?
#
# The smoother (I + mu L)^-1 damps a mode of eigenvalue lambda by
# 1/(1 + mu lambda) and lambda_max ~ 1/h^2, so the natural rule is
# mu ~ h^2. At 12,24,12 h^2 falls by (16/24)^2 = 0.444 against the
# measured optimum mu=1e-3 at 8,16,8, predicting mu_opt ~ 4.4e-4. If the
# ranking flips, mu ~ h^2 holds; if 1e-3 still wins, mu is resolution
# independent. Judged on |dH|/H per unit energy removed.
#
# Usage: run from anywhere; MRX_ROOT defaults to the repository containing
# this file. Site settings come from slurm/site.env or the environment
# (slurm/README.md). Each arm is one single-GPU job through slurm/run.sh
# running scripts/relax.py; logs land under outputs/sweep_mu_scaling/<date>/<time>/,
# results (relax.json, B.h5) under $OUT/<tag>/.
#
# The arms stop on the energy floor (relax.py --floor-tol) or on --steps /
# --seconds, whichever comes first. relax.py does not render Poincare
# sections; B.h5 holds the initial and final fields for that.
set -u
MRX_ROOT=${MRX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
OUT=${OUT:-outputs/sweep_mu_scaling/results}

# sub <tag> <walltime minutes> <relax.py args...>
sub () {
  tag=$1; minutes=$2; shift 2
  mkdir -p "$MRX_ROOT/$OUT/$tag"
  SCRIPT=scripts/relax.py JOB_NAME="$tag" OUTSUB="sweep_mu_scaling" TIMEOUT_MIN="$minutes" \
    ARGS="$* --out $OUT/$tag" MRX_ROOT="$MRX_ROOT" bash "$MRX_ROOT/slurm/run.sh"
}

G="--geometry w7x-fmm002 --ic clebsch --ns 12,24,12 --p 3 --gamma 1 --method cg --steps 3000 --diag-every 250 --seconds 18000"

sub H1_r12_mu4e4 360 $G --mu 4.4e-4
sub H2_r12_mu1e3 360 $G --mu 1e-3

echo "submitted 2 arms"
