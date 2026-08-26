#!/bin/bash
# Second batch around W1: the combinations and the missing control.
#
# S13/S14: resolution on the chaotic w7x-ini-clebsch case (discretisation
# or ideal instability?). S15: does the gamma benefit survive refinement?
# S16: the longest run on the best-conditioned setting. S17: eta on the
# chaotic case (reconnection should not make it notably worse).
#
# Usage: run from anywhere; MRX_ROOT defaults to the repository containing
# this file. Site settings come from slurm/site.env or the environment
# (slurm/README.md). Each arm is one single-GPU job through slurm/run.sh
# running scripts/relax.py; logs land under outputs/sweep_w1b/<date>/<time>/,
# results (relax.json, B.h5) under $OUT/<tag>/.
#
# The arms stop on the energy floor (relax.py --floor-tol) or on --steps /
# --seconds, whichever comes first. relax.py does not render Poincare
# sections; B.h5 holds the initial and final fields for that.
set -u
MRX_ROOT=${MRX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
OUT=${OUT:-outputs/sweep_w1b/results}

# sub <tag> <walltime minutes> <relax.py args...>
sub () {
  tag=$1; minutes=$2; shift 2
  mkdir -p "$MRX_ROOT/$OUT/$tag"
  SCRIPT=scripts/relax.py JOB_NAME="$tag" OUTSUB="sweep_w1b" TIMEOUT_MIN="$minutes" \
    ARGS="$* --out $OUT/$tag" MRX_ROOT="$MRX_ROOT" bash "$MRX_ROOT/slurm/run.sh"
}

PC="--method cg --p 3 --diag-every 250"

sub S13_ini_res12 360 --geometry w7x-ini-clebsch --ic clebsch --ns 12,24,12 --steps 3000 --seconds 12000 $PC
sub S14_ini_res16 360 --geometry w7x-ini-clebsch --ic clebsch --ns 16,32,16 --steps 1500 --seconds 12000 $PC
sub S15_res12_g1  360 --geometry w7x-fmm002 --ic clebsch --ns 12,24,12 --steps 3000 --gamma 1 --mu 1e-3 --seconds 12000 $PC
sub S16_g1_long   240 --geometry w7x-fmm002 --ic clebsch --ns 8,16,8 --steps 12000 --gamma 1 --mu 1e-3 --seconds 11000 --method cg --p 3 --diag-every 500
sub S17_ini_eta3  240 --geometry w7x-ini-clebsch --ic clebsch --ns 8,16,8 --steps 4000 --eta-max 1e-3 --seconds 9000 $PC

echo "submitted 5 arms"
