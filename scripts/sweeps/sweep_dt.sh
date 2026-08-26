#!/bin/bash
# Step-size sweep on the case that went chaotic (w7x-ini-clebsch).
#
# Step counts scale inversely with dt so each arm removes comparable
# energy; otherwise "small dt keeps its surfaces" is confounded with
# "small dt barely moved". D4 repeats the bracket on the case that survived
# (w7x-fmm002) to check the trade is a property of the scheme.
#
# Usage: run from anywhere; MRX_ROOT defaults to the repository containing
# this file. Site settings come from slurm/site.env or the environment
# (slurm/README.md). Each arm is one single-GPU job through slurm/run.sh
# running scripts/relax.py; logs land under outputs/sweep_dt/<date>/<time>/,
# results (relax.json, B.h5) under $OUT/<tag>/.
#
# The arms stop on the energy floor (relax.py --floor-tol) or on --steps /
# --seconds, whichever comes first. relax.py does not render Poincare
# sections; B.h5 holds the initial and final fields for that.
set -u
MRX_ROOT=${MRX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
OUT=${OUT:-outputs/sweep_dt/results}

# sub <tag> <walltime minutes> <relax.py args...>
sub () {
  tag=$1; minutes=$2; shift 2
  mkdir -p "$MRX_ROOT/$OUT/$tag"
  SCRIPT=scripts/relax.py JOB_NAME="$tag" OUTSUB="sweep_dt" TIMEOUT_MIN="$minutes" \
    ARGS="$* --out $OUT/$tag" MRX_ROOT="$MRX_ROOT" bash "$MRX_ROOT/slurm/run.sh"
}

G="--geometry w7x-ini-clebsch --ic clebsch --ns 8,16,8 --p 3 --method cg --dt-mode fixed"

sub D1_dt3e3 240 $G --dt0 3e-3 --steps 3000  --diag-every 250 --seconds 11000
sub D2_dt3e4 240 $G --dt0 3e-4 --steps 11000 --diag-every 500 --seconds 11000
sub D3_dt1e4 240 $G --dt0 1e-4 --steps 12000 --diag-every 500 --seconds 11000
sub D4_fmm_dt 240 --geometry w7x-fmm002 --ic clebsch --ns 8,16,8 --p 3 --method cg --dt-mode fixed --dt0 1e-3 --steps 3000 --diag-every 250 --seconds 11000

echo "submitted 4 arms"
