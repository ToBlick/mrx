#!/bin/bash
# p-refinement at fixed h on the W1 case.
#
# S03 showed p=3 -> p=4 buys 3.4x less helicity lost per unit energy
# removed while h-refinement buys nothing: n2_dbc is unchanged by p, so p
# raises the fidelity per step where h only lowers the rate. These fill in
# the curve either side. p=1 exercises the degree-0 D-spline and polar
# extraction at the lowest order; p=5 is above the resolution where the
# k=2 vector mass preconditioner is known to hold on W7-X (k2 <= p4).
#
# Usage: run from anywhere; MRX_ROOT defaults to the repository containing
# this file. Site settings come from slurm/site.env or the environment
# (slurm/README.md). Each arm is one single-GPU job through slurm/run.sh
# running scripts/relax.py; logs land under outputs/sweep_p/<date>/<time>/,
# results (relax.json, B.h5) under $OUT/<tag>/.
#
# The arms stop on the energy floor (relax.py --floor-tol) or on --steps /
# --seconds, whichever comes first. relax.py does not render Poincare
# sections; B.h5 holds the initial and final fields for that.
set -u
MRX_ROOT=${MRX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
OUT=${OUT:-outputs/sweep_p/results}

# sub <tag> <walltime minutes> <relax.py args...>
sub () {
  tag=$1; minutes=$2; shift 2
  mkdir -p "$MRX_ROOT/$OUT/$tag"
  SCRIPT=scripts/relax.py JOB_NAME="$tag" OUTSUB="sweep_p" TIMEOUT_MIN="$minutes" \
    ARGS="$* --out $OUT/$tag" MRX_ROOT="$MRX_ROOT" bash "$MRX_ROOT/slurm/run.sh"
}

G="--geometry w7x-fmm002 --ic clebsch --ns 8,16,8 --method cg --steps 3000 --diag-every 250 --seconds 12000"

sub P1 360 $G --p 1
sub P2 360 $G --p 2
sub P5 360 $G --p 5

echo "submitted 3 arms"
