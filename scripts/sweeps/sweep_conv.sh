#!/bin/bash
# w7x-ini-conv, the converged W7-X equilibrium: ~20 GPU-h of hyperparameters.
#
# w7x-ini-conv and w7x-ini-clebsch come from the same GVEC run
# (State_0000_00020000.dat, converged, against _00000000.dat, the initial
# guess). Only on the converged case is "does the relaxed pressure approach
# the file's?" a meaningful question, which is why every arm samples the
# diagnostics every 250 steps.
#
# Arms: dt bracket (C1-C4: the linesearch and three fixed steps between
# 3e-3 and the linesearch's ~3e-2), hyperregularisation (C5-C7: gamma=1
# with mu bracketing the M1 optimum from above), composition (C8: dt cap
# and mu together), resolution (C9, C10), resistivity (C11: eta=1e-3 with
# the tanh ramp) and length (C12: C2 at 3.3x the steps).
#
# Budget, from a 20-step smoke run on this geometry: ~1.0 s/step steady
# state at 8^3 after ~90 s of compilation, ~100 s of setup; gamma=1 costs
# ~1.4x per step. The --seconds budget sits below the walltime on purpose.
#
# Usage: run from anywhere; MRX_ROOT defaults to the repository containing
# this file. Site settings come from slurm/site.env or the environment
# (slurm/README.md). Each arm is one single-GPU job through slurm/run.sh
# running scripts/relax.py; logs land under outputs/sweep_conv/<date>/<time>/,
# results (relax.json, B.h5) under $OUT/<tag>/.
#
# The arms stop on the energy floor (relax.py --floor-tol) or on --steps /
# --seconds, whichever comes first. relax.py does not render Poincare
# sections; B.h5 holds the initial and final fields for that.
set -u
MRX_ROOT=${MRX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
OUT=${OUT:-outputs/sweep_conv/results}

# sub <tag> <walltime minutes> <relax.py args...>
sub () {
  tag=$1; minutes=$2; shift 2
  mkdir -p "$MRX_ROOT/$OUT/$tag"
  SCRIPT=scripts/relax.py JOB_NAME="$tag" OUTSUB="sweep_conv" TIMEOUT_MIN="$minutes" \
    ARGS="$* --out $OUT/$tag" MRX_ROOT="$MRX_ROOT" bash "$MRX_ROOT/slurm/run.sh"
}

G="--geometry w7x-ini-conv --ic clebsch --p 3 --method cg --diag-every 250"

sub C1_ls          120 $G --ns 8,16,8 --steps 3000 --seconds 5400
sub C2_dt3e3       120 $G --ns 8,16,8 --steps 3000 --seconds 5400 --dt-mode fixed --dt0 3e-3
sub C3_dt1e2       120 $G --ns 8,16,8 --steps 3000 --seconds 5400 --dt-mode fixed --dt0 1e-2
sub C4_dt3e2       120 $G --ns 8,16,8 --steps 3000 --seconds 5400 --dt-mode fixed --dt0 3e-2
sub C5_mu1e4       150 $G --ns 8,16,8 --steps 3000 --seconds 7200 --gamma 1 --mu 1e-4
sub C6_mu1e3       150 $G --ns 8,16,8 --steps 3000 --seconds 7200 --gamma 1 --mu 1e-3
sub C7_mu1e2       150 $G --ns 8,16,8 --steps 3000 --seconds 7200 --gamma 1 --mu 1e-2
sub C8_mu1e4_dt3e3 150 $G --ns 8,16,8 --steps 3000 --seconds 7200 --gamma 1 --mu 1e-4 --dt-mode fixed --dt0 3e-3
sub C9_r12_ls      300 $G --ns 12,24,12 --steps 3000 --seconds 14400
sub C10_r12_dt3e3  300 $G --ns 12,24,12 --steps 3000 --seconds 14400 --dt-mode fixed --dt0 3e-3
sub C11_eta3       120 $G --ns 8,16,8 --steps 3000 --seconds 5400 --eta-max 1e-3 --eta-schedule tanh
sub C12_dt3e3_long 300 $G --ns 8,16,8 --steps 10000 --seconds 14400 --dt-mode fixed --dt0 3e-3

echo "submitted 12 arms, ~18.8 GPU-h estimated"
