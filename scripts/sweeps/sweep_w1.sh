#!/bin/bash
# Sweep around the W1 case (w7x-fmm002, Clebsch IC, cg).
#
# W1 is the arm that kept its nested surfaces. The axes: resolution
# (S01, S02), degree (S03), hyperregularisation gamma/mu (S04-S06), length
# (S07), resistivity eta (S08-S10), optimiser (S11) and lambda off (S12:
# fluxes, iota and helicity must not move; the force and pressure must).
# Rank arms on |dH| per unit energy removed. ~45 GPU-h in total; the
# 8,16,8 arms are ~1 GPU-h each, the resolution arms are the expensive ones.
#
# Usage: run from anywhere; MRX_ROOT defaults to the repository containing
# this file. Site settings come from slurm/site.env or the environment
# (slurm/README.md). Each arm is one single-GPU job through slurm/run.sh
# running scripts/relax.py; logs land under outputs/sweep_w1/<date>/<time>/,
# results (relax.json, B.h5) under $OUT/<tag>/.
#
# The arms stop on the energy floor (relax.py --floor-tol) or on --steps /
# --seconds, whichever comes first. relax.py does not render Poincare
# sections; B.h5 holds the initial and final fields for that.
set -u
MRX_ROOT=${MRX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
OUT=${OUT:-outputs/sweep_w1/results}

# sub <tag> <walltime minutes> <relax.py args...>
sub () {
  tag=$1; minutes=$2; shift 2
  mkdir -p "$MRX_ROOT/$OUT/$tag"
  SCRIPT=scripts/relax.py JOB_NAME="$tag" OUTSUB="sweep_w1" TIMEOUT_MIN="$minutes" \
    ARGS="$* --out $OUT/$tag" MRX_ROOT="$MRX_ROOT" bash "$MRX_ROOT/slurm/run.sh"
}

G="--geometry w7x-fmm002 --ic clebsch --method cg"

sub S01_res12 360 $G --ns 12,24,12 --p 3 --steps 3000 --diag-every 250 --seconds 12000
sub S02_res16 360 $G --ns 16,32,16 --p 3 --steps 2000 --diag-every 250 --seconds 12000
sub S03_p4    360 $G --ns 8,16,8 --p 4 --steps 3000 --diag-every 250 --seconds 12000
sub S04_g1mu3 240 $G --ns 8,16,8 --p 3 --steps 3000 --diag-every 250 --gamma 1 --mu 1e-3 --seconds 9000
sub S05_g1mu2 240 $G --ns 8,16,8 --p 3 --steps 3000 --diag-every 250 --gamma 1 --mu 1e-2 --seconds 9000
sub S06_g2mu3 240 $G --ns 8,16,8 --p 3 --steps 2000 --diag-every 250 --gamma 2 --mu 1e-3 --seconds 9000
sub S07_long  240 $G --ns 8,16,8 --p 3 --steps 14000 --diag-every 500 --seconds 11000
sub S08_eta4  240 $G --ns 8,16,8 --p 3 --steps 4000 --diag-every 250 --eta-max 1e-4 --seconds 9000
sub S09_eta3  240 $G --ns 8,16,8 --p 3 --steps 4000 --diag-every 250 --eta-max 1e-3 --seconds 9000
sub S10_eta2  240 $G --ns 8,16,8 --p 3 --steps 4000 --diag-every 250 --eta-max 1e-2 --seconds 9000
sub S11_grad  240 --geometry w7x-fmm002 --ic clebsch --method gradient --ns 8,16,8 --p 3 --steps 3000 --diag-every 250 --seconds 4500
sub S11_lbfgs 240 --geometry w7x-fmm002 --ic clebsch --method lbfgs --ns 8,16,8 --p 3 --steps 3000 --diag-every 250 --seconds 4500
sub S12_nolam 240 $G --ns 8,16,8 --p 3 --steps 3000 --diag-every 250 --no-lambda --seconds 9000

echo "submitted 13 arms"
