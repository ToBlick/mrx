#!/bin/bash
# Implicit midpoint against explicit Euler on li383 (nfp=3): helicity drift,
# energy descent, force residual and cost per step, one slurm GPU job per arm
# through slurm/run.sh. Every arm samples the helicity every 100 steps.
#
#   bash scripts/midpoint_sweep.sh main                  # float32 (12,24,24) p=3 pair
#   bash scripts/midpoint_sweep.sh f64                   # float64 pairs, production + smoke mesh
#   bash scripts/midpoint_sweep.sh arm NAME "RELAX ARGS" TIMEOUT_MIN
#
# Job ids and the launch command go to outputs/midpoint_sweep/jobs.tsv;
# scripts/midpoint_figures.py reads the arms back.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
SWEEP=outputs/midpoint_sweep
mkdir -p "$SWEEP"
LEDGER=$SWEEP/jobs.tsv

GEOM=data/wout_li383_low_res_reference.nc
COMMON="--geometry $GEOM --ic clebsch --ns 12,24,24 --p 3 --floor-tol 1e-4 --qoi-every 100"

arm() {  # arm NAME "EXTRA RELAX ARGS" TIMEOUT_MIN
    local name=$1 args=$2 tmin=$3
    local out jid log
    out=$(SCRIPT=scripts/relax.py ARGS="$COMMON $args --out $SWEEP/$name" \
          JOB_NAME="mp_$name" OUTSUB=midpoint_sweep TIMEOUT_MIN="$tmin" bash slurm/run.sh)
    jid=$(echo "$out" | sed -n 's/Submitted batch job \([0-9]*\)/\1/p')
    log=$(echo "$out" | sed -n 's/^log: //p')
    printf '%s\t%s\t%s\t%s\t%s\n' "$jid" "$name" "$tmin" "$log" "$COMMON $args" >> "$LEDGER"
    echo "$jid $name -> $log"
}

main() {
    # float32, the production precision: cost per step, descent.  The
    # helicity floor here is the solver tolerance (sqrt eps) in both schemes.
    arm ex_lbfgs "--scheme explicit --method lbfgs --steps 3000 --seconds 2400"    90
    arm mp_lbfgs "--scheme midpoint --method lbfgs --steps 3000 --seconds 5400"    150
}

f64() {
    # float64: the O(dt) drift of the explicit scheme against the exact
    # discrete helicity of the midpoint scheme, production mesh and the
    # smoke mesh (an h-contrast).
    arm ex_lbfgs_f64 "--scheme explicit --method lbfgs --steps 1500 --seconds 3600 --precision float64" 120
    arm mp_lbfgs_f64 "--scheme midpoint --method lbfgs --steps 1500 --seconds 5400 --precision float64" 150
    arm ex_small_f64 "--scheme explicit --method lbfgs --steps 1500 --seconds 1800 --precision float64 --ns 8,16,16 --p 2" 60
    arm mp_small_f64 "--scheme midpoint --method lbfgs --steps 1500 --seconds 2700 --precision float64 --ns 8,16,16 --p 2" 75
}

case ${1:-} in
    main) main ;;
    f64) f64 ;;
    arm) arm "$2" "$3" "$4" ;;
    *) echo "usage: $0 main | f64 | arm NAME ARGS TMIN" >&2; exit 2 ;;
esac
