#!/bin/bash
# Implicit midpoint against explicit Euler on li383 (nfp=3): helicity drift,
# energy descent, force residual and cost per step, one slurm GPU job per arm
# through slurm/run.sh. Every arm samples the helicity every 100 steps.
#
#   bash scripts/midpoint_sweep.sh main                  # float32 (12,24,24) p=3 pair
#   bash scripts/midpoint_sweep.sh small                 # float64 smoke-mesh 2x2, scheme x H-space
#   bash scripts/midpoint_sweep.sh small32               # their float32 twins (natural H, B only)
#   bash scripts/midpoint_sweep.sh f64                   # float64 production pair, Dirichlet H
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

small() {
    # float64 on the smoke mesh, scheme x H-space: the O(dt) drift of the
    # explicit scheme against the midpoint's, with the natural H (E and H in
    # different spaces, the wall layer of load(u x H) leaks) and the
    # Dirichlet H (exact to the solves).
    local c="--method lbfgs --steps 1000 --precision float64 --ns 8,16,16 --p 2"
    arm ex_small_f64    "--scheme explicit $c --seconds 1800"               60
    arm mp_small_f64    "--scheme midpoint $c --seconds 2700"               75
    arm ex_small_f64_Hd "--scheme explicit $c --seconds 1800 --dirichlet-H" 60
    arm mp_small_f64_Hd "--scheme midpoint $c --seconds 2700 --dirichlet-H" 75
    # the B-only route (J x B, u x B, no proxy): its midpoint drift is the
    # grid's projection error alone, the time error being gone
    arm ex_small_f64_bonly "--scheme explicit $c --seconds 1800 --stepper bonly" 60
    arm mp_small_f64_bonly "--scheme midpoint $c --seconds 2700 --stepper bonly" 75
}

small32() {
    # the float32 twins of the natural-H and B-only small arms: the
    # (route) x (precision) x (scheme) figure, scripts/midpoint_figures.py eight_figure
    local c="--method lbfgs --steps 1000 --ns 8,16,16 --p 2"
    arm ex_small_f32       "--scheme explicit $c --seconds 1800"                 60
    arm mp_small_f32       "--scheme midpoint $c --seconds 2700"                 75
    arm ex_small_f32_bonly "--scheme explicit $c --seconds 1800 --stepper bonly" 60
    arm mp_small_f32_bonly "--scheme midpoint $c --seconds 2700 --stepper bonly" 75
}

f64() {
    # float64 on the production mesh, Dirichlet H
    arm ex_lbfgs_f64_Hd "--scheme explicit --method lbfgs --steps 1500 --seconds 3600 --precision float64 --dirichlet-H" 120
    arm mp_lbfgs_f64_Hd "--scheme midpoint --method lbfgs --steps 1500 --seconds 5400 --precision float64 --dirichlet-H" 150
}

case ${1:-} in
    main) main ;;
    small) small ;;
    small32) small32 ;;
    f64) f64 ;;
    arm) arm "$2" "$3" "$4" ;;
    *) echo "usage: $0 main | small | small32 | f64 | arm NAME ARGS TMIN" >&2; exit 2 ;;
esac
