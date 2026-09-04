#!/bin/bash
# Implicit midpoint against explicit Euler on li383 (nfp=3): helicity drift,
# energy descent, force residual and cost per step, one slurm GPU job per arm
# through slurm/run.sh. Every arm samples the helicity every 100 steps (the
# chunk). The study is docs/research/implicit_midpoint_2026-09-04.md; its
# natural-H arms (H without the wall condition) are not reproducible since
# the 2026-09-04 prune: the auxiliary field is always the Dirichlet H now,
# and the default step reads B itself.
#
#   bash scripts/midpoint_sweep.sh small                 # float64 smoke mesh: scheme x (B, auxiliary H)
#   bash scripts/midpoint_sweep.sh small32               # their float32 twins
#   bash scripts/midpoint_sweep.sh f64                   # float64 production pair, auxiliary H
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
COMMON="--geometry $GEOM --ns 12,24,24 --p 3 --floor-tol 1e-4 --chunk 100"
AUX="--auxiliary-B-field true"

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

small() {
    # float64 on the smoke mesh: the helicity drift of each scheme with the
    # 2-form B in the cross products and with the auxiliary (Dirichlet) H.
    # Exact conservation needs the midpoint scheme AND the auxiliary field.
    local c="--steps 1000 --precision float64 --ns 8,16,16 --p 2"
    arm ex_small_f64_bonly "--scheme explicit $c --seconds 1800"      60
    arm mp_small_f64_bonly "--scheme midpoint $c --seconds 2700"      75
    arm ex_small_f64_Hd    "--scheme explicit $c --seconds 1800 $AUX" 60
    arm mp_small_f64_Hd    "--scheme midpoint $c --seconds 2700 $AUX" 75
}

small32() {
    # The float32 twins: the helicity floor is the solver tolerance there.
    local c="--steps 1000 --ns 8,16,16 --p 2"
    arm ex_small_f32_bonly "--scheme explicit $c --seconds 1800"      60
    arm mp_small_f32_bonly "--scheme midpoint $c --seconds 2700"      75
    arm ex_small_f32_Hd    "--scheme explicit $c --seconds 1800 $AUX" 60
    arm mp_small_f32_Hd    "--scheme midpoint $c --seconds 2700 $AUX" 75
}

f64() {
    # float64 on the production mesh, auxiliary H: the publication pair.
    arm ex_lbfgs_f64_Hd "--scheme explicit --steps 1500 --seconds 3600 --precision float64 $AUX" 120
    arm mp_lbfgs_f64_Hd "--scheme midpoint --steps 1500 --seconds 5400 --precision float64 $AUX" 150
}

case ${1:-} in
    small) small ;;
    small32) small32 ;;
    f64) f64 ;;
    arm) arm "$2" "$3" "$4" ;;
    *) echo "usage: $0 small | small32 | f64 | arm NAME ARGS TMIN" >&2; exit 2 ;;
esac
