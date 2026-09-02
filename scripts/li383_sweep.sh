#!/bin/bash
# li383 (NCSX, nfp=3) relaxation sweep: resolution, degree, gamma, precision.
#
# One slurm GPU job per arm through slurm/run.sh, every arm writing B
# snapshots every 100 steps. Usage:
#
#   bash scripts/li383_sweep.sh wave1          # launch the first wave (wave2: the second)
#   bash scripts/li383_sweep.sh arm NAME "RELAX ARGS" TIMEOUT_MIN
#   bash scripts/li383_sweep.sh sections NAME [TIMEOUT_MIN]   # ic,final at 0,0.25,0.5
#   bash scripts/li383_sweep.sh movie NAME PLANES STEPSPEC [TIMEOUT_MIN]
#
# Job ids and the launch command go to outputs/li383_sweep/jobs.tsv.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
SWEEP=outputs/li383_sweep
mkdir -p "$SWEEP"
LEDGER=$SWEEP/jobs.tsv

GEOM=data/wout_li383_low_res_reference.nc
COMMON="--geometry $GEOM --ic clebsch --method cg --floor-tol 1e-3 --save-every 100"

submit() {  # submit KIND NAME SCRIPT ARGS TIMEOUT_MIN
    local kind=$1 name=$2 script=$3 args=$4 tmin=$5
    local out
    out=$(SCRIPT="$script" ARGS="$args" JOB_NAME="li383_${kind}_${name}" OUTSUB=li383_sweep \
          TIMEOUT_MIN="$tmin" bash slurm/run.sh)
    local jid log
    jid=$(echo "$out" | sed -n 's/Submitted batch job \([0-9]*\)/\1/p')
    log=$(echo "$out" | sed -n 's/^log: //p')
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$jid" "$kind" "$name" "$tmin" "$log" "$script $args" >> "$LEDGER"
    echo "$jid $kind $name -> $log"
}

arm() {  # arm NAME "EXTRA RELAX ARGS" TIMEOUT_MIN
    submit relax "$1" scripts/relax.py "$COMMON $2 --out $SWEEP/$1" "$3"
}

sections() {  # sections NAME [TIMEOUT_MIN]
    submit sec "$1" scripts/poincare_relax.py \
        "$SWEEP/$1/B.h5 --fields ic,final --planes 0,0.25,0.5 --out $SWEEP/$1/poincare" "${2:-30}"
}

movie() {  # movie NAME PLANES STEPSPEC [TIMEOUT_MIN]
    submit mov "$1" scripts/poincare_relax.py \
        "$SWEEP/$1/B.h5 --fields snapshots --snapshot-steps $3 --planes $2 --out $SWEEP/$1/movie" "${4:-120}"
}

wave1() {
    # mu = 0.064 h^2 with h = 1/n_r (the rule of the 2026-08-26 W7-X study)
    arm r12_p3_g0     "--ns 12,24,12 --p 3 --steps 6000"                                   180
    arm r16_p3_g0     "--ns 16,32,16 --p 3 --steps 6000 --seconds 14000"                   360
    arm r24_p3_g0     "--ns 24,48,24 --p 3 --steps 6000 --seconds 30000"                   1200
    arm r12_p1_g0     "--ns 12,24,12 --p 1 --steps 6000"                                   120
    arm r12_p2_g0     "--ns 12,24,12 --p 2 --steps 6000"                                   150
    arm r12_p4_g0     "--ns 12,24,12 --p 4 --steps 6000 --seconds 12000"                   300
    arm r12_p3_g1     "--ns 12,24,12 --p 3 --steps 6000 --velocity-smoothing-order 1 --velocity-smoothing-scale 4.4e-4 --seconds 14000" 360
    arm r16_p3_g1     "--ns 16,32,16 --p 3 --steps 6000 --velocity-smoothing-order 1 --velocity-smoothing-scale 2.5e-4 --seconds 25000" 720
    arm r12_p3_g0_f64 "--ns 12,24,12 --p 3 --steps 6000 --precision float64 --seconds 12000" 300
}

wave2() {
    # decided from the wave-1 costs (r24 p=3: 1.9 s/step; gamma=1 ~2x; p=4 ~1.6x)
    arm r24_p3_g1     "--ns 24,48,24 --p 3 --steps 6000 --velocity-smoothing-order 1 --velocity-smoothing-scale 1.1e-4 --seconds 30000" 1200
    arm r16_p4_g0     "--ns 16,32,16 --p 4 --steps 6000 --seconds 14000"                   360
    arm r12_p4_g1     "--ns 12,24,12 --p 4 --steps 6000 --velocity-smoothing-order 1 --velocity-smoothing-scale 4.4e-4 --seconds 16000" 480
    arm r16_p2_g0     "--ns 16,32,16 --p 2 --steps 6000 --seconds 8000"                    180
}

case ${1:-} in
    wave1) wave1 ;;
    wave2) wave2 ;;
    arm) arm "$2" "$3" "$4" ;;
    sections) sections "$2" "${3:-30}" ;;
    movie) movie "$2" "$3" "$4" "${5:-120}" ;;
    *) echo "usage: $0 wave1 | arm NAME ARGS TMIN | sections NAME [TMIN] | movie NAME PLANES STEPSPEC [TMIN]" >&2; exit 2 ;;
esac
