#!/bin/bash
# li383 (NCSX, nfp=3) publication runs, 2026-09-02: the sweep arms behind the
# figures rerun on the current VMEC reader, plus the first seeded-island arms.
#
#   bash scripts/li383_pub.sh reader     # reread arms (about 9 GPU-h)
#   bash scripts/li383_pub.sh seeded     # seeded arms (about 9 GPU-h)
#   bash scripts/li383_pub.sh sections NAME [TIMEOUT_MIN]
#   bash scripts/li383_pub.sh movie NAME PLANES STEPSPEC [TIMEOUT_MIN]
#
# MRX_ROOT selects the checkout the jobs run (default: this one). Outputs go
# to $MRX_ROOT/outputs/li383_pub/<arm>; job ids to .../jobs.tsv.
set -euo pipefail
ROOT=${MRX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export MRX_ROOT=$ROOT
cd "$ROOT"
PUB=$ROOT/outputs/li383_pub
mkdir -p "$PUB"
LEDGER=$PUB/jobs.tsv

GEOM=$ROOT/data/wout_li383_low_res_reference.nc
GEOM_HI=$ROOT/data/wout_li383_1.4m.nc
COMMON="--ic clebsch --floor-tol 1e-3 --save-every 100 --steps 6000"
G1_12="--velocity-smoothing-order 1 --velocity-smoothing-scale 4.4e-4"   # mu = 0.064 / n_r^2
G1_16="--velocity-smoothing-order 1 --velocity-smoothing-scale 2.5e-4"

submit() {  # submit KIND NAME SCRIPT ARGS TIMEOUT_MIN
    local kind=$1 name=$2 script=$3 args=$4 tmin=$5
    local out
    out=$(SCRIPT="$script" ARGS="$args" JOB_NAME="li383_${kind}_${name}" OUTSUB=li383_pub \
          TIMEOUT_MIN="$tmin" bash slurm/run.sh)
    local jid log
    jid=$(echo "$out" | sed -n 's/Submitted batch job \([0-9]*\)/\1/p')
    log=$(echo "$out" | sed -n 's/^log: //p')
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$jid" "$kind" "$name" "$tmin" "$log" "$script $args" >> "$LEDGER"
    echo "$jid $kind $name -> $log"
}

arm() {  # arm NAME GEOMETRY "EXTRA RELAX ARGS" TIMEOUT_MIN
    submit relax "$1" scripts/relax.py "--geometry $2 $COMMON $3 --out $PUB/$1" "$4"
}

reader() {
    arm r12_p1_g0     "$GEOM" "--ns 12,24,12 --p 1"                                60
    arm r12_p2_g0     "$GEOM" "--ns 12,24,12 --p 2"                                90
    arm r12_p3_g0     "$GEOM" "--ns 12,24,12 --p 3"                               150
    arm r12_p4_g0     "$GEOM" "--ns 12,24,12 --p 4 --seconds 7200"                240
    arm r12_p3_g0_f64 "$GEOM" "--ns 12,24,12 --p 3 --precision float64 --seconds 9000" 300
    arm r12_p3_g1     "$GEOM" "--ns 12,24,12 --p 3 $G1_12 --seconds 9000"         300
    arm r16_p3_g1     "$GEOM" "--ns 16,32,16 --p 3 $G1_16 --seconds 12000"        480
    arm hi_r12_p3_g0  "$GEOM_HI" "--ns 12,24,12 --p 3"                            120
}

seeded() {
    # (m, n) = (6, 1): the iota = 1/2 surface (rho 0.55); (5, 1): iota = 3/5 (rho 0.80).
    # eps = |dB^rho| / |B^zeta| at the surface; island width ~ 1.6 sqrt(eps nfp / (m iota')).
    local s61="--seed 6,1,0.551,0.1" s51="--seed 5,1,0.798,0.1"
    arm s61_e1e-3_g0  "$GEOM" "--ns 12,24,12 --p 3 $s61 --seed-eps 1e-3"          150
    arm s61_e3e-3_g0  "$GEOM" "--ns 12,24,12 --p 3 $s61 --seed-eps 3e-3"          150
    arm s61_e1e-2_g0  "$GEOM" "--ns 12,24,12 --p 3 $s61 --seed-eps 1e-2"          150
    arm s61_e3e-3_g1  "$GEOM" "--ns 12,24,12 --p 3 $s61 --seed-eps 3e-3 $G1_12 --seconds 9000" 300
    arm s51_e3e-3_g0  "$GEOM" "--ns 12,24,12 --p 3 $s51 --seed-eps 3e-3"          150
    arm s51_e3e-3_g1  "$GEOM" "--ns 12,24,12 --p 3 $s51 --seed-eps 3e-3 $G1_12 --seconds 9000" 300
    arm r16_s61_e3e-3_g1 "$GEOM" "--ns 16,32,16 --p 3 $s61 --seed-eps 3e-3 $G1_16 --seconds 10800" 480
}

sections() {  # sections NAME [TIMEOUT_MIN]
    submit sec "$1" scripts/poincare_relax.py \
        "$PUB/$1/B.h5 --fields ic,final --planes 0,0.25,0.5 --out $PUB/$1/poincare" "${2:-30}"
}

movie() {  # movie NAME PLANES STEPSPEC [TIMEOUT_MIN]
    submit mov "$1" scripts/poincare_relax.py \
        "$PUB/$1/B.h5 --fields snapshots --snapshot-steps $3 --planes $2 --out $PUB/$1/movie" "${4:-120}"
}

case ${1:-} in
    reader) reader ;;
    seeded) seeded ;;
    sections) sections "$2" "${3:-30}" ;;
    movie) movie "$2" "$3" "$4" "${5:-120}" ;;
    *) echo "usage: $0 reader | seeded | sections NAME [TMIN] | movie NAME PLANES STEPSPEC [TMIN]" >&2; exit 2 ;;
esac
