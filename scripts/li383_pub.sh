#!/bin/bash
# li383 (NCSX, nfp=3) publication runs, 2026-09-02: the sweep arms behind the
# figures rerun on the current VMEC reader, plus the first seeded-island arms.
#
#   bash scripts/li383_pub.sh reader     # reread arms (about 9 GPU-h)
#   bash scripts/li383_pub.sh seeded     # seeded arms (about 9 GPU-h)
#   bash scripts/li383_pub.sh deep       # three arms past the 1e-3 floor
#   bash scripts/li383_pub.sh hsweep_p2  # gamma = 1 h-sweep at p = 2 (about 16 GPU-h)
#   bash scripts/li383_pub.sh eta [ETA|ARM...]  # tanh resistivity sweep, outputs/li383_eta (about 8 GPU-h)
#   bash scripts/li383_pub.sh bonly [smoke]  # B-only step (no H), outputs/li383_bonly
#   bash scripts/li383_pub.sh sections NAME [TIMEOUT_MIN]
#   bash scripts/li383_pub.sh movie NAME PLANES STEPSPEC [TIMEOUT_MIN]
#
# MRX_ROOT selects the checkout the jobs run (default: this one). Outputs go
# to $MRX_ROOT/outputs/$SUB/<arm> (SUB defaults to li383_pub; the eta sweep
# sets li383_eta, and `SUB=li383_eta ... sections NAME` addresses it); job ids
# to $MRX_ROOT/outputs/$SUB/jobs.tsv.
set -euo pipefail
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # the checkout this launcher lives in
ROOT=${MRX_ROOT:-$WT}
export MRX_ROOT=$ROOT
cd "$ROOT"
SUB=${SUB:-li383_pub}
PUB=$ROOT/outputs/$SUB
mkdir -p "$PUB"
LEDGER=$PUB/jobs.tsv

GEOM=$ROOT/data/wout_li383_low_res_reference.nc
GEOM_HI=$ROOT/data/wout_li383_1.4m.nc
# floor 1e-4 since 2026-09-02 (the arms of that day ran at 1e-3 and all stopped
# there): let --steps / --seconds end the run and show whether it bottoms out.
COMMON="--ic clebsch --floor-tol 1e-4 --save-every 100 --steps 6000"
G1_12="--velocity-smoothing-order 1 --velocity-smoothing-scale 4.4e-4"   # mu = 0.064 / n_r^2
G1_16="--velocity-smoothing-order 1 --velocity-smoothing-scale 2.5e-4"

submit() {  # submit KIND NAME SCRIPT ARGS TIMEOUT_MIN
    local kind=$1 name=$2 script=$3 args=$4 tmin=$5
    local out
    out=$(SCRIPT="$script" ARGS="$args" JOB_NAME="li383_${kind}_${name}" OUTSUB=$SUB \
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
    # New arms follow the 2026-09-02 mesh rule (n, 2n, 2n) and use the ns = 49
    # reference (IC residual 0.013 instead of the ns = 16 file's own 0.054).
    # (m, n) = (6, 1): the iota = 1/2 surface (rho 0.544); (5, 1): iota = 3/5 (rho 0.794).
    # eps = |dB^rho| / |B^zeta| at the surface; island width ~ 1.6 sqrt(eps nfp / (m iota')).
    local s61="--seed 6,1,0.544,0.1" s51="--seed 5,1,0.794,0.1"
    arm hi_r12x24_p3_g0  "$GEOM_HI" "--ns 12,24,24 --p 3 --seconds 5400"                       180
    arm s61_e1e-3_g0     "$GEOM_HI" "--ns 12,24,24 --p 3 $s61 --seed-eps 1e-3 --seconds 5400"  180
    arm s61_e3e-3_g0     "$GEOM_HI" "--ns 12,24,24 --p 3 $s61 --seed-eps 3e-3 --seconds 5400"  180
    arm s61_e1e-2_g0     "$GEOM_HI" "--ns 12,24,24 --p 3 $s61 --seed-eps 1e-2 --seconds 5400"  180
    arm s61_e3e-3_g1     "$GEOM_HI" "--ns 12,24,24 --p 3 $s61 --seed-eps 3e-3 $G1_12 --seconds 9000" 300
    arm s51_e3e-3_g0     "$GEOM_HI" "--ns 12,24,24 --p 3 $s51 --seed-eps 3e-3 --seconds 5400"  180
    arm s51_e3e-3_g1     "$GEOM_HI" "--ns 12,24,24 --p 3 $s51 --seed-eps 3e-3 $G1_12 --seconds 9000" 300
    arm r16_s61_e3e-3_g1 "$GEOM_HI" "--ns 16,32,32 --p 3 $s61 --seed-eps 3e-3 $G1_16 --seconds 12000" 480
}

deep() {
    # The 2026-09-02 arms all stopped at the 1e-3 floor; these three run on to
    # the step / wall cap (floor 1e-4) to show where the residual bottoms out.
    local s61="--seed 6,1,0.544,0.1"
    arm hi_r12x24_p3_g0_f4 "$GEOM_HI" "--ns 12,24,24 --p 3 --seconds 5400"                            180
    arm s61_e3e-3_g0_f4    "$GEOM_HI" "--ns 12,24,24 --p 3 $s61 --seed-eps 3e-3 --seconds 5400"       180
    arm s61_e3e-3_g1_f4    "$GEOM_HI" "--ns 12,24,24 --p 3 $s61 --seed-eps 3e-3 $G1_12 --seconds 7200" 240
}

hsweep_p2() {
    # 2026-09-02: gamma = 1 under h-refinement at fixed p = 2 on the ns = 49
    # reference, mesh (n, 2n, 2n), mu = 0.064 / n^2, floor 1e-5 so the 5000-step
    # cap (or the wall cap) ends the run and the floor versus h is what is measured.
    # Wall caps sum to 18.5 GPU-h; the n = 32 rung takes about 10 h of it.
    # These follow COMMON on the command line, so they override its floor and step cap.
    local common="--p 2 --floor-tol 1e-5 --steps 5000 --velocity-smoothing-order 1"
    arm h8_p2_g1  "$GEOM_HI" "--ns 8,16,16  $common --velocity-smoothing-scale 1.0e-3 --seconds 1800"    90
    arm h12_p2_g1 "$GEOM_HI" "--ns 12,24,24 $common --velocity-smoothing-scale 4.44e-4 --seconds 3600"   150
    arm h16_p2_g1 "$GEOM_HI" "--ns 16,32,32 $common --velocity-smoothing-scale 2.5e-4 --seconds 7200"    300
    arm h24_p2_g1 "$GEOM_HI" "--ns 24,48,48 $common --velocity-smoothing-scale 1.11e-4 --seconds 18000"  660
    arm h32_p2_g1 "$GEOM_HI" "--ns 32,64,64 $common --velocity-smoothing-scale 6.25e-5 --seconds 36000" 1260
}

eta() {
    # 2026-09-03: resistivity sweep mirroring the h-sweep's (16,32,32) rung:
    # p = 2, gamma = 1, tanh schedule (eta_max for the first third of the 5000
    # steps, dropped to ~0 over the middle third, ideal at the end), unseeded
    # and with the (6, 1) chain at eps 3e-3. --eta-every keeps eta K dt >= 2e-5
    # per resistive solve (dt is about 2 under gamma = 1). SUB=li383_eta.
    # Floor 0 since 2026-09-03 (the first launch used 1e-5 and the eta >= 1e-5
    # arms tripped it inside the resistive phase, ending resistive instead of
    # ideal); `eta ETA...` reruns only the named etas.
    local common="--ns 16,32,32 --p 2 --floor-tol 0 --steps 5000 $G1_16 --eta-schedule tanh --seconds 7200"
    local s61="--seed 6,1,0.544,0.1 --seed-eps 3e-3"
    local e K
    for e in 1e-7:100 1e-6:10 1e-5:1 1e-4:1; do
        K=${e#*:}; e=${e%:*}
        # `eta 1e-5` reruns both arms of that rung, `eta s61_eta1e-5` one arm.
        if [ $# -eq 0 ] || [[ " $* " == *" $e "* ]] || [[ " $* " == *" eta$e "* ]]; then
            arm eta$e     "$GEOM_HI" "$common --eta-max $e --eta-every $K"      300
        fi
        if [ $# -eq 0 ] || [[ " $* " == *" $e "* ]] || [[ " $* " == *" s61_eta$e "* ]]; then
            arm s61_eta$e "$GEOM_HI" "$common --eta-max $e --eta-every $K $s61" 300
        fi
    done
}

bonly() {  # bonly [smoke]
    # 2026-09-03: the B-only step (J x B, u x B, no auxiliary H) of
    # mrx/experimental/bonly_relaxation.py, twin of h16_p2_g1, helicity sampled
    # every 50 steps. The code comes from THIS checkout (worktree): SCRIPT is
    # absolute and PYTHONPATH is overridden; data and outputs stay under $ROOT.
    local wt=$WT
    export EXTRA_ENV="PYTHONPATH=$wt"
    local common="--ic clebsch --ns 16,32,32 --p 2 $G1_16 --floor-tol 1e-5 --save-every 100 --qoi-every 50 --stepper bonly"
    if [ "${1:-}" = smoke ]; then
        submit relax bonly_smoke "$wt/scripts/relax.py" "--geometry $GEOM_HI --ic clebsch --ns 8,16,16 --p 2 --floor-tol 1e-5 --steps 200 --qoi-every 20 --stepper bonly --out $PUB/bonly_smoke" 30
    else
        submit relax bonly_h16_p2_g1 "$wt/scripts/relax.py" "--geometry $GEOM_HI $common --steps 5000 --seconds 7200 --out $PUB/bonly_h16_p2_g1" 300
    fi
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
    deep) deep ;;
    hsweep_p2) hsweep_p2 ;;
    eta) SUB=li383_eta; PUB=$ROOT/outputs/$SUB; LEDGER=$PUB/jobs.tsv; mkdir -p "$PUB"; shift; eta "$@" ;;
    bonly) SUB=li383_bonly; PUB=$ROOT/outputs/$SUB; LEDGER=$PUB/jobs.tsv; mkdir -p "$PUB"; bonly "${2:-}" ;;
    sections) sections "$2" "${3:-30}" ;;
    movie) movie "$2" "$3" "$4" "${5:-120}" ;;
    *) echo "usage: $0 reader | seeded | deep | hsweep_p2 | eta | bonly [smoke] | sections NAME [TMIN] | movie NAME PLANES STEPSPEC [TMIN]" >&2; exit 2 ;;
esac
