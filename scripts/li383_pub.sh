#!/bin/bash
# li383 (NCSX, nfp=3) publication runs, 2026-09-02: the sweep arms behind the
# figures rerun on the current VMEC reader, plus the first seeded-island arms.
#
#   bash scripts/li383_pub.sh reader     # reread arms (about 9 GPU-h)
#   bash scripts/li383_pub.sh seeded     # seeded arms (about 9 GPU-h)
#   bash scripts/li383_pub.sh deep       # three arms past the 1e-3 floor
#   bash scripts/li383_pub.sh anchor     # m = 0, 3, 5 and gamma = 0 departures from h16_p2_g1
#   bash scripts/li383_pub.sh hsweep_p2  # gamma = 1 h-sweep at p = 2 (about 16 GPU-h)
#   bash scripts/li383_pub.sh psweep_p16 # gamma = 1 p-sweep at (16,32,32) (about 7 GPU-h)
#   bash scripts/li383_pub.sh aux [smoke|pairs]  # the auxiliary-H step next to the B step, outputs/li383_bonly
#   bash scripts/li383_pub.sh reconnect [smoke|ladder|ladder5k|refine_smoke]  # reconnection series (--reconnect-every), outputs/li383_pulse; ladder = nine equilibria at 1.25% of H each
#   bash scripts/li383_pub.sh sections NAME [TIMEOUT_MIN]
#   bash scripts/li383_pub.sh movie NAME PLANES STEPSPEC [TIMEOUT_MIN]
#
# Every arm before 2026-09-04 ran the H step with the natural H (no wall
# condition) and the in-loop resistivity schedule; both are gone (the
# auxiliary H is Dirichlet now, the default step reads B itself, reconnection
# is one solve between chunks). The tanh resistivity sweep (eta) and the
# scheduled pulses (pulse) have no successor and were removed with them; their
# results stay in docs/research/li383_sweep_results_2026-09-02.md.
#
# MRX_ROOT selects the checkout the jobs run (default: this one). Outputs go
# to $MRX_ROOT/outputs/$SUB/<arm> (SUB defaults to li383_pub; `SUB=li383_pulse
# ... sections NAME` addresses another); job ids to $MRX_ROOT/outputs/$SUB/jobs.tsv.
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
COMMON="--floor-tol 1e-4 --steps 6000"
G1="--velocity-smoothing-order 1"   # mu = 0.02 / n_r^2, the driver's default (SMOOTHING_C)
G1_12=$G1
G1_16=$G1
G1_32=$G1
AUX="--auxiliary-B-field true"

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

anchor_sweeps() {
    # 2026-09-04: one-flag departures from the anchor h16_p2_g1 (the (16,32,32)
    # p = 2 gamma = 1 rung of hsweep_p2): the L-BFGS history m = 0 (steepest
    # descent), 3, 5 (m = 1 is the anchor), and gamma = 0. 5000 steps, no
    # floor, so every run ends on the step cap; the anchor's --seconds cap is
    # dropped for the same reason.
    local common="--ns 16,32,32 --p 2 --floor-tol 0 --steps 5000"
    arm h16_p2_g1_m0 "$GEOM_HI" "$common $G1_16 --history 0"        300
    arm h16_p2_g1_m3 "$GEOM_HI" "$common $G1_16 --history 3"        300
    arm h16_p2_g1_m5 "$GEOM_HI" "$common $G1_16 --history 5"        300
    arm h16_p2_g0    "$GEOM_HI" "$common --velocity-smoothing-order 0" 300
}

hsweep_p2() {
    # 2026-09-02: gamma = 1 under h-refinement at fixed p = 2 on the ns = 49
    # reference, mesh (n, 2n, 2n), mu = 0.02 / n^2 (the default), floor 1e-5 so the 5000-step
    # cap (or the wall cap) ends the run and the floor versus h is what is measured.
    # Wall caps sum to 18.5 GPU-h; the n = 32 rung takes about 10 h of it.
    # These follow COMMON on the command line, so they override its floor and step cap.
    local common="--p 2 --floor-tol 1e-5 --steps 5000 --velocity-smoothing-order 1"
    arm h8_p2_g1  "$GEOM_HI" "--ns 8,16,16  $common --seconds 1800"    90
    arm h12_p2_g1 "$GEOM_HI" "--ns 12,24,24 $common --seconds 3600"   150
    arm h16_p2_g1 "$GEOM_HI" "--ns 16,32,32 $common --seconds 7200"    300
    arm h24_p2_g1 "$GEOM_HI" "--ns 24,48,48 $common --seconds 18000"  660
    arm h32_p2_g1 "$GEOM_HI" "--ns 32,64,64 $common --seconds 36000" 1260
}

psweep_p16() {
    # 2026-09-03: the p-sweep complementary to hsweep_p2, at its (16,32,32) rung
    # (also the resistivity sweep's mesh): p = 1, 3, 4 with the same recipe;
    # p = 2 is h16_p2_g1.
    local common="--ns 16,32,32 --floor-tol 1e-5 --steps 5000 $G1_16"
    arm h16_p1_g1 "$GEOM_HI" "$common --p 1 --seconds 1800"   60
    arm h16_p3_g1 "$GEOM_HI" "$common --p 3 --seconds 10800" 360
    arm h16_p4_g1 "$GEOM_HI" "$common --p 4 --seconds 18000" 600
}

aux() {  # aux [smoke|pairs]
    # 2026-09-03/04: the auxiliary-H step (J x H, u x H with the Dirichlet
    # 1-form H = M_1^-1 P B) next to the default B step, twin of h16_p2_g1,
    # helicity sampled every chunk. The code comes from THIS checkout
    # (worktree): SCRIPT is absolute and PYTHONPATH is overridden; data and
    # outputs stay under $ROOT.
    local wt=$WT
    export EXTRA_ENV="PYTHONPATH=$wt"
    local common="--ns 16,32,32 --p 2 $G1_16 --floor-tol 1e-5 $AUX"
    if [ "${1:-}" = smoke ]; then
        submit relax aux_smoke "$wt/scripts/relax.py" "--geometry $GEOM_HI --ns 8,16,16 --p 2 --floor-tol 1e-5 --steps 200 --chunk 100 $AUX --out $PUB/aux_smoke" 30
    elif [ "${1:-}" = pairs ]; then
        # Two pairs, each with the B step and the auxiliary-H step: float64 at
        # (12,24,24) p = 2 and float32 at (8,16,16) p = 1.
        local base="--floor-tol 1e-5 --steps 5000 --seconds 7200"
        local st flag
        for st in B H; do
            flag=""; [ $st = H ] && flag=$AUX
            submit relax h12_p2_f64_$st "$wt/scripts/relax.py" "--geometry $GEOM_HI $base --ns 12,24,24 --p 2 $G1_12 --precision float64 $flag --out $PUB/h12_p2_f64_$st" 300
            submit relax h8_p1_$st      "$wt/scripts/relax.py" "--geometry $GEOM_HI $base --ns 8,16,16  --p 1 --velocity-smoothing-order 1 $flag --out $PUB/h8_p1_$st" 120
        done
    else
        submit relax aux_h16_p2_g1 "$wt/scripts/relax.py" "--geometry $GEOM_HI $common --steps 5000 --seconds 7200 --out $PUB/aux_h16_p2_g1" 300
    fi
}

reconnect() {  # reconnect [smoke|ladder|ladder5k|refine_smoke]
    # 2026-09-03: the ideal descent, checkpointed and reconnected with one
    # resistive solve every K steps (--reconnect-every; the descent is a
    # power law, there is no stall to wait for); the outcome is the series
    # under <arm>/reconnect/<k>/ plus the final field. (16,32,32) p = 2
    # gamma = 1, 8000 steps = 4000 ideal + three intervals of 2000.
    # 2026-09-04 ladder: the same rung, every 2000 steps from step 0, 1.25% of
    # H per solve (the dose c = 0.02 of the original run), eight solves + the
    # final field = nine ideal equilibria spanning ideal to fully reconnected.
    # 18000 steps, about 3 h at 0.57 s/step.
    # 2026-09-04 ladder5k: three meshes, 10000 steps with ONE solve at 5000
    # spending 1% of H (the dose is set from the helicity since the prune;
    # the original runs used eps = 6.25e-5, calibrated to the same 1%).
    # Meshes: (16,32,32); (32,32,32) uniform; (32,32,32) with the radial
    # cells of the n_r = 16 grid outside two windows and 6 cells of 0.025 +
    # 15 cells of 0.017 inside [0.47, 0.62] (iota = 1/2) and [0.68, 0.94]
    # (iota = 3/5), the outer chain finer.
    export EXTRA_ENV="PYTHONPATH=$WT"
    L5="--floor-tol 0 --steps 10000 --reconnect-every 5000 --reconnect-helicity 0.01"
    R32="--r-refine 0.47:0.62:6,0.68:0.94:15"
    if [ "${1:-}" = ladder5k ]; then
        submit relax reconnect_l5_h16_p2_g1 "$WT/scripts/relax.py" "--geometry $GEOM_HI --ns 16,32,32 --p 2 $G1_16 $L5 --out $PUB/reconnect_l5_h16_p2_g1" 240
        submit relax reconnect_l5_h32u_p2_g1 "$WT/scripts/relax.py" "--geometry $GEOM_HI --ns 32,32,32 --p 2 $G1_32 $L5 --out $PUB/reconnect_l5_h32u_p2_g1" 480
        submit relax reconnect_l5_h32r_p2_g1 "$WT/scripts/relax.py" "--geometry $GEOM_HI --ns 32,32,32 --p 2 $G1_32 $R32 $L5 --out $PUB/reconnect_l5_h32r_p2_g1" 540
    elif [ "${1:-}" = refine_smoke ]; then
        # The refined radial grid end to end at n_r = 16: 5 + 5 window cells, 300 steps, one solve.
        submit relax refine_smoke "$WT/scripts/relax.py" "--geometry $GEOM_HI --ns 16,16,16 --p 2 $G1_16 --r-refine 0.45:0.65:5,0.68:0.92:5 --floor-tol 0 --steps 300 --chunk 100 --reconnect-every 200 --reconnect-helicity 0.02 --out $PUB/refine_smoke" 40
    elif [ "${1:-}" = ladder ]; then
        submit relax reconnect_ladder_h16_p2_g1 "$WT/scripts/relax.py" "--geometry $GEOM_HI --ns 16,32,32 --p 2 $G1_16 --floor-tol 0 --steps 18000 --reconnect-every 2000 --reconnect-helicity 0.0125 --out $PUB/reconnect_ladder_h16_p2_g1" 420
    elif [ "${1:-}" = smoke ]; then
        submit relax reconnect_smoke "$WT/scripts/relax.py" "--geometry $GEOM_HI --ns 8,16,16 --p 2 --floor-tol 0 --steps 3000 --chunk 100 --reconnect-every 600 --out $PUB/reconnect_smoke" 40
    else
        submit relax reconnect_h16_p2_g1 "$WT/scripts/relax.py" "--geometry $GEOM_HI --ns 16,32,32 --p 2 $G1_16 --floor-tol 0 --steps 8000 --seconds 10800 --reconnect-every 2000 --out $PUB/reconnect_h16_p2_g1" 400
    fi
}

# Sections and movies run THIS checkout's plotter (branch poincare-plotter merged
# 2026-09-03: .pgf output, logical-r profile) in float32, as slurm/regen_poincare.sh
# does; TeX for the .pgf comes from the same place as there.
PLOTTER_ENV="PYTHONPATH=$WT PATH=$HOME/texlive/2026/bin/x86_64-linux:$PATH"

sections() {  # sections NAME [TIMEOUT_MIN]: ic, final and the reconnection series of an arm, one call, one colour scale
    export EXTRA_ENV="$PLOTTER_ENV"; submit sec "$1" "$WT/scripts/poincare_relax.py" \
        "$PUB/$1 --fields ic,final,reconnect --planes 0,0.125,0.25,0.375,0.5 --precision float32 --out $PUB/$1/poincare" "${2:-30}"
}

movie() {  # movie NAME PLANES STEPSPEC [TIMEOUT_MIN]
    export EXTRA_ENV="$PLOTTER_ENV"; submit mov "$1" "$WT/scripts/poincare_relax.py" \
        "$PUB/$1 --fields snapshots --snapshot-steps $3 --planes $2 --precision float32 --out $PUB/$1/movie" "${4:-120}"
}

case ${1:-} in
    reader) reader ;;
    seeded) seeded ;;
    deep) deep ;;
    hsweep_p2) hsweep_p2 ;;
    anchor) anchor_sweeps ;;
    psweep_p16) psweep_p16 ;;
    reconnect) SUB=li383_pulse; PUB=$ROOT/outputs/$SUB; LEDGER=$PUB/jobs.tsv; mkdir -p "$PUB"; reconnect "${2:-}" ;;
    aux) SUB=li383_bonly; PUB=$ROOT/outputs/$SUB; LEDGER=$PUB/jobs.tsv; mkdir -p "$PUB"; aux "${2:-}" ;;
    sections) sections "$2" "${3:-30}" ;;
    movie) movie "$2" "$3" "$4" "${5:-120}" ;;
    *) echo "usage: $0 reader | seeded | deep | hsweep_p2 | anchor | psweep_p16 | reconnect [smoke|ladder|ladder5k|refine_smoke] | aux [smoke|pairs] | sections NAME [TMIN] | movie NAME PLANES STEPSPEC [TMIN]" >&2; exit 2 ;;
esac
