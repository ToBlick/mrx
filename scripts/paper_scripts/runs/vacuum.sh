#!/bin/bash
# Tab. 2 and Figs. 3-6: vacuum fields in the Landreman-Paul QA device, float64. The high-resolution wout (201
# surfaces, 16/12 modes) is not in the repository: data/wout_LandremanPaul2021_QA_highres.nc.
#   bash scripts/paper_scripts/runs/vacuum.sh
. "$(dirname "$0")/common.sh"
LOWRES=data/wout_LandremanPaul2021_QA_lowres.nc
HIGHRES=data/wout_LandremanPaul2021_QA_highres.nc

# Tab. 2: the stellarator-symmetric manufactured field in the three symmetry models, 24 and 32 cells, p = 2
O=$RECORDS/vacuum_symmetry
S="--geometry $LOWRES --field coil-ss --p 2 --routes A,C"
for m in 26 34; do
  sub vs_torus$m 90 "" scripts/paper_scripts/analytic_vacuum.py $S --ns $m,$((2 * m)),$((2 * m)) --symmetry none --nfp 1 \
    --out $O/torus_$m
  sub vs_period$m 60 "" scripts/paper_scripts/analytic_vacuum.py $S --ns $m,$((2 * m)),$m --symmetry field-period --out $O/period_$m
  sub vs_half$m 60 "" scripts/paper_scripts/analytic_vacuum.py $S --ns $m,$((2 * m)),$m --symmetry stellarator --out $O/half_$m
done

# Fig. 4: the manufactured field against the analytic one on one field period, (m, 2m, m) splines, m = n + p:
# the coarse meshes as one ladder per degree, every finer one its own job (no gap diagnostics at n >= 48)
O=$RECORDS/vacuum_analytic
A="--geometry $LOWRES --field coil --lam 1.0 --routes A,C"
for p in 1 2 3 4; do
  ladder=$(for n in 4 6 8 10 12 14 16; do m=$((n + p)); printf "%d,%d,%d:" $m $((2 * m)) $m; done)
  sub va_p$p 240 "" scripts/paper_scripts/analytic_vacuum.py $A --p $p --ns ${ladder%:} --out $O/p$p
  for n in 20 24 28 32 48 64; do
    m=$((n + p))
    if [ $n -lt 48 ]; then minutes=240 gaps=""; else minutes=600 gaps="--gap-sweeps 0"; fi
    sub va_p${p}_n$n $minutes "" scripts/paper_scripts/analytic_vacuum.py $A --p $p --ns $m,$((2 * m)),$m $gaps --out $O/p${p}_n$n
  done
done

# Fig. 5: the discrete harmonic 2-form against the VMEC vacuum field, (n + p, 2 (n + 3), n + 3) splines on the
# low-resolution wout, (m, 2m, m) with m = n + p on the high-resolution one
O=$RECORDS/vacuum_vmec
for p in 2 3 4; do
  ladder=$(for n in 5 7 9 11 13 17 21; do printf "%d,%d,%d:" $((n + p)) $((2 * n + 6)) $((n + 3)); done)
  sub vl_p$p 360 "" scripts/paper_scripts/qa_vacuum_sweep.py --geometry $LOWRES --p $p --ns ${ladder%:} --out $O/lowres
  for n in 25 29 37 45; do
    sub vl_p${p}_n$n 300 "" scripts/paper_scripts/qa_vacuum_sweep.py --geometry $LOWRES --p $p \
      --ns $((n + p)),$((2 * n + 6)),$((n + 3)) --out $O/lowres
  done
  ladder=$(for n in 5 9 13 17 21 29; do m=$((n + p)); printf "%d,%d,%d:" $m $((2 * m)) $m; done)
  VH=$(sub vh_p$p 600 "" scripts/paper_scripts/qa_vacuum_sweep.py --geometry $HIGHRES --p $p --ns ${ladder%:} --out $O/highres)
  if [ $p = 3 ]; then VH3=$VH; fi
  for n in 37 45; do
    m=$((n + p))
    sub vh_p${p}_n$n 420 "" scripts/paper_scripts/qa_vacuum_sweep.py --geometry $HIGHRES --p $p --ns $m,$((2 * m)),$m \
      --out $O/highres
  done
done

# Figs. 3 and 6: the (32, 64, 32) p = 3 rung of the high-resolution reference, |B| on the boundary and its sections
R=$O/highres/rung_32x64x32_p3
RUN=$(sub vq_run 10 afterok:$VH3 scripts/paper_scripts/vacuum_run.py $R)
sub vq_trace 60 afterok:$RUN scripts/poincare_trace.py --geometry $HIGHRES $R/run/checkpoints/state_000000.h5 --lines 160 --periods 400
sub vq_bmag 40 afterok:$VH3 scripts/paper_scripts/plot_vacuum_qa.py --geometry $HIGHRES --ns 32,64,32 --p 3 \
  --field-npz $R/fields.npz --out $O/bmag
