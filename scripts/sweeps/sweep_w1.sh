#!/bin/bash
# Sweep around the W1 case (w7x-fmm002, Clebsch IC, cg) -- the one that kept
# its nested surfaces.  Every arm renders Poincare, because "did the surfaces
# survive" is the question and no scalar invariant answers it (handoff s12).
#
# Budget: each job is one GPU.  The 8,16,8 baseline is ~1 GPU-h for 3000
# steps + tracing; the resolution arms are the expensive ones.  Total well
# under the 200 GPU-h ceiling -- estimated ~45.
set -u
# RUN FROM THE REPO ROOT: the paths below (slurm/job_relax_prelim.sh)
# are relative to it, not to this file's directory.
G="--geometry w7x-fmm002 --ic clebsch"
PC="--poincare --pc-seeds 40 --pc-periods 150"
O=/scratch/tblickhan/mrx/out/relax_prelim
S=slurm/job_relax_prelim.sh

sub () {  # sub <tag> <extra sbatch args> -- <script args>
  tag=$1; shift
  sbatch_args=""
  while [ "$1" != "--" ]; do sbatch_args="$sbatch_args $1"; shift; done
  shift
  mkdir -p "$O/$tag"
  # shellcheck disable=SC2086
  jid=$(sbatch $sbatch_args "$S" "$@" --save-b "$O/$tag/B.h5" \
        --out "$O/$tag/$tag.json" | awk '{print $4}')
  echo "$tag -> $jid"
  ln -sfn "/scratch/tblickhan/mrx/logs/relaxprelim_$jid.out" \
     "$O/logs/live_$tag.out"
}

# --- RESOLUTION: the axis the handoff flags as gap #1 --------------------
sub S01_res12  --time=6:00:00 -- $G --ns 12,24,12 --p 3 --steps 3000 \
    --helicity-every 250 --seconds-per-arm 12000 --arms cg $PC
sub S02_res16  --time=6:00:00 -- $G --ns 16,32,16 --p 3 --steps 2000 \
    --helicity-every 250 --seconds-per-arm 12000 --arms cg $PC
sub S03_p4     --time=6:00:00 -- $G --ns 8,16,8 --p 4 --steps 3000 \
    --helicity-every 250 --seconds-per-arm 12000 --arms cg $PC

# --- GAMMA / MU ----------------------------------------------------------
sub S04_g1mu3  --time=4:00:00 -- $G --ns 8,16,8 --p 3 --steps 3000 \
    --helicity-every 250 --gamma 1 --mu 1e-3 --seconds-per-arm 9000 --arms cg $PC
sub S05_g1mu2  --time=4:00:00 -- $G --ns 8,16,8 --p 3 --steps 3000 \
    --helicity-every 250 --gamma 1 --mu 1e-2 --seconds-per-arm 9000 --arms cg $PC
sub S06_g2mu3  --time=4:00:00 -- $G --ns 8,16,8 --p 3 --steps 2000 \
    --helicity-every 250 --gamma 2 --mu 1e-3 --seconds-per-arm 9000 --arms cg $PC

# --- LENGTH: does it converge, or keep grinding? --------------------------
sub S07_long   --time=4:00:00 -- $G --ns 8,16,8 --p 3 --steps 14000 \
    --helicity-every 500 --seconds-per-arm 11000 --arms cg $PC

# --- ETA: relax the topological constraint --------------------------------
sub S08_eta4   --time=4:00:00 -- $G --ns 8,16,8 --p 3 --steps 4000 \
    --helicity-every 250 --eta-max 1e-4 --seconds-per-arm 9000 --arms cg $PC
sub S09_eta3   --time=4:00:00 -- $G --ns 8,16,8 --p 3 --steps 4000 \
    --helicity-every 250 --eta-max 1e-3 --seconds-per-arm 9000 --arms cg $PC
sub S10_eta2   --time=4:00:00 -- $G --ns 8,16,8 --p 3 --steps 4000 \
    --helicity-every 250 --eta-max 1e-2 --seconds-per-arm 9000 --arms cg $PC

# --- OPTIMIZER, on a case known to behave --------------------------------
sub S11_opt    --time=4:00:00 -- $G --ns 8,16,8 --p 3 --steps 3000 \
    --helicity-every 250 --seconds-per-arm 4500 --arms gradient,lbfgs $PC

# --- STRUCTURE: lambda off.  Fluxes, iota and helicity must NOT move;
#     the force and the pressure must.  A gate the IC route asserts. --------
sub S12_nolam  --time=4:00:00 -- $G --ns 8,16,8 --p 3 --steps 3000 \
    --helicity-every 250 --no-lambda --seconds-per-arm 9000 --arms cg $PC

echo "submitted"
