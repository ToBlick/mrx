#!/bin/bash
# Poincare sections of a relaxation state file, initial and final on one scale.
#
#   sbatch slurm/job_poincare_relaxed.sh [state.h5] [extra poincare_relaxed args]
#
# Defaults to the w7x_fmm002 100-iteration state in the MAIN checkout's data/,
# which is gitignored and so never existed in this worktree.
#
# PYTHONPATH is not optional. `python scripts/debug/x.py` puts the SCRIPT's
# directory on sys.path, never the working directory, so `import mrx` resolves
# to the venv's editable install -- which points at the main checkout. Without
# this the job silently runs the worktree's script against the main checkout's
# library and reports the result as a test of the branch.
set -euo pipefail
WT=/kfs3/scratch/tblickhan/mrx/.claude/worktrees/poincare-plotter
STATE=${1:-/kfs3/scratch/tblickhan/mrx/data/w7x_fmm002_relaxed_100.h5}
shift || true
ARGS=${*:---seeds 64 --periods 600 --n-planes 4}
STAMP=$(date +%Y-%m-%d/%H-%M-%S); OUT="outputs/poincare_relaxed/${STAMP}"
mkdir -p "${WT}/${OUT}"
sbatch --exclude=x3101c0s17b0n0 --partition=gpu-h100 --account=extremedata \
  --gpus-per-node=1 --cpus-per-task=32 --time=90 --mem=64G \
  --job-name=poincrelax --output="${WT}/${OUT}/poincare.log" \
  --wrap="set -euo pipefail; cd ${WT}; \
source /kfs3/scratch/tblickhan/mrx/.venv/bin/activate; \
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 PYTHONPATH=${WT}; \
python -u scripts/debug/poincare_relaxed.py ${STATE} --out ${OUT} ${ARGS}"
echo "Log: ${WT}/${OUT}/poincare.log"
