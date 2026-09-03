#!/bin/bash
# The GPU column of the factorization A/B.
#
# The v5e achieves 41.6 GFLOP/s on mass_core_apply against 349 GFLOP/s for a
# plain matmul of the same contraction width and batch on the same chip, so
# the element transform is losing 8.4x to something that is not arithmetic.
# This runs the same five formulations on one GPU, because a formulation that
# wins on a TPU and loses badly on a GPU cannot go into the shared kernel.
#
#   sbatch tpu/gpu_factorization.slurm
#
#SBATCH --job-name=mrx-gpu-factor
#SBATCH --partition=h200_courant
#SBATCH --account=torch_pr_292_courant
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/aak572/mrx_bench/%x-%j.out
#SBATCH --error=/scratch/aak572/mrx_bench/%x-%j.err
set -uo pipefail

OVERLAY=${OVERLAY:-/scratch/aak572/mrx.ext3}
IMAGE=${IMAGE:-/share/apps/images/ubuntu-24.04.3.sif}
REPO=${REPO:-/scratch/aak572/mrx}
OUT=${OUT:-/scratch/aak572/mrx_bench}

mkdir -p "${OUT}"
module purge && module load singularity-ce/4.3.3

nvidia-smi --query-gpu=name,memory.total --format=csv || true

singularity exec --nv --overlay "${OVERLAY}:ro" "${IMAGE}" /bin/bash -c "
source /ext3/env.sh
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
cd ${REPO} || exit 1

for c in 0 1; do
  echo \"============= factorization A/B, k=2 component \${c} =============\"
  MRX_DTYPE=float32 python -u tpu/factorization_ab.py \
      --ns 12,24,12 --p 3 --k 2 --component \${c} \
      --scan-length 50 --map-batch-size 4096 \
      --out ${OUT}/factor_gpu_c\${c}.json
done
"
