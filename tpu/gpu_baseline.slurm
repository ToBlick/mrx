#!/bin/bash
# The GPU column of the TPU comparison, measured rather than quoted.
#
# Every GPU number the TPU work has been compared against came from a
# different measurement: 0.41 s/step is a warm-start A/B on W7-X, and the
# 0.7-0.9 s/step in the concepts docs is a smaller resolution reported as
# slower. Neither is like-for-like, so this runs the same matvec_bench.py and
# tpu_bench_mrx.py that produced the v5e numbers, at the same li383
# (12,24,12) p=3, on one GPU.
#
# The GPU is an H200, not an H100: NYU Torch has h100 partitions but they are
# not open to this account, and h200_courant is. For this comparison that is
# a conservative substitution -- an H200 is the same GH100 compute (identical
# FP64 and BF16 throughput) with 4.8 TB/s of HBM3e against the H100 SXM's
# 3.35 TB/s. Sum-factorised matvecs are bandwidth-bound, so the H200 is the
# faster of the two here and any TPU-versus-GPU gap measured against it is an
# upper bound on the gap against an H100.
#
# Both dtypes, because the existing GPU figures are float64 and the TPU
# figures are float32, and that difference is not incidental: mrx pins
# jax_default_matmul_precision='highest', which is free in float64 and costs
# six bf16 passes per float32 matmul on a TPU MXU.
#
#   sbatch tpu/gpu_baseline.slurm
#
#SBATCH --job-name=mrx-gpu-baseline
#SBATCH --partition=h200_courant
#SBATCH --account=torch_pr_292_courant
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/aak572/mrx_bench/%x-%j.out
#SBATCH --error=/scratch/aak572/mrx_bench/%x-%j.err
set -uo pipefail

OVERLAY=${OVERLAY:-/scratch/aak572/mrx.ext3}
IMAGE=${IMAGE:-/share/apps/images/ubuntu-24.04.3.sif}
REPO=${REPO:-/scratch/aak572/mrx}
OUT=${OUT:-/scratch/aak572/mrx_bench}

mkdir -p "${OUT}"
module purge && module load singularity-ce/4.3.3

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

singularity exec --nv --overlay "${OVERLAY}:ro" "${IMAGE}" /bin/bash -c "
source /ext3/env.sh
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd ${REPO} || exit 1
python -c 'import jax; print(\"jax\", jax.__version__, jax.devices())'

for dt in float32 float64; do
  echo \"===================== matvec baseline, \${dt} =====================\"
  MRX_DTYPE=\${dt} python -u tpu/matvec_bench.py \
      --ns 12,24,12 --p 3 --scan-length 50 --map-batch-size 4096 \
      --out ${OUT}/matvec_gpu_\${dt}.json
done

for dt in float32 float64; do
  echo \"==================== relaxation step, \${dt} =====================\"
  python -u tpu/tpu_bench_mrx.py \
      --ns 12,24,12 --p 3 --precision \${dt} --map-batch-size 4096 \
      --out ${OUT}/relax_gpu_\${dt}.json
done
"
