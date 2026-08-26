#!/bin/bash
# The raw_kron vs metric-lumped-atom Schur probe A/B, on a GPU node.
#
# This is a ONE-SHOT measurement: raw_kron is deleted immediately after, so the
# comparison becomes unreproducible and the log IS the permanent record. Hence
# toroid AND w7x rather than a single geometry, and hence the log is committed
# to docs/research/ rather than left in outputs/ where a scratch purge takes it.
#
#   STAMP=ab bash slurm/job_schur_probe_ab.sh
set -euo pipefail
module load python/3.11.4
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
OUTDIR="outputs/schur_probe_ab/${STAMP:-run}"
mkdir -p "${OUTDIR}"

CMD="set -euo pipefail; cd ${ROOT}; source .venv/bin/activate; \
export PYTHONUNBUFFERED=1 PYTHONPATH=${ROOT} OMP_NUM_THREADS=8; \
python -u scripts/debug/schur_probe_ab.py \
--geometries ${GEOMETRIES:-toroid,w7x} --ns ${NS:-12,24,12} --p ${P:-3} \
--ks ${KS:-1,2,3}"

sbatch --exclude=x3101c0s17b0n0 --partition=gpu-h100 --account=extremedata \
  --gpus-per-node=1 --cpus-per-task=32 --time=180 --mem=128G \
  --job-name=schur_ab --output="${OUTDIR}/ab.log" --wrap="${CMD}"
echo "Submitted. Log: ${OUTDIR}/ab.log"
