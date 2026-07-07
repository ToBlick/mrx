#!/bin/bash
# ============================================================================
# k=0 Laplacian geometric-multigrid prototype sweep.
# Per geometry: h-sweep of MG-PCG (smoothers jacobi/fd/fdax, m in {1,2})
# vs the single-level FD tensor-hodge baseline, both BCs.
# See scripts/debug/laplacian_mg_k0.py and the plan in
# docs/laplacian_mg_k0_plan.md.
# One sbatch per geometry on gpu-h100s. CSVs -> outputs/laplacian_mg_k0/<stamp>/.
#   GEOMETRIES="w7x" bash slurm/job_laplacian_mg_k0.sh
# ============================================================================
set -euo pipefail
cd /scratch/tblickhan/mrx

GEOMETRIES=${GEOMETRIES:-"cylinder toroid w7x"}
NS_LIST=${NS_LIST:-"12,24,24 16,32,32"}
P=${P:-3}
LEVELS=${LEVELS:-2}
COARSEN=${COARSEN:-"2,2,2"}
SMOOTHERS=${SMOOTHERS:-"jacobi,fd,fdax"}
M_LIST=${M_LIST:-"1 2"}
SCHUR=${SCHUR:-rebuild}
TOL=${TOL:-1e-10}
MAXITER=${MAXITER:-3000}

PARTITION=${PARTITION:-gpu-h100s}
ACCOUNT=${ACCOUNT:-extremedata}
TIMEOUT_MIN=${TIMEOUT_MIN:-240}
MEM_GB=${MEM_GB:-128}
CPUS_PER_TASK=${CPUS_PER_TASK:-32}
THREADS=${THREADS:-8}

STAMP=$(date +%Y-%m-%d/%H-%M-%S)
OUTDIR="outputs/laplacian_mg_k0/${STAMP}"
LOGDIR="${OUTDIR}/slurm_logs"
mkdir -p "${OUTDIR}" "${LOGDIR}"
echo "MG k=0 Laplacian sweep -> ${OUTDIR}"

for GEO in ${GEOMETRIES}; do
  case "${GEO}" in
    w7x) NFP=5; ZDIAG="--zeta-diag" ;;
    *)   NFP=3; ZDIAG="" ;;
  esac
  CSV="${OUTDIR}/mg_${GEO}.csv"
  LOG="${LOGDIR}/${GEO}.log"
  CMD="set -euo pipefail; cd /scratch/tblickhan/mrx; source .venv/bin/activate; \
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false W7X_MAP_BATCH=128 \
OMP_NUM_THREADS=${THREADS} OPENBLAS_NUM_THREADS=${THREADS} MKL_NUM_THREADS=${THREADS} \
NUMEXPR_NUM_THREADS=${THREADS}; \
for NS in ${NS_LIST}; do \
  for M in ${M_LIST}; do \
    python -u scripts/debug/laplacian_mg_k0.py --geometry ${GEO} --ns \${NS//,/ } \
      --p ${P} --nfp ${NFP} --levels ${LEVELS} --coarsen ${COARSEN//,/ } \
      --smoothers ${SMOOTHERS} --smooth-steps \${M} --schur ${SCHUR} \
      --tol ${TOL} --maxiter ${MAXITER} ${ZDIAG} --csv ${CSV} || true; \
  done; \
done"
  sbatch --partition="${PARTITION}" --account="${ACCOUNT}" --gpus-per-node=1 \
    --cpus-per-task="${CPUS_PER_TASK}" --time="${TIMEOUT_MIN}" --mem="${MEM_GB}G" \
    --job-name="mg_${GEO}" --output="${LOG}" --wrap="${CMD}"
done
echo "Watch: tail -f ${LOGDIR}/*.log ; merge: awk 'FNR==1&&NR!=1{next}1' ${OUTDIR}/mg_*.csv > ${OUTDIR}/merged.csv"
