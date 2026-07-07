#!/bin/bash
# ============================================================================
# Mass off-diagonal-coupling CEILING test + timings.
# Per geometry: solve M_k x = b (k=0..3, dbc+free) with the greville tensor
# preconditioner against BOTH the true mass (Mfull) and the block-diagonal
# operator (Mbd, off-diagonal metric blocks zeroed). Reports iters AND wall
# time (+ ms/iter) for each, plus the operator-level coupling ||offdiag||/||diag||.
# Mbd_it is the ceiling for any block-diagonal (block-SGS) strategy; the gap
# Mbd_it - k0_it is the residual diagonal-lump fidelity cost.
# One sbatch per geometry (debug-gpu, 4h cap). See
# scripts/debug/greville_mass_coupling_ceiling.py.
# ============================================================================
set -euo pipefail
cd /scratch/tblickhan/mrx

GEOMETRIES=${GEOMETRIES:-"cylinder toroid w7x"}
NS_LIST=${NS_LIST:-"12,24,12 16,32,16"}
P=${P:-3}
TOL=${TOL:-1e-10}
MAXITER=${MAXITER:-5000}

PARTITION=${PARTITION:-debug-gpu}
ACCOUNT=${ACCOUNT:-extremedata}
TIMEOUT_MIN=${TIMEOUT_MIN:-240}
MEM_GB=${MEM_GB:-128}
CPUS_PER_TASK=${CPUS_PER_TASK:-32}
THREADS=${THREADS:-8}

STAMP=$(date +%Y-%m-%d/%H-%M-%S)
OUTDIR="outputs/mass_coupling_ceiling/${STAMP}"
LOGDIR="${OUTDIR}/slurm_logs"
mkdir -p "${OUTDIR}" "${LOGDIR}"
echo "Mass coupling ceiling -> ${OUTDIR}"

for GEO in ${GEOMETRIES}; do
  case "${GEO}" in
    w7x) NFP=5 ;;
    *)   NFP=3 ;;
  esac
  CSV="${OUTDIR}/ceiling_${GEO}.csv"
  LOG="${LOGDIR}/${GEO}.log"
  CMD="set -euo pipefail; cd /scratch/tblickhan/mrx; source .venv/bin/activate; \
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false W7X_MAP_BATCH=128 \
OMP_NUM_THREADS=${THREADS} OPENBLAS_NUM_THREADS=${THREADS} MKL_NUM_THREADS=${THREADS} \
NUMEXPR_NUM_THREADS=${THREADS}; \
for NS in ${NS_LIST}; do \
  python -u scripts/debug/greville_mass_coupling_ceiling.py --geometry ${GEO} \
    --ns \${NS//,/ } --p ${P} --nfp ${NFP} --tol ${TOL} --maxiter ${MAXITER} \
    --csv ${CSV} || true; \
done"
  sbatch --partition="${PARTITION}" --account="${ACCOUNT}" --gpus-per-node=1 \
    --cpus-per-task="${CPUS_PER_TASK}" --time="${TIMEOUT_MIN}" --mem="${MEM_GB}G" \
    --job-name="ceil_${GEO}" --output="${LOG}" --wrap="${CMD}"
done
echo "Watch: tail -f ${LOGDIR}/*.log ; merge: awk 'FNR==1&&NR!=1{next}1' ${OUTDIR}/ceiling_*.csv > ${OUTDIR}/merged.csv"
