#!/bin/bash
# Launch one MRX script as a single-GPU slurm job.
#
# Usage:
#   SCRIPT=scripts/tutorials/1_qa_geometry.py bash slurm/run.sh
#   SCRIPT="-m pytest -q test" JOB_NAME=tests TIMEOUT_MIN=120 bash slurm/run.sh
#
# Site settings never live in the repository. They are read from one file in the
# user's home, ~/.config/mrx/site.env, or from the file MRX_SITE_ENV names:
#
#   export SLURM_ACCOUNT=<account> SLURM_PARTITION=<gpu partition>
#   export SLURM_EXCLUDE=<comma-separated nodes to avoid>   # optional
#   export MRX_DATA=<data root outside the repository>      # optional
#
# Variables:
#   SCRIPT       (required) path relative to MRX_ROOT, or "-m module"
#   ARGS         arguments passed to the script
#   JOB_NAME     slurm job name and log file stem          (default: run)
#   OUTSUB       log directory under the runs root         (default: JOB_NAME)
#                The runs root is $MRX_DATA/runs/mrx if MRX_DATA is set, MRX_ROOT/outputs otherwise.
#   MRX_ROOT     checkout to run; defaults to the repo containing this file.
#                Also exported as PYTHONPATH so an editable install pinned to
#                another checkout does not shadow it.
#   TIMEOUT_MIN  wall time in minutes                     (default: 60)
#   MEM_GB       host memory                              (default: 64)
#   CPUS         cpus per task                            (default: 32)
#   EXTRA_ENV    space-separated VAR=VALUE pairs exported in the job
#   DEPENDENCY   an sbatch --dependency spec, e.g. afterany:123:124 (optional)
#
# The job prints `mrx.__file__` before running anything, so the log shows
# which checkout was tested.

set -euo pipefail

MRX_ROOT=${MRX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SITE_ENV=${MRX_SITE_ENV:-${HOME}/.config/mrx/site.env}
if [ ! -f "${SITE_ENV}" ]; then
  echo "run.sh: no site settings at ${SITE_ENV} (see docs/source/cluster.md, Site settings)" >&2
  exit 1
fi
source "${SITE_ENV}"

SCRIPT=${SCRIPT:?set SCRIPT}
ARGS=${ARGS:-}
JOB_NAME=${JOB_NAME:-run}
OUTSUB=${OUTSUB:-${JOB_NAME}}
TIMEOUT_MIN=${TIMEOUT_MIN:-60}
MEM_GB=${MEM_GB:-64}
CPUS=${CPUS:-32}
EXTRA_ENV=${EXTRA_ENV:-}
DEPENDENCY=${DEPENDENCY:-}
ACCOUNT=${SLURM_ACCOUNT:?set SLURM_ACCOUNT}
PARTITION=${SLURM_PARTITION:?set SLURM_PARTITION}
EXCLUDE=${SLURM_EXCLUDE:-}
VENV=${MRX_VENV:-${MRX_ROOT}/.venv}
[ -d "${VENV}" ] || VENV=$(cd "${MRX_ROOT}" && git rev-parse --path-format=absolute --git-common-dir)/../.venv

STAMP=$(date +%Y-%m-%d/%H-%M-%S)
RUNS=${MRX_DATA:+${MRX_DATA}/runs/mrx}
OUTDIR="${RUNS:-${MRX_ROOT}/outputs}/${OUTSUB}/${STAMP}"
mkdir -p "${OUTDIR}"
LOG="${OUTDIR}/${JOB_NAME}.log"

CMD="set -euo pipefail; cd ${MRX_ROOT}; source ${VENV}/bin/activate; \
export MRX_ROOT=${MRX_ROOT} PYTHONPATH=${MRX_ROOT} PYTHONUNBUFFERED=1; \
${MRX_DATA:+export MRX_DATA=${MRX_DATA}; } \
${EXTRA_ENV:+export ${EXTRA_ENV}; } \
python -c 'import mrx; print(\"mrx from:\", mrx.__file__)'; \
python -u ${SCRIPT} ${ARGS}"

sbatch \
  ${EXCLUDE:+--exclude="${EXCLUDE}"} \
  ${DEPENDENCY:+--dependency="${DEPENDENCY}"} \
  --partition="${PARTITION}" \
  --account="${ACCOUNT}" \
  --gpus-per-node=1 \
  --cpus-per-task="${CPUS}" \
  --time="${TIMEOUT_MIN}" \
  --mem="${MEM_GB}G" \
  --job-name="${JOB_NAME}" \
  --output="${LOG}" \
  --wrap="${CMD}"

echo "log: ${LOG}"
