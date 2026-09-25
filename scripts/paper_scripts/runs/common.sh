# Read by every launcher (bash scripts/paper_scripts/runs/<experiment>.sh, from the repository root): the records root, the li383
# case, and sub, one GPU job through slurm/run.sh (site settings as there).
#   RECORDS   the records root, <RECORDS>/<experiment>/<arm>, relative to the repository root [outputs]
set -euo pipefail
RECORDS=${RECORDS:-outputs}
# NCSX li383, p = 2, half a period, mixed precision, no floor stop: what every relaxation run of the paper shares
LI383="--geometry data/wout_li383_1.4m.nc --spline-degree 2 --precision mixed --symmetry stellarator --floor-tol 0"
#: the geometry file again, for the tracer (it rebuilds a checkpoint's sequence over it)
GEOMETRY=data/wout_li383_1.4m.nc

sub() {  # name minutes dependency script args... -> prints the job id
  local name=$1 minutes=$2 dependency=$3 script=$4
  shift 4
  DEPENDENCY=$dependency SCRIPT=$script JOB_NAME=$name OUTSUB=paper_logs TIMEOUT_MIN=$minutes ARGS="$*" \
    bash slurm/run.sh | grep -o 'Submitted batch job [0-9]*' | grep -o '[0-9]*$'
}
