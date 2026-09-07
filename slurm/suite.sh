#!/usr/bin/env bash
# The test suite in the three precision configurations, three GPU jobs:
#
#   refined float32   MRX_DTYPE=float32                              (the production default, tol 1e-8)
#   plain float64     MRX_DTYPE=float64                              (tol 1e-10)
#   plain float32     MRX_DTYPE=float32 MRX_RESIDUAL_DTYPE=float32   (the TPU configuration, tol sqrt(eps))
#
# A change to the solvers, the precision module or the atoms is verified by
# all three; report the count per configuration. Logs under
# outputs/suite/<stamp>/suite_<config>.log.
#
#   bash slurm/suite.sh            # all three
#   bash slurm/suite.sh mixed      # one of mixed | float64 | plain32
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
declare -A ENVS=([mixed]="MRX_DTYPE=float32" [float64]="MRX_DTYPE=float64"
                 [plain32]="MRX_DTYPE=float32 MRX_RESIDUAL_DTYPE=float32")
for cfg in ${@:-mixed float64 plain32}; do
    SCRIPT="-m pytest -q -rxX test" JOB_NAME=suite_$cfg OUTSUB=suite TIMEOUT_MIN=${TIMEOUT_MIN:-40} \
        EXTRA_ENV="${ENVS[$cfg]}" MRX_ROOT=$ROOT bash "$ROOT/slurm/run.sh"
done
