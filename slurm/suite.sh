#!/usr/bin/env bash
# The test suite in the two precisions, two GPU jobs. Logs under outputs/suite/<stamp>/suite_<dtype>.log.
#
#   bash slurm/suite.sh            # both
#   bash slurm/suite.sh float32    # one of float64 | float32
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
for dtype in ${@:-float64 float32}; do
    SCRIPT="-m pytest -q -rxX test" JOB_NAME=suite_$dtype OUTSUB=suite TIMEOUT_MIN=${TIMEOUT_MIN:-60} \
        EXTRA_ENV="MRX_DTYPE=$dtype" MRX_ROOT=$ROOT bash "$ROOT/slurm/run.sh"
done
