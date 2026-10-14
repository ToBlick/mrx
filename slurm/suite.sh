#!/usr/bin/env bash
# The test suite as one GPU job. Log under outputs/suite/<stamp>/suite.log.
#
#   bash slurm/suite.sh
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCRIPT="-m pytest -q -rxX test" JOB_NAME=suite OUTSUB=suite TIMEOUT_MIN=${TIMEOUT_MIN:-40} \
    MRX_ROOT=$ROOT bash "$ROOT/slurm/run.sh"
