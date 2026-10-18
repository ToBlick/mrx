#!/usr/bin/env bash
# Runs a list of relax.py runs on GKE TPU nodes, four to a node (one per chip), on one set of clusters that is
# created once and deleted at the end (gke/ephemeral.sh), so a scan pays the ~10-15 min cluster start once.
#   bash gke/scan.sh runs.txt [run|deploy]
# runs.txt holds one run per line: the relax.py arguments that differ from ARGS (as for RUNS, gke/README.md); blank
# lines and lines starting with # are skipped. Lines 1-4 go to the first job, 5-8 to the second, and so on: group runs
# of similar size, since a node is billed until its slowest run ends. The first job runs make <target> (default
# deploy: build this checkout first), the others make run. Settings as for make (ARGS, TPU_GEN, TIMEOUT_MIN, ...).
# SCAN: the prefix of the job names and output directories (default scan-<time>): outputs/<SCAN>-<n>/run-<k>/.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIST="${1:?usage: bash gke/scan.sh runs.txt [run|deploy]}"
TARGET="${2:-deploy}"
SCAN="${SCAN:-scan-$(date +%Y%m%d-%H%M%S)}"

LINES=()
while IFS= read -r line; do LINES+=("${line}"); done < <(grep -vE '^[[:space:]]*(#|$)' "${LIST}")
(( ${#LINES[@]} > 0 )) || { echo "[scan] ERROR: no runs in ${LIST}" >&2; exit 1; }
for line in "${LINES[@]}"; do
    [[ "${line}" != *";"* ]] || { echo "[scan] ERROR: one run per line, without ';': ${line}" >&2; exit 1; }
done
N_JOBS=$(( (${#LINES[@]} + 3) / 4 ))
echo "[scan] ${SCAN}: ${#LINES[@]} runs in ${N_JOBS} job(s) of up to 4"

# Whatever happens, the clusters are deleted at the end (once no MRX job of anyone is active on them)
trap 'bash "${HERE}/ephemeral.sh" down' EXIT

FAILED=0
for (( j = 0; j < N_JOBS; j++ )); do
    group=("${LINES[@]:j*4:4}")
    runs="$(printf '%s; ' "${group[@]}")"
    target="run"
    (( j == 0 )) && target="${TARGET}"
    echo "[scan] job $((j + 1))/${N_JOBS}: ${#group[@]} run(s): ${runs%; }"
    if ! KEEP_CLUSTERS=1 JOB_NAME="${SCAN}-$((j + 1))" OUTSUB="${SCAN}-$((j + 1))" RUNS="${runs%; }" \
        bash "${HERE}/ephemeral.sh" "${target}"; then
        echo "[scan] job $((j + 1)) failed"
        FAILED=$((FAILED + 1))
        # a failed build or a missing TPU node fails every later job too
        (( j == 0 )) && break
    fi
done
echo "[scan] ${SCAN}: $((N_JOBS - FAILED)) of ${N_JOBS} job(s) succeeded; outputs in outputs/${SCAN}-*/"
(( FAILED == 0 ))
