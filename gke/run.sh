#!/usr/bin/env bash
# Submits one MRX job to the GKE cluster(s) set up from gcp.env. Run by make run, make deploy and make sharded, which
# pass these settings through (each wins over gcp.env):
#   SCRIPT, ARGS           the script and its arguments (default: scripts/relax.py with the ARGS of gcp.env)
#   RUNS="<args>; <args>"  up to one relax.py run per chip, each with ARGS plus its own arguments (gke/README.md)
#   TPU_GEN=v5p|v6e        the TPU family (default: either). The job waits in every TPU pool of the family on the
#                          clusters of gcp.env and runs in the first to get a node; the others are deleted
#   COMPUTE_CLASS          a ComputeClass (gke/compute-classes.yaml) instead of TPU_GEN
#   FOLLOW=1               stream the logs, wait for the end, copy the outputs to ./outputs/<job>/, exit 0 on success
#   TIMEOUT_MIN            deadline of the job in minutes (default 480)
#   EXTRA_ENV="A=1 B=2"    environment variables exported in the container
#   JOB_NAME, OUTSUB       job name (default mrx-<MRX_USER>-<timestamp>) and output directory (default: the job name);
#                          reusing an output directory resumes the runs in it and skips those that finished
#   IMAGE_NAME             the image (default: mrx-tpu:<MRX_USER>, which make build pushes)

set -euo pipefail

MRX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Values given on the command line win over gcp.env, which is sourced next
_CLI_PROJECT_ID="${PROJECT_ID:-}"
_CLI_REGION="${REGION:-}"
_CLI_COMPUTE_CLASS="${COMPUTE_CLASS:-}"
_CLI_TPU_GEN="${TPU_GEN:-}"
_CLI_SCRIPT="${SCRIPT:-}"
_CLI_ARGS="${ARGS:-}"
_CLI_RUNS="${RUNS:-}"
_CLI_ENABLE_V5P="${ENABLE_TPU_V5P:-}"
_CLI_ENABLE_V6E="${ENABLE_TPU_V6E:-}"

# Unset first, so that what follows is gcp.env's own choice of class
unset COMPUTE_CLASS TPU_GEN
if [[ -f "${MRX_ROOT}/gcp.env" ]]; then
    source "${MRX_ROOT}/gcp.env"
elif [[ -f "${MRX_ROOT}/gcp.env.example" ]]; then
    source "${MRX_ROOT}/gcp.env.example"
fi
_ENV_COMPUTE_CLASS="${COMPUTE_CLASS:-}"
_ENV_TPU_GEN="${TPU_GEN:-}"

[[ -n "${_CLI_PROJECT_ID}" ]] && export PROJECT_ID="${_CLI_PROJECT_ID}"
[[ -n "${_CLI_REGION}" ]] && export REGION="${_CLI_REGION}"
[[ -n "${_CLI_ENABLE_V5P}" ]] && export ENABLE_TPU_V5P="${_CLI_ENABLE_V5P}"
[[ -n "${_CLI_ENABLE_V6E}" ]] && export ENABLE_TPU_V6E="${_CLI_ENABLE_V6E}"
eval "$(bash "${MRX_ROOT}/gke/settings.sh" --export)"
: "${PROJECT_ID:?Copy gcp.env.example to gcp.env and set PROJECT_ID}"

# Pin the job to the image digest, so that a retried pod runs the same image even after the next make build. A tag
# make build has just pushed can take a few seconds to show up, hence the retries
if [[ "${IMAGE_NAME}" != *"@sha256:"* ]]; then
    for attempt in 1 2 3 4 5 6; do
        IMAGE_DIGEST="$(gcloud artifacts docker images describe "${IMAGE_NAME}" --project="${PROJECT_ID}" --format='value(image_summary.digest)' 2>/dev/null || true)"
        [[ -n "${IMAGE_DIGEST}" ]] && break
        (( attempt < 6 )) && { echo "Image ${IMAGE_NAME} not visible yet; retrying in 10 s" >&2; sleep 10; }
    done
    if [[ -z "${IMAGE_DIGEST}" ]]; then
        echo "ERROR: image ${IMAGE_NAME} not found. Build it with make deploy or make build" >&2
        exit 1
    fi
    IMAGE_NAME="${IMAGE_NAME%%:*}@${IMAGE_DIGEST}"
fi

SCRIPT="${_CLI_SCRIPT:-${SCRIPT:-scripts/relax.py}}"
ARGS="${_CLI_ARGS:-${ARGS:-}}"
# RUNS describes one job: only the command line sets it, never gcp.env
RUNS="${_CLI_RUNS}"
# ARGS and RUNS are pasted into the container's shell script (gke/job-tpu.yaml), so they must be plain words
if [[ "${ARGS}${RUNS}" == *[\"\'\$\`\\]* || "${ARGS}${RUNS}" == *$'\n'* ]]; then
    echo "ERROR: ARGS and RUNS cannot contain quotes, \$, backticks, backslashes or line breaks" >&2
    exit 1
fi
STAMP="$(date +%Y%m%d-%H%M%S)"
RAW_JOB_NAME="${JOB_NAME:-mrx-${MRX_USER}-${STAMP}}"
# Kubernetes Job names must be lowercase RFC 1123 subdomains
JOB_NAME="$(printf '%s' "${RAW_JOB_NAME}" | tr '[:upper:]_' '[:lower:]-' | tr -cd 'a-z0-9-')"
OUTSUB="${OUTSUB:-${JOB_NAME}}"
TIMEOUT_MIN="${TIMEOUT_MIN:-480}"
ACTIVE_DEADLINE_SECONDS="$(( TIMEOUT_MIN * 60 ))"
FOLLOW="${FOLLOW:-0}"

# The class: COMPUTE_CLASS, else TPU_GEN, from the command line, then from gcp.env; none means either family
for gen in "${_CLI_TPU_GEN}" "${_ENV_TPU_GEN}"; do
    if [[ -n "${gen}" && "${gen}" != "v5p" && "${gen}" != "v6e" ]]; then
        echo "ERROR: TPU_GEN must be v5p or v6e, not '${gen}'" >&2
        exit 1
    fi
done
CLASS="${_CLI_COMPUTE_CLASS:-${_CLI_TPU_GEN:+mrx-tpu-${_CLI_TPU_GEN}}}"
CLASS="${CLASS:-${_ENV_COMPUTE_CLASS:-${_ENV_TPU_GEN:+mrx-tpu-${_ENV_TPU_GEN}}}}"

# Every ComputeClass has 4-chip nodes (gke/compute-classes.yaml)
TPU_CHIPS=4

N_RUNS=0
if [[ -n "${RUNS}" ]]; then
    [[ "${SCRIPT}" == *"relax.py"* ]] || { echo "ERROR: RUNS starts scripts/relax.py runs, but SCRIPT is ${SCRIPT}" >&2; exit 1; }
    if [[ " ${ARGS} " == *" --output.out"* ]]; then
        echo "ERROR: with RUNS each run writes to run-<n>/ of the job's output directory: remove --output.out from ARGS" >&2
        exit 1
    fi
    N_RUNS="$(tr ';' '\n' <<< "${RUNS}" | grep -c '[^[:space:]]' || true)"
    if (( N_RUNS == 0 || N_RUNS > TPU_CHIPS )); then
        echo "ERROR: RUNS lists ${N_RUNS} runs; a TPU node takes 1 to ${TPU_CHIPS} (one per chip). Submit the others as another job" >&2
        exit 1
    fi
fi

# The region of the v6e cluster: make setup creates it in V6E_BACKUP_REGION when it cannot in V6E_REGION
v6e_region() {
    if gcloud container clusters describe "${V6E_CLUSTER_NAME}" --region="${V6E_REGION}" --project="${PROJECT_ID}" \
        >/dev/null 2>&1; then
        echo "${V6E_REGION}"
    else
        echo "${V6E_BACKUP_REGION}"
    fi
}

# The TPU pools the job can run in, as "<class> <cluster> <region>": every pool of its family, or of the families
# gcp.env enables, on the clusters gcp.env enables. make setup gives each cluster a pool for every enabled family its
# region offers (terraform/main.tf). v6e comes first: it is the cheaper one when two pools get a node at once. Another
# ComputeClass runs where its name says
POOLS=()
if [[ -n "${CLASS}" && "${CLASS}" != "mrx-tpu-v5p" && "${CLASS}" != "mrx-tpu-v6e" ]]; then
    if [[ "${CLASS}" == *"v6e"* ]]; then
        POOLS+=("${CLASS} ${V6E_CLUSTER_NAME} $(v6e_region)")
    else
        POOLS+=("${CLASS} ${V5P_CLUSTER_NAME} ${V5P_REGION}")
    fi
else
    CLUSTERS=() CLUSTER_POOLS=()
    if [[ "${ENABLE_TPU_V6E}" == "true" ]]; then CLUSTERS+=("${V6E_CLUSTER_NAME} $(v6e_region)"); fi
    if [[ "${ENABLE_TPU_V5P}" == "true" ]]; then CLUSTERS+=("${V5P_CLUSTER_NAME} ${V5P_REGION}"); fi
    for entry in "${CLUSTERS[@]}"; do
        read -r cluster region <<< "${entry}"
        CLUSTER_POOLS+=("$(gcloud container node-pools list --cluster="${cluster}" --region="${region}" \
            --project="${PROJECT_ID}" --format='value(name)' 2>/dev/null | tr '\n' ' ' || true)")
    done
    # The pinned family, else the families gcp.env enables
    FAMILIES=()
    if [[ -n "${CLASS}" ]]; then FAMILIES=("${CLASS#mrx-tpu-}"); fi
    if [[ -z "${CLASS}" && "${ENABLE_TPU_V6E}" == "true" ]]; then FAMILIES+=(v6e); fi
    if [[ -z "${CLASS}" && "${ENABLE_TPU_V5P}" == "true" ]]; then FAMILIES+=(v5p); fi
    for family in "${FAMILIES[@]}"; do
        for i in "${!CLUSTERS[@]}"; do
            if [[ " ${CLUSTER_POOLS[i]} " == *" tpu-${family}-4t-spot-pool "* ]]; then
                POOLS+=("mrx-tpu-${family} ${CLUSTERS[i]}")
            fi
        done
    done
fi
if (( ${#POOLS[@]} == 0 )); then
    echo "ERROR: no TPU pool for ${CLASS:-a job} on the clusters of gcp.env. Are they up (make status)? An admin creates them with: make setup" >&2
    exit 1
fi

EXTRA_EXPORT="${EXTRA_ENV:+export ${EXTRA_ENV};}"
export NAMESPACE KSA_NAME IMAGE_NAME GCS_BUCKET MRX_USER TPU_CHIPS ACTIVE_DEADLINE_SECONDS SCRIPT ARGS RUNS EXTRA_EXPORT

# One job per pool, named after it when there are several: "<context> <job> <output directory>"
JOBS=()
for pool in "${POOLS[@]}"; do
    read -r class cluster region <<< "${pool}"
    job="${JOB_NAME}" outsub="${OUTSUB}"
    if (( ${#POOLS[@]} > 1 )); then
        job="${JOB_NAME}-${class#mrx-tpu-}-${region}" outsub="${OUTSUB}-${class#mrx-tpu-}-${region}"
    fi
    gcloud container clusters get-credentials "${cluster}" --region="${region}" --project="${PROJECT_ID}" >/dev/null 2>&1 \
        || { echo "ERROR: cannot reach cluster ${cluster} in ${region}. Is it up (make status)? An admin creates it with: make setup" >&2; exit 1; }
    ctx="gke_${PROJECT_ID}_${region}_${cluster}"
    # A reused JOB_NAME replaces its old job; its outputs stay
    kubectl --context="${ctx}" delete job "${job}" --namespace="${NAMESPACE}" --ignore-not-found >/dev/null 2>&1 || true
    # Only these variables: the other ${...} in job-tpu.yaml belong to the container's shell
    JOB_NAME="${job}" COMPUTE_CLASS="${class}" REMOTE_OUTDIR="/gcs/outputs/${outsub}" \
        envsubst '${JOB_NAME} ${NAMESPACE} ${KSA_NAME} ${IMAGE_NAME} ${GCS_BUCKET} ${MRX_USER} ${COMPUTE_CLASS} ${TPU_CHIPS} ${ACTIVE_DEADLINE_SECONDS} ${REMOTE_OUTDIR} ${SCRIPT} ${ARGS} ${RUNS} ${EXTRA_EXPORT}' \
        < "${MRX_ROOT}/gke/job-tpu.yaml" | kubectl --context="${ctx}" apply -f - >/dev/null
    JOBS+=("${ctx} ${job} ${outsub}")
done

echo "=================================================================="
echo "Submitted GKE TPU Job: ${JOB_NAME}"
if (( ${#POOLS[@]} > 1 )); then
    echo "  TPU pools:    ${#POOLS[@]}, one job in each; the first to get a node runs and the others are deleted"
fi
for i in "${!POOLS[@]}"; do
    read -r class cluster region <<< "${POOLS[i]}"
    read -r ctx job outsub <<< "${JOBS[i]}"
    echo "  Job:          ${job}: ${class} (${TPU_CHIPS} chips, Spot) on ${cluster} (${region})"
done
echo "  Image:        ${IMAGE_NAME}"
echo "  Script:       ${SCRIPT} ${ARGS}"
if (( N_RUNS > 0 )); then
    echo "  Runs:         ${N_RUNS}, one per chip, in run-1/ to run-${N_RUNS}/: ${RUNS}"
fi
echo "  GCS Output:   gs://${GCS_BUCKET}/outputs/${OUTSUB}$( (( ${#POOLS[@]} > 1 )) && echo "-<family>-<region>")/"
echo "  XLA Cache:    gs://${GCS_BUCKET}/xla_cache/"
echo "  Logs:         make logs, or kubectl --context=<context> logs -n ${NAMESPACE} -f job/<job>"
echo "=================================================================="

# Waits until the pod of one of the jobs is on a node, and prints that job's index; the first in JOBS wins a tie.
# Fails once every job has been gone or failed (deleted, or past its deadline) for two checks in a row
race() {
    local i ctx job outsub nodes conditions alive gone=0
    while true; do
        alive=0
        for i in "${!JOBS[@]}"; do
            read -r ctx job outsub <<< "${JOBS[i]}"
            nodes="$(kubectl --context="${ctx}" get pods -n "${NAMESPACE}" -l "job-name=${job}" \
                -o jsonpath='{.items[*].spec.nodeName}' 2>/dev/null || true)"
            if [[ -n "${nodes// /}" ]]; then
                echo "${i}"
                return 0
            fi
            conditions="$(kubectl --context="${ctx}" get job "${job}" -n "${NAMESPACE}" \
                -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null || echo "Gone")"
            [[ "${conditions}" == "Gone" || " ${conditions} " == *" Failed "* ]] || alive=1
        done
        if (( alive )); then gone=0; else gone=$((gone + 1)); fi
        (( gone < 2 )) || return 1
        sleep 10
    done
}

# Deletes every job but the one of index $1
delete_others() {
    local i ctx job outsub
    for i in "${!JOBS[@]}"; do
        if (( i == $1 )); then continue; fi
        read -r ctx job outsub <<< "${JOBS[i]}"
        kubectl --context="${ctx}" delete job "${job}" -n "${NAMESPACE}" --ignore-not-found --wait=false >/dev/null 2>&1 || true
    done
    return 0
}

WINNER=0
if (( ${#JOBS[@]} > 1 )); then
    if [[ "${FOLLOW}" != "1" ]]; then
        # Without FOLLOW a watcher on this machine settles the race; it stops if the machine sleeps or shuts down,
        # and then make stop deletes the jobs left over
        mkdir -p "${MRX_ROOT}/outputs"
        (
            trap '' HUP
            if w="$(race)"; then
                delete_others "${w}"
                read -r class cluster region <<< "${POOLS[w]}"
                echo "$(date '+%F %T') ${class} on ${cluster} (${region}) got a node first; the other jobs are deleted"
            else
                echo "$(date '+%F %T') no job got a node (deleted, or past the deadline)"
            fi
        ) > "${MRX_ROOT}/outputs/${JOB_NAME}-race.log" 2>&1 &
        echo "A watcher on this machine keeps the first job to get a node and deletes the others (outputs/${JOB_NAME}-race.log)"
        exit 0
    fi
    echo "Waiting for a TPU node in any of the ${#JOBS[@]} pools (a node takes about 3 minutes when a zone has capacity)"
    WINNER="$(race)" || { echo "ERROR: no job got a node: all were deleted or failed (make status)" >&2; exit 1; }
    delete_others "${WINNER}"
    read -r class cluster region <<< "${POOLS[WINNER]}"
    echo "${class} on ${cluster} (${region}) got a node first; the other jobs are deleted"
fi
read -r KUBE_CTX JOB_NAME OUTSUB <<< "${JOBS[WINNER]}"

# Wait up to 10 minutes for the pod, then stream its logs. A job still Pending (no Spot capacity yet), or retried
# after a preemption, is waited for below without logs
if [[ "${FOLLOW}" == "1" ]]; then
    (( ${#JOBS[@]} > 1 )) || echo "Waiting for a TPU node and the pod to start (a node takes about 3 minutes)"
    PHASE=""
    for _ in $(seq 1 60); do
        PHASE="$(kubectl --context="${KUBE_CTX}" get pods -l "job-name=${JOB_NAME}" -n "${NAMESPACE}" -o jsonpath='{.items[0].status.phase}' 2>/dev/null || true)"
        if [[ "${PHASE}" == "Running" || "${PHASE}" == "Succeeded" || "${PHASE}" == "Failed" ]]; then
            break
        fi
        sleep 10
    done
    if [[ "${PHASE}" == "Running" || "${PHASE}" == "Succeeded" || "${PHASE}" == "Failed" ]]; then
        kubectl --context="${KUBE_CTX}" logs -n "${NAMESPACE}" -f "job/${JOB_NAME}" || true
    else
        echo "No TPU node after 10 minutes (Spot capacity): waiting for the job without logs (make logs shows them)"
    fi

    # The job's true conditions: a finished job also keeps its interim one (SuccessCriteriaMet or FailureTarget)
    CONDITIONS=""
    until [[ " ${CONDITIONS} " == *" Complete "* || " ${CONDITIONS} " == *" Failed "* ]]; do
        sleep 5
        CONDITIONS="$(kubectl --context="${KUBE_CTX}" get job "${JOB_NAME}" -n "${NAMESPACE}" \
            -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null || true)"
    done
    STATUS="Failed"
    [[ " ${CONDITIONS} " == *" Complete "* ]] && STATUS="Complete"
    mkdir -p "${MRX_ROOT}/outputs/${OUTSUB}"
    gcloud storage rsync -r "gs://${GCS_BUCKET}/outputs/${OUTSUB}" "${MRX_ROOT}/outputs/${OUTSUB}" --project="${PROJECT_ID}" 2>/dev/null || true
    echo "Job ${JOB_NAME} finished with status: ${STATUS} (local outputs: outputs/${OUTSUB}/)"
    [[ "${STATUS}" == "Complete" ]]
fi
