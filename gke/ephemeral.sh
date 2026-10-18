#!/usr/bin/env bash
# One MRX job on GKE Cloud TPUs, with clusters that exist only while jobs run:
#   1. make infra       create the clusters and TPU pools of gcp.env (Terraform, ~10-15 min), unless they are up
#   2. make <target>    build and/or run the job, follow it to the end and copy its outputs (FOLLOW=1)
#   3. make destroy     delete the clusters (the bucket, outputs and images stay) once no MRX job of anyone is
#                       active on them, also after a failure or Ctrl-C. Another user's job, or a second run of this
#                       script, keeps them up; the last one to finish deletes them
# Usage (settings as for make, gke/README.md; make ephemeral runs it with TARGET=deploy):
#   bash gke/ephemeral.sh                          # make deploy: build this checkout and run the ARGS of gcp.env
#   ARGS="..." TPU_GEN=v6e bash gke/ephemeral.sh   # one family only
#   RUNS="<args>; <args>" bash gke/ephemeral.sh run   # last build, up to 4 relax.py runs on one node
#   bash gke/ephemeral.sh down                     # only step 3: delete idle clusters (e.g. after a sleep or a crash)
# KEEP_CLUSTERS=1: skip step 3 (gke/scan.sh does, to run several jobs on one set of clusters).
# NODE_WAIT_MIN (default 45): if no TPU node has started the job by then, the job is stopped and the clusters deleted;
# with one cluster this happens at once when GKE cannot start a node for lack of quota ("GCE quota exceeded").
# Cost: it warns when gcp.env's TPU regions are not the cheapest Spot regions that offer them, or the saved prices are
# over 30 days old (python3 gke/spot_prices.py refreshes them), and prints each job's Spot cost at the end.
# Log: outputs/logs/ephemeral_<time>.log (also printed)
set -uo pipefail

MRX_GKE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-deploy}"
LOGDIR="${MRX_GKE}/outputs/logs"
mkdir -p "${LOGDIR}"
LOG="${LOGDIR}/ephemeral_$(date +%Y%m%d-%H%M%S).log"
START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PRICES="${MRX_GKE}/gke/spot_prices.py"
exec > >(tee -a "${LOG}") 2>&1
cd "${MRX_GKE}" || exit 1

# The project, cluster names and regions of gcp.env, without exporting the rest into make's environment
read -r PROJECT_ID V5P V5P_REGION V6E V6E_REGION V6E_BACKUP ENABLE_V5P ENABLE_V6E NAMESPACE JOB_USER < <(
    unset ENABLE_TPU_V5P ENABLE_TPU_V6E COMPUTE_CLASS
    eval "$(bash gke/settings.sh --export)"
    echo "${PROJECT_ID} ${V5P_CLUSTER_NAME} ${V5P_REGION} ${V6E_CLUSTER_NAME} ${V6E_REGION} ${V6E_BACKUP_REGION}" \
         "${ENABLE_TPU_V5P} ${ENABLE_TPU_V6E} ${NAMESPACE} ${MRX_USER}")
[[ -n "${PROJECT_ID}" ]] || { echo "[ephemeral] ERROR: set PROJECT_ID in ${MRX_GKE}/gcp.env" >&2; exit 1; }

say() { echo "[ephemeral $(date '+%H:%M:%S')] $*"; }
notify() { osascript -e "display notification \"$1\" with title \"MRX GKE\"" >/dev/null 2>&1 || true; }

# The MRX clusters that exist, as "cluster:region"
live_clusters() {
    gcloud container clusters list --project="${PROJECT_ID}" --format='value(name,location)' 2>/dev/null \
        | awk -v a="${V5P}" -v b="${V6E}" '$1 == a || $1 == b {print $1 ":" $2}'
}

# Active MRX jobs (any user) on the live clusters, one "cluster/job" per line
active_jobs() {
    local pair c r
    for pair in $(live_clusters); do
        c="${pair%%:*}" r="${pair##*:}"
        gcloud container clusters get-credentials "${c}" --region="${r}" --project="${PROJECT_ID}" >/dev/null 2>&1 || continue
        kubectl --context="gke_${PROJECT_ID}_${r}_${c}" get jobs -l app=mrx -n "${NAMESPACE}" \
            -o jsonpath='{range .items[?(@.status.active>0)]}{.metadata.name}{"\n"}{end}' 2>/dev/null \
            | sed "s|^|${c}/|"
    done
}

# The full logs of $MRX_USER's jobs created since this script started, to outputs/<job>.log (the stream of make run
# can drop, and the logs are gone with the cluster)
save_logs() {
    local pair c r ctx job
    for pair in $(live_clusters); do
        c="${pair%%:*}" r="${pair##*:}" ctx="gke_${PROJECT_ID}_${r}_${c}"
        for job in $(kubectl --context="${ctx}" get jobs -n "${NAMESPACE}" -l "app=mrx,mrx-user=${JOB_USER}" -o json 2>/dev/null \
            | python3 -c 'import json, sys; [print(j["metadata"]["name"]) for j in json.load(sys.stdin)["items"]
                          if j["metadata"]["creationTimestamp"] >= sys.argv[1]]' "${START_UTC}"); do
            mkdir -p outputs
            kubectl --context="${ctx}" logs -n "${NAMESPACE}" "job/${job}" -c mrx-tpu > "outputs/${job}.log" 2>&1 \
                && say "log: ${MRX_GKE}/outputs/${job}.log"
        done
    done
}

# TERM to a process and everything it started
kill_tree() {
    local child
    for child in $(pgrep -P "$1"); do kill_tree "${child}"; done
    kill -TERM "$1" 2>/dev/null || true
}

# Watches the job until a TPU node runs it. Without one after NODE_WAIT_MIN minutes, or (one cluster) on a failed
# scale-up for quota, it stops the job (make stop) and the make process $1, so the teardown deletes the clusters
watchdog() {
    local deadline=$(( SECONDS + ${NODE_WAIT_MIN:-45} * 60 )) pair c r ctx phases quota n reason
    while sleep 60; do
        phases="" quota=0 n=0
        for pair in $(live_clusters); do
            c="${pair%%:*}" r="${pair##*:}" ctx="gke_${PROJECT_ID}_${r}_${c}" n=$((n + 1))
            phases+=" $(kubectl --context="${ctx}" get pods -n "${NAMESPACE}" -l "app=mrx,mrx-user=${JOB_USER}" -o json 2>/dev/null \
                | python3 -c 'import json, sys; print(" ".join(p["status"].get("phase", "") for p in json.load(sys.stdin)["items"]
                              if p["metadata"]["creationTimestamp"] >= sys.argv[1]))' "${START_UTC}")"
            kubectl --context="${ctx}" get events -n "${NAMESPACE}" --field-selector reason=FailedScaleUp -o json 2>/dev/null \
                | python3 -c 'import json, sys; sys.exit(0 if any("quota exceeded" in e.get("message", "")
                              and (e.get("lastTimestamp") or "") >= sys.argv[1] for e in json.load(sys.stdin)["items"]) else 1)' \
                "${START_UTC}" && quota=$((quota + 1))
        done
        [[ "${phases}" == *Running* || "${phases}" == *Succeeded* || "${phases}" == *Failed* ]] && return 0
        if (( n > 0 && quota == n )); then
            reason="GKE cannot start a TPU node: GCE quota exceeded in $(live_clusters | tr '\n' ' ')(run: python3 ${PRICES})"
        elif (( SECONDS > deadline )); then
            reason="no TPU node after ${NODE_WAIT_MIN:-45} min (no Spot capacity)"
        else
            continue
        fi
        say "${reason}: stopping the job"
        notify "No TPU node: job stopped, clusters deleted"
        make stop </dev/null >/dev/null 2>&1
        kill_tree "$1"
        return 1
    done
}

TORN_DOWN=0
teardown() {
    (( TORN_DOWN )) && return
    TORN_DOWN=1
    if [[ "${KEEP_CLUSTERS:-0}" == "1" ]]; then
        say "KEEP_CLUSTERS=1: clusters left up (delete them with: $0 down)"
        return
    fi
    if [[ -z "$(live_clusters)" ]]; then
        say "no MRX cluster is up"
        return
    fi
    local active
    active="$(active_jobs)"
    if [[ -n "${active}" ]]; then
        say "clusters left up: active MRX job(s): $(tr '\n' ' ' <<< "${active}")"
        say "the last job's run of this script deletes them, or later: $0 down"
        notify "Clusters left up: other MRX jobs are active"
        return
    fi
    say "deleting the clusters (make destroy; bucket, outputs and images stay)"
    if FORCE=1 make destroy </dev/null; then
        [[ -z "$(live_clusters)" ]] && say "clusters deleted" && notify "Clusters deleted" && return
    fi
    say "ERROR: clusters still up: $(live_clusters | tr '\n' ' ')- run: $0 down (or gcloud container clusters delete)"
    notify "ERROR: clusters still up, run gke_ephemeral.sh down"
}

if [[ "${TARGET}" == "down" ]]; then
    teardown
    exit 0
fi

# gcloud and Terraform must both be logged in before anything is created, so the teardown can run at the end
gcloud auth print-access-token >/dev/null 2>&1 \
    || { echo "[ephemeral] ERROR: gcloud is not logged in: gcloud auth login --update-adc" >&2; exit 1; }
gcloud auth application-default print-access-token >/dev/null 2>&1 \
    || { echo "[ephemeral] ERROR: no Application Default Credentials: gcloud auth login --update-adc" >&2; exit 1; }

python3 "${PRICES}" --check 2>/dev/null | sed 's/^/[ephemeral] price: /'

trap teardown EXIT
trap 'say "interrupted"; exit 130' INT TERM

WANT=()
[[ "${ENABLE_V5P}" == "true" ]] && WANT+=("${V5P}")
[[ "${ENABLE_V6E}" == "true" ]] && WANT+=("${V6E}")
LIVE="$(live_clusters)"
MISSING=0
for c in "${WANT[@]}"; do grep -q "^${c}:" <<< "${LIVE}" || MISSING=1; done
if (( MISSING )); then
    say "creating the clusters (make infra): ${WANT[*]}"
    START=${SECONDS}
    make infra </dev/null || { say "ERROR: make infra failed"; exit 1; }
    say "clusters up in $(( (SECONDS - START) / 60 )) min"
else
    say "clusters already up: $(tr '\n' ' ' <<< "${LIVE}")"
fi

say "make ${TARGET}"
START=${SECONDS}
FOLLOW=1 make "${TARGET}" </dev/null &
MAKE_PID=$!
watchdog "${MAKE_PID}" &
WATCHDOG_PID=$!
wait "${MAKE_PID}"
RC=$?
kill_tree "${WATCHDOG_PID}"
say "make ${TARGET} ended with code ${RC} after $(( (SECONDS - START) / 60 )) min"
say "Spot cost of the jobs started since ${START_UTC}:"
python3 "${PRICES}" --cost-since "${START_UTC}" 2>&1
save_logs
(( RC == 0 )) && notify "Job finished: outputs in ${MRX_GKE}/outputs" || notify "Job FAILED (code ${RC})"
exit "${RC}"
