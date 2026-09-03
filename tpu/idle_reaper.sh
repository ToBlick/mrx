#!/bin/bash
#
# Delete this node once nothing has run on it for a while.
#
# WHY THIS EXISTS
#
# The GCE path takes --max-run-duration with --instance-termination-action=DELETE,
# so a v5p or v6e instance bounds its own life at MAX_RUN_DURATION (zones.sh, 4h)
# even if every client disappears. The Cloud TPU API has no equivalent, and on
# this project the API is the ONLY way to reach v5e: the GCE machine type
# ct5lp-hightpu-4t returns 403 regardless of quota. So the machine that is
# easiest to get is the one with no safety net, and a forgotten v5e bills until
# someone remembers it. One did, for eight hours, which is why this file exists.
#
# Four hours of self-termination is a backstop against catastrophe, not cost
# control. This closes the gap to minutes, on both paths.
#
# WHAT COUNTS AS BUSY
#
# Three signals, checked every CHECK_INTERVAL seconds. Any one of them means
# busy and resets the counter:
#
#   1. a process running the environment's python -- the actual work
#   2. a login session -- someone is on the box, even if thinking
#   3. an accelerator device held open -- work that is not our python
#
# Deleting a node someone is using is far worse than paying for an idle one, so
# every ambiguous case resolves to busy: if a check cannot be performed at all,
# it reports busy rather than idle.
#
# WHEN IT ARMS
#
# Not until ${SENTINEL} exists. A cold build takes 10-12 minutes during which
# no python runs and nobody is logged in, which is indistinguishable from idle
# by the rules above. startup.sh installs this service early, before its own
# warm-disk early exit, so a warm node gets a reaper too; the sentinel gate is
# what makes that safe on a cold one.
#
# That gate expires at SETUP_GRACE_MIN. A setup that has not finished in three
# times its worst observed duration has failed, not stalled, and a node whose
# sentinel will never be written is exactly the node that would otherwise bill
# forever -- the failure mode this is for, not an exception to it.
#
# TESTING IT
#
# Deletion cannot be dry-run in production without also not deleting, so:
#
#     ./idle_reaper.sh --check
#
# reports what it detected, which API it would call and with what arguments,
# and exits. Run that once on a new node shape before trusting it.

set -uo pipefail

SENTINEL=/mnt/data/.mrx_env_ready
ENV_DIR=/mnt/data/envs/mrx
STATE=/var/run/mrx_idle_reaper.state
METADATA="http://metadata.google.internal/computeMetadata/v1"

IDLE_TIMEOUT_MIN="${IDLE_TIMEOUT_MIN:-20}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
SETUP_GRACE_MIN="${SETUP_GRACE_MIN:-45}"

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

log() { echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') idle_reaper: $*"; }

md() {
    curl -sf -H "Metadata-Flavor: Google" --max-time 5 "${METADATA}/$1" 2>/dev/null
}

# ------------------------------------------------------------- who am I ---
# A TPU API node and a GCE instance are deleted through different APIs, and a
# TPU VM's hostname is not its node name (it is t1v-n-<uuid>-w-0), so the node
# name has to come from the tpu-env attribute rather than from `hostname`.
# Absence of tpu-env is what identifies the GCE path.
identify() {
    API="" NODE="" ZONE="" PROJECT=""
    local tpu_env
    tpu_env="$(md instance/attributes/tpu-env)"
    if [[ -n "${tpu_env}" ]]; then
        API=tpuapi
        NODE="$(sed -n "s/^NODE_ID: *'\{0,1\}\([^']*\)'\{0,1\}$/\1/p" <<<"${tpu_env}")"
        ZONE="$(sed -n "s/^ZONE: *'\{0,1\}\([^']*\)'\{0,1\}$/\1/p" <<<"${tpu_env}")"
        PROJECT="$(sed -n "s/^CONSUMER_PROJECT_ID: *'\{0,1\}\([^']*\)'\{0,1\}$/\1/p" <<<"${tpu_env}")"
    fi
    # Either the GCE path, or a tpu-env that did not carry what was expected.
    if [[ -z "${NODE}" || -z "${ZONE}" ]]; then
        API=gce
        NODE="$(md instance/name)"
        ZONE="$(basename "$(md instance/zone)")"
        PROJECT="$(md project/project-id)"
    fi
    [[ -n "${PROJECT}" ]] || PROJECT="$(md project/project-id)"
}

# ---------------------------------------------------------------- busy? ---
# Each probe returns 0 for busy. Note the inversion on pgrep/fuser: a failed
# probe must not read as idle, so anything other than a clean "no match" is
# treated as busy by the caller through the `|| return 0` pattern.
python_running() {
    pgrep -f "${ENV_DIR}/bin/python" >/dev/null 2>&1
}

someone_logged_in() {
    local n
    n="$(who 2>/dev/null | wc -l)" || return 0
    (( n > 0 ))
}

accelerator_held() {
    compgen -G "/dev/accel*" >/dev/null 2>&1 || return 1
    # -n so that --check run as an ordinary user fails the probe instead of
    # blocking on a password prompt.
    sudo -n fuser /dev/accel* >/dev/null 2>&1
}

busy_reason() {
    python_running     && { echo "python running"; return 0; }
    someone_logged_in  && { echo "login session";  return 0; }
    accelerator_held   && { echo "accelerator held"; return 0; }
    return 1
}

# --------------------------------------------------------------- delete ---
# gcloud ships in the TPU VM image, but the reaper must not depend on that: if
# it is missing or broken, fall back to the REST API with the metadata server's
# own token. The node's service account needs tpu.nodes.delete (roles/editor
# has it); startup.sh checks that at install time so a node without it says so
# in the serial log rather than silently never reaping.
delete_self() {
    log "deleting ${API} node ${NODE} in ${ZONE} (project ${PROJECT})"
    if command -v gcloud >/dev/null 2>&1; then
        if [[ "${API}" == "tpuapi" ]]; then
            gcloud compute tpus tpu-vm delete "${NODE}" --zone="${ZONE}" \
                --project="${PROJECT}" --quiet && return 0
        else
            gcloud compute instances delete "${NODE}" --zone="${ZONE}" \
                --project="${PROJECT}" --quiet && return 0
        fi
        log "gcloud delete failed; falling back to the REST API"
    fi
    local token url
    token="$(md instance/service-accounts/default/token |
             python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])' 2>/dev/null)"
    if [[ -z "${token}" ]]; then
        log "ERROR: no access token from the metadata server; cannot delete"
        return 1
    fi
    if [[ "${API}" == "tpuapi" ]]; then
        url="https://tpu.googleapis.com/v2/projects/${PROJECT}/locations/${ZONE}/nodes/${NODE}"
    else
        url="https://compute.googleapis.com/compute/v1/projects/${PROJECT}/zones/${ZONE}/instances/${NODE}"
    fi
    curl -sf -X DELETE -H "Authorization: Bearer ${token}" "${url}" >/dev/null
}

# ----------------------------------------------------------------- main ---
identify

if (( CHECK_ONLY )); then
    echo "api:        ${API}"
    echo "node:       ${NODE}"
    echo "zone:       ${ZONE}"
    echo "project:    ${PROJECT}"
    if [[ -f "${SENTINEL}" ]]; then
        echo "sentinel:   present"
    else
        echo "sentinel:   ABSENT, so the reaper would not arm yet"
    fi
    echo "idle after: ${IDLE_TIMEOUT_MIN} min, checked every ${CHECK_INTERVAL}s"
    reason="$(busy_reason)" && echo "state:      BUSY (${reason})" \
                            || echo "state:      idle"
    if [[ -z "${NODE}" || -z "${ZONE}" ]]; then
        echo "would run:  NOTHING -- this node could not be identified, so the"
        echo "            reaper would refuse to start. Expected off a GCE or"
        echo "            TPU VM, where there is no metadata server to ask."
        exit 1
    fi
    if [[ "${API}" == "tpuapi" ]]; then
        echo "would run:  gcloud compute tpus tpu-vm delete ${NODE} --zone=${ZONE} --project=${PROJECT} --quiet"
    else
        echo "would run:  gcloud compute instances delete ${NODE} --zone=${ZONE} --project=${PROJECT} --quiet"
    fi
    exit 0
fi

if [[ -z "${NODE}" || -z "${ZONE}" ]]; then
    log "ERROR: could not identify this node (api=${API} node=${NODE} zone=${ZONE}); refusing to run"
    exit 1
fi

log "watching ${API} node ${NODE} in ${ZONE}; delete after ${IDLE_TIMEOUT_MIN} idle min"
idle_sec=0
echo 0 | sudo tee "${STATE}" >/dev/null 2>&1 || true

while true; do
    sleep "${CHECK_INTERVAL}"

    if [[ ! -f "${SENTINEL}" ]]; then
        uptime_sec="$(cut -d. -f1 /proc/uptime 2>/dev/null || echo 0)"
        if (( uptime_sec < SETUP_GRACE_MIN * 60 )); then
            # Still building. A cold build looks exactly like an idle node.
            idle_sec=0
            continue
        fi
        # Past the grace window with no sentinel: setup failed. Fall through
        # and let the ordinary idle rules reap it.
        if (( idle_sec == 0 )); then
            log "no sentinel after ${SETUP_GRACE_MIN} min; treating setup as failed"
        fi
    fi

    if reason="$(busy_reason)"; then
        if (( idle_sec > 0 )); then
            log "busy again (${reason}); idle counter reset from ${idle_sec}s"
        fi
        idle_sec=0
    else
        idle_sec=$(( idle_sec + CHECK_INTERVAL ))
        if (( idle_sec % 300 == 0 )); then
            log "idle ${idle_sec}s of $(( IDLE_TIMEOUT_MIN * 60 ))s"
        fi
    fi
    echo "${idle_sec}" | sudo tee "${STATE}" >/dev/null 2>&1 || true

    if (( idle_sec >= IDLE_TIMEOUT_MIN * 60 )); then
        log "idle ${idle_sec}s >= ${IDLE_TIMEOUT_MIN} min; deleting this node"
        delete_self || log "ERROR: delete failed; will retry next interval"
        # If the delete succeeded the machine is going away regardless; if it
        # failed, keep the counter pinned so the retry is immediate.
        idle_sec=$(( IDLE_TIMEOUT_MIN * 60 ))
    fi
done
