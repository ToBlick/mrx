#!/usr/bin/env bash
#
# Keep trying until a TPU lands, then run MRX on it unattended.
#
# TPU capacity appears and disappears unpredictably, and a parked flex-start
# request expires silently after two hours. Watching for that by hand is the
# actual bottleneck, so this loops: sweep the ladder, park a request if nothing
# is free, re-park when a window lapses, and the moment an instance reaches
# RUNNING, build the environment, run MRX and send a desktop notification.
#
# Usage:
#   ./acquire_tpu.sh                  # loop until acquired or budget spent
#   ./acquire_tpu.sh --once           # a single sweep, no parking or looping
#   ./acquire_tpu.sh --gc             # delete stray my-data-disk copies
#   GENERATIONS=v5e ./acquire_tpu.sh
#   MAX_HOURS=6 SWEEP_INTERVAL=300 ./acquire_tpu.sh
#
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=zones.sh
source ./zones.sh

SWEEP_INTERVAL="${SWEEP_INTERVAL:-180}"
MAX_HOURS="${MAX_HOURS:-12}"
MAX_SESSIONS="${MAX_SESSIONS:-3}"
LOCK_FILE="${LOCK_FILE:-.acquire.lock}"
ACQUIRE_LOG="${ACQUIRE_LOG:-acquire.log}"
KEEP_VM="${KEEP_VM:-0}"

ONCE=0
GC_ONLY=0
# --acquire-only stops after the hardware lands instead of running MRX. Used
# when the intended job is not the Poisson report -- otherwise acquiring a node
# for, say, a relaxation first burns several minutes and dollars on a run that
# is about to be superseded.
ACQUIRE_ONLY=0
for arg in "$@"; do
    case "${arg}" in
        --once)         ONCE=1 ;;
        --gc)           GC_ONLY=1 ;;
        --acquire-only) ACQUIRE_ONLY=1 ;;
    esac
done

START_EPOCH="$(date +%s)"
SESSIONS=0

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}"
    echo "${msg}" >>"${ACQUIRE_LOG}"
}

notify() {
    local title="$1" body="$2"
    command -v osascript >/dev/null 2>&1 && \
        osascript -e "display notification \"${body}\" with title \"${title}\"" \
        >/dev/null 2>&1
    printf '\a'
}

# ------------------------------------------------------------------ locking ---
# A second daemon would double-book quota and race on the VM name.
if [[ -f "${LOCK_FILE}" ]]; then
    old_pid="$(cat "${LOCK_FILE}" 2>/dev/null)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
        echo "ERROR: another daemon is running (pid ${old_pid})." >&2
        echo "Kill it or remove ${LOCK_FILE} if it is stale." >&2
        exit 1
    fi
    rm -f "${LOCK_FILE}"
fi
echo "$$" >"${LOCK_FILE}"
trap 'rm -f "${LOCK_FILE}"' EXIT INT TERM

# ----------------------------------------------------------- disk gc helper ---
# The sweep used to pre-create a data disk per candidate zone, which left six
# 100 GB volumes billing at once. That is fixed, but a --gc mode is cheap
# insurance and useful after an interrupted run.
gc_disks() {
    local keep_zone="${1:-}"
    local zones z
    zones="$(gcloud compute disks list --filter="name=${DATA_DISK}" \
        --format="value(zone.basename())" 2>/dev/null)"
    for z in ${zones}; do
        [[ "${z}" == "${keep_zone}" ]] && continue
        # Never touch a disk that has data on it or is attached.
        local users
        users="$(gcloud compute disks describe "${DATA_DISK}" --zone="${z}" \
            --format="value(users)" 2>/dev/null)"
        if [[ -n "${users}" ]]; then
            log "  keeping ${DATA_DISK} in ${z} (attached)"
            continue
        fi
        log "  deleting unused ${DATA_DISK} in ${z}"
        gcloud compute disks delete "${DATA_DISK}" --zone="${z}" \
            --quiet >/dev/null 2>&1
    done
}

if (( GC_ONLY )); then
    log "Garbage-collecting stray ${DATA_DISK} volumes..."
    echo "This deletes every unattached ${DATA_DISK} except those you keep."
    read -r -p "Zone to keep (blank for none): " keep
    gc_disks "${keep}"
    log "done"
    exit 0
fi

# ------------------------------------------------------------- current state ---
# Returns "name zone status" for an existing VM_NAME, if any.
find_existing() {
    gcloud compute instances list --filter="name=${VM_NAME}" \
        --format="value(name,zone.basename(),status)" 2>/dev/null | head -n 1
}

running_zone() {
    gcloud compute instances list --filter="name=${VM_NAME} AND status=RUNNING" \
        --format="value(zone.basename())" 2>/dev/null | head -n 1
}

# Cloud TPU API nodes are a different resource type and are not returned by
# `compute instances list`. Their zone is embedded in the resource name and
# their healthy state is READY, not RUNNING.
# Find a READY Cloud TPU API node, returning its zone.
#
# Two traps here, both of which cost a live TPU once. `tpu-vm list` without a
# zone errors out rather than returning nothing, and with --zone=- it prints
# the *short* name, so there is no path to parse a zone out of. Describing each
# candidate zone directly avoids both. The node also spends a minute or two in
# CREATING after the create call returns, so poll rather than check once --
# checking once is what made the daemon walk away from a TPU it had just won.
tpu_running_zone() {
    local name="${1:-${VM_NAME}}" tries="${2:-1}"
    local entry gen mt zone model api state i zones seen

    zones=""
    for entry in "${CANDIDATES[@]}"; do
        IFS=':' read -r gen mt zone model api <<<"${entry}"
        [[ "${api}" == "tpuapi" ]] || continue
        case " ${seen:-} " in *" ${zone} "*) continue ;; esac
        seen="${seen:-} ${zone}"
        zones="${zones} ${zone}"
    done

    for (( i = 0; i < tries; i++ )); do
        for zone in ${zones}; do
            state="$(gcloud compute tpus tpu-vm describe "${name}" \
                --zone="${zone}" --format="value(state)" 2>/dev/null)"
            case "${state}" in
                READY)    echo "${zone}"; return 0 ;;
                CREATING) ;;   # keep polling
            esac
        done
        (( i + 1 < tries )) && sleep 20
    done
    return 1
}

pending_zone() {
    gcloud compute instances list --filter="name=${VM_NAME} AND status=PENDING" \
        --format="value(zone.basename())" 2>/dev/null | head -n 1
}

# --------------------------------------------------------------- the payload ---
run_session() {
    local zone="$1"

    if (( ACQUIRE_ONLY )); then
        log "TPU is RUNNING in ${zone}; --acquire-only, so not starting a job."
        notify "TPU acquired" "${VM_NAME} ready in ${zone}"
        echo ""
        echo "  Node ${VM_NAME} is ready in ${zone} and is BILLING until deleted."
        echo "  Run something on it:  VM_NAME=${VM_NAME} ZONE=${zone} ./run_on_tpu.sh"
        echo ""
        exit 0
    fi

    SESSIONS=$(( SESSIONS + 1 ))
    log "TPU is RUNNING in ${zone}; starting MRX session ${SESSIONS}/${MAX_SESSIONS}"
    notify "TPU acquired" "Running MRX in ${zone}"

    if ZONE="${zone}" ./run_on_tpu.sh; then
        log "MRX run completed; results in script_outputs/"
        notify "MRX finished" "Results pulled to script_outputs/"
    else
        log "MRX run exited non-zero; check script_outputs/ and acquire.log"
        notify "MRX finished with warnings" "Check summary.md for deviations"
    fi

    # The snapshot was taken from a raw disk, so it carries no environment.
    # Refresh it once the build has actually happened, so a future zone starts
    # warm instead of paying the 10-12 minute install again.
    refresh_snapshot "${zone}"
}

refresh_snapshot() {
    local zone="$1" size
    # Nothing to snapshot when the run used the boot-disk fallback, which is
    # what happens in any zone without a my-data-disk (us-south1-a, where the
    # first successful v5e session landed). Without this the create below
    # always failed and reported a refresh failure that was not one.
    if ! data_disk_exists "${zone}"; then
        log "  no ${DATA_DISK} in ${zone} (boot-disk run); nothing to snapshot"
        return 0
    fi
    size="$(gcloud compute snapshots describe "${DATA_SNAPSHOT}" \
        --format="value(storageBytes)" 2>/dev/null || echo 0)"
    # Only worth redoing while the snapshot is still effectively empty.
    if [[ -n "${size}" ]] && (( size > 1000000000 )); then
        return 0
    fi
    log "Refreshing ${DATA_SNAPSHOT} from ${zone} so future zones start warm..."
    gcloud compute snapshots delete "${DATA_SNAPSHOT}" --quiet >/dev/null 2>&1
    gcloud compute snapshots create "${DATA_SNAPSHOT}" \
        --source-disk="${DATA_DISK}" --source-disk-zone="${zone}" \
        --quiet >/dev/null 2>&1 \
        && log "  snapshot refreshed" \
        || log "  snapshot refresh failed (non-fatal)"
}

# ---------------------------------------------------------------- main loop ---
log "=============================================================="
log "acquire_tpu.sh starting (max ${MAX_HOURS}h, ${MAX_SESSIONS} session(s))"
build_candidates
log "${#CANDIDATES[@]} candidates; best is ${CANDIDATES[0]}"

while true; do
    elapsed=$(( $(date +%s) - START_EPOCH ))
    if (( elapsed > MAX_HOURS * 3600 )); then
        log "Budget of ${MAX_HOURS}h reached; stopping."
        notify "TPU daemon stopped" "No capacity within ${MAX_HOURS}h"
        exit 1
    fi
    if (( SESSIONS >= MAX_SESSIONS )); then
        log "Completed ${SESSIONS} session(s); stopping."
        exit 0
    fi

    # Something may already exist from a previous loop or a manual launch.
    zone="$(running_zone)"
    if [[ -n "${zone}" ]]; then
        run_session "${zone}"
        (( ONCE )) && exit 0
        continue
    fi

    # A parked flex-start request is free and holds a place in the DWS queue, so
    # keep it -- but do not let it stop us trying the Cloud TPU API, which is a
    # separate pool and the only route to v5e here. The tpuapi sweep uses its
    # own name so it cannot collide with the parked GCE instance.
    zone="$(pending_zone)"
    if [[ -n "${zone}" ]]; then
        log "Request queued in ${zone}; meanwhile trying the Cloud TPU API"
        APIS=tpuapi VM_NAME="${VM_NAME}-v5e" ./launch_tpu.sh \
            >>"${ACQUIRE_LOG}" 2>&1
        tzone="$(tpu_running_zone "${VM_NAME}-v5e" 12)"
        if [[ -n "${tzone}" ]]; then
            log "Acquired a Cloud TPU API node in ${tzone}"
            VM_NAME="${VM_NAME}-v5e" TPU_API=1 run_session "${tzone}"
            (( ONCE )) && exit 0
            continue
        fi
        log "Still queued; sleeping ${SWEEP_INTERVAL}s"
        sleep "${SWEEP_INTERVAL}"
        continue
    fi

    log "Sweeping the ladder..."
    if ./launch_tpu.sh >>"${ACQUIRE_LOG}" 2>&1; then
        zone="$(running_zone)"
        if [[ -z "${zone}" ]]; then
            zone="$(tpu_running_zone "${VM_NAME}" 12)"
            [[ -n "${zone}" ]] && export TPU_API=1
        fi
        if [[ -n "${zone}" ]]; then
            log "Acquired in ${zone}"
            run_session "${zone}"
            (( ONCE )) && exit 0
            continue
        fi
        # launch_tpu.sh parks a 2h request when every zone is out; that shows up
        # as PENDING on the next pass rather than as a failure here.
        log "Nothing immediately available; a request is parked."
    else
        log "Sweep failed outright; see ${ACQUIRE_LOG}"
    fi

    if (( ONCE )); then
        log "--once given; not looping."
        exit 1
    fi

    log "Sleeping ${SWEEP_INTERVAL}s before the next attempt"
    sleep "${SWEEP_INTERVAL}"
done
