#!/usr/bin/env bash
#
# Keep trying until a TPU lands, then run MRX on it unattended.
#
# Capacity appears and disappears unpredictably, so this loops: sweep the
# ladder, sleep, sweep again, and the moment a node reaches RUNNING build the
# environment, run MRX and send a desktop notification.
#
# Every create in the sweep fails fast rather than queueing. The command in
# Google's setup doc blocks for up to two hours, because
# --request-valid-for-duration leaves the instance PENDING while Dynamic
# Workload Scheduler waits for capacity and gcloud waits with it. A success in
# the sweep is therefore always a real VM, and there is no probe-then-claim
# race.
#
# --queue adds the other half of that trade rather than replacing it: standing
# requests through the Queued Resources API, which wait in Google's admission
# queue while the sweep goes on asking. Worth turning on in a stockout, when the
# sweep has been losing the instant-capacity race for hours.
#
# Usage:
#   ./acquire_tpu.sh                  # loop until acquired or budget spent
#   ./acquire_tpu.sh --once           # a single sweep, no looping
#   ./acquire_tpu.sh --acquire-only   # stop once the hardware lands
#   ./acquire_tpu.sh --queue          # also file standing queued requests
#   GENERATIONS=v5e ./acquire_tpu.sh
#   MAX_HOURS=6 SWEEP_INTERVAL=300 ./acquire_tpu.sh
#   QUEUE_VALID_FOR=12h ./acquire_tpu.sh --queue
#
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
# shellcheck source=zones.sh
source ./zones.sh

SWEEP_INTERVAL="${SWEEP_INTERVAL:-180}"
MAX_HOURS="${MAX_HOURS:-12}"
MAX_SESSIONS="${MAX_SESSIONS:-3}"
LOCK_FILE="${LOCK_FILE:-.acquire.lock}"
ACQUIRE_LOG="${ACQUIRE_LOG:-acquire.log}"

ONCE=0
# --acquire-only stops after the hardware lands instead of running MRX. Used
# when the intended job is not the Poisson report -- otherwise acquiring a node
# for, say, a relaxation first burns several minutes and dollars on a run that
# is about to be superseded.
ACQUIRE_ONLY=0
# --queue additionally files standing requests through the Queued Resources
# API, which the sweep alone cannot do. The requests live exactly as long as
# this daemon: they are cancelled on exit, so nothing it filed can be fulfilled
# after nobody is left watching for it.
QUEUE=0
QUEUE_VALID_FOR="${QUEUE_VALID_FOR:-6h}"
for arg in "$@"; do
    case "${arg}" in
        --once)         ONCE=1 ;;
        --acquire-only) ACQUIRE_ONLY=1 ;;
        --queue)        QUEUE=1 ;;
    esac
done

if [[ ! -f startup.sh ]]; then
    echo "ERROR: startup.sh not found next to this script." >&2
    exit 1
fi

START_EPOCH="$(date +%s)"
SESSIONS=0

log() {
    local msg
    msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
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

# auto-delete=no on the data disk is mandatory, not defensive. gcloud documents
# the --disk auto-delete default as *yes*, so combined with
# --instance-termination-action=DELETE the original command would have destroyed
# my-data-disk the moment the run duration expired.
#
# Set NO_DISK=1 to build a create that does not attach the data disk. Used to
# retry machine types that reject the disk's type outright.
NO_DISK=0

build_args() {
    local mt="$1" zone="$2" model="$3"
    # The --metadata flags take comma-separated key=value lists inside a single
    # argument, which is not the array separator shellcheck reads them as.
    # shellcheck disable=SC2054
    CREATE_ARGS=(
        "${VM_NAME}"
        --zone="${zone}"
        --machine-type="${mt}"
        --provisioning-model="${model}"
        --max-run-duration="${MAX_RUN_DURATION}"
        --instance-termination-action=DELETE
        --image-project="${IMAGE_PROJECT}"
        --image-family="${IMAGE_FAMILY}"
        --maintenance-policy=TERMINATE
        --metadata-from-file=startup-script=startup.sh,idle-reaper=idle_reaper.sh
        --metadata=idle-timeout-min="${IDLE_TIMEOUT_MIN}",mrx-branch="${MRX_BRANCH}"
    )
    if (( ! NO_DISK )) && data_disk_exists "${zone}"; then
        CREATE_ARGS+=(
            "--disk=name=${DATA_DISK},device-name=data-disk,mode=rw,boot=no,auto-delete=no"
        )
    fi
}

# Winning a zone with no data disk is the good problem to have: create the disk
# from the snapshot and hot-attach it, then re-run the startup script so it
# formats, mounts and builds the environment. Deliberately done only after a
# win -- pre-seeding disks across a sweep that usually finds nothing once left
# six 100 GB volumes billing in six zones.
#
# The two surfaces differ. `tpu-vm update --attach-disk` takes no device name,
# so the disk lands as google-persistent-disk-N rather than google-data-disk;
# startup.sh looks for both. And force=1 means the create *refused* a disk that
# exists, which is what DISK_INCOMPATIBLE is: v5p rejects hyperdisk-balanced as
# a boot-time attachment but may accept the hot attach, which is a separate
# code path. If it does not, the warning is the point: that node has no
# persistent environment.
attach_disk_after_create() {
    local zone="$1" api="$2" force="${3:-0}"
    if (( ! force )); then
        # Either we deliberately dropped the disk, or the create already
        # attached it.
        (( NO_DISK )) && return 0
        data_disk_exists "${zone}" && return 0
    fi

    if data_disk_exists "${zone}"; then
        echo "  ${DATA_DISK} was refused at create time; trying a hot attach..."
    else
        echo "  no ${DATA_DISK} in ${zone}; creating it from ${DATA_SNAPSHOT}..."
        if ! ensure_data_disk "${zone}"; then
            echo "  WARNING: could not create ${DATA_DISK}; the environment will" >&2
            echo "           be rebuilt from scratch and will not persist" >&2
            return 1
        fi
    fi

    echo "  attaching ${DATA_DISK} to the node..."
    if [[ "${api}" == "tpuapi" ]]; then
        gcloud compute tpus tpu-vm update "${VM_NAME}" --zone="${zone}" \
            --attach-disk "source=projects/${PROJECT}/zones/${zone}/disks/${DATA_DISK},mode=read-write" \
            --quiet >/dev/null 2>&1 || {
            echo "  WARNING: attach-disk failed; the environment will not persist" >&2
            return 1; }
    else
        gcloud compute instances attach-disk "${VM_NAME}" \
            --zone="${zone}" --disk="${DATA_DISK}" \
            --device-name=data-disk --mode=rw --quiet >/dev/null 2>&1 || {
            echo "  WARNING: attach-disk failed" >&2; return 1; }
        gcloud compute instances set-disk-auto-delete "${VM_NAME}" \
            --zone="${zone}" --no-auto-delete --device-name=data-disk \
            --quiet >/dev/null 2>&1
    fi

    # The startup script has already run against the boot disk by now. Re-run it
    # so it mounts what we just attached: if the disk came from the snapshot the
    # sentinel is on it and the build is skipped entirely.
    echo "  attached; re-running the startup script against the persistent disk"
    if [[ "${api}" == "tpuapi" ]]; then
        gcloud compute tpus tpu-vm ssh "${VM_NAME}" --zone="${zone}" \
            --command="sudo google_metadata_script_runner startup" \
            --quiet >/dev/null 2>&1 &
    else
        gcloud compute ssh "${VM_NAME}" --zone="${zone}" \
            --command="sudo google_metadata_script_runner startup" \
            -- -o StrictHostKeyChecking=no >/dev/null 2>&1 &
    fi
}

# The Cloud TPU API is a separate surface from `compute instances create`, and
# on this project it is the only way to reach v5e: the GCE machine type
# ct5lp-hightpu-4t returns 403 "user agent is not allowed to use the machine
# type" regardless of the 512 chips of quota.
#
# It has no --max-run-duration, so the only self-termination it gets is
# idle_reaper.sh, shipped in as a second metadata file and installed by
# startup.sh. That makes the reaper load-bearing here rather than a convenience:
# on this path it is the sole thing standing between a forgotten node and an
# unbounded bill.
create_tpuapi() {
    local accel="$1" zone="$2" model="$3" log="$4"
    # shellcheck disable=SC2054  # as in build_args, the commas are inside one arg
    local args=(
        "${VM_NAME}"
        --zone="${zone}"
        --accelerator-type="${accel}"
        --version="$(runtime_for "${accel}")"
        --metadata-from-file=startup-script=startup.sh,idle-reaper=idle_reaper.sh
        --metadata=idle-timeout-min="${IDLE_TIMEOUT_MIN}",mrx-branch="${MRX_BRANCH}"
    )
    [[ "${model}" == "SPOT" ]] && args+=(--spot)
    if (( ! NO_DISK )) && data_disk_exists "${zone}"; then
        args+=("--data-disk=source=projects/${PROJECT}/zones/${zone}/disks/${DATA_DISK},mode=read-write")
    fi
    gcloud compute tpus tpu-vm create "${args[@]}" >"${log}" 2>&1
}

# Standing requests through the Queued Resources API.
#
# A sweep asks "is there capacity this instant", and in a stockout it loses that
# race to whoever else is asking, for hours at a time. A queued resource asks
# instead to be placed in Google's own admission queue and filled when a slice
# frees, whether or not we happen to be awake for the moment it does. The two
# are complementary and run together: the sweep can still win a node outright
# while the requests wait.
#
# A fulfilled request creates a node named VM_NAME carrying the same metadata,
# runtime and data disk a direct create gives it, so idle_reaper.sh is installed
# exactly as it would be otherwise. That matters more here than on the sweep,
# because the node can appear at a moment nobody is watching, and the reaper is
# the only thing bounding the bill.
queue_request_name() {
    local accel="$1" model="$2"
    echo "${VM_NAME}-q-${accel}-$(printf '%s' "${model}" | tr '[:upper:]' '[:lower:]')"
}

# One request per queueable ZONE, not per rung. Every request asks for a node
# called VM_NAME, and a zone holds only one node of that name, so a second
# request there can do nothing but fail:
#
#   us-south1-a  mrx-remeasure-q-v5litepod-1-ondemand
#     FAILED  already_exists: node 'mrx-remeasure' already exists
#
# measured 2026-09-05, after a second request in that zone had already
# taken the name. CANDIDATES is ordered best-first, so keeping the first rung
# seen per zone keeps the best one and costs nothing. Only the Cloud TPU API
# path has a queue at all; the GCE rungs are skipped. Re-filing an existing
# request is not an error, so this is safe to call on every pass.
file_queued_requests() {
    local entry gen mt zone model api name errlog kind seen=""
    local filed=0 already=0 refused=0
    errlog="$(mktemp)"

    for entry in "${CANDIDATES[@]}"; do
        IFS=':' read -r gen mt zone model api <<<"${entry}"
        [[ "${api}" == "tpuapi" ]] || continue
        case " ${seen} " in *" ${zone} "*) continue ;; esac
        seen="${seen} ${zone}"
        name="$(queue_request_name "${mt}" "${model}")"

        # shellcheck disable=SC2054  # commas are inside one argument, as above
        local args=(
            "${name}"
            --zone="${zone}"
            --node-id="${VM_NAME}"
            --accelerator-type="${mt}"
            --runtime-version="$(runtime_for "${mt}")"
            --valid-until-duration="${QUEUE_VALID_FOR}"
            --metadata-from-file=startup-script=startup.sh,idle-reaper=idle_reaper.sh
            --metadata=idle-timeout-min="${IDLE_TIMEOUT_MIN}",mrx-branch="${MRX_BRANCH}"
        )
        [[ "${model}" == "SPOT" ]] && args+=(--spot)
        if data_disk_exists "${zone}"; then
            args+=("--data-disk=source=projects/${PROJECT}/zones/${zone}/disks/${DATA_DISK},mode=read-write")
        fi

        if gcloud compute tpus queued-resources create "${args[@]}" >"${errlog}" 2>&1; then
            filed=$(( filed + 1 ))
            continue
        fi

        if rg -q "already exists|ALREADY_EXISTS" "${errlog}"; then
            already=$(( already + 1 ))
        else
            # NO_QUEUEING is the expected answer for rungs whose accelerator
            # cannot be queued in that location; it says nothing about capacity
            # and is not worth a line each pass.
            kind="$(classify_failure "${errlog}")"
            refused=$(( refused + 1 ))
            [[ "${kind}" == "NO_QUEUEING" ]] || \
                printf '  %-16s %-14s %-8s %s\n' "${zone}" "${mt}" "${model}" "${kind}"
        fi
    done

    rm -f "${errlog}"
    log "queued requests: ${filed} filed, ${already} already standing, ${refused} refused"
}

# Delete every request we filed. Deliberately without --force: the API refuses
# to delete a request that holds a node, so a request that has just been
# fulfilled survives this untouched and cannot be cancelled out from under the
# node it won.
cancel_queued_requests() {
    local entry gen mt zone model api name
    for entry in "${CANDIDATES[@]}"; do
        IFS=':' read -r gen mt zone model api <<<"${entry}"
        [[ "${api}" == "tpuapi" ]] || continue
        name="$(queue_request_name "${mt}" "${model}")"
        gcloud compute tpus queued-resources delete "${name}" --zone="${zone}" \
            --quiet >/dev/null 2>&1 &
    done
    wait
}

# First fulfilment wins; everything else is torn down at once.
#
# Requests are only cancelled by the EXIT trap, so between a fulfilment and the
# daemon exiting every other request is still standing and can be filled too.
# That window is not theoretical: on 2026-09-05 a v5litepod-1 came up in
# us-east1-c while another request was coming up in us-west1-c, and two
# nodes billed at once. Cancelling on the spot closes it.
#
# The winner's own request is left alone. cancel_queued_requests deletes
# without --force and the API refuses to delete a request holding a node, so
# the winner survives that call; the losers are deleted here by name, since a
# request that has already produced a node cannot be cancelled out of it.
claim_first_node() {
    local winner="$1"
    local entry gen mt zone model api seen=""

    cancel_queued_requests

    for entry in "${CANDIDATES[@]}"; do
        IFS=':' read -r gen mt zone model api <<<"${entry}"
        [[ "${api}" == "tpuapi" ]] || continue
        [[ "${zone}" == "${winner}" ]] && continue
        case " ${seen} " in *" ${zone} "*) continue ;; esac
        seen="${seen} ${zone}"
        if gcloud compute tpus tpu-vm describe "${VM_NAME}" --zone="${zone}" \
             --format="value(name)" >/dev/null 2>&1; then
            log "Tearing down a second node in ${zone} (${winner} won)"
            gcloud compute tpus tpu-vm delete "${VM_NAME}" --zone="${zone}" \
                --quiet >/dev/null 2>&1 &
        fi
    done
    wait
}

announce_success() {
    local gen="$1" mt="$2" zone="$3" model="$4" api="$5"
    echo ""
    if [[ "${api}" == "tpuapi" ]]; then
        gcloud compute tpus tpu-vm describe "${VM_NAME}" --zone="${zone}" \
            --format="table(name.basename(),state,acceleratorType,runtimeVersion)"
        echo ""
        echo "NOTE: Cloud TPU API nodes have no max-run-duration. This one is"
        echo "      bounded only by idle_reaper.sh, which deletes it after"
        echo "      ${IDLE_TIMEOUT_MIN} idle minutes. To go now, or if the reaper"
        echo "      did not install:"
        echo "        gcloud compute tpus tpu-vm delete ${VM_NAME} --zone=${zone}"
    else
        gcloud compute instances describe "${VM_NAME}" --zone="${zone}" \
            --format="table(name,status,machineType.basename(),scheduling.provisioningModel)"
    fi
    echo ""
    echo "Generation: ${gen} (${mt}), zone ${zone}, ${model} via ${api}"
}

try_create() {
    local mt="$1" zone="$2" model="$3" api="$4" log="$5"
    if [[ "${api}" == "tpuapi" ]]; then
        create_tpuapi "${mt}" "${zone}" "${model}" "${log}"
    else
        build_args "${mt}" "${zone}" "${model}"
        gcloud compute instances create "${CREATE_ARGS[@]}" >"${log}" 2>&1
    fi
}

# Walk the ladder once, fail-fast on each candidate. Returns 0 the moment a node
# is created, 1 if nothing had capacity.
#
# Always called inside a subshell, because it sets NO_DISK as it goes; in bash a
# `VAR=x func` prefix on a *function* leaks the assignment into the caller, so
# isolating it in a process is the only way the daemon's own state survives a
# sweep intact. The EXIT trap set here likewise replaces the lock-file trap only
# within it.
walk_ladder() {
    build_candidates
    if (( ${#CANDIDATES[@]} == 0 )); then
        echo "No candidates match the current filters." >&2
        return 1
    fi

    printf 'Walking %d candidates, fail-fast on each\n\n' "${#CANDIDATES[@]}"

    local log_dir
    log_dir="$(mktemp -d)"
    # Expanded now, not at trap time: log_dir is local to this function, which
    # has already returned by the time the subshell fires its EXIT trap, so the
    # deferred form died on `set -u` and leaked the directory every sweep.
    # shellcheck disable=SC2064
    trap "rm -rf '${log_dir}'" EXIT

    local entry gen mt zone model api log kind
    for entry in "${CANDIDATES[@]}"; do
        IFS=':' read -r gen mt zone model api <<<"${entry}"
        printf '[%-3s %-16s %-11s %-6s] ' "${gen}" "${zone}" "${mt}" "${api}"

        log="${log_dir}/${gen}-${zone}-${mt}-${model}.log"

        if try_create "${mt}" "${zone}" "${model}" "${api}" "${log}"; then
            echo "SUCCESS"
            attach_disk_after_create "${zone}" "${api}"
            announce_success "${gen}" "${mt}" "${zone}" "${model}" "${api}"
            return 0
        fi

        kind="$(classify_failure "${log}")"

        # An expired credential fails every zone identically and instantly, so
        # walking the rest of the ladder learns nothing. Returning 2 stops the
        # daemon rather than letting it sweep all night reporting "no capacity"
        # when nothing was ever asked. It swept for four hours that way once.
        if [[ "${kind}" == "AUTH" ]]; then
            explain_failure "${kind}" "${log}"
            return 2
        fi

        # A transient Google-side blip says nothing about capacity; a disk-type
        # rejection is worth retrying without the disk, since losing persistence
        # beats losing the zone. v5p rejects hyperdisk-balanced outright.
        if [[ "${kind}" == "TRANSIENT" || "${kind}" == "DISK_INCOMPATIBLE" ]]; then
            explain_failure "${kind}" "${log}"
            printf '%*s' 45 ''
            [[ "${kind}" == "TRANSIENT" ]] && sleep 5
            [[ "${kind}" == "DISK_INCOMPATIBLE" ]] && NO_DISK=1

            if try_create "${mt}" "${zone}" "${model}" "${api}" "${log}"; then
                echo "SUCCESS (retry)"
                # A DISK_INCOMPATIBLE retry deliberately created without the
                # disk, so without the force flag the node comes up with no
                # persistent environment and nothing says so.
                attach_disk_after_create "${zone}" "${api}" "${NO_DISK}"
                announce_success "${gen}" "${mt}" "${zone}" "${model}" "${api}"
                return 0
            fi
            kind="$(classify_failure "${log}")"
            NO_DISK=0
        fi
        explain_failure "${kind}" "${log}"
    done

    echo ""
    echo "No candidate has capacity right now."
    return 1
}

running_zone() {
    gcloud compute instances list --filter="name=${VM_NAME} AND status=RUNNING" \
        --format="value(zone.basename())" 2>/dev/null | head -n 1
}

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
}

log "=============================================================="
log "acquire_tpu.sh starting (max ${MAX_HOURS}h, ${MAX_SESSIONS} session(s))"
build_candidates
log "${#CANDIDATES[@]} candidates; best is ${CANDIDATES[0]}"

if (( QUEUE )); then
    # Replaces the lock-file trap rather than adding to it, so both actions are
    # restated here.
    trap 'cancel_queued_requests; rm -f "${LOCK_FILE}"' EXIT INT TERM
    log "Filing standing queued-resource requests (valid ${QUEUE_VALID_FOR})"
    file_queued_requests
fi

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
    if [[ -z "${zone}" ]] && (( QUEUE )); then
        # A fulfilled request arrives as a Cloud TPU API node named VM_NAME,
        # which the GCE listing above cannot see.
        zone="$(tpu_running_zone "${VM_NAME}" 1)"
        if [[ -n "${zone}" ]]; then
            export TPU_API=1
            log "A queued request was fulfilled in ${zone}"
            claim_first_node "${zone}"
        fi
    fi
    if [[ -n "${zone}" ]]; then
        run_session "${zone}"
        (( ONCE )) && exit 0
        continue
    fi

    log "Sweeping the ladder..."
    ( walk_ladder ) >>"${ACQUIRE_LOG}" 2>&1
    sweep_status=$?

    if (( sweep_status == 2 )); then
        log "gcloud credential expired. Run 'gcloud auth login' and start again."
        notify "TPU daemon stopped" "gcloud credential expired"
        exit 1
    fi

    if (( sweep_status == 0 )); then
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
        log "A node was created but no zone reports it RUNNING yet."
    else
        log "No capacity on the ladder."
    fi

    if (( ONCE )); then
        log "--once given; not looping."
        exit 1
    fi

    # Re-filing is a no-op for a request already standing, so this costs one
    # ALREADY_EXISTS per rung and keeps the set whole as requests expire or are
    # cancelled Google-side.
    (( QUEUE )) && file_queued_requests

    log "Sleeping ${SWEEP_INTERVAL}s before the next attempt"
    sleep "${SWEEP_INTERVAL}"
done
