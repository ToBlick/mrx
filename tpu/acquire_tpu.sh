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
# The ladder walk itself is here too rather than in its own script. The command
# from the setup doc blocks for up to two hours because
# --request-valid-for-duration=2h puts the instance in PENDING while Dynamic
# Workload Scheduler waits for capacity, and gcloud waits with it. Every attempt
# below fails fast instead: STANDARD and SPOT do so naturally, and FLEX_START is
# given --request-valid-for-duration=0 so it allocates only if resources are
# free right now. A success is the real VM, so there is no probe-then-claim race.
#
# Usage:
#   ./acquire_tpu.sh                  # loop until acquired or budget spent
#   ./acquire_tpu.sh --once           # a single sweep, no parking or looping
#   ./acquire_tpu.sh --park           # skip the sweep, park a 2h queued request
#   ./acquire_tpu.sh --watch          # report on queued requests, then stop
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

ONCE=0
GC_ONLY=0
PARK_ONLY=0
WATCH=0
# --acquire-only stops after the hardware lands instead of running MRX. Used
# when the intended job is not the Poisson report -- otherwise acquiring a node
# for, say, a relaxation first burns several minutes and dollars on a run that
# is about to be superseded.
ACQUIRE_ONLY=0
for arg in "$@"; do
    case "${arg}" in
        --once)         ONCE=1 ;;
        --gc)           GC_ONLY=1 ;;
        --park)         PARK_ONLY=1 ;;
        --watch)        WATCH=1 ;;
        --acquire-only) ACQUIRE_ONLY=1 ;;
    esac
done

if [[ ! -f startup.sh ]]; then
    echo "ERROR: startup.sh not found next to this script." >&2
    exit 1
fi

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

# ------------------------------------------------------------------- watch ---
# A PENDING flex-start instance and a RUNNING insert operation with progress 0
# are what a queued DWS request looks like from outside. Google publishes no
# queue position, so this shows everything that is actually observable: which
# requests are live, how long they have left before their wait time expires, and
# the failure reason on anything that has already resolved. Read-only, so it
# runs before the lock and does not disturb a daemon that is already looping.
watch_report() {
    echo "=================================================================="
    date '+%Y-%m-%d %H:%M:%S %Z'
    echo "=================================================================="

    echo ""
    echo "--- Instances ---"
    local instances
    instances="$(gcloud compute instances list \
        --format="table(name,zone.basename(),machineType.basename(),status,creationTimestamp)" 2>&1)"
    if [[ -z "${instances}" ]]; then
        echo "(none)"
    else
        echo "${instances}"
    fi

    echo ""
    echo "--- Live queued requests (RUNNING insert operations) ---"
    local ops
    ops="$(gcloud compute operations list \
        --filter="status=RUNNING AND operationType=insert" \
        --format="value(name,zone.basename(),targetLink.basename(),insertTime)" 2>&1)"

    if [[ -z "${ops}" ]]; then
        echo "(none queued)"
    else
        printf '%-14s %-22s %-10s %s\n' "TARGET" "SUBMITTED" "ELAPSED" "ZONE"
        while IFS=$'\t' read -r name zone target inserted; do
            [[ -z "${name}" ]] && continue
            local start_epoch now_epoch elapsed
            # BSD date on macOS, GNU date elsewhere.
            start_epoch="$(date -j -f "%Y-%m-%dT%H:%M:%S" "${inserted:0:19}" +%s 2>/dev/null \
                || date -d "${inserted}" +%s 2>/dev/null || echo 0)"
            now_epoch="$(date +%s)"
            if [[ "${start_epoch}" != "0" ]]; then
                elapsed=$(( (now_epoch - start_epoch) / 60 ))
                elapsed="${elapsed}m"
            else
                elapsed="?"
            fi
            printf '%-14s %-22s %-10s %s\n' \
                "${target}" "${inserted:0:19}" "${elapsed}" "${zone}"
        done <<<"${ops}"
        echo ""
        echo "A flex-start wait time is capped at 2h; Compute Engine deletes the"
        echo "instance when it expires without capacity."
    fi

    echo ""
    echo "--- Most recent resolved insert operations ---"
    # CSV rather than the default tab-separated `value(...)`: a missing
    # httpErrorStatusCode yields an empty field, and consecutive tabs collapse
    # under IFS word-splitting, which shifts every later column.
    gcloud compute operations list \
        --filter="operationType=insert AND status=DONE" \
        --sort-by=~insertTime --limit=8 \
        --format="csv[no-heading](targetLink.basename(),zone.basename(),httpErrorStatusCode,error.errors[0].code,insertTime)" 2>&1 \
        | while IFS=',' read -r target zone code errcode inserted; do
            [[ -z "${target}" ]] && continue
            if [[ -z "${code}" || "${code}" == "None" ]]; then
                printf '  %-16s %-16s %-5s %-19s %s\n' \
                    "${target}" "${zone}" "OK" "${inserted:0:19}" ""
            else
                printf '  %-16s %-16s %-5s %-19s %s\n' \
                    "${target}" "${zone}" "${code}" "${inserted:0:19}" "${errcode}"
            fi
        done
    echo ""
}

if (( WATCH )); then
    watch_report
    exit 0
fi

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

# =============================================================== launching ===

# Build the argv for a create.
#
# auto-delete=no on the data disk is mandatory, not defensive. gcloud documents
# the --disk auto-delete default as *yes*, so combined with
# --instance-termination-action=DELETE the original command would have destroyed
# my-data-disk the moment the run duration expired.
# Set NO_DISK=1 to build a create that does not attach the data disk. Used to
# retry machine types that reject the disk's type outright.
NO_DISK=0

build_args() {
    local mt="$1" zone="$2" model="$3" valid_for="${4:-}"
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
    # Only FLEX_START takes a wait time; passing it elsewhere is rejected.
    if [[ "${model}" == "FLEX_START" ]]; then
        CREATE_ARGS+=(--request-valid-for-duration="${valid_for:-0}")
    fi
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
# With force=1 the disk already exists but was refused at create time, which is
# what DISK_INCOMPATIBLE means: v5p rejects hyperdisk-balanced as a boot-time
# attachment. Skip the existence short-circuit and try a hot attach, which is a
# separate code path and may be accepted where the create-time one was not. If
# it is not, the warning is the point: that node has no persistent environment.
attach_disk_after_create() {
    local zone="$1" force="${2:-0}"
    (( ! force )) && data_disk_exists "${zone}" && return 0

    if data_disk_exists "${zone}"; then
        echo "  ${DATA_DISK} was refused at create time; trying a hot attach..."
    else
        echo "  no ${DATA_DISK} in ${zone}; creating and attaching it now..."
        if ! ensure_data_disk "${zone}"; then
            echo "  WARNING: could not create ${DATA_DISK}; environment will not persist" >&2
            return 1
        fi
    fi
    gcloud compute instances attach-disk "${VM_NAME}" \
        --zone="${zone}" --disk="${DATA_DISK}" \
        --device-name=data-disk --mode=rw --quiet >/dev/null 2>&1 || {
        echo "  WARNING: attach-disk failed" >&2; return 1; }

    gcloud compute instances set-disk-auto-delete "${VM_NAME}" \
        --zone="${zone}" --no-auto-delete --device-name=data-disk \
        --quiet >/dev/null 2>&1

    echo "  attached; re-running the startup script to build the environment"
    gcloud compute ssh "${VM_NAME}" --zone="${zone}" \
        --command="sudo google_metadata_script_runner startup" \
        -- -o StrictHostKeyChecking=no >/dev/null 2>&1 &
}

# The same job on the Cloud TPU API, which needs its own function because that
# surface differs twice over. `tpu-vm create --data-disk` can only reference a
# disk that already exists, so a zone that has never held one gets nothing;
# and `tpu-vm update --attach-disk` takes no device name, so the disk lands as
# google-persistent-disk-N rather than google-data-disk (startup.sh looks for
# both).
#
# Deferred to here rather than folded into create_tpuapi because the ladder is
# fail-fast across a dozen zones: creating a 100 GB disk per attempt would cost
# half a minute each and leave copies in zones that never produced a node. By
# this point exactly one zone has, so the disk is created where it will be used.
attach_disk_after_create_tpuapi() {
    local zone="$1"
    (( NO_DISK )) && return 0

    if ! data_disk_exists "${zone}"; then
        echo "  no ${DATA_DISK} in ${zone}; creating it from ${DATA_SNAPSHOT}..."
        if ! ensure_data_disk "${zone}"; then
            echo "  WARNING: could not create ${DATA_DISK}; the environment will" >&2
            echo "           be rebuilt from scratch and will not persist" >&2
            return 1
        fi
    fi

    echo "  attaching ${DATA_DISK} to the node..."
    gcloud compute tpus tpu-vm update "${VM_NAME}" --zone="${zone}" \
        --attach-disk "source=projects/${PROJECT}/zones/${zone}/disks/${DATA_DISK},mode=read-write" \
        --quiet >/dev/null 2>&1 || {
        echo "  WARNING: attach-disk failed; the environment will not persist" >&2
        return 1; }

    # The startup script has already run against the boot disk by now. Re-run it
    # so it mounts what we just attached: if the disk came from the snapshot the
    # sentinel is on it and the build is skipped entirely.
    echo "  attached; re-running the startup script against the persistent disk"
    gcloud compute tpus tpu-vm ssh "${VM_NAME}" --zone="${zone}" \
        --command="sudo google_metadata_script_runner startup" \
        --quiet >/dev/null 2>&1 &
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
    local args=(
        "${VM_NAME}"
        --zone="${zone}"
        --accelerator-type="${accel}"
        --version="${TPU_RUNTIME}"
        --metadata-from-file=startup-script=startup.sh,idle-reaper=idle_reaper.sh
        --metadata=idle-timeout-min="${IDLE_TIMEOUT_MIN}",mrx-branch="${MRX_BRANCH}"
    )
    case "${model}" in
        SPOT)        args+=(--spot) ;;
        PREEMPTIBLE) args+=(--preemptible) ;;
    esac
    if (( ! NO_DISK )) && data_disk_exists "${zone}"; then
        args+=("--data-disk=source=projects/${PROJECT}/zones/${zone}/disks/${DATA_DISK},mode=read-write")
    fi
    gcloud compute tpus tpu-vm create "${args[@]}" >"${log}" 2>&1
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

# Parking only exists on the GCE flex-start path; the Cloud TPU API has no
# equivalent queue. Pick the first gce candidate rather than the first candidate
# overall, which is now a v5p entry.
park_request() {
    local entry gen mt zone model api
    for entry in "${CANDIDATES[@]}"; do
        IFS=':' read -r gen mt zone model api <<<"${entry}"
        [[ "${api}" == "gce" ]] && break
        api=""
    done
    if [[ -z "${api:-}" ]]; then
        echo "No GCE candidate to park a queued request against." >&2
        return 1
    fi

    echo ""
    echo "Parking a 2h queued request: ${gen} (${mt}) in ${zone}..."
    ensure_data_disk "${zone}" >/dev/null 2>&1 || true

    local plog
    plog="$(mktemp)"
    local attempt
    for attempt in 1 2; do
        build_args "${mt}" "${zone}" FLEX_START 2h
        if gcloud compute instances create "${CREATE_ARGS[@]}" --async \
            >"${plog}" 2>&1; then
            tail -n 2 "${plog}"
            echo ""
            echo "Request queued; this terminal is not blocked."
            echo "Check it with:  ./acquire_tpu.sh --watch"
            rm -f "${plog}"; NO_DISK=0
            return 0
        fi
        # Same disk-type rejection as the sweep: drop the disk and try again.
        if [[ "$(classify_failure "${plog}")" == "DISK_INCOMPATIBLE" && "${attempt}" == "1" ]]; then
            echo "  ${DATA_DISK} rejected by ${mt}; parking without it"
            NO_DISK=1
            continue
        fi
        break
    done
    echo "  park failed: $(explain_failure "$(classify_failure "${plog}")" "${plog}")"
    rm -f "${plog}"; NO_DISK=0
    return 1
}

try_create() {
    local mt="$1" zone="$2" model="$3" api="$4" log="$5"
    if [[ "${api}" == "tpuapi" ]]; then
        create_tpuapi "${mt}" "${zone}" "${model}" "${log}"
    else
        build_args "${mt}" "${zone}" "${model}" 0
        gcloud compute instances create "${CREATE_ARGS[@]}" >"${log}" 2>&1
    fi
}

# Walk the ladder once, fail-fast on each candidate. Returns 0 the moment a node
# is created, 1 if nothing had capacity.
#
# Always called inside a subshell, because the loop below overrides VM_NAME and
# APIS for the v5e pass and sets NO_DISK as it goes; in bash a `VAR=x func` prefix
# on a *function* leaks the assignment into the caller, so isolating it in a
# process is the only way the daemon's own state survives a sweep intact. The
# EXIT trap set here likewise replaces the lock-file trap only within it.
walk_ladder() {
    build_candidates
    if (( ${#CANDIDATES[@]} == 0 )); then
        echo "No candidates match the current filters." >&2
        return 1
    fi

    printf 'Walking %d candidates, fail-fast on each\n\n' "${#CANDIDATES[@]}"

    local log_dir
    log_dir="$(mktemp -d)"
    trap 'rm -rf "${log_dir}"' EXIT

    local entry gen mt zone model api log kind
    for entry in "${CANDIDATES[@]}"; do
        IFS=':' read -r gen mt zone model api <<<"${entry}"
        printf '[%-3s %-16s %-11s %-6s] ' "${gen}" "${zone}" "${mt}" "${api}"

        log="${log_dir}/${gen}-${zone}-${mt}-${model}.log"

        if try_create "${mt}" "${zone}" "${model}" "${api}" "${log}"; then
            echo "SUCCESS"
            if [[ "${api}" == "gce" ]]; then
                attach_disk_after_create "${zone}"
            else
                attach_disk_after_create_tpuapi "${zone}"
            fi
            announce_success "${gen}" "${mt}" "${zone}" "${model}" "${api}"
            return 0
        fi

        kind="$(classify_failure "${log}")"

        # A transient Google-side blip says nothing about capacity; a disk-type
        # rejection is worth retrying without the disk, since losing persistence
        # beats losing the zone. v5p rejects hyperdisk-balanced outright.
        if [[ "${kind}" == "TRANSIENT" || "${kind}" == "DISK_INCOMPATIBLE" ]]; then
            echo "$(explain_failure "${kind}" "${log}")"
            printf '%*s' 45 ''
            [[ "${kind}" == "TRANSIENT" ]] && sleep 5
            [[ "${kind}" == "DISK_INCOMPATIBLE" ]] && NO_DISK=1

            if try_create "${mt}" "${zone}" "${model}" "${api}" "${log}"; then
                echo "SUCCESS (retry)"
                # The retry needs the same disk handling as the first attempt,
                # and more of it: a DISK_INCOMPATIBLE retry deliberately created
                # without the disk, so without this the node comes up with no
                # persistent environment and nothing says so.
                if [[ "${api}" == "gce" ]]; then
                    attach_disk_after_create "${zone}" "${NO_DISK}"
                else
                    attach_disk_after_create_tpuapi "${zone}"
                fi
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

if (( PARK_ONLY )); then
    build_candidates
    park_request
    exit $?
fi

# ------------------------------------------------------------- current state ---
# Returns "name zone status" for an existing VM_NAME, if any.
running_zone() {
    gcloud compute instances list --filter="name=${VM_NAME} AND status=RUNNING" \
        --format="value(zone.basename())" 2>/dev/null | head -n 1
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
        ( APIS=tpuapi VM_NAME="${VM_NAME}-v5e"; walk_ladder ) \
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
    if ( walk_ladder ) >>"${ACQUIRE_LOG}" 2>&1; then
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
    elif (( ONCE )); then
        log "No capacity on the ladder; --once given, so not parking."
    else
        # Nothing had capacity. Park a 2h queued request so the DWS queue holds
        # a place while we sleep; it shows up as PENDING on the next pass.
        log "No capacity on the ladder; parking a 2h queued request"
        park_request >>"${ACQUIRE_LOG}" 2>&1 \
            && log "  request parked" \
            || log "  park failed; see ${ACQUIRE_LOG}"
    fi

    if (( ONCE )); then
        log "--once given; not looping."
        exit 1
    fi

    log "Sleeping ${SWEEP_INTERVAL}s before the next attempt"
    sleep "${SWEEP_INTERVAL}"
done
