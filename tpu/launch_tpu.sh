#!/usr/bin/env bash
#
# Launch a TPU VM by walking the candidate ladder, without hanging.
#
# The command from the setup doc blocks for up to two hours because
# --request-valid-for-duration=2h puts the instance in PENDING while Dynamic
# Workload Scheduler waits for capacity, and gcloud waits with it. Here every
# attempt fails fast instead: STANDARD and SPOT do so naturally, and FLEX_START
# is given --request-valid-for-duration=0 so it allocates only if resources are
# free right now. A success is the real VM, so there is no probe-then-claim race.
#
# Usage:
#   ./launch_tpu.sh                        # walk the whole ladder
#   GENERATIONS=v5e ./launch_tpu.sh        # v5e only
#   ZONES=us-east5-b ./launch_tpu.sh
#   ./launch_tpu.sh --park                 # skip the sweep, park a 2h request
#
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=zones.sh
source ./zones.sh

PARK_ONLY=0
[[ "${1:-}" == "--park" ]] && PARK_ONLY=1

if [[ ! -f startup.sh ]]; then
    echo "ERROR: startup.sh not found next to this script." >&2
    exit 1
fi

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
    echo "Next:  ZONE=${zone} TPU_API=$([[ ${api} == tpuapi ]] && echo 1 || echo 0) ./run_on_tpu.sh"
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
            echo "Check it with:  ./watch_request.sh"
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

build_candidates
if (( ${#CANDIDATES[@]} == 0 )); then
    echo "No candidates match the current filters." >&2
    exit 1
fi

if (( PARK_ONLY )); then
    park_request
    exit $?
fi

printf 'Walking %d candidates, fail-fast on each\n\n' "${#CANDIDATES[@]}"

LOG_DIR="$(mktemp -d)"
trap 'rm -rf "${LOG_DIR}"' EXIT

try_create() {
    local mt="$1" zone="$2" model="$3" api="$4" log="$5"
    if [[ "${api}" == "tpuapi" ]]; then
        create_tpuapi "${mt}" "${zone}" "${model}" "${log}"
    else
        build_args "${mt}" "${zone}" "${model}" 0
        gcloud compute instances create "${CREATE_ARGS[@]}" >"${log}" 2>&1
    fi
}

for entry in "${CANDIDATES[@]}"; do
    IFS=':' read -r gen mt zone model api <<<"${entry}"
    printf '[%-3s %-16s %-11s %-6s] ' "${gen}" "${zone}" "${mt}" "${api}"

    log="${LOG_DIR}/${gen}-${zone}-${mt}-${model}.log"

    if try_create "${mt}" "${zone}" "${model}" "${api}" "${log}"; then
        echo "SUCCESS"
        [[ "${api}" == "gce" ]] && attach_disk_after_create "${zone}"
        announce_success "${gen}" "${mt}" "${zone}" "${model}" "${api}"
        exit 0
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
            # The retry needs the same disk handling as the first attempt, and
            # more of it: a DISK_INCOMPATIBLE retry deliberately created without
            # the disk, so without this the node comes up with no persistent
            # environment and nothing says so.
            [[ "${api}" == "gce" ]] && \
                attach_disk_after_create "${zone}" "${NO_DISK}"
            announce_success "${gen}" "${mt}" "${zone}" "${model}" "${api}"
            exit 0
        fi
        kind="$(classify_failure "${log}")"
        NO_DISK=0
    fi
    explain_failure "${kind}" "${log}"
done

echo ""
echo "No candidate has capacity right now."
echo "Falling back to a parked 2h queued request."
park_request
