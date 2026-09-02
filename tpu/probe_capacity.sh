#!/usr/bin/env bash
#
# Probe TPU capacity across the candidate ladder.
#
# Google exposes no queue-position or global-capacity API: `gcloud alpha compute
# advice capacity` needs Compute Alpha allowlisting and `advice calendar-mode`
# rejects the ct6e family. A Spot create is the usable substitute. Spot never
# queues, so it fails within seconds and names the reason, which separates a
# hardware stockout from a quota problem -- something a PENDING flex-start
# request hides completely.
#
# Run check_quota.sh first; it rules out quota, subnet and machine-type
# problems for free, leaving this script to answer the one question that
# genuinely requires touching the API.
#
# Usage:
#   ./probe_capacity.sh                    # probe every viable candidate
#   GENERATIONS=v5e ./probe_capacity.sh
#   ZONES=us-east5-b,us-east5-c ./probe_capacity.sh
#   KEEP_LOGS=1 ./probe_capacity.sh        # preserve raw gcloud output
#   MAX_PARALLEL=8 ./probe_capacity.sh
#
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=zones.sh
source ./zones.sh

MAX_PARALLEL="${MAX_PARALLEL:-6}"
KEEP_LOGS="${KEEP_LOGS:-0}"

if [[ "${KEEP_LOGS}" == "1" ]]; then
    LOG_DIR="probe_logs/$(date +%Y%m%dT%H%M%S)"
    mkdir -p "${LOG_DIR}"
else
    LOG_DIR="$(mktemp -d)"
fi

RESULT_DIR="$(mktemp -d)"

# Any probe that succeeds has created a real, billable TPU VM. Sweep up every
# instance this script could possibly have made, no matter how we exit.
# Any probe that succeeds has created a real, billable TPU. Sweep up every
# resource this script could possibly have made, no matter how we exit. The
# ".created" file records the api too, because a Cloud TPU API node is not a
# GCE instance and `compute instances delete` will not touch it.
cleanup() {
    local f zone name api
    for f in "${RESULT_DIR}"/*.created; do
        [[ -f "${f}" ]] || continue
        IFS=' ' read -r name zone api <"${f}"
        [[ -z "${name}" ]] && continue
        echo "Cleaning up ${name} in ${zone}..."
        delete_probe "${name}" "${zone}" "${api}"
        rm -f "${f}"
    done
    rm -rf "${RESULT_DIR}"
    [[ "${KEEP_LOGS}" == "1" ]] || rm -rf "${LOG_DIR}"
}
trap cleanup EXIT INT TERM

delete_probe() {
    local name="$1" zone="$2" api="$3"
    if [[ "${api}" == "tpuapi" ]]; then
        gcloud compute tpus tpu-vm delete "${name}" --zone="${zone}" \
            --quiet >/dev/null 2>&1
    else
        gcloud compute instances delete "${name}" --zone="${zone}" \
            --quiet >/dev/null 2>&1
    fi
}

# Spot regardless of the candidate's own provisioning model: this is a capacity
# question, and spot answers it fastest and cheapest.
create_probe() {
    local name="$1" mt="$2" zone="$3" api="$4" log="$5"
    if [[ "${api}" == "tpuapi" ]]; then
        # No --max-run-duration exists on this API, so the cleanup trap is the
        # only safety net; keep it in mind before adding an early return above.
        gcloud compute tpus tpu-vm create "${name}" \
            --zone="${zone}" \
            --accelerator-type="${mt}" \
            --version="${TPU_RUNTIME}" \
            --spot \
            >"${log}" 2>&1
    else
        gcloud compute instances create "${name}" \
            --zone="${zone}" \
            --machine-type="${mt}" \
            --provisioning-model=SPOT \
            --instance-termination-action=DELETE \
            --max-run-duration=10m \
            --image-project="${IMAGE_PROJECT}" \
            --image-family="${IMAGE_FAMILY}" \
            --maintenance-policy=TERMINATE \
            >"${log}" 2>&1
    fi
}

# Probe one candidate. Writes "<verdict>\t<detail>" to the result file.
probe_one() {
    local entry="$1" idx="$2"
    local gen mt zone model api
    IFS=':' read -r gen mt zone model api <<<"${entry}"

    local name="capacity-probe-$$-${idx}"
    local log="${LOG_DIR}/${gen}-${zone}-${mt}-${api}.log"
    local result="${RESULT_DIR}/${idx}.result"
    local created="${RESULT_DIR}/${idx}.created"

    local attempt kind detail
    for attempt in 1 2; do
        if create_probe "${name}" "${mt}" "${zone}" "${api}" "${log}"; then
            echo "${name} ${zone} ${api}" >"${created}"
            printf 'AVAILABLE\tcapacity exists (spot create succeeded)\n' >"${result}"
            delete_probe "${name}" "${zone}" "${api}"
            rm -f "${created}"
            return
        fi

        kind="$(classify_failure "${log}")"
        # A Google-side internal error says nothing about capacity; us-east1-b
        # produced one on the first sweep and was recorded as a dead zone.
        if [[ "${kind}" == "TRANSIENT" && "${attempt}" == "1" ]]; then
            sleep 5
            continue
        fi
        break
    done

    detail="$(explain_failure "${kind}" "${log}")"
    printf '%s\t%s\n' "${kind}" "${detail}" >"${result}"
}

build_candidates

# One probe per (generation, zone) pair; the provisioning model does not change
# the answer, so probing both STANDARD and SPOT rows would just double the cost.
# Dedup via a string rather than an associative array: macOS ships bash 3.2.
UNIQUE=()
seen=""
for entry in "${CANDIDATES[@]}"; do
    IFS=':' read -r gen mt zone model api <<<"${entry}"
    # Keyed on the type too, not just the generation: v5litepod-1 and
    # v5litepod-4 draw on different amounts of the same pool, so one being
    # stocked out says little about the other.
    key="${gen}:${mt}:${zone}:${api}"
    case " ${seen} " in *" ${key} "*) continue ;; esac
    seen="${seen} ${key}"
    UNIQUE+=("${gen}:${mt}:${zone}:SPOT:${api}")
done

printf 'Probing %d type/zone combinations, %d at a time\n' \
    "${#UNIQUE[@]}" "${MAX_PARALLEL}"
printf 'Any probe that succeeds is deleted immediately.\n'
[[ "${KEEP_LOGS}" == "1" ]] && printf 'Logs: %s\n' "${LOG_DIR}"
echo ""

# Batched rather than a rolling pool, because `wait -n` needs bash 4.3.
total="${#UNIQUE[@]}"
i=0
while (( i < total )); do
    batch_end=$(( i + MAX_PARALLEL ))
    (( batch_end > total )) && batch_end="${total}"
    for (( j = i; j < batch_end; j++ )); do
        probe_one "${UNIQUE[$j]}" "$j" &
    done
    wait
    i="${batch_end}"
done

printf '%-5s %-17s %-15s %-11s %s\n' "GEN" "TYPE" "ZONE" "RESULT" "DETAIL"
printf -- '----------------------------------------------------------------------------------------\n'

available=()
for i in "${!UNIQUE[@]}"; do
    IFS=':' read -r gen mt zone model api <<<"${UNIQUE[$i]}"
    result="${RESULT_DIR}/${i}.result"
    if [[ -f "${result}" ]]; then
        IFS=$'\t' read -r verdict detail <"${result}"
    else
        verdict="OTHER"; detail="no result recorded"
    fi
    printf '%-5s %-17s %-15s %-11s %s\n' "${gen}" "${mt}" "${zone}" "${verdict}" "${detail}"
    [[ "${verdict}" == "AVAILABLE" ]] && available+=("${gen}:${mt}:${zone}")
done

echo ""
if (( ${#available[@]} > 0 )); then
    printf 'Capacity found: %s\n' "${available[*]}"
    IFS=':' read -r gen mt zone <<<"${available[0]}"
    printf 'Launch with:  GENERATIONS=%s MACHINE_TYPE=%s ZONES=%s ./launch_tpu.sh\n' \
        "${gen}" "${mt}" "${zone}"
else
    printf 'No candidate has capacity right now.\n'
    printf 'Start the daemon to keep trying:  ./acquire_tpu.sh\n'
fi
