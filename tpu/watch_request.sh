#!/usr/bin/env bash
#
# Report on queued flex-start requests without blocking.
#
# A PENDING flex-start instance and a RUNNING insert operation with progress 0
# are what a queued DWS request looks like from outside. Google publishes no
# queue position, so this shows everything that is actually observable: which
# requests are live, how long they have left before their wait time expires, and
# the failure reason on anything that has already resolved.
#
# Usage:
#   ./watch_request.sh          # one-shot report
#   ./watch_request.sh -w       # refresh every 30s
#
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

WATCH=0
[[ "${1:-}" == "-w" ]] && WATCH=1

report() {
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
    while true; do
        clear
        report
        echo "Refreshing in 30s (Ctrl-C to stop)..."
        sleep 30
    done
else
    report
fi
