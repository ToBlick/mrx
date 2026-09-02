#!/usr/bin/env bash
#
# Read-only preflight for the candidate ladder.
#
# Every create failure this project has hit falls out of three facts that can
# all be read without touching a VM: the effective quota for the generation in
# that region, whether the default VPC has a subnet there, and whether the
# machine type is offered in the zone. Checking them first turns what was a
# 25-minute blind sweep into a couple of seconds, and it is what surfaced that
# v6e -- not capacity in general -- was the blocker.
#
# Usage:
#   ./check_quota.sh                 # table plus the viable candidate list
#   ./check_quota.sh --list          # just the candidates, for scripts
#   ./check_quota.sh --refresh       # ignore the cache
#   GENERATIONS=v5e ./check_quota.sh
#
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=zones.sh
source ./zones.sh

CACHE_FILE="${CACHE_FILE:-.quota_cache.json}"
CACHE_TTL="${CACHE_TTL:-3600}"
LIST_ONLY=0
REFRESH=0
for arg in "$@"; do
    case "${arg}" in
        --list)    LIST_ONLY=1 ;;
        --refresh) REFRESH=1 ;;
    esac
done

PY="${PY:-python3}"
command -v "${PY}" >/dev/null 2>&1 || PY=/usr/bin/python3

# `gcloud beta quotas info describe` rejects the call without an explicit
# container flag, even when a default project is configured.
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
if [[ -z "${PROJECT}" ]]; then
    echo "ERROR: no project set; run 'gcloud config set project ...'" >&2
    exit 1
fi

log() { (( LIST_ONLY )) || echo "$@"; }

# --------------------------------------------------------------- cache ---
cache_age() {
    [[ -f "${CACHE_FILE}" ]] || { echo 999999; return; }
    local mtime now
    mtime="$(stat -f %m "${CACHE_FILE}" 2>/dev/null || stat -c %Y "${CACHE_FILE}" 2>/dev/null || echo 0)"
    now="$(date +%s)"
    echo $(( now - mtime ))
}

# The distinct zones in the ladder that use the Cloud TPU API.
tpuapi_zones() {
    local entry gen mt zone model api seen=""
    for entry in "${DEFAULT_CANDIDATES[@]}"; do
        IFS=':' read -r gen mt zone model api <<<"${entry}"
        [[ "${api}" == "tpuapi" ]] || continue
        case " ${seen} " in *" ${zone} "*) continue ;; esac
        seen="${seen} ${zone}"
        echo "${zone}"
    done
}

build_cache() {
    log "Querying quota, subnets, machine types and accelerator types..."
    local tmp
    tmp="$(mktemp)"

    {
        echo '{'
        # Quota buckets, one per generation in play.
        echo '"quota": {'
        local first=1 gen bucket
        for gen in v5e v5p v6e; do
            bucket="$(quota_bucket_for "${gen}")"
            (( first )) || echo ','
            first=0
            printf '"%s": ' "${gen}"
            gcloud beta quotas info describe "${bucket}" \
                --service=compute.googleapis.com \
                --project="${PROJECT}" \
                --format=json 2>/dev/null || echo 'null'
        done
        echo '},'

        echo '"subnet_regions": '
        gcloud compute networks subnets list --network=default \
            --format="json(region)" 2>/dev/null || echo '[]'
        echo ','

        echo '"machine_zones": '
        gcloud compute machine-types list \
            --filter="name~ct5lp-hightpu-4t OR name~ct5p-hightpu-4t OR name~ct6e-standard-4t" \
            --format="json(name,zone)" 2>/dev/null || echo '[]'
        echo ','

        # Cloud TPU API accelerator types (v5litepod-*, v6e-*) are NOT GCE
        # machine types and never appear in `machine-types list`, so they need
        # their own per-zone query. `accelerator-types list` takes one zone at
        # a time, hence the loop over the tpuapi zones in the ladder.
        echo '"accel_types": {'
        local afirst=1 az
        for az in $(tpuapi_zones); do
            (( afirst )) || echo ','
            afirst=0
            printf '"%s": ' "${az}"
            gcloud compute tpus accelerator-types list --zone="${az}" \
                --format="json(type)" 2>/dev/null || echo '[]'
        done
        echo '},'

        echo '"disk_zones": '
        gcloud compute disks list --filter="name=${DATA_DISK}" \
            --format="json(zone)" 2>/dev/null || echo '[]'

        echo '}'
    } >"${tmp}"

    mv "${tmp}" "${CACHE_FILE}"
}

if (( REFRESH )) || [[ ! -f "${CACHE_FILE}" ]] || (( $(cache_age) > CACHE_TTL )); then
    build_cache
else
    log "Using cached facts ($(cache_age)s old; --refresh to requery)"
fi

# --------------------------------------------------------------- report ---
build_candidates

export CACHE_FILE LIST_ONLY

printf '%s\n' "${CANDIDATES[@]}" | "${PY}" -c '
import json, os, sys

cache = json.load(open(os.environ["CACHE_FILE"]))
list_only = os.environ["LIST_ONLY"] == "1"

subnet_regions = {
    s["region"].rsplit("/", 1)[-1] for s in (cache.get("subnet_regions") or [])
}
machine_zones = {
    (m["name"], m["zone"].rsplit("/", 1)[-1])
    for m in (cache.get("machine_zones") or [])
}
accel_types = {
    zone: {a["type"] for a in (types or [])}
    for zone, types in (cache.get("accel_types") or {}).items()
}
disk_zones = {
    d["zone"].rsplit("/", 1)[-1] for d in (cache.get("disk_zones") or [])
}

# Measured 2026-09-01: creating v5e through Compute Engine returns
#   403 This user agent is not allowed to use the machine type
#       [ct5lp-hightpu-4t]
# on this project, with 512 chips of TPU-LITE-PODSLICE-V5 quota sitting unused.
# The allowlist is a separate gate from quota, and no read-only API exposes it,
# so it has to be hardcoded from observation. v5e is reachable only through the
# Cloud TPU API.
GCE_BLOCKED = {"ct5lp-hightpu-4t": "not allowlisted for this project (403)"}


def quota_for(gen, region):
    """Effective limit for a generation in a region, or None if unset.

    Cloud Quotas returns one dimensionsInfo per region override plus a
    catch-all whose applicableLocations lists every region sharing the default.
    A region-specific entry wins; otherwise fall back to the catch-all that
    lists the region.
    """
    info = (cache.get("quota") or {}).get(gen)
    if not info:
        return None
    fallback = None
    for di in info.get("dimensionsInfos", []):
        dims = di.get("dimensions", {}) or {}
        value = (di.get("details") or {}).get("value")
        if dims.get("region") == region:
            return int(value) if value is not None else None
        if not dims.get("region") and region in (di.get("applicableLocations") or []):
            fallback = int(value) if value is not None else None
    return fallback


rows, viable = [], []
for line in sys.stdin:
    entry = line.strip()
    if not entry:
        continue
    gen, mt, zone, model, api = entry.split(":")
    region = zone.rsplit("-", 1)[0]

    limit = quota_for(gen, region)
    has_disk = zone in disk_zones

    # The two APIs need different availability questions asked of them.
    if api == "tpuapi":
        # Cloud TPU API nodes do not consume a subnet in the default VPC the
        # way a GCE instance does, so the subnet check does not apply.
        offered = mt in accel_types.get(zone, set())
        missing = "accelerator type not offered in zone"
        blocked = None
    else:
        offered = (mt, zone) in machine_zones
        missing = "machine type not offered in zone"
        blocked = GCE_BLOCKED.get(mt)
        if region not in subnet_regions:
            blocked = blocked or "no subnet in region (org policy)"

    if blocked:
        verdict, why = "BLOCKED", blocked
    elif not offered:
        verdict, why = "BLOCKED", missing
    elif limit is None:
        verdict, why = "UNKNOWN", "quota unset; may resolve to 0"
    elif limit < 4:
        verdict, why = "BLOCKED", f"quota {limit} < 4 chips needed"
    else:
        verdict, why = "OK", f"quota {limit}" + (", disk present" if has_disk else "")

    rows.append((gen, mt, zone, model, api, verdict, why))
    if verdict == "OK":
        viable.append(entry)

if list_only:
    print("\n".join(viable))
    sys.exit(0)

print()
print("GEN   ZONE            TYPE              API     VERDICT  DETAIL")
print("-" * 92)
seen = set()
for gen, mt, zone, model, api, verdict, why in rows:
    key = (gen, zone, mt, api)
    if key in seen:
        continue
    seen.add(key)
    print(f"{gen:<5} {zone:<15} {mt:<17} {api:<7} {verdict:<8} {why}")

print()
print(f"{len(viable)} of {len(rows)} candidates are viable.")
if viable:
    print("Best target: " + viable[0])
else:
    print("Nothing viable. A quota increase is required before any create can")
    print("succeed; see the per-generation buckets in zones.sh.")
' 2>&1
