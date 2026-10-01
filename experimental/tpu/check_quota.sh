#!/usr/bin/env bash
#
# Read-only quota preflight for the candidate ladder.
#
# Quota is the one gate worth checking before a sweep, because it is the only
# failure retrying cannot fix: a region with a hard 0 will never produce a node,
# and a blind sweep against one costs 25 minutes to learn that. This is what
# surfaced that v6e -- not capacity in general -- was the blocker.
#
# Everything else the ladder discovers by trying, in seconds per candidate, and
# classify_failure in zones.sh names it. Quota is also only a ceiling, never an
# allocation, so an OK here says a create is permitted, not that hardware is
# free.
#
# Usage:
#   ./check_quota.sh             # effective quota per generation and region
#   ./check_quota.sh --refresh   # ignore the cache
#   GENERATIONS=v5e ./check_quota.sh
#
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
# shellcheck source=zones.sh
source ./zones.sh

CACHE_FILE="${CACHE_FILE:-.quota_cache.json}"
CACHE_TTL="${CACHE_TTL:-3600}"
REFRESH=0
for arg in "$@"; do
    case "${arg}" in
        --refresh) REFRESH=1 ;;
    esac
done

PY="${PY:-python3}"
command -v "${PY}" >/dev/null 2>&1 || PY=/usr/bin/python3

if [[ -z "${PROJECT}" ]]; then
    echo "ERROR: no project set; run 'gcloud config set project ...'" >&2
    exit 1
fi

cache_age() {
    [[ -f "${CACHE_FILE}" ]] || { echo 999999; return; }
    local mtime now
    mtime="$(stat -f %m "${CACHE_FILE}" 2>/dev/null || stat -c %Y "${CACHE_FILE}" 2>/dev/null || echo 0)"
    now="$(date +%s)"
    echo $(( now - mtime ))
}

# The generations actually in the ladder, so a bucket is never queried for a
# generation that was filtered out or removed.
ladder_generations() {
    local entry gen rest seen=""
    for entry in "${DEFAULT_CANDIDATES[@]}"; do
        IFS=':' read -r gen rest <<<"${entry}"
        case " ${seen} " in *" ${gen} "*) continue ;; esac
        seen="${seen} ${gen}"
        echo "${gen}"
    done
}

build_cache() {
    echo "Querying quota buckets..."
    local tmp first gen bucket
    tmp="$(mktemp)"
    {
        echo '{'
        first=1
        for gen in $(ladder_generations); do
            bucket="$(quota_bucket_for "${gen}")"
            (( first )) || echo ','
            first=0
            printf '"%s": ' "${gen}"
            # `gcloud beta quotas info describe` rejects the call without an
            # explicit container flag, even with a default project configured.
            gcloud beta quotas info describe "${bucket}" \
                --service=compute.googleapis.com \
                --project="${PROJECT}" \
                --format=json 2>/dev/null || echo 'null'
        done
        echo '}'
    } >"${tmp}"
    mv "${tmp}" "${CACHE_FILE}"
}

if (( REFRESH )) || [[ ! -f "${CACHE_FILE}" ]] || (( $(cache_age) > CACHE_TTL )); then
    build_cache
else
    echo "Using cached quota ($(cache_age)s old; --refresh to requery)"
fi

build_candidates
export CACHE_FILE

printf '%s\n' "${CANDIDATES[@]}" | "${PY}" -c '
import json, os, sys

cache = json.load(open(os.environ["CACHE_FILE"]))


def quota_for(gen, region):
    """Effective limit for a generation in a region, or None if unset.

    Cloud Quotas returns one dimensionsInfo per region override plus a
    catch-all whose applicableLocations lists every region sharing the default.
    A region-specific entry wins; otherwise fall back to the catch-all that
    lists the region.
    """
    info = cache.get(gen)
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


rows, blocked = [], 0
seen = set()
for line in sys.stdin:
    entry = line.strip()
    if not entry:
        continue
    gen, mt, zone, model, api = entry.split(":")
    region = zone.rsplit("-", 1)[0]
    if (gen, region) in seen:
        continue
    seen.add((gen, region))

    limit = quota_for(gen, region)
    if limit is None:
        verdict, why = "UNKNOWN", "quota unset; may resolve to 0"
    elif limit < 4:
        verdict, why = "BLOCKED", f"quota {limit} < 4 chips needed"
        blocked += 1
    else:
        verdict, why = "OK", f"quota {limit}"
    rows.append((gen, region, verdict, why))

print()
print("GEN   REGION          VERDICT  DETAIL")
print("-" * 62)
for gen, region, verdict, why in rows:
    print(f"{gen:<5} {region:<15} {verdict:<8} {why}")

print()
ok = sum(1 for r in rows if r[2] != "BLOCKED")
print(f"{ok} of {len(rows)} generation/region pairs permit a create.")
if not ok:
    print("Nothing can succeed. A quota increase is required; see the")
    print("per-generation buckets in zones.sh.")
else:
    print("Quota is a ceiling, not an allocation. Sweep for real capacity:")
    print("  ./acquire_tpu.sh --once")
' 2>&1
