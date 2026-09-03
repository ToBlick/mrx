#!/usr/bin/env bash
# Shared configuration and helpers for the TPU probe, launcher and daemon.

# GCE-path self-termination. This is paired with --instance-termination-action
# =DELETE in launch_tpu.sh, so it does not merely stop the job at expiry, it
# destroys the VM and anything on its boot disk. It must therefore stay
# comfortably above RUN_TIMEOUT in run_on_tpu.sh: the 30m that suited the
# Poisson example would delete the machine partway through a relaxation.
# The Cloud TPU API has no equivalent flag, which is why v5e sessions are
# unaffected by this setting. They are covered instead by idle_reaper.sh, which
# both paths carry.
MAX_RUN_DURATION="${MAX_RUN_DURATION:-4h}"
# Minutes of no python, no login and no accelerator activity before a node
# deletes itself. This, not MAX_RUN_DURATION, is what actually bounds the bill:
# four hours is a backstop against catastrophe, twenty minutes is cost control.
# Long enough to read output, think, and start another run against a warm
# compilation cache; short enough that forgetting costs one coffee. Set to 0 to
# install the reaper but never let it fire.
IDLE_TIMEOUT_MIN="${IDLE_TIMEOUT_MIN:-20}"
# The branch a fresh node checks out. Override to measure a feature branch
# without SYNC_LOCAL_MRX; the node otherwise clones the development branch.
MRX_BRANCH="${MRX_BRANCH:-static-dynamic-refactor}"
DATA_DISK="${DATA_DISK:-my-data-disk}"
DATA_SNAPSHOT="${DATA_SNAPSHOT:-my-data-snapshot}"
IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-accelerator-images}"
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e}"
VM_NAME="${VM_NAME:-my-tpu-vm}"
# Runtime for the Cloud TPU API path. The -base images ship the TPU driver and
# leave Python to us, which is what the conda build on /mnt/data expects.
TPU_RUNTIME="${TPU_RUNTIME:-tpu-ubuntu2204-base}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"

# ---------------------------------------------------------------------------
# Candidate ladder
# ---------------------------------------------------------------------------
# Entries are generation:type:zone:model:api, tried in order.
#
# `api` is either `gce` (gcloud compute instances create) or `tpuapi`
# (gcloud compute tpus tpu-vm create). That distinction is not cosmetic: each
# generation is reachable through exactly one of them on this project.
#
# Measured 2026-09-01, all of it the hard way:
#
#   v5e via gce      403 "This user agent is not allowed to use the machine
#                    type [ct5lp-hightpu-4t]". Quota is 512, but the
#                    TPU-as-GCE-instance path is not allowlisted here.
#   v5e via tpuapi   works; currently "Insufficient capacity". This is what
#                    the TPU-LITE-PODSLICE-V5 quota of 512 actually governs.
#   v5p via gce      works; currently ZONE_RESOURCE_POOL_EXHAUSTED. Quota 768.
#   v6e via gce      quota is a hard 0.0 in us-east5 and us-east4, and every
#                    other reachable zone has been in continuous stockout.
#
# Also constraining things: constraints/gcp.resourceLocations restricts this
# project to US locations, so the non-US regions carrying a CT6E limit of 48
# are unreachable -- the auto-mode default VPC has no subnet there and a create
# dies at network validation before capacity is ever consulted.
#
# my-data-disk exists in us-central1-a, us-west4-a, us-east5-a and us-east5-b;
# landing in one of those skips restoring from the snapshot.
#
# Single-chip entries are interleaved late because one chip is far likelier to
# be free than four, and the MRX solve is single-device anyway.
DEFAULT_CANDIDATES=(
    # v5p, 4 chips, 95 GB HBM each. Highest quota that is actually usable.
    v5p:ct5p-hightpu-4t:us-east5-b:STANDARD:gce
    v5p:ct5p-hightpu-4t:us-east5-a:STANDARD:gce
    v5p:ct5p-hightpu-4t:us-central1-a:STANDARD:gce
    v5p:ct5p-hightpu-4t:us-east5-c:STANDARD:gce
    v5p:ct5p-hightpu-4t:us-east1-d:STANDARD:gce
    v5p:ct5p-hightpu-4t:us-south1-a:STANDARD:gce

    # v5e through the Cloud TPU API, disk zones first
    v5e:v5litepod-4:us-east5-b:ONDEMAND:tpuapi
    v5e:v5litepod-4:us-east5-a:ONDEMAND:tpuapi
    v5e:v5litepod-4:us-central1-a:ONDEMAND:tpuapi
    v5e:v5litepod-4:us-west4-a:ONDEMAND:tpuapi
    v5e:v5litepod-4:us-east5-c:ONDEMAND:tpuapi
    v5e:v5litepod-4:us-south1-a:ONDEMAND:tpuapi

    # Single chip: much likelier to be free, and enough for the MRX solve
    v5e:v5litepod-1:us-east5-b:ONDEMAND:tpuapi
    v5e:v5litepod-1:us-east5-a:ONDEMAND:tpuapi
    v5e:v5litepod-1:us-central1-a:ONDEMAND:tpuapi
    v5e:v5litepod-1:us-east5-c:ONDEMAND:tpuapi

    # Spot: cheapest, preemptible, same pools
    v5e:v5litepod-4:us-east5-b:SPOT:tpuapi
    v5e:v5litepod-1:us-east5-b:SPOT:tpuapi
    v5e:v5litepod-1:us-east5-a:SPOT:tpuapi

    # v6e last: needs a CT6E quota grant in us-east5/us-east4 to work at all.
    v6e:ct6e-standard-4t:us-central1-a:FLEX_START:gce
    v6e:ct6e-standard-4t:us-central1-b:FLEX_START:gce
    v6e:ct6e-standard-4t:us-central1-c:FLEX_START:gce
    v6e:ct6e-standard-4t:us-east1-d:FLEX_START:gce
    v6e:ct6e-standard-4t:us-east1-b:FLEX_START:gce
    v6e:ct6e-standard-4t:us-west1-c:FLEX_START:gce
)

# Cloud Quotas bucket that actually governs each generation. The generic
# TPUS-PER-TPU-FAMILY bucket is a poor guide: it reads "unset" for regions that
# in practice permit creates and for regions that hard-fail at 0.
quota_bucket_for() {
    case "$1" in
        v5e) echo "TPU-LITE-PODSLICE-V5-per-project-region" ;;
        v5p) echo "TPU-V5P-per-project-region" ;;
        v6e) echo "PREEMPTIBLE-TPU-V6E-per-project-region" ;;
        *)   echo "TPUS-PER-TPU-FAMILY-per-project-region" ;;
    esac
}

# Build CANDIDATES from the defaults, honouring the env overrides.
#   GENERATIONS=v5e,v5p     restrict to these generations
#   ZONES=us-east5-b,...    restrict to these zones
#   MACHINE_TYPE=...        restrict to one machine type
#   MODELS=STANDARD,SPOT    restrict to these provisioning models
#   APIS=gce,tpuapi         restrict to these create paths
build_candidates() {
    local src=("${DEFAULT_CANDIDATES[@]}")
    CANDIDATES=()

    local gen_filter="${GENERATIONS:-}" zone_filter="${ZONES:-}"
    local mt_filter="${MACHINE_TYPE:-}" model_filter="${MODELS:-}"
    local api_filter="${APIS:-}"

    local entry gen mt zone model api
    for entry in "${src[@]}"; do
        IFS=':' read -r gen mt zone model api <<<"${entry}"
        [[ -n "${gen_filter}"   && ",${gen_filter//[[:space:]]/},"   != *",${gen},"*   ]] && continue
        [[ -n "${zone_filter}"  && ",${zone_filter//[[:space:]]/},"  != *",${zone},"*  ]] && continue
        [[ -n "${mt_filter}"    && ",${mt_filter//[[:space:]]/},"    != *",${mt},"*    ]] && continue
        [[ -n "${model_filter}" && ",${model_filter//[[:space:]]/}," != *",${model},"* ]] && continue
        [[ -n "${api_filter}"   && ",${api_filter//[[:space:]]/},"   != *",${api},"*   ]] && continue
        CANDIDATES+=("${entry}")
    done
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
#
# Lives here rather than in the daemon because run_on_tpu.sh needs it too: its
# zone lookup used to ask `compute instances list` only, so it could not find a
# v5e at all.
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

# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------
# Echoes one of:
#   STOCKOUT   no hardware in the zone
#   QUOTA      permitted machine type, but the quota bucket is exhausted or 0
#   NO_SUBNET  the default VPC has no subnet in this region
#   POLICY     an org policy forbids the location outright
#   PERMISSION IAM or API enablement
#   UNSUPPORTED the machine type or image genuinely is not offered here
#   NOT_ALLOWLISTED  the machine type exists and quota exists, but this project
#              is not permitted to reach it by this API (v5e via GCE)
#   DISK_INCOMPATIBLE  the machine type refuses the data disk's type; worth
#              retrying without it (v5p rejects hyperdisk-balanced)
#   TRANSIENT  a Google-side internal error, worth one retry
#   OTHER      unrecognised
#
# NO_SUBNET and POLICY used to be folded into UNSUPPORTED, which hid the real
# reason eight zones were failing behind a label that implied the hardware did
# not exist there.
classify_failure() {
    local log="$1"
    if rg -q "reason: stockout|ZONE_RESOURCE_POOL_EXHAUSTED|Insufficient capacity" "${log}"; then
        echo "STOCKOUT"
    elif rg -q "user agent is not allowed to use the machine type|not allowed to use" "${log}"; then
        echo "NOT_ALLOWLISTED"
    elif rg -q "disk type cannot be used by|not supported for the machine type" "${log}"; then
        echo "DISK_INCOMPATIBLE"
    elif rg -q "QUOTA_EXCEEDED|Quota .* exceeded|quotaExceeded" "${log}"; then
        echo "QUOTA"
    elif rg -q "No default subnetwork was found|does not have a subnetwork" "${log}"; then
        echo "NO_SUBNET"
    elif rg -q "resourceLocations|violates constraint|Constraint .* violated|orgpolicy" "${log}"; then
        echo "POLICY"
    elif rg -q "Internal error|backend error|Try again later|deadline exceeded" "${log}"; then
        echo "TRANSIENT"
    elif rg -q "PERMISSION_DENIED|Required .* permission|403" "${log}"; then
        echo "PERMISSION"
    elif rg -q "not found|does not exist|Invalid value|UNSUPPORTED" "${log}"; then
        echo "UNSUPPORTED"
    else
        echo "OTHER"
    fi
}

# One-line human explanation for a classification.
explain_failure() {
    local kind="$1" log="$2" hint detail
    case "${kind}" in
        STOCKOUT)
            hint="$(zones_available_hint "${log}")"
            if [[ -n "${hint}" ]]; then
                echo "no hardware; google suggests: ${hint}"
            else
                echo "no hardware (zonesAvailable empty)"
            fi
            ;;
        QUOTA)
            detail="$(rg -o "Quota '[^']*' exceeded[^.]*" "${log}" | head -n 1)"
            # A quota rejection means the request cleared the capacity check
            # first, so the hardware is probably there. That makes it a better
            # target for a quota increase than a stockout zone.
            echo "${detail:-quota exhausted} (capacity likely present)"
            ;;
        NO_SUBNET)   echo "default VPC has no subnet in this region" ;;
        POLICY)      echo "blocked by org policy (gcp.resourceLocations)" ;;
        TRANSIENT)   echo "google-side internal error" ;;
        PERMISSION)  echo "permission denied - check IAM / API enablement" ;;
        NOT_ALLOWLISTED)
            echo "machine type not allowlisted for this project (quota is irrelevant)" ;;
        DISK_INCOMPATIBLE)
            echo "${DATA_DISK} type rejected by this machine type; retrying without it" ;;
        UNSUPPORTED) echo "machine type or image not offered here" ;;
        *)
            # gcloud puts "Could not fetch resource:" on the ERROR line and the
            # useful text on a following " - ..." or "message:" line, so keying
            # off ERROR alone reports nothing of value.
            detail="$(rg -o '^\s*-\s+\S.*|"message":.*' "${log}" | head -n 1 \
                | sed 's/^[[:space:]]*-[[:space:]]*//' | cut -c1-90)"
            if [[ -z "${detail}" ]]; then
                detail="$(rg -o 'ERROR:.*' "${log}" | head -n 1 | cut -c1-90)"
            fi
            echo "${detail:-unrecognised failure}"
            ;;
    esac
}

# Pull the zonesAvailable hint out of a stockout error, if Google supplied one.
# Google leaves this empty when no zone has capacity, which is itself useful.
zones_available_hint() {
    local log="$1" raw
    raw="$(rg -o "zonesAvailable: '?[^'\"]*" "${log}" 2>/dev/null | head -n 1)"
    # Strip up to the first colon, then quotes and whitespace. Avoids sed
    # escape-syntax differences between BSD and GNU.
    raw="${raw#*: }"
    raw="${raw//\'/}"
    printf '%s' "${raw}" | tr -d '[:space:]'
}

# ---------------------------------------------------------------------------
# Disk helpers
# ---------------------------------------------------------------------------
data_disk_exists() {
    gcloud compute disks describe "${DATA_DISK}" --zone="$1" \
        --format="value(name)" >/dev/null 2>&1
}

data_disk_count() {
    gcloud compute disks list --filter="name=${DATA_DISK}" \
        --format="value(name)" 2>/dev/null | grep -c . | tr -d ' '
}

# How many DATA_DISK volumes may exist across all zones at once.
#
# A persistent disk is zonal, capacity decides the zone, and capacity moves, so
# creating one wherever a sweep happens to win accumulates a 100 GB volume per
# zone that ever succeeded -- four of them had built up before this cap existed,
# billing continuously while attached to nothing. The cap is deliberately small:
# a cold environment build is only about four minutes, so a warm disk in the
# wrong zone is worth much less than it costs.
MAX_DATA_DISKS="${MAX_DATA_DISKS:-2}"

# Create DATA_DISK in a zone from the snapshot when it is not already there.
ensure_data_disk() {
    local zone="$1" count
    data_disk_exists "${zone}" && return 0
    if ! gcloud compute snapshots describe "${DATA_SNAPSHOT}" \
        --format="value(name)" >/dev/null 2>&1; then
        return 1
    fi

    count="$(data_disk_count)"
    if [[ -n "${count}" ]] && (( count >= MAX_DATA_DISKS )); then
        echo "  ${count} ${DATA_DISK} volume(s) already exist (cap ${MAX_DATA_DISKS});" >&2
        echo "  not creating another in ${zone}. This run uses the boot disk." >&2
        echo "  Free one with './acquire_tpu.sh --gc', or raise MAX_DATA_DISKS." >&2
        return 1
    fi

    gcloud compute disks create "${DATA_DISK}" \
        --zone="${zone}" \
        --source-snapshot="${DATA_SNAPSHOT}" \
        --type=hyperdisk-balanced \
        --quiet >/dev/null 2>&1
}
