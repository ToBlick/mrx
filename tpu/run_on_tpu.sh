#!/usr/bin/env bash
#
# Drive one TPU session: wait for the environment, run MRX, pull results back.
#
# The first session on a fresh data disk spends ~4-12 minutes in startup.sh
# building miniforge, jax[tpu] and mrx; every later session finds
# /mnt/data/.mrx_env_ready already present and starts computing within a minute.
#
# SCRIPT defaults to the toroidal Poisson driver, which checks TPU results
# against a CPU float32 reference. Set it to run anything else in the repo, and
# set OUTDIR to the directory that script writes.
#
# Usage:
#   ZONE=us-east5-b ./run_on_tpu.sh --n 6 8 --p 2
#   SCRIPT=scripts/tutorials/li383_relaxation.py \
#     OUTDIR=outputs/tutorials/li383_relaxation \
#     ZONE=us-south1-a ./run_on_tpu.sh --ns 12,24,12 --p 3
#
# RUN_TIMEOUT bounds the remote command. It defaulted to 1500s when this only
# ever ran the Poisson example; a relaxation needs far more, and on the GCE path
# MAX_RUN_DURATION in zones.sh must stay above it or the VM is deleted first.
#
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
# For tpu_running_zone: a v5e is a Cloud TPU API node, which is invisible to
# `compute instances list`, so the zone lookup below needs both surfaces.
# shellcheck source=zones.sh
source ./zones.sh

VM_NAME="${VM_NAME:-my-tpu-vm}"
ZONE="${ZONE:-}"
OUT_ROOT="script_outputs"
SENTINEL=/mnt/data/.mrx_env_ready
SETUP_LOG=/mnt/data/setup.log
POLL_SECONDS="${POLL_SECONDS:-20}"
SETUP_TIMEOUT="${SETUP_TIMEOUT:-2400}"
RUN_TIMEOUT="${RUN_TIMEOUT:-7200}"

MRX_DIR=/mnt/data/mrx
PYBIN=/mnt/data/envs/mrx/bin/python

# SCRIPT is relative to the mrx repo; OUTDIR is where it writes, absolute or
# relative to the checkout.
SCRIPT="${SCRIPT:-scripts/benchmark/poisson_regression.py}"
OUTDIR="${OUTDIR:-/mnt/data/mrx_tpu_results}"

# Persistent XLA compilation cache, on the data disk so it survives the VM. A
# gs:// path also works and is the only way to carry compiled kernels to a node
# with no data disk; see docs/source/tpu.md for the caveats.
JAX_CACHE_DIR="${JAX_CACHE_DIR:-/mnt/data/jax_cache}"
# Which JAX backend the remote command should use. JAX_PLATFORMS=cpu lets a
# float64 stage run on the host CPU, since TPUs have no usable float64.
RUN_PLATFORM="${RUN_PLATFORM:-}"
RUN_DTYPE="${RUN_DTYPE:-float32}"

if [[ -z "${ZONE}" ]]; then
    echo "Locating ${VM_NAME}..."
    ZONE="$(gcloud compute instances list \
        --filter="name=${VM_NAME}" \
        --format="value(zone.basename())" 2>/dev/null | head -n 1)"
    # A v5e is not a GCE instance, so the list above cannot see it. Ask the
    # Cloud TPU API before giving up, or this errors out on a live node.
    if [[ -z "${ZONE}" ]]; then
        build_candidates
        ZONE="$(tpu_running_zone "${VM_NAME}")"
    fi
    if [[ -z "${ZONE}" ]]; then
        echo "ERROR: could not find ${VM_NAME} as a GCE instance or as a" >&2
        echo "Cloud TPU API node in any candidate zone." >&2
        echo "Set ZONE explicitly, or acquire one with ./acquire_tpu.sh" >&2
        exit 1
    fi
    echo "  found in ${ZONE}"
fi

# v5e is only reachable through the Cloud TPU API on this project, and a node
# created that way is not a GCE instance: it needs `tpus tpu-vm ssh/scp` and
# reports `state`, not `status`. Detect rather than require the caller to say.
TPU_API="${TPU_API:-auto}"
if [[ "${TPU_API}" == "auto" ]]; then
    if gcloud compute instances describe "${VM_NAME}" --zone="${ZONE}" \
        --format="value(name)" >/dev/null 2>&1; then
        TPU_API=0
    elif gcloud compute tpus tpu-vm describe "${VM_NAME}" --zone="${ZONE}" \
        --format="value(name)" >/dev/null 2>&1; then
        TPU_API=1
    else
        echo "ERROR: no GCE instance or TPU node named ${VM_NAME} in ${ZONE}." >&2
        exit 1
    fi
fi
(( TPU_API )) && echo "  (Cloud TPU API node)"

# stderr is dropped: gcloud's SSH banner noise would otherwise repeat on every
# poll. The run's own output comes back through the log file, not through here.
ssh_vm() {
    if (( TPU_API )); then
        gcloud compute tpus tpu-vm ssh "${VM_NAME}" --zone="${ZONE}" \
            --command="$1" -- -o ConnectTimeout=15 -o StrictHostKeyChecking=no 2>/dev/null
    else
        gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --tunnel-through-iap=false \
            --command="$1" -- -o ConnectTimeout=15 -o StrictHostKeyChecking=no 2>/dev/null
    fi
}

# Run a command on the VM detached from the SSH session, streaming its log back.
#
# A foreground `ssh --command` looked fine for the 2-minute Poisson example but
# is wrong for anything long. `gcloud ... ssh` silently retries a failed
# connection, so the command can be launched twice: the second copy then dies
# with "The TPU is already in use by process with pid N" while the first keeps
# running, detached, with its stdout going nowhere. That is exactly how a
# calibration run was lost. Any dropped connection over a multi-hour relaxation
# would do the same.
#
# So: start it once under setsid, record the exit status in a marker file, and
# poll the log. The job now survives a dropped connection instead of being
# orphaned by one, and its output is on the VM's disk either way.
run_detached() {
    local tag="$1" cmd="$2"
    local dir="/mnt/data/runs"
    local logf="${dir}/${tag}.log"
    local donef="${dir}/${tag}.done"
    local pidf="${dir}/${tag}.pid"
    local remote_runner="${dir}/${tag}.sh"

    # Ship the command as a file rather than interpolating it into an ssh
    # argument. The payload contains both quote styles (the awk timestamper,
    # and whatever the caller passed), and nesting those inside `bash -lc '...'`
    # inside an `ssh --command` string does not survive.
    local runner
    runner="$(mktemp -t mrxrun)"
    cat >"${runner}" <<EOF
#!/usr/bin/env bash
# Record the process group leader so the poller can kill the whole job on
# timeout. Matching on the command line instead does not work: two runs of the
# same script are indistinguishable, so a pattern kill either matches nothing
# or matches every python on the box.
echo \$\$ >"${pidf}"
{
    ${cmd}
    echo \$? >"${donef}"
# Timestamp every line: these runs have long silent stretches inside a single
# library call, and without a clock there is no telling slow from hung.
} 2>&1 | awk '{ print strftime("[%H:%M:%S]"), \$0; fflush() }' >"${logf}"
EOF

    ssh_vm "mkdir -p ${dir} && rm -f '${logf}' '${donef}' '${pidf}'" >/dev/null
    if ! scp_to_vm "${runner}" "${remote_runner}"; then
        rm -f "${runner}"
        echo "ERROR: could not copy the runner script to the VM." >&2
        return 1
    fi
    rm -f "${runner}"

    # setsid detaches the job into its own session, so a dropped SSH cannot
    # take it down and a later reconnect cannot start a second copy.
    ssh_vm "chmod +x '${remote_runner}' && \
        setsid '${remote_runner}' </dev/null >/dev/null 2>&1 & sleep 3; echo ok" \
        >/dev/null

    local pid
    pid="$(ssh_vm "cat '${pidf}' 2>/dev/null" | tr -d '[:space:]')"
    if [[ -z "${pid}" ]]; then
        echo "ERROR: the job never started; no PID file at ${pidf}." >&2
        ssh_vm "tail -n 20 '${logf}' 2>/dev/null" >&2
        return 1
    fi
    echo "  [detached as pid ${pid}; log ${logf}]"

    # One round trip per poll, not two: the tail and the done-marker read are
    # batched, with a sentinel separating them. At ~3 s per SSH that halves the
    # effective poll interval.
    local seen=0 waited=0 batch chunk status=""
    local marker="__MRX_DONE__"
    while true; do
        batch="$(ssh_vm "tail -n +$(( seen + 1 )) '${logf}' 2>/dev/null; \
                         echo '${marker}'; cat '${donef}' 2>/dev/null")"
        chunk="${batch%%"${marker}"*}"
        status="$(printf '%s' "${batch#*"${marker}"}" | tr -d '[:space:]')"

        if [[ -n "${chunk}" ]]; then
            printf '%s' "${chunk}"
            seen=$(( seen + $(printf '%s' "${chunk}" | grep -c '') ))
        fi
        [[ -n "${status}" ]] && return "${status}"

        if (( waited >= RUN_TIMEOUT )); then
            echo "ERROR: RUN_TIMEOUT (${RUN_TIMEOUT}s) reached; killing pid ${pid}." >&2
            ssh_vm "kill -TERM -${pid} 2>/dev/null || kill -TERM ${pid} 2>/dev/null" >/dev/null
            return 124
        fi
        sleep "${RUN_POLL:-20}"
        waited=$(( waited + ${RUN_POLL:-20} ))
    done
}

scp_to_vm() {
    if (( TPU_API )); then
        gcloud compute tpus tpu-vm scp "$1" "${VM_NAME}:$2" --zone="${ZONE}" >/dev/null 2>&1
    else
        gcloud compute scp "$1" "${VM_NAME}:$2" --zone="${ZONE}" >/dev/null 2>&1
    fi
}

scp_from_vm() {
    if (( TPU_API )); then
        gcloud compute tpus tpu-vm scp --recurse "${VM_NAME}:$1" "$2" \
            --zone="${ZONE}" >/dev/null 2>&1
    else
        gcloud compute scp --recurse "${VM_NAME}:$1" "$2" \
            --zone="${ZONE}" >/dev/null 2>&1
    fi
}

vm_status() {
    if (( TPU_API )); then
        gcloud compute tpus tpu-vm describe "${VM_NAME}" --zone="${ZONE}" \
            --format="value(state)" 2>/dev/null
    else
        gcloud compute instances describe "${VM_NAME}" --zone="${ZONE}" \
            --format="value(status)" 2>/dev/null
    fi
}

echo ""
echo "Waiting for ${VM_NAME} to reach RUNNING..."
while true; do
    status="$(vm_status)"
    case "${status}" in
        RUNNING|READY) echo "  ${status}"; break ;;
        PENDING) echo "  still PENDING (queued for capacity); sleeping ${POLL_SECONDS}s" ;;
        "")      echo "ERROR: the node is gone." >&2
                 exit 1 ;;
        *)       echo "  status=${status}; sleeping ${POLL_SECONDS}s" ;;
    esac
    sleep "${POLL_SECONDS}"
done

print_remaining() {
    local created max_secs
    # Cloud TPU API nodes have no max-run-duration to report against. Their
    # bound is the idle reaper, whose countdown starts when this run ends.
    if (( TPU_API )); then
        local idle
        idle="$(ssh_vm "systemctl show -p Environment --value mrx-idle-reaper 2>/dev/null" |
                sed -n 's/.*IDLE_TIMEOUT_MIN=\([0-9]*\).*/\1/p')"
        if [[ -n "${idle}" ]]; then
            echo "Session: Cloud TPU API node; self-deletes after ${idle} idle min"
        else
            echo "Session: Cloud TPU API node with NO reaper running -- delete when done"
        fi
        return
    fi
    created="$(gcloud compute instances describe "${VM_NAME}" --zone="${ZONE}" \
        --format="value(creationTimestamp)" 2>/dev/null)"
    max_secs="$(gcloud compute instances describe "${VM_NAME}" --zone="${ZONE}" \
        --format="value(scheduling.maxRunDuration.seconds)" 2>/dev/null)"
    [[ -z "${created}" || -z "${max_secs}" ]] && return
    local start_epoch now_epoch used left
    # BSD date on macOS, GNU date elsewhere.
    start_epoch="$(date -j -f "%Y-%m-%dT%H:%M:%S" "${created:0:19}" +%s 2>/dev/null \
        || date -d "${created}" +%s 2>/dev/null || echo 0)"
    [[ "${start_epoch}" == "0" ]] && return
    now_epoch="$(date +%s)"
    used=$(( now_epoch - start_epoch ))
    left=$(( max_secs - used ))
    printf 'Session: %dm used, %dm remaining of %dm\n' \
        $(( used / 60 )) $(( left / 60 )) $(( max_secs / 60 ))
}

print_remaining

echo ""
echo "Waiting for the environment (${SENTINEL})..."
waited=0
while true; do
    if ssh_vm "test -f ${SENTINEL} && echo READY" | rg -q READY; then
        echo "  environment ready"
        break
    fi
    if (( waited >= SETUP_TIMEOUT )); then
        echo "ERROR: environment not ready after ${SETUP_TIMEOUT}s." >&2
        echo "Last 40 lines of ${SETUP_LOG}:" >&2
        ssh_vm "sudo tail -n 40 ${SETUP_LOG}" >&2
        exit 1
    fi
    tail_line="$(ssh_vm "sudo tail -n 1 ${SETUP_LOG} 2>/dev/null")"
    printf '  [%4ds] %s\n' "${waited}" "${tail_line:-(waiting for startup-script)}"
    sleep "${POLL_SECONDS}"
    waited=$(( waited + POLL_SECONDS ))
done

# The branch is baked into instance metadata at create time and read by
# startup.sh, so a node acquired with the default and then driven with
# MRX_BRANCH=<feature> would run the default checkout and report the numbers as
# if they were the feature branch's. That is the failure startup.sh warns about,
# and it is silent, so honour MRX_BRANCH here as well.
echo ""
echo "Checking out ${MRX_BRANCH} in ${MRX_DIR}..."
# startup.sh clones as root, so the login user hits git's "dubious ownership"
# refusal on every command. The exception is per-user and idempotent, and a node
# is single-tenant, so it costs nothing to assert it.
#
# Checked out from FETCH_HEAD rather than origin/<branch>, because startup.sh
# clones shallow and single-branch: remote.origin.fetch names only the branch it
# cloned, so no other origin/* ref can ever exist and `checkout --track` fails
# with "not a branch".
VM_SHA="$(ssh_vm "git config --global --add safe.directory ${MRX_DIR} 2>/dev/null; \
        git -C ${MRX_DIR} fetch --quiet origin ${MRX_BRANCH} && \
        git -C ${MRX_DIR} checkout --quiet -B ${MRX_BRANCH} FETCH_HEAD && \
        git -C ${MRX_DIR} rev-parse --short HEAD" \
    | tr -d '\r' | rg -o '^[0-9a-f]{7,40}$' | tail -n 1)"
if [[ -z "${VM_SHA}" ]]; then
    echo "ERROR: could not check out ${MRX_BRANCH} on the VM." >&2
    exit 1
fi
echo "  at ${VM_SHA}"

RUN_ARGS="$*"

# TPU_STDERR_LOG_LEVEL silences libtpu driver chatter that would otherwise bury
# the results. The compilation cache is not a minor tuning knob: MRX's inner
# solves run as eager lax.while_loops, so each call traces a fresh program and
# XLA compiles it again, and turning the cache on took one apply_laplacian k=1
# call from 9854 ms to 105 ms. The two thresholds are lowered because their
# defaults would skip nearly every kernel this workload compiles.
COMMON_ENV="export PATH=/mnt/data/envs/mrx/bin:\$PATH; \
    export MPLBACKEND=Agg MRX_REPO=${MRX_DIR}; \
    export TPU_STDERR_LOG_LEVEL=3 TPU_MIN_LOG_LEVEL=3; \
    export JAX_COMPILATION_CACHE_DIR=${JAX_CACHE_DIR}; \
    export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0; \
    export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.1;"
[[ -n "${RUN_PLATFORM}" ]] && COMMON_ENV="${COMMON_ENV} export JAX_PLATFORMS=${RUN_PLATFORM};"

case "${OUTDIR}" in
    /*) REMOTE_OUT="${OUTDIR}" ;;
    *)  REMOTE_OUT="${MRX_DIR}/${OUTDIR}" ;;
esac

echo ""
echo "Running ${SCRIPT} with MRX_DTYPE=${RUN_DTYPE}${RUN_PLATFORM:+ on ${RUN_PLATFORM}} [VM checkout ${VM_SHA}]..."
echo "=================================================================="

# cd into the repo: scripts such as li383_relaxation.py default to a relative
# geometry path (data/wout_li383_low_res_reference.nc), which does not resolve
# from the login shell's home directory.
run_detached "$(basename "${SCRIPT}" .py)" \
    "cd ${MRX_DIR} && ${COMMON_ENV} export MRX_DTYPE=${RUN_DTYPE}; \
     ${PYBIN} -u ${SCRIPT} ${RUN_ARGS}"
RUN_STATUS=$?

echo "=================================================================="
echo ""

mkdir -p "${OUT_ROOT}"
echo "Pulling results into ${OUT_ROOT}/..."
scp_from_vm "${REMOTE_OUT}" "${OUT_ROOT}/" \
    && echo "  ${OUT_ROOT}/$(basename "${REMOTE_OUT}")/" \
    || echo "  WARNING: no results directory to copy"

print_remaining

if (( RUN_STATUS != 0 )); then
    echo ""
    echo "The run exited ${RUN_STATUS}. That is ${SCRIPT}'s own exit code;"
    echo "the pulled-back ${OUTDIR} and the log above are the evidence."
    if [[ "${SCRIPT}" == *poisson_regression.py ]]; then
        echo "For that driver a non-zero exit usually means the TPU results"
        echo "deviated from the CPU float32 reference; check summary.md."
    fi
fi
exit "${RUN_STATUS}"
