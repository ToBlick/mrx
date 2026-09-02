#!/usr/bin/env bash
#
# Drive one TPU session: wait for the environment, run MRX, pull results back.
#
# The first session on a fresh data disk spends ~4-12 minutes in startup.sh
# building miniforge, jax[tpu] and mrx; every later session finds
# /mnt/data/.mrx_env_ready already present and starts computing within a minute.
#
# Two modes:
#
#   default        run mrx_tpu_report.py, the toroidal Poisson driver that
#                  checks TPU results against the CPU float32 reference
#   SCRIPT=...     run an arbitrary script from the mrx repo and pull back the
#                  directory named by OUTDIR
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

cd "$(dirname "${BASH_SOURCE[0]}")"

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

# SCRIPT is relative to the mrx repo; OUTDIR is where it writes.
SCRIPT="${SCRIPT:-}"
OUTDIR="${OUTDIR:-}"
# Overlay a local mrx working tree onto the VM's checkout, so a fix can be
# measured on real hardware before it is committed to anything.
SYNC_LOCAL_MRX="${SYNC_LOCAL_MRX:-0}"
LOCAL_MRX="${LOCAL_MRX:-${HOME}/mrx}"
# Space-separated local files copied into the repo directory before the run, so
# a driver that lives in this kit rather than in mrx (tpu_bench_mrx.py) can be
# used as SCRIPT and still resolve the repo's relative data paths.
PUSH_FILES="${PUSH_FILES:-}"
# Persistent XLA compilation cache. On the data disk when there is one, so it
# survives the VM; set to empty to disable.
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
    if [[ -z "${ZONE}" ]]; then
        echo "ERROR: could not find an instance named ${VM_NAME}." >&2
        echo "Set ZONE explicitly, or launch one with ./launch_tpu.sh" >&2
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

# $2="keep-stderr" surfaces stderr; the default hides gcloud's SSH banner noise
# during polling. Hiding it during the actual run would discard the traceback.
ssh_vm() {
    if [[ "${2:-}" == "keep-stderr" ]]; then
        if (( TPU_API )); then
            gcloud compute tpus tpu-vm ssh "${VM_NAME}" --zone="${ZONE}" \
                --command="$1" -- -o ConnectTimeout=15 -o StrictHostKeyChecking=no 2>&1 \
                | rg -v "Could not open|log file|batch size|SSH: Attempting|Warning: Permanently"
            return "${PIPESTATUS[0]}"
        fi
        gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --tunnel-through-iap=false \
            --command="$1" -- -o ConnectTimeout=15 -o StrictHostKeyChecking=no 2>&1
        return $?
    fi
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
# timeout. Matching on the command line instead does not work: the report mode
# has no distinguishing SCRIPT string, so a pattern kill either matches nothing
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
        chunk="${batch%%${marker}*}"
        status="$(printf '%s' "${batch#*${marker}}" | tr -d '[:space:]')"

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

# ------------------------------------------------------- wait for RUNNING ---
echo ""
echo "Waiting for ${VM_NAME} to reach RUNNING..."
while true; do
    status="$(vm_status)"
    case "${status}" in
        RUNNING|READY) echo "  ${status}"; break ;;
        PENDING) echo "  still PENDING (queued for capacity); sleeping ${POLL_SECONDS}s" ;;
        "")      echo "ERROR: instance is gone. Its flex-start wait time probably expired." >&2
                 exit 1 ;;
        *)       echo "  status=${status}; sleeping ${POLL_SECONDS}s" ;;
    esac
    sleep "${POLL_SECONDS}"
done

# ------------------------------------------------- report session lifetime ---
print_remaining() {
    local created max_secs
    # Cloud TPU API nodes have no max-run-duration to report against.
    if (( TPU_API )); then
        echo "Session: Cloud TPU API node (no max-run-duration; delete when done)"
        return
    fi
    created="$(gcloud compute instances describe "${VM_NAME}" --zone="${ZONE}" \
        --format="value(creationTimestamp)" 2>/dev/null)"
    max_secs="$(gcloud compute instances describe "${VM_NAME}" --zone="${ZONE}" \
        --format="value(scheduling.maxRunDuration.seconds)" 2>/dev/null)"
    [[ -z "${created}" || -z "${max_secs}" ]] && return
    local start_epoch now_epoch used left
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

# --------------------------------------------------- wait for the sentinel ---
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
    # Show progress so a stalled pip is visible rather than silent.
    tail_line="$(ssh_vm "sudo tail -n 1 ${SETUP_LOG} 2>/dev/null")"
    printf '  [%4ds] %s\n' "${waited}" "${tail_line:-(waiting for startup-script)}"
    sleep "${POLL_SECONDS}"
    waited=$(( waited + POLL_SECONDS ))
done

# ------------------------------------------------------- sync local mrx ---
# Overlay the local working tree onto /mnt/data/mrx. Only git-tracked files are
# sent: the full directory is 568 MB because of untracked HDF5 output and
# notebooks, while the 170 tracked files are 1.7 MB, so this stays a few-second
# operation you can repeat every iteration.
#
# The tree is overlaid, not swapped in, so the VM keeps its editable install and
# its data/ files. The local git SHA and dirty-file count go into the log,
# because a measurement taken against an unrecorded working tree is not a
# measurement of anything.
if [[ "${SYNC_LOCAL_MRX}" == "1" ]]; then
    echo ""
    echo "Syncing ${LOCAL_MRX} -> ${MRX_DIR} (tracked files only)..."
    if [[ ! -d "${LOCAL_MRX}/.git" ]]; then
        echo "ERROR: ${LOCAL_MRX} is not a git checkout." >&2
        exit 1
    fi
    sync_sha="$(git -C "${LOCAL_MRX}" rev-parse --short HEAD)"
    sync_dirty="$(git -C "${LOCAL_MRX}" status --porcelain --untracked-files=no \
        | grep -c '' | tr -d ' ')"
    sync_tar="$(mktemp -t mrxsync).tgz"
    ( cd "${LOCAL_MRX}" && git ls-files -z | tar czf "${sync_tar}" --null -T - ) || {
        echo "ERROR: could not build the source tarball." >&2; exit 1; }
    echo "  ${sync_sha} (+${sync_dirty} modified tracked file(s)), $(du -h "${sync_tar}" | cut -f1)"

    if ! scp_to_vm "${sync_tar}" "/mnt/data/mrx_sync.tgz"; then
        rm -f "${sync_tar}"
        echo "ERROR: could not copy the source tarball to the VM." >&2
        exit 1
    fi
    rm -f "${sync_tar}"

    ssh_vm "cd ${MRX_DIR} && tar xzf /mnt/data/mrx_sync.tgz && \
            rm -f /mnt/data/mrx_sync.tgz && \
            find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null; \
            echo synced" | rg -q synced \
        || { echo "ERROR: unpacking the tarball on the VM failed." >&2; exit 1; }
    echo "  synced (pycache cleared)"
    SYNC_TAG="local ${sync_sha}+${sync_dirty}"
else
    SYNC_TAG="VM checkout"
fi

# --------------------------------------------------- push extra scripts ---
if [[ -n "${PUSH_FILES}" ]]; then
    echo ""
    echo "Pushing extra files into ${MRX_DIR}..."
    for f in ${PUSH_FILES}; do
        if [[ ! -f "${f}" ]]; then
            echo "ERROR: PUSH_FILES entry '${f}' is not a file." >&2
            exit 1
        fi
        scp_to_vm "${f}" "${MRX_DIR}/$(basename "${f}")" \
            || { echo "ERROR: could not push ${f}" >&2; exit 1; }
        echo "  ${f}"
    done
fi

# ------------------------------------------------------------ push + run ---
RUN_ARGS="$*"

# TPU_STDERR_LOG_LEVEL silences the libtpu driver chatter, which otherwise emits
# a "Could not open the log file ... Permission denied" pair several times a
# second and buries the actual results.
#
# The compilation cache is not a minor tuning knob here. MRX's inner solves run
# as eager lax.while_loops, so each call traces a fresh program and XLA compiles
# it again; on a v5e one apply_laplacian k=1 call costs ~10 s of compilation
# against ~20 ms of actual work. Turning the cache on takes that call from
# 9854 ms to 105 ms, a 93x difference, and it is the single largest effect
# measured on this hardware. The two thresholds are lowered because their
# defaults are tuned for a handful of large training programs and would skip
# the many small kernels this workload compiles.
COMMON_ENV="export PATH=/mnt/data/envs/mrx/bin:\$PATH; \
    export MPLBACKEND=Agg MRX_REPO=${MRX_DIR}; \
    export TPU_STDERR_LOG_LEVEL=3 TPU_MIN_LOG_LEVEL=3; \
    export JAX_COMPILATION_CACHE_DIR=${JAX_CACHE_DIR}; \
    export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0; \
    export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.1;"
[[ -n "${RUN_PLATFORM}" ]] && COMMON_ENV="${COMMON_ENV} export JAX_PLATFORMS=${RUN_PLATFORM};"

if [[ -n "${SCRIPT}" ]]; then
    if [[ -z "${OUTDIR}" ]]; then
        echo "ERROR: SCRIPT requires OUTDIR (the directory the script writes)." >&2
        exit 1
    fi
    echo ""
    print_remaining
    echo ""
    echo "Running ${SCRIPT} with MRX_DTYPE=${RUN_DTYPE}${RUN_PLATFORM:+ on ${RUN_PLATFORM}} [${SYNC_TAG}]..."
    echo "=================================================================="

    # cd into the repo: scripts such as li383_relaxation.py default to a
    # relative geometry path (data/wout_li383_low_res_reference.nc), which does
    # not resolve from the login shell's home directory.
    run_detached "$(basename "${SCRIPT}" .py)" \
        "cd ${MRX_DIR} && ${COMMON_ENV} export MRX_DTYPE=${RUN_DTYPE}; \
         ${PYBIN} -u ${SCRIPT} ${RUN_ARGS}"
    RUN_STATUS=$?
    REMOTE_OUT="${MRX_DIR}/${OUTDIR}"
else
    echo ""
    echo "Copying mrx_tpu_report.py to the VM..."
    scp_to_vm mrx_tpu_report.py "~/mrx_tpu_report.py" || {
        echo "ERROR: scp failed" >&2; exit 1; }

    print_remaining
    echo ""
    echo "Running MRX with MRX_DTYPE=${RUN_DTYPE}..."
    echo "=================================================================="

    # Detached here too. This mode is short, but it ran foreground until now and
    # so carried the same double-launch exposure that run_detached exists to
    # remove; there is no reason to keep two code paths with different failure
    # modes.
    run_detached "mrx_tpu_report" \
        "cd ${MRX_DIR} && ${COMMON_ENV} export MRX_DTYPE=${RUN_DTYPE}; \
         ${PYBIN} -u ~/mrx_tpu_report.py ${RUN_ARGS}"
    RUN_STATUS=$?
    REMOTE_OUT=/mnt/data/mrx_tpu_results
fi

echo "=================================================================="
echo ""

# --------------------------------------------------------- pull results ---
mkdir -p "${OUT_ROOT}"
echo "Pulling results into ${OUT_ROOT}/..."
scp_from_vm "${REMOTE_OUT}" "${OUT_ROOT}/" \
    && echo "  ${OUT_ROOT}/$(basename "${REMOTE_OUT}")/" \
    || echo "  WARNING: no results directory to copy"

print_remaining

if (( RUN_STATUS != 0 )); then
    echo ""
    echo "The run exited ${RUN_STATUS}. A non-zero exit here usually means the"
    echo "TPU results deviated from the CPU float32 reference; check summary.md."
fi
exit "${RUN_STATUS}"
