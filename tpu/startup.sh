#!/bin/bash
#
# TPU VM startup script.
#
# Builds the whole MRX toolchain on /mnt/data rather than the boot disk. The boot
# disk dies with the instance at max-run-duration; the data disk is attached with
# auto-delete=no and survives, so a 30-minute session spends ~1 minute on setup
# instead of the ~10-12 minutes a cold install costs.
#
# Everything below the mount is idempotent and guarded by a sentinel, so a
# re-created VM reuses the existing environment.

set -uo pipefail

# ---------------------------------------------------------------- data disk ---
# Everything downstream uses /mnt/data. Normally that is the persistent
# my-data-disk, but not every machine type can take it: ct5p-hightpu-4t (v5p)
# rejects hyperdisk-balanced outright with "hyperdisk-balanced disk type cannot
# be used by ct5p-hightpu-4t machine type". Rather than forfeit those zones,
# fall back to a boot-disk directory at the same path and carry on. The only
# thing lost is persistence, which is an optimisation, not a requirement.
sudo mkdir -p /mnt/data
PERSISTENT=0

if [ -e /dev/disk/by-id/google-data-disk ]; then
    # Format only if there is no filesystem yet, so we never wipe a disk that
    # already carries the environment.
    if [ -z "$(sudo blkid /dev/disk/by-id/google-data-disk)" ]; then
        sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/disk/by-id/google-data-disk
    fi
    sudo mount -o discard,defaults /dev/disk/by-id/google-data-disk /mnt/data

    if mountpoint -q /mnt/data; then
        PERSISTENT=1
        # Survive a reboot. The startup script re-mounts anyway, but an fstab
        # entry means a manual reboot does not silently drop the environment.
        if ! grep -q "google-data-disk" /etc/fstab 2>/dev/null; then
            echo "/dev/disk/by-id/google-data-disk /mnt/data ext4 discard,defaults,nofail 0 2" \
                | sudo tee -a /etc/fstab >/dev/null
        fi
    else
        # The device exists but would not mount. Building on top of a failed
        # mount would put the environment and the sentinel on the boot disk
        # while everything reported success, so say so loudly in the log.
        echo "WARNING: google-data-disk present but failed to mount; using boot disk" \
            | sudo tee /var/log/mrx-setup-warning.log >&2
    fi
fi

sudo chmod a+w /mnt/data

# ------------------------------------------------------------------ logging ---
SETUP_LOG=/mnt/data/setup.log
SENTINEL=/mnt/data/.mrx_env_ready
exec > >(sudo tee -a "${SETUP_LOG}") 2>&1

echo "=================================================================="
echo "startup.sh  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
if [ "${PERSISTENT}" -eq 1 ]; then
    echo "/mnt/data is the persistent data disk; the environment survives."
else
    echo "/mnt/data is on the BOOT disk; the environment dies with this VM."
fi
echo "=================================================================="

FORGE_DIR=/mnt/data/miniforge3
ENV_DIR=/mnt/data/envs/mrx
REPO_DIR=/mnt/data/mrx
MRX_BRANCH=static-dynamic-refactor
PY_VERSION=3.12
# The one geometry file the finite-beta tutorial needs. It is whitelisted in
# .gitignore (data/* is otherwise ignored), so it arrives with the clone.
GEOM=data/wout_li383_low_res_reference.nc

# libtpu writes here, and the directory is created by this script running as
# root. Without the mode bits the ordinary login user cannot open a log file,
# and libtpu emits a "Could not open the log file ... Permission denied" pair
# several times a second, which buries real tracebacks in the run output.
sudo mkdir -p /tmp/tpu_logs
sudo chmod 1777 /tmp/tpu_logs

if [ -f "${SENTINEL}" ]; then
    echo "Environment already present (${SENTINEL}); skipping build."
    # A warm data disk can carry a checkout older than the data commits, so
    # refresh it even when the build itself is skipped.
    if [ -d "${REPO_DIR}/.git" ]; then
        echo "--- refreshing ${REPO_DIR} (${MRX_BRANCH}) ---"
        git -C "${REPO_DIR}" fetch --quiet origin "${MRX_BRANCH}" || true
        git -C "${REPO_DIR}" checkout --quiet "${MRX_BRANCH}" || true
        git -C "${REPO_DIR}" pull --quiet --ff-only origin "${MRX_BRANCH}" || true
        if [ -f "${REPO_DIR}/${GEOM}" ]; then
            echo "    ${GEOM} present"
        else
            echo "    WARNING: ${GEOM} missing; the li383 tutorial will not run"
        fi
        sudo chmod -R a+rwX "${REPO_DIR}"
    fi
    echo "Delete the sentinel to force a rebuild."
    exit 0
fi

# ----------------------------------------------------------------- miniforge ---
if [ ! -x "${FORGE_DIR}/bin/conda" ]; then
    echo "--- installing Miniforge to ${FORGE_DIR} ---"
    curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash /tmp/miniforge.sh -b -p "${FORGE_DIR}"
    rm -f /tmp/miniforge.sh
else
    echo "--- Miniforge already installed ---"
fi

export PATH="${FORGE_DIR}/bin:${PATH}"

# ----------------------------------------------------------------- conda env ---
if [ ! -x "${ENV_DIR}/bin/python" ]; then
    echo "--- creating conda env at ${ENV_DIR} (python ${PY_VERSION}) ---"
    "${FORGE_DIR}/bin/conda" create -y -p "${ENV_DIR}" "python=${PY_VERSION}"
else
    echo "--- conda env already exists ---"
fi

PY="${ENV_DIR}/bin/python"

# ------------------------------------------------------------------- jax/tpu ---
# The documented install path for Cloud TPU. This pulls jax, jaxlib and libtpu
# matched to each other; it works inside a conda env as long as this is the only
# process holding the TPU.
echo "--- installing jax[tpu] ---"
"${PY}" -m pip install --upgrade pip
"${PY}" -m pip install "jax[tpu]" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# etils is what lets JAX put its persistent compilation cache on a gs:// path,
# which is the only way to carry compiled kernels onto a node with no data
# disk. It is installed unconditionally because without it JAX does not fail on
# a gs:// cache dir -- it writes nothing, reads nothing and reports nothing, so
# the cache silently does not exist and the only symptom is a slow run.
"${PY}" -m pip install "etils[epath,epath-gcs]"

# ----------------------------------------------------------------------- mrx ---
# The branch matters. `main` is ten months stale and its Laplacian assembly path
# raises a reshape TypeError inside the k=0 tensor Hodge preconditioner;
# static-dynamic-refactor passes its full suite and is the only branch where
# float32 is a supported, first-class mode.
if [ ! -d "${REPO_DIR}/.git" ]; then
    echo "--- cloning mrx @ ${MRX_BRANCH} ---"
    git clone --branch "${MRX_BRANCH}" --depth 1 \
        https://github.com/ToBlick/mrx.git "${REPO_DIR}"
else
    echo "--- mrx repo present; fast-forwarding ${MRX_BRANCH} ---"
    # The clone is shallow, so `pull --ff-only` can refuse; fetch the tip and
    # reset onto it instead. Nothing on the VM is worth preserving.
    git -C "${REPO_DIR}" fetch --depth 1 origin "${MRX_BRANCH}" \
        && git -C "${REPO_DIR}" reset --hard "origin/${MRX_BRANCH}" \
        || echo "    WARNING: could not refresh the checkout"
fi

# The tutorial geometry is committed but sits under an otherwise-ignored data/
# directory, so a stale checkout silently lacks it and the run fails much later
# with a confusing file-not-found from inside the VMEC reader.
if [ -f "${REPO_DIR}/${GEOM}" ]; then
    echo "    ${GEOM} present ($(du -h "${REPO_DIR}/${GEOM}" | cut -f1))"
else
    echo "    WARNING: ${GEOM} missing; the li383 tutorial will not run"
fi

# This script runs as root, so the clone lands root-owned and the ordinary
# login user cannot create the tutorials' outputs/ directory. Without this the
# first write fails with PermissionError several minutes into a run.
sudo chmod -R a+rwX "${REPO_DIR}"

echo "--- installing mrx (editable) ---"
cd "${REPO_DIR}"
"${PY}" -m pip install -e .

# ------------------------------------------------------------------- env vars ---
# MRX reads MRX_DTYPE at import to pick its working precision; float32 is the
# only sane choice on TPU, which has no native 64-bit path. Agg because the VM
# is headless and the tutorials save figures.
cat <<'PROFILE' | sudo tee /etc/profile.d/mrx.sh >/dev/null
export PATH="/mnt/data/envs/mrx/bin:/mnt/data/miniforge3/bin:${PATH}"
export MRX_DTYPE=float32
export MPLBACKEND=Agg
export MRX_REPO=/mnt/data/mrx
PROFILE
sudo chmod 0644 /etc/profile.d/mrx.sh

# ------------------------------------------------------------------ smoke test ---
echo "--- smoke test: jax devices + mrx precision ---"
MRX_DTYPE=float32 "${PY}" -c "
import jax, mrx
print('jax', jax.__version__)
print('devices', jax.devices())
print('device_count', jax.device_count())
print('mrx DTYPE', mrx.DTYPE, 'EPS', mrx.EPS)
print('matmul precision', jax.config.jax_default_matmul_precision)
"
SMOKE=$?

if [ "${SMOKE}" -eq 0 ]; then
    sudo touch "${SENTINEL}"
    echo "=================================================================="
    echo "SETUP COMPLETE  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "=================================================================="
else
    echo "=================================================================="
    echo "SETUP FAILED (smoke test exit ${SMOKE}); sentinel not written"
    echo "=================================================================="
    exit "${SMOKE}"
fi
