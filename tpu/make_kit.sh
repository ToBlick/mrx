#!/usr/bin/env bash
#
# Build tpu_access_kit.zip, the self-contained bundle to hand to the group.
#
# This exists because the kit was assembled by hand once and there was then no
# way to tell which version of which script was inside it. Now the manifest is
# in one place and rebuilding is one command.
#
#   ./make_kit.sh
#
set -euo pipefail

KIT_NAME="tpu_access_kit"
OUT_ZIP="${KIT_NAME}.zip"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

# What ships is what someone else would RUN. The acquisition and session
# tooling, the benchmarks that answer "is my node healthy and how fast is it",
# and the precision guard. The one-off investigations that produced the numbers
# in results/benchmark_v5e_vs_cpu.md have since been deleted outright: each
# answered a question once, and a kit full of settled questions reads as a
# reading list rather than something to run. gpu_baseline.slurm is the one file
# still in the repo but out of the kit, because it needs a SLURM cluster.
FILES=(
    README.md
    TPU_GUIDE.md
    zones.sh
    check_quota.sh
    probe_capacity.sh
    launch_tpu.sh
    acquire_tpu.sh
    run_on_tpu.sh
    startup.sh
    watch_request.sh
    mrx_tpu_report.py
    tpu_bench_mrx.py
    matvec_bench.py
    summarize_matvec.py
    roofline.py
    map_precision.py
    profile_top_ops.py
    pmap_sweep.py
    gcs_cache_smoke.py
    make_kit.sh
)

missing=0
for f in "${FILES[@]}"; do
    [[ -f "${f}" ]] || { echo "MISSING: ${f}" >&2; missing=1; }
done
(( missing == 0 )) || { echo "Refusing to build an incomplete kit." >&2; exit 1; }

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT INT TERM
DEST="${STAGE}/${KIT_NAME}"
mkdir -p "${DEST}"

for f in "${FILES[@]}"; do
    cp "${f}" "${DEST}/"
done
# The shell entry points have to arrive executable; a zip preserves the mode.
chmod +x "${DEST}"/*.sh

if [[ -d results ]]; then
    mkdir -p "${DEST}/results"
    cp results/* "${DEST}/results/" 2>/dev/null || true
fi

rm -f "${OUT_ZIP}"
( cd "${STAGE}" && zip -qr "${HERE}/${OUT_ZIP}" "${KIT_NAME}" )

echo "Built ${OUT_ZIP}"
unzip -l "${OUT_ZIP}" | tail -n +2
