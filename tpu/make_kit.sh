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
# and the precision guard. Deliberately NOT the one-off investigations that
# back results/benchmark_v5e_vs_cpu.md -- factorization_ab.py, mxu_occupancy.py,
# shifted_atom_measure.py, dispatch_ab.py and the two NYU .slurm files. Those
# answered a question once, are cited from the write-up, and live in the repo;
# putting them in the kit would make it look like a reading list rather than
# something to run. The README's file table marks the split.
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
