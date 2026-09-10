#!/bin/bash
# ============================================================================
# Re-trace every Poincare section under the given output roots -- one GPU job
# per run (relax.json + checkpoints/), scripts/poincare_trace.py in float32,
# writing each run's trace.npz. The pages are then drawn on the login node:
#   python scripts/poincare_plot.py <run>          (no GPU, seconds per page)
#
#   bash slurm/regen_poincare.sh                 # submit all
#   DRYRUN=1 bash slurm/regen_poincare.sh        # list what it would submit
#   ROOTS="outputs/li383_eta" bash slurm/regen_poincare.sh   # one root
#
# f32 is fine here: on li383 the trace matches f64 on every regular surface
# (see the session notes), and these states were relaxed in f32 anyway.
# ============================================================================
set -euo pipefail

# REPO holds the outputs (and the .venv); CODE holds the tracer that runs
# (mrx + scripts/poincare_trace.py). They differ only when tracing from a worktree.
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CODE=${CODE:-$REPO}                    # checkout the tracer imports/runs from
VENV=${VENV:-$REPO/.venv}              # main checkout's venv

ROOTS=${ROOTS:-"outputs/li383_eta outputs/li383_sweep"}
NAME_GLOB=${NAME_GLOB:-}              # if set, only state dirs matching it, e.g. "*_g1"
PLANES=${PLANES:-0,0.125,0.25,0.375,0.5}   # the standard five planes (half a period)
PERIODS=${PERIODS:-400}
FIELDS=${FIELDS:-ic,final}

PARTITION=${PARTITION:-gpu-h100}
ACCOUNT=${ACCOUNT:-extremedata}
TIMEOUT_MIN=${TIMEOUT_MIN:-60}
MEM_GB=${MEM_GB:-64}
CPUS_PER_TASK=${CPUS_PER_TASK:-32}
# Node with a broken cuSolver (handle creation fails); override with EXCLUDE=.
EXCLUDE=${EXCLUDE:-x3101c0s17b0n0}
DRYRUN=${DRYRUN:-0}

n=0
for root in $ROOTS; do
  [ -d "$REPO/$root" ] || { echo "skip (no dir): $root"; continue; }
  while IFS= read -r bh5; do          # bh5 is absolute
    dir=$(dirname "$bh5")
    if [ -n "$NAME_GLOB" ]; then       # filter by state-dir basename
      case "$(basename "$dir")" in $NAME_GLOB) ;; *) continue ;; esac
    fi
    n=$((n + 1))
    name="poinc_$(echo "${dir#"$REPO"/}" | sed 's#outputs/##; s#/#_#g')"
    CMD="set -euo pipefail; source $VENV/bin/activate; \
export PYTHONPATH=$CODE; export PYTHONUNBUFFERED=1; \
python -u $CODE/scripts/poincare_trace.py --run $dir --precision float32 \
--fields $FIELDS --planes $PLANES --periods $PERIODS"
    if [ "$DRYRUN" = "1" ]; then
      echo "[dryrun] $name  <-  ${bh5#"$REPO"/}  ->  ${dir#"$REPO"/}/trace.npz  (planes $PLANES, f32)"
      continue
    fi
    sbatch \
      ${EXCLUDE:+--exclude="${EXCLUDE}"} \
      --partition="${PARTITION}" --account="${ACCOUNT}" \
      --gpus-per-node=1 --cpus-per-task="${CPUS_PER_TASK}" \
      --time="${TIMEOUT_MIN}" --mem="${MEM_GB}G" \
      --job-name="${name}" --output="$dir/poincare_regen.log" \
      --wrap="${CMD}"
  done < <(find "$REPO/$root" -name relax.json | sort)
done
echo
echo "$([ "$DRYRUN" = "1" ] && echo would submit || echo submitted) $n job(s)."
