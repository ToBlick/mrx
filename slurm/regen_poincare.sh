#!/bin/bash
# ============================================================================
# Regenerate every Poincare section under the given output roots -- one GPU
# job per run (relax.json + checkpoints/) -- in float32, with the presentation .pgf next to each PNG.
#
#   bash slurm/regen_poincare.sh                 # submit all
#   DRYRUN=1 bash slurm/regen_poincare.sh        # list what it would submit
#   ROOTS="outputs/li383_eta" bash slurm/regen_poincare.sh   # one root
#
# Needs the poincare-plotter plotter (logical-r profile + .pgf); it runs the
# scripts/poincare_relax.py sitting next to this driver, so run it from a
# checkout (or worktree) that has those. The .pgf needs xelatex on PATH -- the
# home TeX Live is used by default (TEXBIN); without it the PNG is still written
# and the PGF is skipped. f32 is fine here: on li383 the trace matches f64 on
# every regular surface (see the session notes), and these states were relaxed
# in f32 anyway.
# ============================================================================
set -euo pipefail

# REPO holds the outputs (and the .venv); CODE holds the plotter that runs
# (mrx + scripts/poincare_relax.py). They are the same after the branch is
# merged; before then, point CODE at the poincare-plotter worktree and REPO at
# the main checkout (its outputs), e.g.
#   REPO=/scratch/tblickhan/mrx CODE=/scratch/tblickhan/mrx/.claude/worktrees/poincare \
#     bash slurm/regen_poincare.sh
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CODE=${CODE:-$REPO}                    # checkout the plotter imports/runs from
VENV=${VENV:-$REPO/.venv}              # main checkout's venv

ROOTS=${ROOTS:-"outputs/li383_eta outputs/li383_sweep"}
NAME_GLOB=${NAME_GLOB:-}              # if set, only state dirs matching it, e.g. "*_g1"
PLANES=${PLANES:-0,0.125,0.25,0.375,0.5}   # the standard five planes (half a period)
PERIODS=${PERIODS:-400}
FIELDS=${FIELDS:-ic,final}
TEXBIN=${TEXBIN:-$HOME/texlive/2026/bin/x86_64-linux}

PARTITION=${PARTITION:-gpu-h100}
ACCOUNT=${ACCOUNT:-extremedata}
TIMEOUT_MIN=${TIMEOUT_MIN:-60}
MEM_GB=${MEM_GB:-64}
CPUS_PER_TASK=${CPUS_PER_TASK:-32}
# Node with a broken cuSolver (handle creation fails); override with EXCLUDE=.
EXCLUDE=${EXCLUDE:-x3101c0s17b0n0}
DRYRUN=${DRYRUN:-0}
FROMNPZ=${FROMNPZ:-0}                 # 1 = re-render plots from an existing sections.npz, no re-trace

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
    fromnpz=""; [ "$FROMNPZ" = "1" ] && fromnpz="--from-npz"   # re-render only
    CMD="set -euo pipefail; source $VENV/bin/activate; \
export PYTHONPATH=$CODE; export PATH=$TEXBIN:\$PATH; export PYTHONUNBUFFERED=1; \
python -u $CODE/scripts/poincare_relax.py $dir $fromnpz --precision float32 \
--fields $FIELDS --planes $PLANES --periods $PERIODS --out $dir/poincare"
    if [ "$DRYRUN" = "1" ]; then
      echo "[dryrun] $name  <-  ${bh5#"$REPO"/}  ->  ${dir#"$REPO"/}/poincare  (planes $PLANES, f32, pgf$([ "$FROMNPZ" = "1" ] && echo ', from-npz re-render'))"
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
