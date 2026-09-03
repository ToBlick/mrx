#!/bin/bash
# ============================================================================
# Regenerate every Poincare section under the given output roots -- one GPU
# job per B.h5 -- in float32, with the presentation .pgf next to each PNG.
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

# The repo this driver lives in (main checkout or a worktree); the job cds here
# and PYTHONPATH points at it so a worktree imports its own mrx, not main's.
# Override REPO= to run the plotter from one checkout against another's outputs.
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

ROOTS=${ROOTS:-"outputs/li383_eta outputs/li383_sweep"}
PLANES=${PLANES:-0,0.25,0.5}          # the standard three planes
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

n=0
for root in $ROOTS; do
  [ -d "$REPO/$root" ] || { echo "skip (no dir): $root"; continue; }
  while IFS= read -r bh5; do
    n=$((n + 1))
    dir=$(dirname "$bh5")
    name="poinc_$(echo "$dir" | sed 's#outputs/##; s#/#_#g')"
    log="$dir/poincare_regen.log"
    CMD="set -euo pipefail; cd $REPO; source .venv/bin/activate; \
export PYTHONPATH=$REPO; export PATH=$TEXBIN:\$PATH; export PYTHONUNBUFFERED=1; \
python -u scripts/poincare_relax.py $bh5 --precision float32 \
--fields $FIELDS --planes $PLANES --periods $PERIODS --out $dir/poincare"
    if [ "$DRYRUN" = "1" ]; then
      echo "[dryrun] $name  <-  $bh5  ->  $dir/poincare  (planes $PLANES, f32, pgf)"
      continue
    fi
    sbatch \
      ${EXCLUDE:+--exclude="${EXCLUDE}"} \
      --partition="${PARTITION}" --account="${ACCOUNT}" \
      --gpus-per-node=1 --cpus-per-task="${CPUS_PER_TASK}" \
      --time="${TIMEOUT_MIN}" --mem="${MEM_GB}G" \
      --job-name="${name}" --output="$REPO/${log}" \
      --wrap="${CMD}"
  done < <(find "$REPO/$root" -name B.h5 | sed "s#^$REPO/##" | sort)
done
echo
echo "$([ "$DRYRUN" = "1" ] && echo would submit || echo submitted) $n job(s)."
