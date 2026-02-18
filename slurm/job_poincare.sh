#!/bin/bash
#SBATCH --job-name=poincare
#SBATCH --nodes=1
#SBATCH --account=extremedata
#SBATCH --time=6:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/tblickhan/mrx/logs/%x_%A_%a.out

# ============================================================================
# Poincaré Plot Generation SLURM Job Script with Hydra
# ============================================================================
#
# Usage:
#   sbatch slurm/job_poincare.sh "run_dir=out/relax_from_nfs/20260206_072421"
#
#   With overrides:
#   sbatch slurm/job_poincare.sh "run_dir=out/relax_from_nfs/20260206_072421 plotting.dpi=300 fieldline.n_vmap=32"
#
# ============================================================================

# --- Load modules and activate environment ---
module load python/3.11.4
cd /scratch/tblickhan/mrx
source .venv/bin/activate

# --- Hydra overrides (required: at least run_dir=...) ---
HYDRA_OVERRIDES="${1:-}"

# --- Run the plotting script ---
echo "Starting Poincaré plot generation at $(date)"
echo "Hydra overrides: $HYDRA_OVERRIDES"

python scripts/config_scripts/poincare_plots.py $HYDRA_OVERRIDES

echo "Job finished at $(date)"
