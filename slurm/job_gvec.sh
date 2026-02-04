#!/bin/bash
#SBATCH --job-name=video_test
#SBATCH --nodes=1
#SBATCH --account=extremedata
#SBATCH --time=6:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/tblickhan/mrx/logs/%x_%A_%a.out

# ============================================================================
# GVEC Relaxation SLURM Job Script with Hydra
# ============================================================================
# 
# Usage:
#   Single run:     sbatch job_gvec.sh
#   With overrides: sbatch job_gvec.sh "fem.ns_r=16 fem.ns_theta=32"
#   Multirun:       sbatch job_gvec.sh "-m fem.ns_r=8,12,16"
#
# ============================================================================

# --- Load modules and activate environment ---
module load python/3.11.4
cd /scratch/tblickhan/mrx
source .venv/bin/activate

# --- Hydra overrides (optional) ---
HYDRA_OVERRIDES="${1:-}"

# --- Run the relaxation script ---
echo "Starting GVEC relaxation at $(date)"
echo "Hydra overrides: $HYDRA_OVERRIDES"

python scripts/config_scripts/relax_gvec.py $HYDRA_OVERRIDES

echo "Job finished at $(date)"
