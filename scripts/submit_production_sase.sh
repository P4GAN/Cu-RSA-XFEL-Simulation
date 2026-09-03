#!/bin/bash
# SLURM array job for the 5 SASE production-comparison configs (2026-09-03 XATOM recompute):
#   Cu-seed-SASE-no-2s, Cu-seed-SASE, Cu-seed-SASE-satellite-no-L2, Cu-seed-SASE-satellite,
#   Cu-seed-SASE-double-satellite
#
# One array task == one config == one independent SLURM job/node -- fully isolated from every
# other task, so queueing (running now vs. hours from now vs. never because the partition is full)
# never causes interference:
#   - stdout/stderr: logs/%x_%A_%a.out/.err (SLURM's own job-id_array-task-id, always unique)
#   - data: DATA_PATH/<config-stem>/ (config-stem is unique per config -- unlike the E_seed-keyed
#     naming run_intensity_sweep.py uses, which would collide here since all 5 configs share
#     E_seed_uJ: 40)
# No task reads or writes any other task's files, so it doesn't matter whether SLURM runs all 5
# concurrently, serially, or anywhere in between.
#
# Before submitting:
#   1. mkdir -p logs
#   2. edit NREP / the #SBATCH block below if you want different resources
#   3. sbatch scripts/submit_production_sase.sh

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-production-sase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"
mkdir -p logs

CONFIGS=(
    config/base/Cu-seed-SASE-no-2s.yaml
    config/base/Cu-seed-SASE.yaml
    config/base/Cu-seed-SASE-satellite-no-L2.yaml
    config/base/Cu-seed-SASE-satellite.yaml
    config/base/Cu-seed-SASE-double-satellite.yaml
)

NREP=200
YAML=${CONFIGS[$SLURM_ARRAY_TASK_ID]:-}
if [[ -z "$YAML" ]]; then
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID has no entry in CONFIGS (${#CONFIGS[@]} configs)" >&2
    exit 1
fi
DATA_PATH=data/production_sase_${SLURM_ARRAY_JOB_ID}

echo "task $SLURM_ARRAY_TASK_ID -> $YAML, $NREP reps, $SLURM_CPUS_PER_TASK workers -> $DATA_PATH"

python scripts/run_production_config.py \
    --yaml "$YAML" \
    --rep-start 0 --rep-end "$NREP" \
    --nproc "$SLURM_CPUS_PER_TASK" \
    --data-path "$DATA_PATH"
