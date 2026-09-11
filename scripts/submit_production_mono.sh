#!/bin/bash
# SLURM array job for the 2 monochromator production-comparison configs (2026-09-03 XATOM recompute):
#   Cu-seed-mono-SASE-original, Cu-seed-mono-SASE-double-satellite
#
# Separate submission from submit_production_sase.sh because these two are much more expensive:
# tgrid=12000 (vs. 2400 for the SASE configs) and, for the double-satellite one, nlevel=8 plus 7
# satellite blocks -- the same "whole node, generous time budget" treatment submit_mono_sweep.sh
# already gives the monochromator sweep, just one config per array task instead of fanning 8 light
# configs across one node (these two are heavy enough that each wants a node to itself).
#
# Same isolation guarantee as the SASE submission: one array task = one config = one independent
# job, logs auto-unique via %A_%a, data isolated by config-stem subfolder -- queueing order/overlap
# with the SASE array (or anything else on the cluster) cannot cause interference.
#
# Before submitting:
#   1. mkdir -p logs
#   2. edit NREP / the #SBATCH block below if you want different resources
#   3. sbatch scripts/submit_production_mono.sh

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-production-mono
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --array=0-1
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
    config/base/Cu-seed-mono-SASE-original.yaml
    config/base/Cu-seed-mono-SASE-double-satellite.yaml
)

NREP=10
YAML=${CONFIGS[$SLURM_ARRAY_TASK_ID]:-}
if [[ -z "$YAML" ]]; then
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID has no entry in CONFIGS (${#CONFIGS[@]} configs)" >&2
    exit 1
fi
DATA_PATH=data/production_mono_${SLURM_ARRAY_JOB_ID}

# NREP=10 < SLURM_CPUS_PER_TASK=40: cap workers at NREP, no point reserving idle workers.
NPROC=$(( NREP < SLURM_CPUS_PER_TASK ? NREP : SLURM_CPUS_PER_TASK ))

echo "task $SLURM_ARRAY_TASK_ID -> $YAML, $NREP reps, $NPROC workers -> $DATA_PATH"

python scripts/run_production_config.py \
    --yaml "$YAML" \
    --rep-start 0 --rep-end "$NREP" \
    --nproc "$NPROC" \
    --data-path "$DATA_PATH"
