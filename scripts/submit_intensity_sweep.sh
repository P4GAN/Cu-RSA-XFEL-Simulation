#!/bin/bash
# SLURM array job for the transmittance-vs-intensity sweep.
#
# One array task == one independent SLURM job == (usually) its own node.
# CHUNKS_PER_CONFIG=4 below splits each config's 300 repetitions into 4
# chunks of 75 (2 parallel rounds on 40 cores each, ~4 min/task at ~2
# min/rep). All tasks run *concurrently* on different nodes instead of
# sequentially on a single JupyterHub session.
#
# Raising CHUNKS_PER_CONFIG further doesn't help unless REPS_PER_CHUNK
# drops to <=40 (one round instead of two) -- e.g. CHUNKS_PER_CONFIG=10
# (30 reps/chunk) hits that floor, but multiplies the task count by
# 2.5x, so only worth it if you know you have that much idle capacity.
# Every task writes into the same runs_seed_<E>_uJ/ folder, so the
# notebook's aggregation step (data_from_folder) doesn't need to change
# either way.
#
# The array size is NOT hardcoded: the block below counts manifest.txt's
# lines and resubmits itself via `sbatch --array=...` with the right bound,
# so it always matches however many configs DEFAULT_E_SEED_VALUES (in
# generate_intensity_sweep_configs.py) currently produces.
#
# Before submitting:
#   1. python scripts/generate_intensity_sweep_configs.py   (writes the manifest)
#   2. mkdir -p logs
#   3. edit the "adjust to your environment" block below
#   4. bash scripts/submit_intensity_sweep.sh   (NOT sbatch -- this computes
#      the array size, then calls sbatch itself)

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-transmittance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MANIFEST=config/generated/transmittance_vs_intensity/manifest.txt
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing $MANIFEST -- run scripts/generate_intensity_sweep_configs.py first" >&2
    exit 1
fi

NREP=300
CHUNKS_PER_CONFIG=4

# Not yet running as a SLURM array task -> compute the array bound from the
# manifest and resubmit ourselves as one.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    NUM_CONFIGS=$(wc -l < "$MANIFEST")
    TOTAL_TASKS=$(( NUM_CONFIGS * CHUNKS_PER_CONFIG ))
    echo "manifest has $NUM_CONFIGS configs -> submitting array 0-$((TOTAL_TASKS - 1)) ($TOTAL_TASKS tasks)"
    exec sbatch --array="0-$((TOTAL_TASKS - 1))" "$0"
fi

mapfile -t YAML_FILES < "$MANIFEST"
REPS_PER_CHUNK=$(( NREP / CHUNKS_PER_CONFIG ))

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / CHUNKS_PER_CONFIG ))
CHUNK_IDX=$(( SLURM_ARRAY_TASK_ID % CHUNKS_PER_CONFIG ))
REP_START=$(( CHUNK_IDX * REPS_PER_CHUNK ))
REP_END=$(( REP_START + REPS_PER_CHUNK ))

YAML=${YAML_FILES[$CONFIG_IDX]:-}
if [[ -z "$YAML" ]]; then
    echo "CONFIG_IDX $CONFIG_IDX has no entry in $MANIFEST (${#YAML_FILES[@]} configs) -- stale --array bound?" >&2
    exit 1
fi
DATA_PATH=data/sweep_${SLURM_ARRAY_JOB_ID}

echo "task $SLURM_ARRAY_TASK_ID -> config $CONFIG_IDX ($YAML), reps [$REP_START, $REP_END)"

python scripts/run_intensity_sweep.py \
    --yaml "$YAML" \
    --rep-start "$REP_START" --rep-end "$REP_END" \
    --nproc "$SLURM_CPUS_PER_TASK" \
    --data-path "$DATA_PATH"
