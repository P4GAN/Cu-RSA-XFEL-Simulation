#!/bin/bash
# SLURM array job for the transmittance-vs-duration sweep.
# Counterpart of submit_intensity_sweep.sh -- see that file's header for
# how CHUNKS_PER_CONFIG trades off node count vs. reps per task.
#
# CHUNKS_PER_CONFIG=4 -> 75 reps/chunk, already <=40 so each task is a
# single parallel round (~2 min) -- the speed floor for this NREP. Going
# finer wastes tasks without going faster.
#
# --array below is NOT authoritative: generate_duration_sweep_configs.py
# prints the exact `sbatch --array=...` command to run after it (re)writes
# the manifest, since only it knows how many configs DEFAULT_DURATION_VALUES
# currently produces. A CLI --array overrides the pragma below, so always
# submit with the printed command -- the pragma is just a stale-safe fallback.
#
# Before submitting:
#   1. python scripts/generate_duration_sweep_configs.py   (writes the
#      manifest and prints the sbatch command to run)
#   2. mkdir -p logs
#   3. edit the "adjust to your environment" block below
#   4. run the sbatch command printed by step 1

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-duration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --array=0-39
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

MANIFEST=config/generated/transmittance_vs_duration/manifest.txt
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing $MANIFEST -- run scripts/generate_duration_sweep_configs.py first" >&2
    exit 1
fi
mapfile -t YAML_FILES < "$MANIFEST"

NREP=300
CHUNKS_PER_CONFIG=4
REPS_PER_CHUNK=$(( NREP / CHUNKS_PER_CONFIG ))

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / CHUNKS_PER_CONFIG ))
CHUNK_IDX=$(( SLURM_ARRAY_TASK_ID % CHUNKS_PER_CONFIG ))
REP_START=$(( CHUNK_IDX * REPS_PER_CHUNK ))
REP_END=$(( REP_START + REPS_PER_CHUNK ))

YAML=${YAML_FILES[$CONFIG_IDX]:-}
if [[ -z "$YAML" ]]; then
    echo "CONFIG_IDX $CONFIG_IDX has no entry in $MANIFEST (${#YAML_FILES[@]} configs) -- --array doesn't match the manifest; rerun scripts/generate_duration_sweep_configs.py and use the sbatch command it prints" >&2
    exit 1
fi
DATA_PATH=data/duration_sweep_${SLURM_ARRAY_JOB_ID}

echo "task $SLURM_ARRAY_TASK_ID -> config $CONFIG_IDX ($YAML), reps [$REP_START, $REP_END)"

python scripts/run_duration_sweep.py \
    --yaml "$YAML" \
    --rep-start "$REP_START" --rep-end "$REP_END" \
    --nproc "$SLURM_CPUS_PER_TASK" \
    --data-path "$DATA_PATH"
