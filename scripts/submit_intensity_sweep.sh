#!/bin/bash
# SLURM array job for the transmittance-vs-intensity sweep.
#
# One array task == one independent SLURM job == (usually) its own node.
# CHUNKS_PER_CONFIG=4 below splits each of the 10 E_seed configs' 300
# repetitions into 4 chunks of 75 (2 parallel rounds on 40 cores each,
# ~4 min/task at ~2 min/rep) -- 40 tasks total, comfortably under a >50
# idle-node budget. All of them run *concurrently* on different nodes
# instead of sequentially on a single JupyterHub session.
#
# Raising CHUNKS_PER_CONFIG further doesn't help unless REPS_PER_CHUNK
# drops to <=40 (one round instead of two) -- e.g. CHUNKS_PER_CONFIG=10
# (30 reps/chunk) hits that floor, but costs 100 tasks to get there, so
# only worth it if you know you have that much idle capacity. Every task
# writes into the same runs_seed_<E>_uJ/ folder, so the notebook's
# aggregation step (data_from_folder) doesn't need to change either way.
#
# Before submitting:
#   1. python scripts/generate_intensity_sweep_configs.py         (writes the manifest)
#   2. mkdir -p logs
#   3. edit the "adjust to your environment" block below
#   4. sbatch scripts/submit_intensity_sweep.sh

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-transmittance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=00:15:00
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

MANIFEST=config/generated/transmittance_vs_intensity/manifest.txt
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing $MANIFEST -- run scripts/generate_intensity_sweep_configs.py first" >&2
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

YAML=${YAML_FILES[$CONFIG_IDX]}
DATA_PATH=data/sweep_${SLURM_ARRAY_JOB_ID}

echo "task $SLURM_ARRAY_TASK_ID -> config $CONFIG_IDX ($YAML), reps [$REP_START, $REP_END)"

python scripts/run_intensity_sweep.py \
    --yaml "$YAML" \
    --rep-start "$REP_START" --rep-end "$REP_END" \
    --nproc "$SLURM_CPUS_PER_TASK" \
    --data-path "$DATA_PATH"
