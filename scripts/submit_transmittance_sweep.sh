#!/bin/bash
# SLURM array job for the transmittance-vs-intensity sweep.
#
# One array task == one independent SLURM job == (usually) its own node.
# With CHUNKS_PER_CONFIG=1 below, each of the 10 E_seed configs gets one
# node and runs its 300 repetitions across that node's cores, same as the
# notebook's multiprocessing.Pool -- but all 10 configs run *concurrently*
# on different nodes instead of one after another on a single JupyterHub
# session.
#
# To spread across MORE nodes than there are configs (e.g. you have queue
# room for 30 nodes, not just 10), raise CHUNKS_PER_CONFIG and widen
# --array to match: CHUNKS_PER_CONFIG=3 with --array=0-29 splits each
# config's 300 repetitions into 3 chunks of 100, each chunk an independent
# task/node. Every task writes into the same runs_seed_<E>_uJ/ folder, so
# the notebook's aggregation step (data_from_folder) doesn't need to change
# either way.
#
# Before submitting:
#   1. python scripts/generate_sweep_configs.py         (writes the manifest)
#   2. mkdir -p logs
#   3. edit the "adjust to your environment" block below
#   4. sbatch scripts/submit_transmittance_sweep.sh

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-transmittance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# --- adjust to your environment ---
# module purge
# module load maxwell/python  # or whatever module gives you the right python
# source ~/miniconda3/bin/activate xlo-sim
# -----------------------------------

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

MANIFEST=config/generated/transmittance_vs_intensity/manifest.txt
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing $MANIFEST -- run scripts/generate_sweep_configs.py first" >&2
    exit 1
fi
mapfile -t YAML_FILES < "$MANIFEST"

NREP=300
CHUNKS_PER_CONFIG=1
REPS_PER_CHUNK=$(( NREP / CHUNKS_PER_CONFIG ))

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / CHUNKS_PER_CONFIG ))
CHUNK_IDX=$(( SLURM_ARRAY_TASK_ID % CHUNKS_PER_CONFIG ))
REP_START=$(( CHUNK_IDX * REPS_PER_CHUNK ))
REP_END=$(( REP_START + REPS_PER_CHUNK ))

YAML=${YAML_FILES[$CONFIG_IDX]}
DATA_PATH=data/sweep_${SLURM_ARRAY_JOB_ID}

echo "task $SLURM_ARRAY_TASK_ID -> config $CONFIG_IDX ($YAML), reps [$REP_START, $REP_END)"

python scripts/run_transmittance_sweep.py \
    --yaml "$YAML" \
    --rep-start "$REP_START" --rep-end "$REP_END" \
    --nproc "$SLURM_CPUS_PER_TASK" \
    --data-path "$DATA_PATH"
