#!/bin/bash
# SLURM array job for the mono-transmittance-vs-intensity sweep.
#
# Counterpart of submit_transmittance_sweep.sh, but the grid has two physical
# parameters (E_seed x absolute target photon energy) instead of one. That
# doesn't change the array-indexing logic below: generate_mono_sweep_configs.py
# flattens the E_seed x energy grid into a single manifest.txt, so each
# array task still just looks up one YAML path by CONFIG_IDX, exactly like
# the 1D sweeps.
#
# With CHUNKS_PER_CONFIG=1, each of the 110 (E_seed, energy) configs gets
# one node and runs its NREP repetitions across that node's cores. Raise
# CHUNKS_PER_CONFIG (and widen --array to match) to spread across more nodes
# if you have queue room -- every task for a given config writes into the
# same runs_seed_<E>_uJ__energy_<E_target>_eV/ folder, so the notebook's
# aggregation step (data_from_folder) doesn't need to change either way.
#
# Before submitting:
#   1. python scripts/generate_mono_sweep_configs.py     (writes the manifest)
#   2. mkdir -p logs
#   3. edit the "adjust to your environment" block below
#   4. sbatch scripts/submit_mono_sweep.sh

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-mono-transmittance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --array=0-109%40
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

MANIFEST=config/generated/mono_transmittance_vs_intensity/manifest.txt
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing $MANIFEST -- run scripts/generate_mono_sweep_configs.py first" >&2
    exit 1
fi
mapfile -t YAML_FILES < "$MANIFEST"

NREP=20
CHUNKS_PER_CONFIG=1
REPS_PER_CHUNK=$(( NREP / CHUNKS_PER_CONFIG ))

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / CHUNKS_PER_CONFIG ))
CHUNK_IDX=$(( SLURM_ARRAY_TASK_ID % CHUNKS_PER_CONFIG ))
REP_START=$(( CHUNK_IDX * REPS_PER_CHUNK ))
REP_END=$(( REP_START + REPS_PER_CHUNK ))

YAML=${YAML_FILES[$CONFIG_IDX]}
DATA_PATH=data/mono_sweep_${SLURM_ARRAY_JOB_ID}

echo "task $SLURM_ARRAY_TASK_ID -> config $CONFIG_IDX ($YAML), reps [$REP_START, $REP_END)"

python scripts/run_mono_transmittance_sweep.py \
    --yaml "$YAML" \
    --rep-start "$REP_START" --rep-end "$REP_END" \
    --nproc "$SLURM_CPUS_PER_TASK" \
    --data-path "$DATA_PATH"
