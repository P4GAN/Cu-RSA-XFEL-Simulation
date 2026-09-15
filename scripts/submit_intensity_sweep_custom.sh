#!/bin/bash
# SLURM array job for a transmittance-vs-intensity sweep into a custom manifest dir, generalizing
# submit_intensity_sweep.sh (which hardcodes config/generated/transmittance_vs_intensity/manifest.txt
# -- fine for the one standing production sweep it's for, unsafe to reuse for an ad hoc E_seed sweep
# without clobbering it). Takes the config out-dir as $1, same convention as
# submit_grid_convergence_sweep.sh / submit_mono_convergence_sweep.sh.
#
# NREP=100 below is a deliberately reduced rep count for an investigative sweep, not full
# production statistics (that's submit_intensity_sweep.sh's NREP=600) -- bump it if the trend
# this is checking needs tighter error bars.
#
# Before submitting:
#   1. python scripts/generate_intensity_sweep_configs.py --out-dir <DIR> --e-seed ...   (prints
#      the exact sbatch command, including --array)
#   2. mkdir -p logs
#   3. edit the "adjust to your environment" block below, after timing the largest config
#   4. run the sbatch command printed by step 1

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-intensity-custom
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-27
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

OUT_DIR=${1:?usage: sbatch submit_intensity_sweep_custom.sh <config-out-dir>}
MANIFEST="$OUT_DIR/manifest.txt"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing $MANIFEST -- run scripts/generate_intensity_sweep_configs.py --out-dir $OUT_DIR first" >&2
    exit 1
fi
mapfile -t YAML_FILES < "$MANIFEST"

NREP=100
CHUNKS_PER_CONFIG=4
REPS_PER_CHUNK=$(( NREP / CHUNKS_PER_CONFIG ))

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / CHUNKS_PER_CONFIG ))
CHUNK_IDX=$(( SLURM_ARRAY_TASK_ID % CHUNKS_PER_CONFIG ))
REP_START=$(( CHUNK_IDX * REPS_PER_CHUNK ))
REP_END=$(( REP_START + REPS_PER_CHUNK ))

YAML=${YAML_FILES[$CONFIG_IDX]:-}
if [[ -z "$YAML" ]]; then
    echo "CONFIG_IDX $CONFIG_IDX has no entry in $MANIFEST (${#YAML_FILES[@]} configs) -- --array doesn't match the manifest" >&2
    exit 1
fi
DATA_PATH=data/$(basename "$OUT_DIR")_${SLURM_ARRAY_JOB_ID}

echo "task $SLURM_ARRAY_TASK_ID -> config $CONFIG_IDX ($YAML), reps [$REP_START, $REP_END)"

python scripts/run_intensity_sweep.py \
    --yaml "$YAML" \
    --rep-start "$REP_START" --rep-end "$REP_END" \
    --nproc "$SLURM_CPUS_PER_TASK" \
    --data-path "$DATA_PATH"
