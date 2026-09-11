#!/bin/bash
# SLURM array job for a t/xy/z grid numerical-convergence sweep, generalized over
# submit_tgrid_sweep.sh / submit_xygrid_sweep.sh / submit_zgrid_sweep.sh (identical apart from
# the hardcoded manifest path, job-name and NREP/resource sizing). Takes the config out-dir as
# $1 so the same script covers every (axis, E_seed) case -- e.g. run once per
# config/generated/convergence_tgrid_2uJ, convergence_tgrid_60uJ, convergence_xygrid_2uJ, ...
# -- without each needing its own manifest path baked in.
#
# CHUNKS_PER_CONFIG=2 below splits each config's NREP repetitions into 2 chunks, matching the
# repo's other convergence sweeps -- this is about grid resolution, not SASE-shot statistics, so
# NREP=20 (paired same-random_seed comparison across grid values, per run_convergence_sweep.py's
# docstring) rather than the 100-300 used for production statistics sweeps.
#
# --time/--mem/--cpus-per-task below are a generic starting budget -- this config
# (double-satellite + L2, 8 coupled blocks) is more expensive per rep than the plain base config
# the original 3 scripts default to, so time the largest config in your manifest locally first
# and adjust before submitting the full array.
#
# Before submitting:
#   1. python scripts/generate_Xgrid_sweep_configs.py --base-yaml <E_seed-specific yaml> \
#        --out-dir <DIR> --Xgrid <values...>   (prints the exact sbatch command, incl. --array)
#   2. mkdir -p logs
#   3. edit the "adjust to your environment" block below, after timing the largest config
#   4. run the sbatch command printed by step 1

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-grid-convergence
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

OUT_DIR=${1:?usage: sbatch submit_grid_convergence_sweep.sh <config-out-dir>}
MANIFEST="$OUT_DIR/manifest.txt"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing $MANIFEST -- run the matching generate_*grid_sweep_configs.py --out-dir $OUT_DIR first" >&2
    exit 1
fi
mapfile -t YAML_FILES < "$MANIFEST"

NREP=20
CHUNKS_PER_CONFIG=2
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

python scripts/run_convergence_sweep.py \
    --yaml "$YAML" \
    --rep-start "$REP_START" --rep-end "$REP_END" \
    --nproc "$SLURM_CPUS_PER_TASK" \
    --data-path "$DATA_PATH"
