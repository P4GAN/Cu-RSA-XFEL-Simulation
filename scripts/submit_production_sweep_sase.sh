#!/bin/bash
# SLURM array job: E_seed intensity sweep (E_seed = 60, 40, 9, 2 uJ, from
# generate_intensity_sweep_configs.py's DEFAULT_E_SEED_VALUES) for EACH of the 5 SASE-family
# 2026-09-03-XATOM-recompute production configs. Run scripts/generate_production_sweeps.sh first.
#
# 5 configs x 4 E_seed = 20 (config, E_seed) pairs, one array task per pair. NREP=200 (matching the
# reps/config already used for submit_production_sase.sh) fits in a single task on one 40-core node
# (~5 rounds x ~2 min/rep, no need for submit_intensity_sweep.sh's CHUNKS_PER_CONFIG fan-out, which
# exists to spread NREP=600 across multiple nodes).
#
# Isolation: DATA_PATH nests under the config name (data/production_sweep_sase_<jobid>/<config>/),
# so run_intensity_sweep.py's own E_seed-keyed subfolder naming (runs_seed_<E>_uJ/) -- which by
# itself would collide across configs, since all 5 share the same E_seed grid -- can't collide here.
# Logs use SLURM's own %A_%a, always unique regardless of queueing/concurrency.
#
# Before submitting:
#   1. mkdir -p logs
#   2. bash scripts/generate_production_sweeps.sh
#   3. sbatch scripts/submit_production_sweep_sase.sh

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-production-sweep-sase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --array=0-19
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
    Cu-seed-SASE-no-2s
    Cu-seed-SASE
    Cu-seed-SASE-satellite-no-L2
    Cu-seed-SASE-satellite
    Cu-seed-SASE-double-satellite
)
N_E_SEED=4  # must match generate_intensity_sweep_configs.py's DEFAULT_E_SEED_VALUES length

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / N_E_SEED ))
E_SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_E_SEED ))
NAME=${CONFIGS[$CONFIG_IDX]:-}
if [[ -z "$NAME" ]]; then
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID has no entry in CONFIGS (${#CONFIGS[@]} configs x $N_E_SEED)" >&2
    exit 1
fi

MANIFEST="config/generated/transmittance_vs_intensity/${NAME}/manifest.txt"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing $MANIFEST -- run scripts/generate_production_sweeps.sh first" >&2
    exit 1
fi
mapfile -t YAML_FILES < "$MANIFEST"
YAML=${YAML_FILES[$E_SEED_IDX]:-}
if [[ -z "$YAML" ]]; then
    echo "E_SEED_IDX $E_SEED_IDX has no entry in $MANIFEST (${#YAML_FILES[@]} entries) -- manifest doesn't match N_E_SEED=$N_E_SEED, rerun generate_production_sweeps.sh" >&2
    exit 1
fi

NREP=200
DATA_PATH=data/production_sweep_sase_${SLURM_ARRAY_JOB_ID}/${NAME}

echo "task $SLURM_ARRAY_TASK_ID -> config $NAME, E_seed entry $E_SEED_IDX ($YAML), $NREP reps -> $DATA_PATH"

python scripts/run_intensity_sweep.py \
    --yaml "$YAML" \
    --rep-start 0 --rep-end "$NREP" \
    --nproc "$SLURM_CPUS_PER_TASK" \
    --data-path "$DATA_PATH"
