#!/bin/bash
# SLURM array job: full E_seed x energy monochromator sweep (5 E_seed x 52 energies, from
# generate_mono_sweep_configs.py's defaults) for EACH of the 2 mono-family 2026-09-03-XATOM-recompute
# production configs (Cu-seed-mono-SASE-original, Cu-seed-mono-SASE-double-satellite). Run
# scripts/generate_production_sweeps.sh first.
#
# Same fan-out design as submit_mono_sweep.sh (CONFIGS_PER_TASK=8 concurrent background processes
# per array task, since NREP=10 is too few reps to fill a 40-core node on its own), doubled with an
# outer loop over the 2 base configs: TASKS_PER_CONFIG = ceil(260/8) = 33 array tasks per config,
# so 66 tasks total (array=0-65). --mem=0 (whole node) per task, same reasoning as
# submit_mono_sweep.sh -- can easily want 100s of GB across 8 concurrent tgrid=12000 configs.
#
# Isolation: DATA_PATH nests under the config name
# (data/production_sweep_mono_<jobid>/<config>/), so run_mono_sweep.py's own
# E_seed+energy-keyed subfolder naming (runs_seed_<E>_uJ__energy_<E>_eV/) -- which by itself would
# collide across the 2 configs, since both share the same E_seed x energy grid -- can't collide
# here. Logs use SLURM's own %A_%a, always unique regardless of queueing/concurrency.
#
# Before submitting:
#   1. mkdir -p logs
#   2. bash scripts/generate_production_sweeps.sh
#   3. sbatch scripts/submit_production_sweep_mono.sh

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-production-sweep-mono
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=16:00:00
#SBATCH --array=0-65
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
    Cu-seed-mono-SASE-original
    Cu-seed-mono-SASE-double-satellite
)

# NREP * CONFIGS_PER_TASK must equal --cpus-per-task above (matches submit_mono_sweep.sh's
# convention), and TASKS_PER_CONFIG must equal ceil(260 / CONFIGS_PER_TASK).
NREP=10
CONFIGS_PER_TASK=8
TASKS_PER_CONFIG=33
NPROC_PER_CONFIG=$(( SLURM_CPUS_PER_TASK / CONFIGS_PER_TASK ))

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / TASKS_PER_CONFIG ))
INNER_TASK_ID=$(( SLURM_ARRAY_TASK_ID % TASKS_PER_CONFIG ))
NAME=${CONFIGS[$CONFIG_IDX]:-}
if [[ -z "$NAME" ]]; then
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID has no entry in CONFIGS (${#CONFIGS[@]} configs x $TASKS_PER_CONFIG)" >&2
    exit 1
fi

MANIFEST="config/generated/mono_transmittance_vs_intensity/${NAME}/manifest.txt"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing $MANIFEST -- run scripts/generate_production_sweeps.sh first" >&2
    exit 1
fi
mapfile -t YAML_FILES < "$MANIFEST"

CONFIG_START=$(( INNER_TASK_ID * CONFIGS_PER_TASK ))
DATA_PATH=data/production_sweep_mono_${SLURM_ARRAY_JOB_ID}/${NAME}

pids=()
n_launched=0
for (( i = 0; i < CONFIGS_PER_TASK; i++ )); do
    ENTRY_IDX=$(( CONFIG_START + i ))
    YAML=${YAML_FILES[$ENTRY_IDX]:-}
    if [[ -z "$YAML" ]]; then
        break  # last task for this config: fewer than CONFIGS_PER_TASK entries remain
    fi
    echo "task $SLURM_ARRAY_TASK_ID ($NAME) slot $i -> entry $ENTRY_IDX ($YAML), reps [0, $NREP), $NPROC_PER_CONFIG workers"
    python scripts/run_mono_sweep.py \
        --yaml "$YAML" \
        --rep-start 0 --rep-end "$NREP" \
        --nproc "$NPROC_PER_CONFIG" \
        --data-path "$DATA_PATH" &
    pids+=("$!")
    n_launched=$(( n_launched + 1 ))
done

if (( n_launched == 0 )); then
    echo "CONFIG_START $CONFIG_START ($NAME) has no entries in $MANIFEST (${#YAML_FILES[@]} entries) -- rerun scripts/generate_production_sweeps.sh" >&2
    exit 1
fi

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
