#!/bin/bash
# SLURM array job for a mono-pulse numerical-convergence sweep (generate_mono_convergence_configs.py).
#
# Takes the config out-dir as $1 (unlike the other submit_*_sweep.sh scripts, which hardcode
# one manifest path) since this is meant to be run once per (grid-axis, E_seed range) case --
# e.g. once for the tgrid check and once for the xygrid check -- each into its own out-dir, so
# they don't clobber a shared manifest.txt the way two generate_*_sweep_configs.py calls at the
# same --out-dir would.
#
# Fan-out pattern copied from submit_mono_sweep.sh: NREP=5 reps/config is too few to fill a node
# on its own, so each array task runs CONFIGS_PER_TASK configs as concurrent background
# processes (CONFIGS_PER_TASK * NPROC_PER_CONFIG = --cpus-per-task). Must match
# CONFIGS_PER_TASK in generate_mono_convergence_configs.py.
#
# Before submitting:
#   1. python scripts/generate_mono_convergence_configs.py --out-dir <DIR> ...   (prints the
#      exact sbatch command to run, including --array)
#   2. mkdir -p logs
#   3. edit the "adjust to your environment" block below, after timing the largest config locally
#   4. run the sbatch command printed by step 1

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-mono-convergence
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

OUT_DIR=${1:?usage: sbatch submit_mono_convergence_sweep.sh <config-out-dir>}
MANIFEST="$OUT_DIR/manifest.txt"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing $MANIFEST -- run scripts/generate_mono_convergence_configs.py --out-dir $OUT_DIR first" >&2
    exit 1
fi
mapfile -t YAML_FILES < "$MANIFEST"

NREP=5
CONFIGS_PER_TASK=4
NPROC_PER_CONFIG=$(( SLURM_CPUS_PER_TASK / CONFIGS_PER_TASK ))

CONFIG_START=$(( SLURM_ARRAY_TASK_ID * CONFIGS_PER_TASK ))
DATA_PATH=data/$(basename "$OUT_DIR")_${SLURM_ARRAY_JOB_ID}

pids=()
n_launched=0
for (( i = 0; i < CONFIGS_PER_TASK; i++ )); do
    CONFIG_IDX=$(( CONFIG_START + i ))
    YAML=${YAML_FILES[$CONFIG_IDX]:-}
    if [[ -z "$YAML" ]]; then
        break  # last task: fewer than CONFIGS_PER_TASK configs remain in the manifest
    fi
    echo "task $SLURM_ARRAY_TASK_ID slot $i -> config $CONFIG_IDX ($YAML), reps [0, $NREP), $NPROC_PER_CONFIG workers"
    python scripts/run_mono_sweep.py \
        --yaml "$YAML" \
        --rep-start 0 --rep-end "$NREP" \
        --nproc "$NPROC_PER_CONFIG" \
        --data-path "$DATA_PATH" &
    pids+=("$!")
    n_launched=$(( n_launched + 1 ))
done

if (( n_launched == 0 )); then
    echo "CONFIG_START $CONFIG_START has no entries in $MANIFEST (${#YAML_FILES[@]} configs) -- --array doesn't match the manifest" >&2
    exit 1
fi

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
