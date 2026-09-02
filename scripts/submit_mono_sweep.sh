#!/bin/bash
# SLURM array job for the mono-transmittance-vs-intensity sweep.
#
# Counterpart of submit_intensity_sweep.sh, but the grid has two physical
# parameters (E_seed x absolute target photon energy) instead of one. That
# doesn't change the array-indexing logic below: generate_mono_sweep_configs.py
# flattens the E_seed x energy grid into a single manifest.txt.
#
# Unlike the 1D sweeps, each array task here handles CONFIGS_PER_TASK=8
# configs, not one: the monochromator only needs NREP=5 repetitions per
# config to average (vs. 25+ for broadband SASE), too few to fill a 40-core
# node on its own -- a pool sized for 40 workers but fed only 5 tasks only
# ever runs 5 of them concurrently. Instead this launches 8 configs'
# run_mono_sweep.py as concurrent background processes, each with its own
# --nproc 5 worker pool, so all 40 cores are busy at once (8 * 5 = 40 =
# --cpus-per-task). Each background process still checkpoints/saves
# independently (see tools.run_sweep_chunk), so one config's worker dying
# doesn't affect the other 7.
#
# The mono base config (config/base/Cu-seed-mono-SASE.yaml) uses tgrid=12000
# vs. 2400 for the 1D sweeps (config/base/Cu-seed-SASE.yaml) -- 5x finer time
# resolution to resolve the monochromator's narrow bandwidth. That makes each
# rep far more expensive,
# both in compute and memory, so --mem=0 (grab the whole node) instead of a
# fixed cap: running 8 configs' worth of workers concurrently (40 total,
# vs. only ever 25 concurrent under the old one-config-per-task scheme) can
# easily want 300+ GB between them.
#
# --array below is NOT authoritative: generate_mono_sweep_configs.py prints
# the exact `sbatch --array=...` command to run after it (re)writes the
# manifest, since only it knows how many configs len(E_seed) x len(energy)
# currently produces, and hence how many ceil(n_configs / 8) array tasks are
# needed. A CLI --array overrides the pragma below, so always submit with
# the printed command -- the pragma is just a stale-safe fallback.
#
# Before submitting:
#   1. python scripts/generate_mono_sweep_configs.py   (writes the manifest
#      and prints the sbatch command to run)
#   2. mkdir -p logs
#   3. edit the "adjust to your environment" block below
#   4. run the sbatch command printed by step 1

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-mono-transmittance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --array=0-32
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

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

# NREP * CONFIGS_PER_TASK must equal --cpus-per-task above, so every config
# gets exactly enough workers for its own reps and all of them run in one
# parallel wave -- these must match generate_mono_sweep_configs.py's
# CONFIGS_PER_TASK.
NREP=5
CONFIGS_PER_TASK=8
NPROC_PER_CONFIG=$(( SLURM_CPUS_PER_TASK / CONFIGS_PER_TASK ))

CONFIG_START=$(( SLURM_ARRAY_TASK_ID * CONFIGS_PER_TASK ))
DATA_PATH=data/mono_sweep_${SLURM_ARRAY_JOB_ID}

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
    echo "CONFIG_START $CONFIG_START has no entries in $MANIFEST (${#YAML_FILES[@]} configs) -- --array doesn't match the manifest; rerun scripts/generate_mono_sweep_configs.py and use the sbatch command it prints" >&2
    exit 1
fi

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
