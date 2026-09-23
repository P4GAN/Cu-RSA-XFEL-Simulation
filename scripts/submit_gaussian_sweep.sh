#!/bin/bash
# SLURM array job for the Gaussian-pulse transmittance-vs-photon-energy sweep
# (scripts/generate_gaussian_sweep_configs.py -> this -> scripts/run_gaussian_sweep.py).
#
# Every config is one deterministic, single-process run (tools.Gaussian_pulse_aniso_seed, no SASE
# repetitions), so each array task runs CONFIGS_PER_TASK=40 configs side by side, one per core.
# The --array printed by generate_gaussian_sweep_configs.py (ceil(n_configs / 40) tasks) is
# authoritative; the pragma below is only a fallback.
#
# Environment (both required):
#   MANIFEST   manifest.txt written by generate_gaussian_sweep_configs.py
#   DATA_TAG   output goes to data/${DATA_TAG}_${SLURM_ARRAY_JOB_ID}/
#
#   mkdir -p logs
#   MANIFEST=config/generated/l2fit_mono_gaussian/manifest.txt DATA_TAG=l2fit_mono_gaussian \
#       sbatch --array=0-6 scripts/submit_gaussian_sweep.sh

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-gaussian-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

: "${MANIFEST:?set MANIFEST=<manifest.txt from generate_gaussian_sweep_configs.py>}"
: "${DATA_TAG:?set DATA_TAG=<name for data/<DATA_TAG>_<jobid>>}"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing $MANIFEST -- run scripts/generate_gaussian_sweep_configs.py first" >&2
    exit 1
fi
mapfile -t YAML_FILES < "$MANIFEST"

CONFIGS_PER_TASK=40      # must match generate_gaussian_sweep_configs.py
CONFIG_START=$(( SLURM_ARRAY_TASK_ID * CONFIGS_PER_TASK ))
DATA_PATH=data/${DATA_TAG}_${SLURM_ARRAY_JOB_ID}
mkdir -p "$DATA_PATH"
if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
    git -C "$REPO_ROOT" log -1 --format='commit %H %cd' > "$DATA_PATH/provenance.txt" 2>/dev/null || true
    git -C "$REPO_ROOT" status --short >> "$DATA_PATH/provenance.txt" 2>/dev/null || true
    cp "$MANIFEST" "$DATA_PATH/manifest.txt"
fi

pids=()
for (( i = 0; i < CONFIGS_PER_TASK; i++ )); do
    CONFIG_IDX=$(( CONFIG_START + i ))
    YAML=${YAML_FILES[$CONFIG_IDX]:-}
    [[ -z "$YAML" ]] && break
    echo "task $SLURM_ARRAY_TASK_ID slot $i -> config $CONFIG_IDX ($YAML)"
    python scripts/run_gaussian_sweep.py --yaml "$YAML" --data-path "$DATA_PATH" &
    pids+=("$!")
done

if (( ${#pids[@]} == 0 )); then
    echo "CONFIG_START $CONFIG_START has no entries in $MANIFEST (${#YAML_FILES[@]} configs) -- use the --array printed by generate_gaussian_sweep_configs.py" >&2
    exit 1
fi

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
