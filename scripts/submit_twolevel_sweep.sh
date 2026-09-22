#!/bin/bash
# SLURM array job: two-level (g, l, u + auxiliary x) Maxwell-Bloch runs for the comparison with
# the analytic RSA model (XLO_sim/twolevel.py, scripts/run_twolevel_sweep.py). Analyse with
# scripts/plot_twolevel_vs_analytic.py.
#
# Array layout: 4 (beam, mode) combinations x NCHUNKS energy chunks, task = combo * NCHUNKS + chunk:
#   combo 0 mono mb | 1 mono re | 2 sase mb | 3 sase re
# Chunk k runs pulse_energies_uJ[k::NCHUNKS], so the expensive high-energy points (smaller time
# step) are spread over the chunks. One .npz per (beam, mode, energy) goes to
# data/twolevel_<jobid>[_<TAG>]/; existing files are skipped, so a resubmission with the same
# DATA_PATH only fills in what is missing.
#
# Before submitting:
#   1. python scripts/run_twolevel_sweep.py --beam mono --mode mb --check-only   (seconds)
#   2. sbatch --array=0-11 scripts/submit_twolevel_sweep.sh      (NCHUNKS=3, the default)
# Overrides (environment): NCHUNKS (array size must be 4*NCHUNKS), CONFIG, TAG,
#   SET="grid.zmax_um=2 sase.n_shots=200" (passed to --set), ENERGIES="0.01 1 10" (instead of the
#   config list), DATA_PATH (resume into an old folder).

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-twolevel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"
mkdir -p logs

COMBOS=("mono mb" "mono re" "sase mb" "sase re")
NCHUNKS=${NCHUNKS:-3}
CONFIG=${CONFIG:-config/base/Cu-2level-analytic.yaml}
TAG=${TAG:-}
DATA_PATH=${DATA_PATH:-data/twolevel_${SLURM_ARRAY_JOB_ID}${TAG:+_$TAG}}

COMBO=$(( SLURM_ARRAY_TASK_ID / NCHUNKS ))
CHUNK=$(( SLURM_ARRAY_TASK_ID % NCHUNKS ))
if (( COMBO >= ${#COMBOS[@]} )); then
    echo "array index $SLURM_ARRAY_TASK_ID is beyond 4 x NCHUNKS=$NCHUNKS tasks" >&2
    exit 1
fi
read -r BEAM MODE <<< "${COMBOS[$COMBO]}"

SET_ARGS=()
if [[ -n "${SET:-}" ]]; then
    read -r -a SET_WORDS <<< "$SET"
    SET_ARGS=(--set "${SET_WORDS[@]}")
fi
if [[ -n "${ENERGIES:-}" ]]; then
    read -r -a E_WORDS <<< "$ENERGIES"
    SET_ARGS+=(--E-uJ "${E_WORDS[@]}")
fi

echo "task $SLURM_ARRAY_TASK_ID -> $BEAM $MODE, energy chunk $CHUNK/$NCHUNKS -> $DATA_PATH"
# ${arr[@]+...} form: an empty array trips `set -u` on bash < 4.4
python scripts/run_twolevel_sweep.py \
    --config "$CONFIG" --beam "$BEAM" --mode "$MODE" \
    --chunk "$CHUNK" --nchunks "$NCHUNKS" \
    --data-path "$DATA_PATH" --nthreads "$SLURM_CPUS_PER_TASK" \
    ${SET_ARGS[@]+"${SET_ARGS[@]}"}
