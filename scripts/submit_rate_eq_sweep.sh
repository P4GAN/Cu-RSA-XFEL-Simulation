#!/bin/bash
# SLURM array job: rate-equation vs full Maxwell-Bloch comparison sweep. One array task per line
# of config/generated/rate_eq/tasks.txt ("<config name> <yaml path>"), written by
# scripts/generate_rate_eq_sweep.sh -- 6 E_seed points each for Cu-seed-SASE-double-satellite-rate-eq
# (tasks 0-5) and the full-MB Cu-seed-SASE-double-satellite (tasks 6-11), same seeds 0..NREP-1, so
# the RE/MB ratio compares identical SASE realisations.
#
# NREP=200 on one 40-core node. The double-satellite model managed ~120 reps in 3 h, so 8 h leaves
# room for all 200 (incomplete tasks still leave a usable .partial.npz checkpoint).
#
# run_intensity_sweep.py checks, before any repetition, that it imports this checkout's XLO_sim and
# that the kernel honours use_rate_equations, and writes a <stem>.provenance.txt next to the output
# -- jobs 24533426/24535882/24541257 silently ran the full model from a stale XLO_sim.
#
# Output: data/rate_eq_sweep_sase_<jobid>/<config name>/runs_seed_<E>_uJ/
#
# Before submitting:
#   1. mkdir -p logs
#   2. bash scripts/generate_rate_eq_sweep.sh
#   3. sbatch --array=0-11 scripts/submit_rate_eq_sweep.sh

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-rate-eq-sase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --array=0-11
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

TASKS=config/generated/rate_eq/tasks.txt
if [[ ! -f "$TASKS" ]]; then
    echo "Missing $TASKS -- run scripts/generate_rate_eq_sweep.sh first" >&2
    exit 1
fi
mapfile -t LINES < "$TASKS"
LINE=${LINES[$SLURM_ARRAY_TASK_ID]:-}
if [[ -z "$LINE" ]]; then
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID has no entry in $TASKS (${#LINES[@]} tasks)" >&2
    exit 1
fi
read -r NAME YAML <<< "$LINE"

NREP=200
DATA_PATH=data/rate_eq_sweep_sase_${SLURM_ARRAY_JOB_ID}/${NAME}

echo "task $SLURM_ARRAY_TASK_ID -> config $NAME ($YAML), $NREP reps -> $DATA_PATH"

python scripts/run_intensity_sweep.py \
    --yaml "$YAML" \
    --rep-start 0 --rep-end "$NREP" \
    --nproc "$SLURM_CPUS_PER_TASK" \
    --data-path "$DATA_PATH"
