#!/bin/bash
# SLURM array job: linear-response vs full Maxwell-Bloch comparison sweep. One array task per line
# of config/generated/linear_response/tasks.txt ("<config name> <yaml path>"), written by
# scripts/generate_linear_response_sweep.sh -- 6 E_seed points for Cu-seed-SASE-double-satellite-linear
# plus 2 low-fluence points for the full Cu-seed-SASE-double-satellite model = 8 tasks.
#
# Same per-task settings as submit_production_sweep_sase.sh (NREP=200 on one 40-core node, 3 h);
# the double-satellite model previously reached ~120 reps in that window and saved them as a
# .partial.npz, which data_from_folder/the plotting scripts read the same way.
#
# Output: data/linear_response_sweep_sase_<jobid>/<config name>/runs_seed_<E>_uJ/
#
# Before submitting:
#   1. mkdir -p logs
#   2. bash scripts/generate_linear_response_sweep.sh
#   3. sbatch --array=0-7 scripts/submit_linear_response_sweep.sh

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-linear-response-sase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=03:00:00
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
mkdir -p logs

TASKS=config/generated/linear_response/tasks.txt
if [[ ! -f "$TASKS" ]]; then
    echo "Missing $TASKS -- run scripts/generate_linear_response_sweep.sh first" >&2
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
DATA_PATH=data/linear_response_sweep_sase_${SLURM_ARRAY_JOB_ID}/${NAME}

echo "task $SLURM_ARRAY_TASK_ID -> config $NAME ($YAML), $NREP reps -> $DATA_PATH"

python scripts/run_intensity_sweep.py \
    --yaml "$YAML" \
    --rep-start 0 --rep-end "$NREP" \
    --nproc "$SLURM_CPUS_PER_TASK" \
    --data-path "$DATA_PATH"
