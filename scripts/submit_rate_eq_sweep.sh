#!/bin/bash
# SLURM array job: rate-equation vs full Maxwell-Bloch comparison sweep. One array task per line
# of config/generated/rate_eq/tasks.txt ("<config name> <yaml path>"), written by
# scripts/generate_rate_eq_sweep.sh -- 6 E_seed points for Cu-seed-SASE-double-satellite-rate-eq.
#
# Same per-task settings as submit_production_sweep_sase.sh (NREP=200 on one 40-core node, 3 h);
# the double-satellite model previously reached ~120 reps in that window and saved them as a
# .partial.npz, which data_from_folder/the plotting scripts read the same way.
#
# Output: data/rate_eq_sweep_sase_<jobid>/<config name>/runs_seed_<E>_uJ/
#
# Before submitting:
#   1. mkdir -p logs
#   2. bash scripts/generate_rate_eq_sweep.sh
#   3. sbatch --array=0-5 scripts/submit_rate_eq_sweep.sh

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-rate-eq-sase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --array=0-5
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

# Preflight: the XLO_sim that python actually imports must know use_rate_equations. Older code
# silently ignores the config key (XLO_sim setattrs every key) and runs the full model -- job
# 24533426 did exactly that with the earlier linear-response flag. Checks the imported file, so a
# stale non-editable install is caught too.
# tail: importing XLO_sim pulls in ocelot, which prints "initializing ocelot..." to stdout first
MODEL_PY=$(python -c "import XLO_sim.Model as M; print(M.__file__)" | tail -n 1)
if ! grep -q "use_rate_equations" "$MODEL_PY"; then
    echo "$MODEL_PY has no use_rate_equations support -- pull the latest code (and pip install -e .)" >&2
    exit 1
fi
echo "using $MODEL_PY"

NREP=200
DATA_PATH=data/rate_eq_sweep_sase_${SLURM_ARRAY_JOB_ID}/${NAME}

echo "task $SLURM_ARRAY_TASK_ID -> config $NAME ($YAML), $NREP reps -> $DATA_PATH"

python scripts/run_intensity_sweep.py \
    --yaml "$YAML" \
    --rep-start 0 --rep-end "$NREP" \
    --nproc "$SLURM_CPUS_PER_TASK" \
    --data-path "$DATA_PATH"
