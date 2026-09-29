#!/bin/bash
# SLURM array job that works through a sweep family (scripts/generate_sweep.py) on the Maxwell cluster.
#
#   sbatch --array=0-<N> --export=ALL,NREP=10,CHUNKS=1,PER_TASK=8 scripts/submit_sweep.sh <family folder>
#
# Each config's NREP shots are split into CHUNKS work items (for sweeps with many shots per point), and each
# array task runs PER_TASK work items side by side on SLURM_CPUS_PER_TASK / PER_TASK processes each. The
# generate_*_sweep.sh scripts print the matching command, including the --array range. Run it from the repo
# root after `mkdir -p logs`. Output: data/<family>_<array job id>/<variant>/runs_.../
#
# The defaults below take a whole 40-core node per task for up to 16 hours; override them on the sbatch
# command line (e.g. --mem=64G --time=08:00:00) if the queue is busy.

#SBATCH --partition=allcpu
#SBATCH --job-name=xraymb-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

FAMILY_DIR=${1:?usage: sbatch ... scripts/submit_sweep.sh <family folder>}
FAMILY=$(basename "$FAMILY_DIR")
MANIFEST="$FAMILY_DIR/manifest.txt"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing $MANIFEST -- run the family's generate script first" >&2
    exit 1
fi
mapfile -t LINES < "$MANIFEST"

NREP=${NREP:-10}
CHUNKS=${CHUNKS:-1}
PER_TASK=${PER_TASK:-8}
REPS_PER_CHUNK=$(( NREP / CHUNKS ))
NPROC=$(( SLURM_CPUS_PER_TASK / PER_TASK ))

pids=()
for (( i = 0; i < PER_TASK; i++ )); do
    ITEM=$(( SLURM_ARRAY_TASK_ID * PER_TASK + i ))
    LINE=${LINES[$(( ITEM / CHUNKS ))]:-}
    if [[ -z "$LINE" ]]; then
        break   # the last task can have fewer than PER_TASK items
    fi
    read -r VARIANT YAML <<< "$LINE"
    REP_START=$(( (ITEM % CHUNKS) * REPS_PER_CHUNK ))
    REP_END=$(( REP_START + REPS_PER_CHUNK ))
    DATA_PATH=data/${FAMILY}_${SLURM_ARRAY_JOB_ID}/${VARIANT}
    echo "task $SLURM_ARRAY_TASK_ID item $ITEM: $VARIANT $YAML, shots [$REP_START, $REP_END) on $NPROC processes -> $DATA_PATH"
    python scripts/run_sweep.py --yaml "$YAML" --rep-start "$REP_START" --rep-end "$REP_END" \
        --nproc "$NPROC" --data-path "$DATA_PATH" &
    pids+=("$!")
done

if (( ${#pids[@]} == 0 )); then
    echo "array index $SLURM_ARRAY_TASK_ID has no work items (${#LINES[@]} configs x $CHUNKS chunks)" >&2
    exit 1
fi
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
