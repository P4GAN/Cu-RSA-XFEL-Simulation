#!/bin/bash
# SLURM array job: population budget runs (scripts/run_population_budget.py) for a family written by
# scripts/generate_population_budget.sh. $1 is the family directory holding a "<variant> <yaml>"
# manifest.txt. Each array task runs CONFIGS_PER_TASK consecutive manifest entries concurrently, each
# with cpus-per-task / CONFIGS_PER_TASK workers and NREP shots (one shot per worker, so NREP should not
# exceed that). Output: data/<family>_<jobid>/<variant>/<config stem>.npz, config and provenance
# alongside. The generator prints the exact sbatch lines (memory, NREP, CONFIGS_PER_TASK, array range).
#
# Before submitting:
#   1. mkdir -p logs
#   2. bash scripts/generate_population_budget.sh

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-population-budget
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=08:00:00
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

if [[ $# -lt 1 ]]; then
    echo "usage: sbatch ... scripts/submit_population_budget.sh <family dir>" >&2
    exit 1
fi
FAMILY_DIR=$1
FAMILY=$(basename "$FAMILY_DIR")
MANIFEST="$FAMILY_DIR/manifest.txt"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing $MANIFEST -- run scripts/generate_population_budget.sh first" >&2
    exit 1
fi
mapfile -t LINES < "$MANIFEST"

NREP=${NREP:-10}
CONFIGS_PER_TASK=${CONFIGS_PER_TASK:-1}
NPROC=$(( SLURM_CPUS_PER_TASK / CONFIGS_PER_TASK ))
START=$(( SLURM_ARRAY_TASK_ID * CONFIGS_PER_TASK ))

pids=()
for (( i = 0; i < CONFIGS_PER_TASK; i++ )); do
    LINE=${LINES[$(( START + i ))]:-}
    [[ -z "$LINE" ]] && break
    read -r VARIANT YAML <<< "$LINE"
    OUT=data/${FAMILY}_${SLURM_ARRAY_JOB_ID}/${VARIANT}/$(basename "$YAML" .yaml).npz
    echo "task $SLURM_ARRAY_TASK_ID slot $i -> $VARIANT $YAML, $NREP shots on $NPROC workers -> $OUT"
    python scripts/run_population_budget.py --yaml "$YAML" --rep-start 0 --rep-end "$NREP" \
        --nproc "$NPROC" --out "$OUT" &
    pids+=("$!")
done

if (( ${#pids[@]} == 0 )); then
    echo "array index $SLURM_ARRAY_TASK_ID has no entries in $MANIFEST (${#LINES[@]} configs)" >&2
    exit 1
fi
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
