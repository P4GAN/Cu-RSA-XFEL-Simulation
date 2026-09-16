#!/bin/bash
# SLURM array job: monochromator E_seed x energy sweep for the 5 pathway-extension variants of the
# double-satellite model (see scripts/generate_pathway_sweeps.sh for what each variant is).
# 5 variants x 5 E_seed x 52 energies = 1300 configs at tgrid 12000, zgrid 15, xgrid=ygrid 5,
# NREP=10 each. Same fan-out as submit_production_sweep_mono.sh: each array task runs
# CONFIGS_PER_TASK=8 configs as concurrent background processes with 5 workers each (8 x 5 = 40 =
# --cpus-per-task), so ceil(1300/8) = 163 array tasks. --mem=0 (whole node), as for the other mono
# sweeps: 40 concurrent tgrid=12000 reps with 13 satellite blocks each want a lot of memory.
#
# DATA_PATH nests under the variant name (data/pathway_sweep_mono_<jobid>/<variant>/), since
# run_mono_sweep.py's own runs_seed_<E>_uJ__energy_<E>_eV/ folders would otherwise collide across
# variants. run_mono_sweep.py refuses to start if the XLO_sim it imports predates
# use_middlemen/use_eii (tools.verify_code), and writes a .provenance.txt next to every output.
#
# Before submitting:
#   1. mkdir -p logs
#   2. bash scripts/generate_pathway_sweeps.sh   (prints the exact sbatch --array command)
#   3. sbatch --array=0-162 scripts/submit_pathway_sweep_mono.sh

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-pathway-sweep-mono
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=16:00:00
#SBATCH --array=0-162
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

# Optional $1: the family directory holding manifest.txt (default: the pathway sweep). Any
# generator that writes "<variant> <yaml>" lines can reuse this script, e.g.
# generate_bracket_sweeps.sh -> sbatch --array=... scripts/submit_pathway_sweep_mono.sh config/generated/bracket_sweep_mono
FAMILY_DIR=${1:-config/generated/pathway_sweep_mono}
FAMILY=$(basename "$FAMILY_DIR")
MANIFEST="$FAMILY_DIR/manifest.txt"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing $MANIFEST -- run the generator for $FAMILY first" >&2
    exit 1
fi
mapfile -t LINES < "$MANIFEST"

# CONFIGS_PER_TASK x NPROC_PER_CONFIG = --cpus-per-task (10 reps on 5 workers = 2 rounds per config).
# CONFIGS_PER_TASK must match MONO_CONFIGS_PER_TASK in generate_pathway_sweeps.sh, which prints
# the --array bound from it.
NREP=10
CONFIGS_PER_TASK=8
NPROC_PER_CONFIG=$(( SLURM_CPUS_PER_TASK / CONFIGS_PER_TASK ))
CONFIG_START=$(( SLURM_ARRAY_TASK_ID * CONFIGS_PER_TASK ))

pids=()
for (( i = 0; i < CONFIGS_PER_TASK; i++ )); do
    IDX=$(( CONFIG_START + i ))
    LINE=${LINES[$IDX]:-}
    if [[ -z "$LINE" ]]; then
        break  # last task: fewer than CONFIGS_PER_TASK configs remain
    fi
    read -r VARIANT YAML <<< "$LINE"
    DATA_PATH=data/${FAMILY}_${SLURM_ARRAY_JOB_ID}/${VARIANT}
    echo "task $SLURM_ARRAY_TASK_ID slot $i -> config $IDX: $VARIANT ($YAML), reps [0, $NREP), $NPROC_PER_CONFIG workers -> $DATA_PATH"
    python scripts/run_mono_sweep.py \
        --yaml "$YAML" \
        --rep-start 0 --rep-end "$NREP" \
        --nproc "$NPROC_PER_CONFIG" \
        --data-path "$DATA_PATH" &
    pids+=("$!")
done

if (( ${#pids[@]} == 0 )); then
    echo "CONFIG_START $CONFIG_START has no entries in $MANIFEST (${#LINES[@]} configs) -- --array doesn't match the manifest" >&2
    exit 1
fi

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
