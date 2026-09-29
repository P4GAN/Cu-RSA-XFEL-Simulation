#!/bin/bash
# SLURM array job: SASE transmittance-vs-intensity sweep for the 5 pathway-extension variants of
# the double-satellite model (see scripts/generate_pathway_sweeps.sh for what each variant is).
# 5 variants x 7 E_seed (2-80 uJ) = 35 configs, one array task per config, NREP=200 on one 40-core node
# (5 rounds of 40 reps). Reps are seeded by index (X.random_seed = rep), so every variant sees the
# same 200 SASE pulses and differences between variants are paired.
#
# DATA_PATH nests under the variant name (data/pathway_sweep_sase_<jobid>/<variant>/), since
# run_intensity_sweep.py's own runs_seed_<E>_uJ/ folders would otherwise collide across variants.
# run_intensity_sweep.py refuses to start if the XLO_sim it imports predates use_middlemen/use_eii
# (tools.verify_code), and writes a .provenance.txt next to every output: check it.
#
# Before submitting:
#   1. mkdir -p logs
#   2. bash scripts/generate_pathway_sweeps.sh   (prints the exact sbatch --array command)
#   3. sbatch --array=0-34 scripts/submit_pathway_sweep_sase.sh

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-pathway-sweep-sase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-34
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
# generate_bracket_sweeps.sh -> sbatch --array=... scripts/submit_pathway_sweep_sase.sh config/generated/bracket_sweep_sase
FAMILY_DIR=${1:-config/generated/pathway_sweep_sase}
FAMILY=$(basename "$FAMILY_DIR")
MANIFEST="$FAMILY_DIR/manifest.txt"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing $MANIFEST -- run the generator for $FAMILY first" >&2
    exit 1
fi
mapfile -t LINES < "$MANIFEST"
LINE=${LINES[$SLURM_ARRAY_TASK_ID]:-}
if [[ -z "$LINE" ]]; then
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID has no entry in $MANIFEST (${#LINES[@]} configs) -- --array doesn't match the manifest" >&2
    exit 1
fi
read -r VARIANT YAML <<< "$LINE"

NREP=200
DATA_PATH=data/${FAMILY}_${SLURM_ARRAY_JOB_ID}/${VARIANT}

echo "task $SLURM_ARRAY_TASK_ID -> $VARIANT ($YAML), $NREP reps -> $DATA_PATH"

python scripts/run_intensity_sweep.py \
    --yaml "$YAML" \
    --rep-start 0 --rep-end "$NREP" \
    --nproc "$SLURM_CPUS_PER_TASK" \
    --data-path "$DATA_PATH"
