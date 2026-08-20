#!/bin/bash
# SLURM array job for the transmittance-vs-spotsize sweep.
#
# One array task == one independent SLURM job == (usually) its own node.
# CHUNKS_PER_CONFIG=4 below splits each config's NREP=300 repetitions into 4
# chunks of 75, matching submit_intensity_sweep.sh / submit_duration_sweep.sh
# -- see those files' headers for how CHUNKS_PER_CONFIG trades off node
# count vs. reps per task.
#
# Unlike the intensity/duration sweeps, cost here is NOT uniform across
# configs: xgrid/ygrid grow with the spot-size scale (see
# generate_spotsize_sweep_configs.py), so the scale=4 config (xgrid=ygrid=32)
# can be orders of magnitude slower/more memory-hungry than scale=0.25
# (xgrid=ygrid=8, floored at the base value). The --time/--mem/--cpus-per-task
# below are a single generous budget sized for the *largest* config in the
# default sweep and will be wasteful for the smallest ones -- time the
# largest case locally first (see generate_spotsize_sweep_configs.py's
# printed caution) and adjust.
#
# --array below is NOT authoritative: generate_spotsize_sweep_configs.py
# prints the exact `sbatch --array=...` command to run after it (re)writes
# the manifest, since only it knows how many configs DEFAULT_SCALE_VALUES
# currently produces. A CLI --array overrides the pragma below, so always
# submit with the printed command -- the pragma is just a stale-safe fallback.
#
# Before submitting:
#   1. python scripts/generate_spotsize_sweep_configs.py   (writes the
#      manifest and prints the sbatch command to run)
#   2. mkdir -p logs
#   3. edit the "adjust to your environment" block below, especially after
#      timing the largest config
#   4. run the sbatch command printed by step 1

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-spotsize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

MANIFEST=config/generated/transmittance_vs_spotsize/manifest.txt
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing $MANIFEST -- run scripts/generate_spotsize_sweep_configs.py first" >&2
    exit 1
fi
mapfile -t YAML_FILES < "$MANIFEST"

NREP=80
CHUNKS_PER_CONFIG=8
REPS_PER_CHUNK=$(( NREP / CHUNKS_PER_CONFIG ))

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / CHUNKS_PER_CONFIG ))
CHUNK_IDX=$(( SLURM_ARRAY_TASK_ID % CHUNKS_PER_CONFIG ))
REP_START=$(( CHUNK_IDX * REPS_PER_CHUNK ))
REP_END=$(( REP_START + REPS_PER_CHUNK ))

YAML=${YAML_FILES[$CONFIG_IDX]:-}
if [[ -z "$YAML" ]]; then
    echo "CONFIG_IDX $CONFIG_IDX has no entry in $MANIFEST (${#YAML_FILES[@]} configs) -- --array doesn't match the manifest; rerun scripts/generate_spotsize_sweep_configs.py and use the sbatch command it prints" >&2
    exit 1
fi
DATA_PATH=data/spotsize_sweep_${SLURM_ARRAY_JOB_ID}

echo "task $SLURM_ARRAY_TASK_ID -> config $CONFIG_IDX ($YAML), reps [$REP_START, $REP_END)"

python scripts/run_spotsize_sweep.py \
    --yaml "$YAML" \
    --rep-start "$REP_START" --rep-end "$REP_END" \
    --nproc "$SLURM_CPUS_PER_TASK" \
    --data-path "$DATA_PATH"
