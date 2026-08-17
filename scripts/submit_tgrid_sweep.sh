#!/bin/bash
# SLURM array job for the tgrid numerical-convergence sweep.
#
# One array task == one independent SLURM job == (usually) its own node.
# CHUNKS_PER_CONFIG=2 below splits each config's NREP=20 repetitions into 2
# chunks of 10. This sweep is about time-grid resolution, not SASE-shot
# statistics, so it uses far fewer repetitions per config than the
# intensity/duration sweeps (300) -- bump NREP below if you need tighter
# statistics per grid setting.
#
# Cost varies a lot across this manifest: tgrid=20000 can be orders of
# magnitude slower/more memory-hungry than tgrid=200. The --time/--mem/
# --cpus-per-task below are a single generous budget sized for the *largest*
# config in the default sweep and will be wasteful for the smallest ones --
# time the largest case locally first (see generate_tgrid_sweep_configs.py's
# printed caution) and adjust.
#
# --array below is NOT authoritative: generate_tgrid_sweep_configs.py prints
# the exact `sbatch --array=...` command to run after it (re)writes the
# manifest. A CLI --array overrides the pragma below, so always submit with
# the printed command -- the pragma is just a stale-safe fallback.
#
# Before submitting:
#   1. python scripts/generate_tgrid_sweep_configs.py   (writes the manifest
#      and prints the sbatch command to run)
#   2. mkdir -p logs
#   3. edit the "adjust to your environment" block below, especially after
#      timing the largest config
#   4. run the sbatch command printed by step 1

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-convergence-tgrid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

MANIFEST=config/generated/convergence_tgrid/manifest.txt
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing $MANIFEST -- run scripts/generate_tgrid_sweep_configs.py first" >&2
    exit 1
fi
mapfile -t YAML_FILES < "$MANIFEST"

NREP=20
CHUNKS_PER_CONFIG=2
REPS_PER_CHUNK=$(( NREP / CHUNKS_PER_CONFIG ))

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / CHUNKS_PER_CONFIG ))
CHUNK_IDX=$(( SLURM_ARRAY_TASK_ID % CHUNKS_PER_CONFIG ))
REP_START=$(( CHUNK_IDX * REPS_PER_CHUNK ))
REP_END=$(( REP_START + REPS_PER_CHUNK ))

YAML=${YAML_FILES[$CONFIG_IDX]:-}
if [[ -z "$YAML" ]]; then
    echo "CONFIG_IDX $CONFIG_IDX has no entry in $MANIFEST (${#YAML_FILES[@]} configs) -- --array doesn't match the manifest; rerun scripts/generate_tgrid_sweep_configs.py and use the sbatch command it prints" >&2
    exit 1
fi
DATA_PATH=data/convergence_tgrid_${SLURM_ARRAY_JOB_ID}

echo "task $SLURM_ARRAY_TASK_ID -> config $CONFIG_IDX ($YAML), reps [$REP_START, $REP_END)"

python scripts/run_convergence_sweep.py \
    --yaml "$YAML" \
    --rep-start "$REP_START" --rep-end "$REP_END" \
    --nproc "$SLURM_CPUS_PER_TASK" \
    --data-path "$DATA_PATH"
