#!/bin/bash
# SLURM array job for the mono-transmittance-vs-intensity sweep.
#
# Counterpart of submit_intensity_sweep.sh, but the grid has two physical
# parameters (E_seed x absolute target photon energy) instead of one. That
# doesn't change the array-indexing logic below: generate_mono_sweep_configs.py
# flattens the E_seed x energy grid into a single manifest.txt, so each
# array task still just looks up one YAML path by CONFIG_IDX, exactly like
# the 1D sweeps.
#
# The mono base config (config/base/Cu-seed-mono-SASE.yaml) uses tgrid=15000
# vs. 800 for the 1D sweeps -- ~19x finer time resolution to resolve the
# monochromator's narrow bandwidth. That makes each rep far more expensive,
# both in compute and memory: rho_ijtxyz alone is ~8.3 GB per worker process
# at this tgrid, so --mem=0 (grab the whole node) instead of a fixed cap,
# since 40 concurrent workers can easily want 300+ GB between them.
#
# CHUNKS_PER_CONFIG=2 splits NREP=50 into 25/chunk, fitting one parallel
# round on 40 cores (the speed floor). Every task for a given config writes
# into the same runs_seed_<E>_uJ__energy_<E_target>_eV/ folder, so the
# notebook's aggregation step (data_from_folder) doesn't need to change
# either way.
#
# The array size is NOT hardcoded: the block below counts manifest.txt's
# lines and resubmits itself via `sbatch --array=...` with the right bound
# (throttled to ARRAY_THROTTLE concurrent tasks), so it always matches
# however many configs len(E_seed) x len(energy) currently produces in
# generate_mono_sweep_configs.py.
#
# Before submitting:
#   1. python scripts/generate_mono_sweep_configs.py     (writes the manifest)
#   2. mkdir -p logs
#   3. edit the "adjust to your environment" block below
#   4. bash scripts/submit_mono_sweep.sh   (NOT sbatch -- this computes the
#      array size, then calls sbatch itself)

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-mono-transmittance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MANIFEST=config/generated/mono_transmittance_vs_intensity/manifest.txt
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing $MANIFEST -- run scripts/generate_mono_sweep_configs.py first" >&2
    exit 1
fi

NREP=50
CHUNKS_PER_CONFIG=2
ARRAY_THROTTLE=50   # max concurrent array tasks

# Not yet running as a SLURM array task -> compute the array bound from the
# manifest and resubmit ourselves as one.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    NUM_CONFIGS=$(wc -l < "$MANIFEST")
    TOTAL_TASKS=$(( NUM_CONFIGS * CHUNKS_PER_CONFIG ))
    echo "manifest has $NUM_CONFIGS configs -> submitting array 0-$((TOTAL_TASKS - 1))%${ARRAY_THROTTLE} ($TOTAL_TASKS tasks)"
    exec sbatch --array="0-$((TOTAL_TASKS - 1))%${ARRAY_THROTTLE}" "$0"
fi

mapfile -t YAML_FILES < "$MANIFEST"
REPS_PER_CHUNK=$(( NREP / CHUNKS_PER_CONFIG ))

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / CHUNKS_PER_CONFIG ))
CHUNK_IDX=$(( SLURM_ARRAY_TASK_ID % CHUNKS_PER_CONFIG ))
REP_START=$(( CHUNK_IDX * REPS_PER_CHUNK ))
REP_END=$(( REP_START + REPS_PER_CHUNK ))

YAML=${YAML_FILES[$CONFIG_IDX]:-}
if [[ -z "$YAML" ]]; then
    echo "CONFIG_IDX $CONFIG_IDX has no entry in $MANIFEST (${#YAML_FILES[@]} configs) -- stale --array bound?" >&2
    exit 1
fi
DATA_PATH=data/mono_sweep_${SLURM_ARRAY_JOB_ID}

echo "task $SLURM_ARRAY_TASK_ID -> config $CONFIG_IDX ($YAML), reps [$REP_START, $REP_END)"

python scripts/run_mono_sweep.py \
    --yaml "$YAML" \
    --rep-start "$REP_START" --rep-end "$REP_END" \
    --nproc "$SLURM_CPUS_PER_TASK" \
    --data-path "$DATA_PATH"
