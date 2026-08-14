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
# --array below is NOT authoritative: generate_mono_sweep_configs.py prints
# the exact `sbatch --array=...%<throttle>` command to run after it
# (re)writes the manifest, since only it knows how many configs
# len(E_seed) x len(energy) currently produces. A CLI --array overrides the
# pragma below, so always submit with the printed command -- the pragma is
# just a stale-safe fallback.
#
# Before submitting:
#   1. python scripts/generate_mono_sweep_configs.py   (writes the manifest
#      and prints the sbatch command to run)
#   2. mkdir -p logs
#   3. edit the "adjust to your environment" block below
#   4. run the sbatch command printed by step 1

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-mono-transmittance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --array=0-419%50
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

MANIFEST=config/generated/mono_transmittance_vs_intensity/manifest.txt
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing $MANIFEST -- run scripts/generate_mono_sweep_configs.py first" >&2
    exit 1
fi
mapfile -t YAML_FILES < "$MANIFEST"

NREP=50
CHUNKS_PER_CONFIG=2
REPS_PER_CHUNK=$(( NREP / CHUNKS_PER_CONFIG ))

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / CHUNKS_PER_CONFIG ))
CHUNK_IDX=$(( SLURM_ARRAY_TASK_ID % CHUNKS_PER_CONFIG ))
REP_START=$(( CHUNK_IDX * REPS_PER_CHUNK ))
REP_END=$(( REP_START + REPS_PER_CHUNK ))

YAML=${YAML_FILES[$CONFIG_IDX]:-}
if [[ -z "$YAML" ]]; then
    echo "CONFIG_IDX $CONFIG_IDX has no entry in $MANIFEST (${#YAML_FILES[@]} configs) -- --array doesn't match the manifest; rerun scripts/generate_mono_sweep_configs.py and use the sbatch command it prints" >&2
    exit 1
fi
DATA_PATH=data/mono_sweep_${SLURM_ARRAY_JOB_ID}

echo "task $SLURM_ARRAY_TASK_ID -> config $CONFIG_IDX ($YAML), reps [$REP_START, $REP_END)"

python scripts/run_mono_sweep.py \
    --yaml "$YAML" \
    --rep-start "$REP_START" --rep-end "$REP_END" \
    --nproc "$SLURM_CPUS_PER_TASK" \
    --data-path "$DATA_PATH"
