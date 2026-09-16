#!/bin/bash
# SLURM array job: centre-pixel populations and flux at every z plane, per shot, for the ORIGINAL
# model (6-level L3/K block only: no 2s, L2, satellites, middlemen, EII or sublevel mixing), to
# compare with the analytic two-level RSA model in ../RSA-derivation. See
# scripts/run_population_record.py for what is saved.
#
# One array task per (beam, pulse energy) point in POINTS below; NREP shots each on one 40-core
# node (SASE shots are seeded by index, so every point sees the same SASE pulses).
#   0-2  SASE 8045 eV centre, config/base/Cu-seed-SASE-no-2s.yaml, E_seed 2 / 9 / 40 uJ
#   3-5  self-seeded + Si(111) DCM on Kalpha1 (8047.91 eV), config/base/Cu-seed-mono-SASE-original.yaml,
#        E_seed 1 / 5 / 20 uJ
# Grids are whatever those base configs currently hold.
#
# Before submitting:
#   1. mkdir -p logs
#   2. python scripts/run_population_record.py --yaml config/base/Cu-seed-SASE-no-2s.yaml --check-only
#      python scripts/run_population_record.py --yaml config/base/Cu-seed-mono-SASE-original.yaml --check-only
#   3. sbatch --array=0-5 scripts/submit_population_record.sh
# Overrides: NREP=... (default 40).

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-population-record
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=08:00:00
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

SASE_YAML=config/base/Cu-seed-SASE-no-2s.yaml
MONO_YAML=config/base/Cu-seed-mono-SASE-original.yaml
POINTS=(
    "sase 2"
    "sase 9"
    "sase 40"
    "mono 1"
    "mono 5"
    "mono 20"
)
NREP=${NREP:-40}

POINT=${POINTS[$SLURM_ARRAY_TASK_ID]:-}
if [[ -z "$POINT" ]]; then
    echo "array index $SLURM_ARRAY_TASK_ID has no entry in POINTS (${#POINTS[@]} points)" >&2
    exit 1
fi
read -r KIND E_UJ <<< "$POINT"

DATA_PATH=data/population_record_${SLURM_ARRAY_JOB_ID}
if [[ "$KIND" == sase ]]; then
    YAML=$SASE_YAML
    SETS=(E_seed_uJ="$E_UJ")
    OUT="$DATA_PATH/sase_${E_UJ}uJ.npz"
else
    YAML=$MONO_YAML
    SETS=(E_seed_uJ="$E_UJ" monochromator_target_energy_eV=8047.91)
    OUT="$DATA_PATH/mono_${E_UJ}uJ_8047.91eV.npz"
fi

echo "task $SLURM_ARRAY_TASK_ID -> $KIND $E_UJ uJ ($YAML), shots [0, $NREP) -> $OUT"

python scripts/run_population_record.py \
    --yaml "$YAML" --set "${SETS[@]}" \
    --rep-start 0 --rep-end "$NREP" \
    --nproc "$SLURM_CPUS_PER_TASK" \
    --out "$OUT"
