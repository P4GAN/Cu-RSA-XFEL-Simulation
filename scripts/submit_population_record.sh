#!/bin/bash
# SLURM array job: centre-pixel populations and flux at every z plane, per shot, for the ORIGINAL
# model (6-level L3/K block only: no 2s, L2, satellites, middlemen, EII or sublevel mixing), to
# compare with the analytic two-level RSA model in ../RSA-derivation. See
# scripts/run_population_record.py for what is saved.
#
# One array task per "<mode> <beam> <pulse energy>" point of the chosen point set, NREP shots each on
# one 40-core node. mode mb = full Maxwell-Bloch, re = use_rate_equations (same level structure,
# coherences adiabatically eliminated). Shots are seeded by index, so MB and RE runs of the same point
# (and job 24728749's MB runs) see identical SASE pulses: the comparison is paired.
#   sase = config/base/Cu-seed-SASE-no-2s.yaml, centre 8045 eV
#   mono = config/base/Cu-seed-mono-SASE-original.yaml, self-seeded + Si(111) DCM on Kalpha1 (8047.91 eV)
# Grids are whatever those base configs currently hold (SASE xgrid 3 puts 0.67x the nominal fluence on
# the centre pixel; kept so these runs pair with job 24728749).
#
# Point sets ($1, default "original"):
#   original    0-5    MB  sase 2/9/40, mono 1/5/20 uJ                 (what job 24728749 ran)
#   lowfluence  0-7    MB  sase 0.1/0.3/1, mono 0.003/0.01/0.03/0.1/0.3 uJ
#               8-15   RE  the same 8 low-fluence points
#               16-21  RE  sase 2/9/40, mono 1/5/20 uJ (pairs with job 24728749's MB runs)
#   Entrance-plane W/W_sat is ~2.6 at mono 1 uJ and ~0.33 at SASE 2 uJ and scales with E, so the
#   low-fluence set reaches W/W_sat ~ 0.01-0.02 (linear) at its lowest point for both beams.
#
# Before submitting:
#   1. mkdir -p logs
#   2. python scripts/run_population_record.py --yaml config/base/Cu-seed-SASE-no-2s.yaml --rate-equations --check-only
#      python scripts/run_population_record.py --yaml config/base/Cu-seed-mono-SASE-original.yaml --rate-equations --check-only
#   3. sbatch --array=0-21 scripts/submit_population_record.sh lowfluence
#      (sbatch --array=0-5 scripts/submit_population_record.sh reproduces job 24728749)
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

SET=${1:-original}
SASE_YAML=config/base/Cu-seed-SASE-no-2s.yaml
MONO_YAML=config/base/Cu-seed-mono-SASE-original.yaml
case "$SET" in
    original)
        POINTS=("mb sase 2" "mb sase 9" "mb sase 40" "mb mono 1" "mb mono 5" "mb mono 20")
        ;;
    lowfluence)
        LOW=("sase 0.1" "sase 0.3" "sase 1" "mono 0.003" "mono 0.01" "mono 0.03" "mono 0.1" "mono 0.3")
        POINTS=()
        for p in "${LOW[@]}"; do POINTS+=("mb $p"); done
        for p in "${LOW[@]}"; do POINTS+=("re $p"); done
        POINTS+=("re sase 2" "re sase 9" "re sase 40" "re mono 1" "re mono 5" "re mono 20")
        ;;
    *)
        echo "unknown point set '$SET' (original | lowfluence)" >&2
        exit 1
        ;;
esac
NREP=${NREP:-40}

POINT=${POINTS[$SLURM_ARRAY_TASK_ID]:-}
if [[ -z "$POINT" ]]; then
    echo "array index $SLURM_ARRAY_TASK_ID has no entry in set '$SET' (${#POINTS[@]} points)" >&2
    exit 1
fi
read -r MODE KIND E_UJ <<< "$POINT"

if [[ "$SET" == original ]]; then
    DATA_PATH=data/population_record_${SLURM_ARRAY_JOB_ID}
else
    DATA_PATH=data/population_record_${SET}_${SLURM_ARRAY_JOB_ID}
fi
if [[ "$KIND" == sase ]]; then
    YAML=$SASE_YAML
    SETS=(E_seed_uJ="$E_UJ")
    STEM="sase_${E_UJ}uJ"
else
    YAML=$MONO_YAML
    SETS=(E_seed_uJ="$E_UJ" monochromator_target_energy_eV=8047.91)
    STEM="mono_${E_UJ}uJ_8047.91eV"
fi
RE_ARGS=()
if [[ "$MODE" == re ]]; then
    RE_ARGS=(--rate-equations)
    STEM="${STEM}_re"
fi
OUT="$DATA_PATH/${STEM}.npz"

echo "set $SET task $SLURM_ARRAY_TASK_ID -> $MODE $KIND $E_UJ uJ ($YAML), shots [0, $NREP) -> $OUT"

# ${arr[@]+...} form: an empty array trips `set -u` on bash < 4.4
python scripts/run_population_record.py \
    --yaml "$YAML" --set "${SETS[@]}" ${RE_ARGS[@]+"${RE_ARGS[@]}"} \
    --rep-start 0 --rep-end "$NREP" \
    --nproc "$SLURM_CPUS_PER_TASK" \
    --out "$OUT"
