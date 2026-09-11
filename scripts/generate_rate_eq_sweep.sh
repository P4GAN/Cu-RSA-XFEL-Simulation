#!/bin/bash
# Generate the E_seed sweep manifest for the rate-equation vs full Maxwell-Bloch comparison
# (use_rate_equations in Model._MB_nlevel_regular_core -- coherences adiabatically eliminated):
#
#   Cu-seed-SASE-double-satellite-rate-eq : 60, 40, 9, 2, 0.5, 0.12 uJ   (tasks 0-5)
#   Cu-seed-SASE-double-satellite         : 60, 40, 9, 2, 0.5, 0.12 uJ   (tasks 6-11, full MB)
#
# The full-MB runs duplicate data/production_sweep_sase_24437070 (120 reps) on purpose: same seeds
# and rep count as the RE runs, so the RE/MB ratio isn't polluted by SASE sampling noise. Submit
# only tasks 0-5 to skip them.
#
# Also writes config/generated/rate_eq/tasks.txt, one "<config name> <yaml path>" line per SLURM
# array task, which submit_rate_eq_sweep.sh indexes directly. Prints the matching sbatch command
# (authoritative over the #SBATCH --array pragma in the submit script).
#
# Run this once (locally or on a login node) before submitting.

set -euo pipefail
cd "$(dirname "$0")/.."

OUT_ROOT=config/generated/rate_eq
TASKS="$OUT_ROOT/tasks.txt"
mkdir -p "$OUT_ROOT"
: > "$TASKS"

gen() {
    local name=$1; shift
    local out_dir="$OUT_ROOT/$name"
    echo "=== $name ==="
    python scripts/generate_intensity_sweep_configs.py \
        --base-yaml "config/base/${name}.yaml" \
        --out-dir "$out_dir" \
        --e-seed "$@" > /dev/null
    while read -r yaml; do
        echo "$name $yaml" >> "$TASKS"
        echo "  $yaml"
    done < "$out_dir/manifest.txt"
}

gen Cu-seed-SASE-double-satellite-rate-eq 60 40 9 2 0.5 0.12
# Full-MB counterpart on the same seeds/rep count, so RE/MB compares identical SASE shots.
gen Cu-seed-SASE-double-satellite 60 40 9 2 0.5 0.12

N=$(wc -l < "$TASKS" | tr -d ' ')
echo
echo "tasks: $TASKS ($N tasks)"
echo "submit with:"
echo "  sbatch --array=0-$((N - 1)) scripts/submit_rate_eq_sweep.sh"
