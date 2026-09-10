#!/bin/bash
# Generate the E_seed sweep manifests for the linear-response vs full Maxwell-Bloch comparison
# (isolates the Rabi nonlinearity of the resonant Kalpha coupling -- see linear_resonant_response
# in Model._MB_nlevel_regular_core):
#
#   Cu-seed-SASE-double-satellite-linear : 60, 40, 9, 2, 0.5, 0.12 uJ  (new, linear response)
#   Cu-seed-SASE-double-satellite        : 0.5, 0.12 uJ                (extends the existing full-model
#                                                                      production sweep, which already
#                                                                      has 60/40/9/2 uJ, down to the
#                                                                      experiment's low-fluence curves)
#
# Also writes config/generated/linear_response/tasks.txt, one "<config name> <yaml path>" line per
# SLURM array task, which submit_linear_response_sweep.sh indexes directly. Prints the matching
# sbatch command (authoritative over the #SBATCH --array pragma in the submit script).
#
# Run this once (locally or on a login node) before submitting.

set -euo pipefail
cd "$(dirname "$0")/.."

OUT_ROOT=config/generated/linear_response
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

gen Cu-seed-SASE-double-satellite-linear 60 40 9 2 0.5 0.12
gen Cu-seed-SASE-double-satellite 0.5 0.12

N=$(wc -l < "$TASKS" | tr -d ' ')
echo
echo "tasks: $TASKS ($N tasks)"
echo "submit with:"
echo "  sbatch --array=0-$((N - 1)) scripts/submit_linear_response_sweep.sh"
