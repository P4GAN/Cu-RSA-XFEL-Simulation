#!/bin/bash
# Grid convergence of the final model (step 5, self-seeded): one grid axis at a time is coarsened and refined
# around the grid of the final sweep (tgrid 12000, xgrid = ygrid 5, zgrid 15), at the Kalpha2 line, between the
# lines and at the Kalpha1 line (and at 8000 eV, the reference wing), at 1 and 20 uJ.
# Plot: scripts/plot_convergence.py.
#
# Run from the repo root on the cluster login node, then submit with the printed command.

set -euo pipefail
cd "$(dirname "$0")/.."

FAMILY=config/generated/convergence_sweep_mono
BASE=config/mono/5-electrons.yaml
E_SEED=(1 20)
ENERGIES=(8000 8028 8038 8048)
NREP=10
PER_TASK=8

gen() {  # gen <variant> [KEY=VALUE ...]
    local name=$1
    shift
    python scripts/generate_sweep.py --family "$FAMILY" --variant "$name" --base "$BASE" \
        --e-seed "${E_SEED[@]}" --energy "${ENERGIES[@]}" --set "$@"
}

rm -rf "$FAMILY"
gen ref     tgrid=12000
gen t6000   tgrid=6000
gen t24000  tgrid=24000
gen xy3     xgrid=3 ygrid=3
gen xy7     xgrid=7 ygrid=7
gen xy9     xgrid=9 ygrid=9
gen z8      zgrid=8
gen z30     zgrid=30

n=$(wc -l < "$FAMILY/manifest.txt")
echo
echo "submit with:"
echo "  mkdir -p logs"
echo "  sbatch --array=0-$(( (n + PER_TASK - 1) / PER_TASK - 1 )) --export=ALL,NREP=$NREP,CHUNKS=1,PER_TASK=$PER_TASK scripts/submit_sweep.sh $FAMILY"
