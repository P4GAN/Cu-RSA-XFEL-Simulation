#!/bin/bash
# Population budget of the final model (step 5): where the atoms and electrons are during the pulse, at every
# z plane (scripts/run_population_budget.py; plots: plot_population_budget.py, plot_eii_diagnostics.py).
#   mono: on the Kalpha1 line (8048 eV) and on the red wing (8000 eV, photoionisation only) at 1/5/20/30 uJ, 10 shots
#   SASE: 2/9/20/40 uJ, 40 shots (populations average over shots far faster than spectra)
#
# Run from the repo root on the cluster login node, then submit with the two printed commands.

set -euo pipefail
cd "$(dirname "$0")/.."

MONO=config/generated/population_budget_mono
SASE=config/generated/population_budget_sase
MONO_PER_TASK=4   # x 10 processes = 40 cores

rm -rf "$MONO" "$SASE"
python scripts/generate_sweep.py --family "$MONO" --variant 5-electrons --base config/mono/5-electrons.yaml \
    --e-seed 1 5 20 30 --energy 8000 8048
python scripts/generate_sweep.py --family "$SASE" --variant 5-electrons --base config/sase/5-electrons.yaml \
    --e-seed 2 9 20 40

n_mono=$(wc -l < "$MONO/manifest.txt")
n_sase=$(wc -l < "$SASE/manifest.txt")
echo
echo "submit with:"
echo "  mkdir -p logs"
echo "  sbatch --array=0-$(( (n_mono + MONO_PER_TASK - 1) / MONO_PER_TASK - 1 )) --mem=0 --export=ALL,NREP=10,CONFIGS_PER_TASK=$MONO_PER_TASK scripts/submit_population_budget.sh $MONO"
echo "  sbatch --array=0-$(( n_sase - 1 )) --mem=64G --export=ALL,NREP=40,CONFIGS_PER_TASK=1 scripts/submit_population_budget.sh $SASE"
