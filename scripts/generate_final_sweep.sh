#!/bin/bash
# The final sweep: the self-seeded absorbance spectrum of the model built up in five steps
# (config/mono/1-bare.yaml .. 5-electrons.yaml), at the 25 photon energies the experiment measured and at
# 1/5/20/30 uJ, 10 shots each. 5 steps x 100 configs = 500 configs, 8 per array task.
# Plots: scripts/plot_final.py.
#
# Run from the repo root on the cluster login node, then submit with the printed command.

set -euo pipefail
cd "$(dirname "$0")/.."

FAMILY=config/generated/final_sweep_mono
E_SEED=(1 5 20 30)
ENERGIES=(8000 8005 8010 8015 8019 8022 8025 8027 8028 8029 8032 8035 8038 8041 8044 8046 8047 8048 8049 8050
          8052 8055 8060 8065 8070)
NREP=10
PER_TASK=8

rm -rf "$FAMILY"
for step in 1-bare 2-single 3-double 4-middlemen 5-electrons; do
    python scripts/generate_sweep.py --family "$FAMILY" --variant "$step" --base "config/mono/$step.yaml" \
        --e-seed "${E_SEED[@]}" --energy "${ENERGIES[@]}"
done

n=$(wc -l < "$FAMILY/manifest.txt")
echo
echo "submit with:"
echo "  mkdir -p logs"
echo "  sbatch --array=0-$(( (n + PER_TASK - 1) / PER_TASK - 1 )) --export=ALL,NREP=$NREP,CHUNKS=1,PER_TASK=$PER_TASK scripts/submit_sweep.sh $FAMILY"
