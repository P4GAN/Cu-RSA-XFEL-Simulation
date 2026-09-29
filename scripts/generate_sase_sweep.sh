#!/bin/bash
# The final model (step 5) with SASE pulses: transmitted spectra against pulse energy, 200 shots per pulse
# energy (a single SASE shot has a spiky, random spectrum, so many are averaged). Each pulse energy's shots
# are split over 5 array tasks of 40 shots, one task per node.
#
# Run from the repo root on the cluster login node, then submit with the printed command.

set -euo pipefail
cd "$(dirname "$0")/.."

FAMILY=config/generated/sase_sweep
E_SEED=(2 5 9 20 40 60 80)
NREP=200
CHUNKS=5

rm -rf "$FAMILY"
python scripts/generate_sweep.py --family "$FAMILY" --variant 5-electrons --base config/sase/5-electrons.yaml \
    --e-seed "${E_SEED[@]}"

n=$(wc -l < "$FAMILY/manifest.txt")
echo
echo "submit with:"
echo "  mkdir -p logs"
echo "  sbatch --array=0-$(( n * CHUNKS - 1 )) --export=ALL,NREP=$NREP,CHUNKS=$CHUNKS,PER_TASK=1 scripts/submit_sweep.sh $FAMILY"
