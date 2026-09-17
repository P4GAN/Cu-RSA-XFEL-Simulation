#!/bin/bash
# Population budget runs: where the atoms and electrons are during the pulse, for the runs the model
# comparisons rest on. The sweeps keep only the exit plane's centre pixel, two electron totals, and the
# satellite blocks as dipole contractions rather than populations. These runs record every tracked
# population (middleman pool, base and each satellite block per manifold, each free-electron energy
# group, ...) at every z plane, centre pixel and beam average, with the transmittance alongside
# (scripts/run_population_budget.py, XLO_sim/population_budget.py; plots: scripts/plot_population_budget.py).
#
#   R        reference model: double satellites + middlemen + CK feeds + L-shell EII
#            (config/base/Cu-seed-{SASE,mono-SASE}-middlemen-eii.yaml)
#   raman10  R with the 2p3/2 dark state removed (sublevel_raman_dephasing_fs_inv = 10, as in
#            generate_coherence_sweeps.sh): the largest lever found so far, and the one that changes
#            how many K holes, Auger electrons and middlemen each hole makes
#
# Mono: on the Kalpha1 line (8048 eV, resonant cycling) and on the red wing (8000 eV, photoionisation
# only) at 1/5/20/50 uJ, 10 shots, 5x5, the bracket grid. SASE: 2/9/20/40 uJ, 40 shots (populations
# average far faster than spectra), 5x5.
#
# Run from the cluster login node, then submit with the two sbatch commands printed at the end.

set -euo pipefail
cd "$(dirname "$0")/.."

SASE_OUT=config/generated/population_budget_sase
MONO_OUT=config/generated/population_budget_mono
BASES=config/generated/population_budget_bases
SASE_E_SEED=(2 9 20 40)
MONO_E_SEED=(1 5 20 50)
MONO_ENERGIES=(8000 8048)
GRID="xgrid=5 ygrid=5"
SASE_NREP=40
MONO_NREP=10
MONO_CONFIGS_PER_TASK=4  # x 10 workers each = 40 cores

SASE_R=config/base/Cu-seed-SASE-middlemen-eii.yaml
MONO_R=config/base/Cu-seed-mono-SASE-middlemen-eii.yaml

mkdir -p "$SASE_OUT" "$MONO_OUT" "$BASES"
: > "$SASE_OUT/manifest.txt"
: > "$MONO_OUT/manifest.txt"

gen() {  # gen <variant> [derive_config transforms...]
    local name=$1
    shift
    echo "=== $name ==="
    python scripts/derive_config.py --base "$SASE_R" --out "$BASES/sase_$name.yaml" --apply "$@"
    python scripts/derive_config.py --base "$MONO_R" --out "$BASES/mono_$name.yaml" --apply "$@"
    python scripts/generate_intensity_sweep_configs.py \
        --base-yaml "$BASES/sase_$name.yaml" --out-dir "$SASE_OUT/$name" \
        --e-seed "${SASE_E_SEED[@]}" --set $GRID > /dev/null
    python scripts/generate_mono_sweep_configs.py \
        --base-yaml "$BASES/mono_$name.yaml" --out-dir "$MONO_OUT/$name" \
        --e-seed "${MONO_E_SEED[@]}" --energy "${MONO_ENERGIES[@]}" --set $GRID > /dev/null
    sed "s|^|$name |" "$SASE_OUT/$name/manifest.txt" >> "$SASE_OUT/manifest.txt"
    sed "s|^|$name |" "$MONO_OUT/$name/manifest.txt" >> "$MONO_OUT/manifest.txt"
}

gen R
gen raman10 raman=10

n_sase=$(( $(wc -l < "$SASE_OUT/manifest.txt") ))
n_mono=$(( $(wc -l < "$MONO_OUT/manifest.txt") ))
echo
echo "SASE: $n_sase configs -> $SASE_OUT/manifest.txt (one array task each, $SASE_NREP shots)"
echo "mono: $n_mono configs -> $MONO_OUT/manifest.txt ($MONO_CONFIGS_PER_TASK per array task, $MONO_NREP shots)"
echo
echo "submit with:"
echo "  sbatch --mem=64G --export=ALL,NREP=$SASE_NREP,CONFIGS_PER_TASK=1 --array=0-$(( n_sase - 1 )) scripts/submit_population_budget.sh $SASE_OUT"
echo "  sbatch --mem=0 --export=ALL,NREP=$MONO_NREP,CONFIGS_PER_TASK=$MONO_CONFIGS_PER_TASK --array=0-$(( (n_mono + MONO_CONFIGS_PER_TASK - 1) / MONO_CONFIGS_PER_TASK - 1 )) scripts/submit_population_budget.sh $MONO_OUT"
