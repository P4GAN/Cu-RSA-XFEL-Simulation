#!/bin/bash
# Model progression, monochromator: the same mono sweep for each stage of the model, from the original
# L3/K model with only the 2p1/2 hole added, up to electron-impact ionisation. One physics addition per
# stage, so each step's effect on the spectrum can be read off by itself.
#
#   1-L2          8 levels: 2p3/2, 1s and 2p1/2 holes in one density matrix, nothing else (no 2s, no
#                 satellites), original GRASP/RATIP cross sections
#                 (config/base/Cu-seed-mono-SASE-L2-original.yaml)
#   2-sat         + 2s hole and the four single-spectator satellite blocks (3d+-, 3p+-), + middlemen
#                 (routed to the 3d satellites) and the metal CK feeds: L2 -> 3d satellites, 2s -> bare
#                 2p + 4s (config/base/Cu-seed-mono-SASE-satellite-middlemen.yaml)
#   3-dsat        + the double-3d satellites fed by the 3p satellites' super-CK; middlemen routed to them
#                 (config/base/Cu-seed-mono-SASE-middlemen.yaml)
#   4a-eii-L      + free electrons and electron-impact ionisation, L shell only (2p3/2, 2p1/2, 2s holes),
#                 non-thermal ladder: 24 energy levels from 7.95 keV down to the 3d threshold (16.5 eV),
#                 each source born at the top of its own level, and the secondary (delta) electron of
#                 every ionisation followed down the ladder
#   4b-eii-LM     4a + M-shell (3s/3p/3d) EII at the atomic cross section: neutral atoms -> middlemen.
#                 Upper bound (in the metal a single 3d vacancy delocalises); M-shell EII of atoms that
#                 already have a core hole is not modelled
#   4c-eii-fixed  4a's L-shell EII with fixed-energy electrons: one level per source (7.95, 7.1, 0.87,
#                 0.06 keV) that never slows down -- the bound. Secondaries go straight to the bin below
#                 16.5 eV (at a fixed energy they would ionise without limit)
#
# 4a-4c are derived from config/base/Cu-seed-mono-SASE-middlemen-eii.yaml (the old 6-group reference,
# which stays unchanged) by scripts/derive_config.py; each generated config records its transforms under
# "derived". The model: docs/theory-eii-electron-ladder-explained.md.
#
# Mono grid: the bracket/coherence grid (15 energies x 1/5/20/30 uJ, 5x5, 10 reps), so the figures of
# scripts/plot_bracket_sweep.py / plot_coherence_sweep.py compare directly with the experiment.
# Population budget (4a-4c only): every population and every electron level at every z plane vs time,
# on the Kalpha1 line (8048 eV) and the red wing (8000 eV), same pulse energies, 10 shots, 5x5
# (scripts/run_population_budget.py; plot with scripts/plot_population_budget.py).
#
# Run from the cluster login node (after pulling this commit), then submit with the two sbatch commands
# printed at the end. Data: data/progression_sweep_mono_<jobid>/<variant>/ and
# data/progression_budget_mono_<jobid>/<variant>/.

set -euo pipefail
cd "$(dirname "$0")/.."

MONO_OUT=config/generated/progression_sweep_mono
BUDGET_OUT=config/generated/progression_budget_mono
BASES=config/generated/progression_bases
MONO_E_SEED=(1 5 20 30)
MONO_ENERGIES=(8000 8015 8020 8025 8028 8032 8036 8040 8044 8046 8048 8050 8055 8060 8070)
BUDGET_ENERGIES=(8000 8048)
GRID="xgrid=5 ygrid=5"
MONO_CONFIGS_PER_TASK=8    # must match CONFIGS_PER_TASK in submit_pathway_sweep_mono.sh
BUDGET_NREP=10
BUDGET_CONFIGS_PER_TASK=4  # x 10 workers each = 40 cores

EII_REF=config/base/Cu-seed-mono-SASE-middlemen-eii.yaml
LADDER=(eii=n_groups:24 eii=E_top_eV:7950 eii=E_bottom_eV:16.5 eii=anchor_birth_energies:true
        eii=secondary_spectrum:true eii=M_shell_scale:0.0 eii=spatial_factor:0.5)

mkdir -p "$MONO_OUT" "$BUDGET_OUT" "$BASES"
: > "$MONO_OUT/manifest.txt"
: > "$BUDGET_OUT/manifest.txt"

gen() {  # gen <variant> <base yaml> <in budget: 0|1> [derive_config transforms...]
    local name=$1 base=$2 budget=$3
    shift 3
    echo "=== $name ==="
    python scripts/derive_config.py --base "$base" --out "$BASES/mono_$name.yaml" --apply "$@"
    python scripts/run_mono_sweep.py --yaml "$BASES/mono_$name.yaml" --rep-end 1 --check-only | tail -n 4
    python scripts/generate_mono_sweep_configs.py \
        --base-yaml "$BASES/mono_$name.yaml" --out-dir "$MONO_OUT/$name" \
        --e-seed "${MONO_E_SEED[@]}" --energy "${MONO_ENERGIES[@]}" --set $GRID > /dev/null
    sed "s|^|$name |" "$MONO_OUT/$name/manifest.txt" >> "$MONO_OUT/manifest.txt"
    if [[ $budget == 1 ]]; then
        python scripts/generate_mono_sweep_configs.py \
            --base-yaml "$BASES/mono_$name.yaml" --out-dir "$BUDGET_OUT/$name" \
            --e-seed "${MONO_E_SEED[@]}" --energy "${BUDGET_ENERGIES[@]}" --set $GRID > /dev/null
        sed "s|^|$name |" "$BUDGET_OUT/$name/manifest.txt" >> "$BUDGET_OUT/manifest.txt"
    fi
}

gen 1-L2         config/base/Cu-seed-mono-SASE-L2-original.yaml        0
gen 2-sat        config/base/Cu-seed-mono-SASE-satellite-middlemen.yaml 0
gen 3-dsat       config/base/Cu-seed-mono-SASE-middlemen.yaml           0
gen 4a-eii-L     "$EII_REF" 1 "${LADDER[@]}"
gen 4b-eii-LM    "$EII_REF" 1 "${LADDER[@]}" eii=M_shell_scale:1.0
gen 4c-eii-fixed "$EII_REF" 1 eii=slowing_down:false eii=E_bottom_eV:16.5 eii=secondary_spectrum:false \
                              eii=M_shell_scale:0.0 eii=spatial_factor:0.5

n_mono=$(( $(wc -l < "$MONO_OUT/manifest.txt") ))
n_budget=$(( $(wc -l < "$BUDGET_OUT/manifest.txt") ))
echo
echo "mono sweep: $n_mono configs -> $MONO_OUT/manifest.txt ($MONO_CONFIGS_PER_TASK per array task, 10 reps)"
echo "population budget: $n_budget configs -> $BUDGET_OUT/manifest.txt ($BUDGET_CONFIGS_PER_TASK per array task, $BUDGET_NREP shots)"
echo
echo "submit with:"
echo "  sbatch --array=0-$(( (n_mono + MONO_CONFIGS_PER_TASK - 1) / MONO_CONFIGS_PER_TASK - 1 )) scripts/submit_pathway_sweep_mono.sh $MONO_OUT"
echo "  sbatch --mem=0 --export=ALL,NREP=$BUDGET_NREP,CONFIGS_PER_TASK=$BUDGET_CONFIGS_PER_TASK --array=0-$(( (n_budget + BUDGET_CONFIGS_PER_TASK - 1) / BUDGET_CONFIGS_PER_TASK - 1 )) scripts/submit_population_budget.sh $BUDGET_OUT"
