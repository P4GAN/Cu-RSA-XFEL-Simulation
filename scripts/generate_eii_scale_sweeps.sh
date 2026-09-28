#!/bin/bash
# Can more electron-impact ionisation reach the measurement? Step 5 of the final family (the full model with
# electrons, every fix, Bote-Salvat) with the L-shell EII cross sections multiplied by 2, 4, 8 and 16
# (eii.L_shell_scale; docs/bote-salvat-and-final-experiments.md Part C). Only the rates that make 2p3/2, 2p1/2
# and 2s holes change; the electrons slow down as before.
#
#   x2    what a no-transport bound (spatial_factor 1) and a +10% cross section would give together: the most
#         that is physically defensible
#   x4, x8
#   x16   the energy ceiling: at x16 a 7.09 keV photoelectron makes ~4.9 L-shell holes (0.178 + 0.084 + 0.050,
#         x16), i.e. spends ~5 of its 7 keV on them. Beyond this the electron would make holes it cannot pay for.
#
# Scale 1 is 5-electrons of the final family (data/final_sweep_mono_<id>), which plot_eii_scale.py reads
# alongside. Same grid as that family: the 25 measured photon energies x 1/5/20/30 uJ, 5x5, 10 reps.
# Population budget of every variant at 8000/8048 eV x 1/5/20/30 uJ (where the extra holes come from, and when).
#
# Run from the cluster login node (after pulling), then submit with the two sbatch commands printed at the end.
# Data: data/eii_scale_sweep_mono_<jobid>/<variant>/, data/eii_scale_budget_mono_<jobid>/<variant>/.
# Plots: python scripts/plot_eii_scale.py

set -euo pipefail
cd "$(dirname "$0")/.."

OUT=config/generated/eii_scale_sweep_mono
BUDGET_OUT=config/generated/eii_scale_budget_mono
BASES=config/generated/eii_scale_bases
E_SEED=(1 5 20 30)
ENERGIES=(8000 8005 8010 8015 8019 8022 8025 8027 8028 8029 8032 8035 8038 8041 8044 8046 8047 8048 8049 8050
          8052 8055 8060 8065 8070)
BUDGET_ENERGIES=(8000 8048)
GRID="xgrid=5 ygrid=5"
CONFIGS_PER_TASK=8          # must match CONFIGS_PER_TASK in submit_pathway_sweep_mono.sh
BUDGET_NREP=10
BUDGET_CONFIGS_PER_TASK=4
SCALES=(2 4 8 16)

# = 5-electrons in scripts/generate_final_sweeps.sh
STEP5=(flag=read_field_after_last_plane:true flag=satellite_detuning_sign_fix:true
       eii=n_groups:24 eii=E_top_eV:7950 eii=E_bottom_eV:16.5 eii=anchor_birth_energies:true
       eii=secondary_spectrum:true eii=M_shell_scale:0.0 eii=spatial_factor:0.5 eii=cross_section:bote_salvat)

mkdir -p "$OUT" "$BUDGET_OUT" "$BASES"
: > "$OUT/manifest.txt"
: > "$BUDGET_OUT/manifest.txt"

for k in "${SCALES[@]}"; do
    name=x$k
    echo "=== $name ==="
    python scripts/derive_config.py --base config/base/Cu-seed-mono-SASE-middlemen-eii.yaml \
        --out "$BASES/$name.yaml" --apply "${STEP5[@]}" eii=L_shell_scale:$k > /dev/null
    python scripts/run_mono_sweep.py --yaml "$BASES/$name.yaml" --rep-end 1 --check-only | grep -E "extensions|check|eii:"
    python scripts/generate_mono_sweep_configs.py \
        --base-yaml "$BASES/$name.yaml" --out-dir "$OUT/$name" \
        --e-seed "${E_SEED[@]}" --energy "${ENERGIES[@]}" --set $GRID > /dev/null
    sed "s|^|$name |" "$OUT/$name/manifest.txt" >> "$OUT/manifest.txt"
    python scripts/generate_mono_sweep_configs.py \
        --base-yaml "$BASES/$name.yaml" --out-dir "$BUDGET_OUT/$name" \
        --e-seed "${E_SEED[@]}" --energy "${BUDGET_ENERGIES[@]}" --set $GRID > /dev/null
    sed "s|^|$name |" "$BUDGET_OUT/$name/manifest.txt" >> "$BUDGET_OUT/manifest.txt"
done

n=$(( $(wc -l < "$OUT/manifest.txt") ))
n_budget=$(( $(wc -l < "$BUDGET_OUT/manifest.txt") ))
echo
echo "spectra: $n configs ($CONFIGS_PER_TASK per array task, 10 reps); population budget: $n_budget configs"
echo
echo "submit with:"
echo "  sbatch --array=0-$(( (n + CONFIGS_PER_TASK - 1) / CONFIGS_PER_TASK - 1 )) scripts/submit_pathway_sweep_mono.sh $OUT"
echo "  sbatch --mem=0 --export=ALL,NREP=$BUDGET_NREP,CONFIGS_PER_TASK=$BUDGET_CONFIGS_PER_TASK --array=0-$(( (n_budget + BUDGET_CONFIGS_PER_TASK - 1) / BUDGET_CONFIGS_PER_TASK - 1 )) scripts/submit_population_budget.sh $BUDGET_OUT"
