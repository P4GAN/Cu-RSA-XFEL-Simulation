#!/bin/bash
# Bracketing sweeps around the reference model R = double-satellite + middlemen + CK feeds + L-shell
# EII (config/base/Cu-seed-{SASE,mono-SASE}-middlemen-eii.yaml). Every variant changes ONE thing
# relative to R, so a difference against R is attributable. Bases are derived with
# scripts/derive_config.py, the sweeps with the usual generators, and the submit scripts are the
# pathway ones with the family directory as $1.
#
#   R           the reference itself (rerun so every comparison is on the same code and grid)
#   mixsat-pop  2p3/2 sublevel mixing 2 fs^-1 in the satellite blocks, population exchange only
#               (coherence factor 0): the dark-state lever without the 0.66 eV of dephasing the
#               Lindblad form adds; compare with the pathway sweep's mid-eii-mixsat (factor 1)
#   mixall-pop  the same in the base block too (upper bound)
#   eii1        EII spatial factor 1.0 (every EII event lands in the focus; upper bound on f)
#   det0        every satellite absorbs at the bare Kalpha1/Kalpha2 lines: the "spectator 3d hole
#               delocalises in ~0.2 fs" limit of the metal (overlap O -> 1)
#   grasp       GRASP-recomputed detunings and cross sections (Cu-seed-*-double-satellite-grasp.yaml)
#               with the same extension blocks: brackets the atomic-data uncertainty, in particular
#               the sign of the 3d-spectator shift (XATOM red, GRASP blue)
#   spot71      focal FWHM x 0.7071 in both directions (half the area, 2x peak flux): the onset test
#   foil10      10 um foil, against Ichiro's 10 um data (excess absorption is front-loaded:
#               260 cm^-1 at 10 um vs ~200 at 20 um)
#
# Grids: SASE as the base configs (tgrid 600, zgrid 30, 3x3) at 2/9/20/40 uJ, 200 reps; SASE_GRID in
# the environment overrides that (e.g. SASE_GRID="xgrid=5 ygrid=5", which the 3x3 on-axis-fluence
# artefact calls for; submit with sbatch --mem=64G then, 40 workers at ~2.8x the 3x3 memory). Mono: tgrid
# 12000, zgrid 15, 5x5 (as the pathway sweep) but only 15 photon energies x 1/5/20/30 uJ, 10 reps:
# enough to read T_wing, both line depths and the between-lines level, at a quarter of the cost.
#
# Run from the cluster login node, then submit with the two sbatch commands printed at the end.

set -euo pipefail
cd "$(dirname "$0")/.."

SASE_OUT=config/generated/bracket_sweep_sase
MONO_OUT=config/generated/bracket_sweep_mono
BASES=config/generated/bracket_bases
SASE_E_SEED=(2 9 20 40)
MONO_E_SEED=(1 5 20 30)
MONO_ENERGIES=(8000 8015 8020 8025 8028 8032 8036 8040 8044 8046 8048 8050 8055 8060 8070)
MONO_GRID="xgrid=5 ygrid=5"
SASE_GRID=${SASE_GRID:-}  # empty: the base configs' own grid
MONO_CONFIGS_PER_TASK=8  # must match CONFIGS_PER_TASK in submit_pathway_sweep_mono.sh

SASE_R=config/base/Cu-seed-SASE-middlemen-eii.yaml
MONO_R=config/base/Cu-seed-mono-SASE-middlemen-eii.yaml

mkdir -p "$SASE_OUT" "$MONO_OUT" "$BASES"
: > "$SASE_OUT/manifest.txt"
: > "$MONO_OUT/manifest.txt"

gen() {  # gen <variant> <SASE base> <mono base> [derive_config transforms...]
    local name=$1 sase=$2 mono=$3
    shift 3
    echo "=== $name ==="
    python scripts/derive_config.py --base "$sase" --out "$BASES/sase_$name.yaml" --apply "$@"
    python scripts/derive_config.py --base "$mono" --out "$BASES/mono_$name.yaml" --apply "$@"
    python scripts/generate_intensity_sweep_configs.py \
        --base-yaml "$BASES/sase_$name.yaml" --out-dir "$SASE_OUT/$name" \
        --e-seed "${SASE_E_SEED[@]}" ${SASE_GRID:+--set $SASE_GRID} > /dev/null
    python scripts/generate_mono_sweep_configs.py \
        --base-yaml "$BASES/mono_$name.yaml" --out-dir "$MONO_OUT/$name" \
        --e-seed "${MONO_E_SEED[@]}" --energy "${MONO_ENERGIES[@]}" --set $MONO_GRID > /dev/null
    sed "s|^|$name |" "$SASE_OUT/$name/manifest.txt" >> "$SASE_OUT/manifest.txt"
    sed "s|^|$name |" "$MONO_OUT/$name/manifest.txt" >> "$MONO_OUT/manifest.txt"
}

gen R          "$SASE_R" "$MONO_R"
gen mixsat-pop "$SASE_R" "$MONO_R" mixing=2.0,sat,0.0
gen mixall-pop "$SASE_R" "$MONO_R" mixing=2.0,all,0.0
gen eii1       "$SASE_R" "$MONO_R" eii_spatial=1.0
gen det0       "$SASE_R" "$MONO_R" detunings_zero
gen grasp      config/base/Cu-seed-SASE-double-satellite-grasp.yaml \
               config/base/Cu-seed-mono-SASE-double-satellite-grasp.yaml \
               "extensions_from=$MONO_R"
gen spot71     "$SASE_R" "$MONO_R" spot_scale=0.7071
gen foil10     "$SASE_R" "$MONO_R" foil_um=10

n_sase=$(( $(wc -l < "$SASE_OUT/manifest.txt") ))
n_mono=$(( $(wc -l < "$MONO_OUT/manifest.txt") ))
echo
echo "SASE: $n_sase configs -> $SASE_OUT/manifest.txt (one array task each)"
echo "mono: $n_mono configs -> $MONO_OUT/manifest.txt ($MONO_CONFIGS_PER_TASK per array task)"
echo
echo "submit with:"
echo "  sbatch ${SASE_GRID:+--mem=64G }--array=0-$(( n_sase - 1 )) scripts/submit_pathway_sweep_sase.sh $SASE_OUT"
echo "  sbatch --array=0-$(( (n_mono + MONO_CONFIGS_PER_TASK - 1) / MONO_CONFIGS_PER_TASK - 1 )) scripts/submit_pathway_sweep_mono.sh $MONO_OUT"
