#!/bin/bash
# Coherence levers on the reference model R (config/base/Cu-seed-{SASE,mono-SASE}-middlemen-eii.yaml),
# after the bracket sweep (scripts/generate_bracket_sweeps.sh) showed no one-change variant closing more
# than ~10% of the saturated gap. Two nonlinear effects are left that set how many resonant photons a
# 2p3/2 hole absorbs and over what width:
#
#   R             the reference (rerun on the same code and grids)
#   raman10       sublevel_raman_dephasing_fs_inv = 10: damps only the 2p-2p and 1s-1s (Raman)
#                 coherences, so the linearly polarised field can no longer park the 2p3/2 hole in its
#                 dark superposition. 0D kernel check: photons absorbed per hole at saturation 0.39 ->
#                 0.60 (rate equations 0.64), weak-field line shape unchanged to 2e-4. The bracket's
#                 mixsat-pop/mixall-pop exchanged populations only and left this coherence intact,
#                 which is why they moved the dip by 5-11% of the gap.
#   broad25       additional_dephasing = 2.5 fs^-1 on every optical (and 2p3/2-2p1/2) coherence: +3.3 eV
#                 homogeneous FWHM, taking the 2 uJ SASE line from 2.9 to ~6.2 eV, the measured width.
#                 In saturation the scan area grows ~sqrt(width) at a fixed ceiling, so this tests
#                 whether width alone buys area.
#   raman10-broad25  both
#
# Both are phenomenological upper bounds (what would it take), not mechanisms: they say whether the
# measured depth, width and area are reachable by these two levers before building the physics that
# would supply them (open-shell multiplets for the dark state; collisions/unresolved satellites for the
# width).
#
# Grids: SASE at xgrid=ygrid=5 (the 3x3 base grid puts 0.67x the fluence on axis and overstates the 2 uJ
# dip by 27%), tgrid 600, zgrid 30, 200 reps, 2/9/20/40 uJ -- submit with --mem=64G. Mono exactly the
# bracket grid (15 energies x 1/5/20/30 uJ, 5x5, 10 reps) so figs/bracket_* compare directly.
#
# Run from the cluster login node, then submit with the two sbatch commands printed at the end.

set -euo pipefail
cd "$(dirname "$0")/.."

SASE_OUT=config/generated/coherence_sweep_sase
MONO_OUT=config/generated/coherence_sweep_mono
BASES=config/generated/coherence_bases
SASE_E_SEED=(2 9 20 40)
MONO_E_SEED=(1 5 20 30)
MONO_ENERGIES=(8000 8015 8020 8025 8028 8032 8036 8040 8044 8046 8048 8050 8055 8060 8070)
SASE_GRID="xgrid=5 ygrid=5"
MONO_GRID="xgrid=5 ygrid=5"
MONO_CONFIGS_PER_TASK=8  # must match CONFIGS_PER_TASK in submit_pathway_sweep_mono.sh

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
        --e-seed "${SASE_E_SEED[@]}" --set $SASE_GRID > /dev/null
    python scripts/generate_mono_sweep_configs.py \
        --base-yaml "$BASES/mono_$name.yaml" --out-dir "$MONO_OUT/$name" \
        --e-seed "${MONO_E_SEED[@]}" --energy "${MONO_ENERGIES[@]}" --set $MONO_GRID > /dev/null
    sed "s|^|$name |" "$SASE_OUT/$name/manifest.txt" >> "$SASE_OUT/manifest.txt"
    sed "s|^|$name |" "$MONO_OUT/$name/manifest.txt" >> "$MONO_OUT/manifest.txt"
}

gen R
gen raman10         raman=10
gen broad25         set=additional_dephasing:2.5
gen raman10-broad25 raman=10 set=additional_dephasing:2.5

n_sase=$(( $(wc -l < "$SASE_OUT/manifest.txt") ))
n_mono=$(( $(wc -l < "$MONO_OUT/manifest.txt") ))
echo
echo "SASE: $n_sase configs -> $SASE_OUT/manifest.txt (one array task each)"
echo "mono: $n_mono configs -> $MONO_OUT/manifest.txt ($MONO_CONFIGS_PER_TASK per array task)"
echo
echo "submit with:"
echo "  sbatch --mem=64G --array=0-$(( n_sase - 1 )) scripts/submit_pathway_sweep_sase.sh $SASE_OUT"
echo "  sbatch --array=0-$(( (n_mono + MONO_CONFIGS_PER_TASK - 1) / MONO_CONFIGS_PER_TASK - 1 )) scripts/submit_pathway_sweep_mono.sh $MONO_OUT"
