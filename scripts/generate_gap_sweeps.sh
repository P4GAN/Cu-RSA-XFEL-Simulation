#!/bin/bash
# Closing the gap: two bug fixes, a better EII cross section, and what electron collisions do to atoms that
# already carry a core hole. Built on the full model of the progression runs (4a-eii-L: double satellites,
# middlemen, CK feeds, L-shell EII on the non-thermal 24-level ladder, config/base/*-middlemen-eii.yaml plus
# the ladder keys below). One change per step, then the mechanisms on top of the corrected reference:
#
#   a-readout     4a + read_field_after_last_plane: the transmitted field is read after all 14 absorption
#                 steps instead of 13 (old off-by-one; ~ +7.7% on every absorbance)
#   b-sign        a + satellite_detuning_sign_fix: satellites absorb at Kalpha1 + detuning_eV (3d spectators
#                 on the red side) instead of the mirror image (0D kernel check: docs/eii-model-evaluation.md
#                 sec 6); the measured lines sit ~2 eV below the model's
#   c-bs          b + Bote-Salvat (DWBA) EII cross sections for 2s/2p/3s/3p: +58% L-shell EII per primary
#                 (BCF is 35-45% low for 2p at 3-8 keV). The corrected reference for the rest
#   d-deph        c + electron-impact dephasing, optical + Raman, k = 1: every valence collision of a free
#                 electron with a core-holed ion randomises the phase (~1 fs^-1 at 20 uJ, grows with fluence)
#   e-deph-raman  c + the Raman (2p-2p, 1s-1s) half only: removes the 2p3/2 dark state, no broadening;
#                 d - e is the broadening
#   f-core        c + M-shell EII of core-holed atoms (atomic picture): the collision leaves a spectator,
#                 base 2p/1s holes -> 3d/3p single satellites, 3d singles -> 3d double satellites, the
#                 rest (3s, triple holes) -> middlemen; the missing EII feed of the satellite ladder
#   g-core-raman  f + Raman dephasing: ionising collisions move the atom, and every collision scrambles
#                 the 2p3/2 sublevels (no double counting of the optical dephasing, which f already has)
#
# Mono: the bracket grid (15 energies x 1/5/20/30 uJ, 5x5, 10 reps), every variant.
# SASE (the fluence-dependent width test: measured FWHM 6.2 -> 14.1 eV from 2 to 40 uJ, model 2.9 -> 4.3):
#   a, c, d, f, g at 2/9/20/40 uJ, 5x5, 200 reps (--mem=64G).
# Population budget: c, d, f at 8000/8048 eV x 1/5/20/30 uJ, 10 shots.
# The evaluation behind these: docs/eii-model-evaluation.md.
#
# Run from the cluster login node (after pulling), then submit with the three sbatch commands printed at
# the end. Data: data/gap_sweep_{mono,sase}_<jobid>/<variant>/, data/gap_budget_mono_<jobid>/<variant>/.

set -euo pipefail
cd "$(dirname "$0")/.."

MONO_OUT=config/generated/gap_sweep_mono
SASE_OUT=config/generated/gap_sweep_sase
BUDGET_OUT=config/generated/gap_budget_mono
BASES=config/generated/gap_bases
MONO_E_SEED=(1 5 20 30)
SASE_E_SEED=(2 9 20 40)
MONO_ENERGIES=(8000 8015 8020 8025 8028 8032 8036 8040 8044 8046 8048 8050 8055 8060 8070)
BUDGET_ENERGIES=(8000 8048)
GRID="xgrid=5 ygrid=5"
MONO_CONFIGS_PER_TASK=8    # must match CONFIGS_PER_TASK in submit_pathway_sweep_mono.sh
BUDGET_NREP=10
BUDGET_CONFIGS_PER_TASK=4

MONO_BASE=config/base/Cu-seed-mono-SASE-middlemen-eii.yaml
SASE_BASE=config/base/Cu-seed-SASE-middlemen-eii.yaml
LADDER=(eii=n_groups:24 eii=E_top_eV:7950 eii=E_bottom_eV:16.5 eii=anchor_birth_energies:true
        eii=secondary_spectrum:true eii=M_shell_scale:0.0 eii=spatial_factor:0.5)
R1=("${LADDER[@]}" flag=read_field_after_last_plane:true)
R2=("${R1[@]}" flag=satellite_detuning_sign_fix:true)
R3=("${R2[@]}" eii=cross_section:bote_salvat)
# spectator hole on a core-holed atom: 3d5/2 : 3d3/2 = 6 : 4 and 3p3/2 : 3p1/2 = 4 : 2 on the base block; a
# second 3d hole by the electrons left in each j-shell (3d+: 5 : 4, 3d-: 6 : 3); anything else -> middlemen
CORE='eii=core_hole_EII:{scale: 1.0, base: {3d: {3d+: 0.6, 3d-: 0.4}, 3p: {3p+: 0.667, 3p-: 0.333}},
      3d+: {3d: {3d+3d+: 0.556, 3d-3d+: 0.444}}, 3d-: {3d: {3d-3d+: 0.667, 3d-3d-: 0.333}}}'
DEPH='eii=dephasing:{optical: 1.0, raman: 1.0}'
RAMAN='eii=dephasing:{optical: 0.0, raman: 1.0}'

mkdir -p "$MONO_OUT" "$SASE_OUT" "$BUDGET_OUT" "$BASES"
: > "$MONO_OUT/manifest.txt"
: > "$SASE_OUT/manifest.txt"
: > "$BUDGET_OUT/manifest.txt"

gen() {  # gen <variant> <sase: 0|1> <budget: 0|1> [derive_config transforms...]
    local name=$1 sase=$2 budget=$3
    shift 3
    echo "=== $name ==="
    python scripts/derive_config.py --base "$MONO_BASE" --out "$BASES/mono_$name.yaml" --apply "$@" > /dev/null
    python scripts/run_mono_sweep.py --yaml "$BASES/mono_$name.yaml" --rep-end 1 --check-only | tail -n 3
    python scripts/generate_mono_sweep_configs.py \
        --base-yaml "$BASES/mono_$name.yaml" --out-dir "$MONO_OUT/$name" \
        --e-seed "${MONO_E_SEED[@]}" --energy "${MONO_ENERGIES[@]}" --set $GRID > /dev/null
    sed "s|^|$name |" "$MONO_OUT/$name/manifest.txt" >> "$MONO_OUT/manifest.txt"
    if [[ $sase == 1 ]]; then
        python scripts/derive_config.py --base "$SASE_BASE" --out "$BASES/sase_$name.yaml" --apply "$@" > /dev/null
        python scripts/run_intensity_sweep.py --yaml "$BASES/sase_$name.yaml" --rep-end 1 --check-only | tail -n 1
        python scripts/generate_intensity_sweep_configs.py \
            --base-yaml "$BASES/sase_$name.yaml" --out-dir "$SASE_OUT/$name" \
            --e-seed "${SASE_E_SEED[@]}" --set $GRID > /dev/null
        sed "s|^|$name |" "$SASE_OUT/$name/manifest.txt" >> "$SASE_OUT/manifest.txt"
    fi
    if [[ $budget == 1 ]]; then
        python scripts/generate_mono_sweep_configs.py \
            --base-yaml "$BASES/mono_$name.yaml" --out-dir "$BUDGET_OUT/$name" \
            --e-seed "${MONO_E_SEED[@]}" --energy "${BUDGET_ENERGIES[@]}" --set $GRID > /dev/null
        sed "s|^|$name |" "$BUDGET_OUT/$name/manifest.txt" >> "$BUDGET_OUT/manifest.txt"
    fi
}

gen a-readout    1 0 "${R1[@]}"
gen b-sign       0 0 "${R2[@]}"
gen c-bs         1 1 "${R3[@]}"
gen d-deph       1 1 "${R3[@]}" "$DEPH"
gen e-deph-raman 0 0 "${R3[@]}" "$RAMAN"
gen f-core       1 1 "${R3[@]}" "$CORE"
gen g-core-raman 1 0 "${R3[@]}" "$CORE" "$RAMAN"

n_mono=$(( $(wc -l < "$MONO_OUT/manifest.txt") ))
n_sase=$(( $(wc -l < "$SASE_OUT/manifest.txt") ))
n_budget=$(( $(wc -l < "$BUDGET_OUT/manifest.txt") ))
echo
echo "mono: $n_mono configs ($MONO_CONFIGS_PER_TASK per task, 10 reps); SASE: $n_sase configs (one per task, 200 reps);"
echo "population budget: $n_budget configs ($BUDGET_CONFIGS_PER_TASK per task, $BUDGET_NREP shots)"
echo
echo "submit with:"
echo "  sbatch --array=0-$(( (n_mono + MONO_CONFIGS_PER_TASK - 1) / MONO_CONFIGS_PER_TASK - 1 )) scripts/submit_pathway_sweep_mono.sh $MONO_OUT"
echo "  sbatch --mem=64G --time=08:00:00 --array=0-$(( n_sase - 1 )) scripts/submit_pathway_sweep_sase.sh $SASE_OUT"
echo "  sbatch --mem=0 --export=ALL,NREP=$BUDGET_NREP,CONFIGS_PER_TASK=$BUDGET_CONFIGS_PER_TASK --array=0-$(( (n_budget + BUDGET_CONFIGS_PER_TASK - 1) / BUDGET_CONFIGS_PER_TASK - 1 )) scripts/submit_population_budget.sh $BUDGET_OUT"
