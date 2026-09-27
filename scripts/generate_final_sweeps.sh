#!/bin/bash
# Final presentation family: the model built up in five physics steps, plus three last experiments, all with
# every fix in place, as self-seeded (monochromator) absorbance spectra on the experiment's own photon
# energies. The model, step by step: docs/final-model-progression.md. Plots: scripts/plot_final.py.
#
# Fixes on in every step (docs/eii-model-evaluation.md):
#   read_field_after_last_plane   the field is read after all 14 absorption steps: a true 20 um foil
#                                 (the old off-by-one simulated 18.6 um)
#   satellite_detuning_sign_fix   satellites absorb at Kalpha1 + detuning, not at the mirror image
#                                 (no effect on step 1, which has no satellites)
#
#   1-bare        2p3/2, 1s and 2p1/2 holes in one density matrix (Kalpha1 + Kalpha2), original cross sections
#                 (config/base/Cu-seed-mono-SASE-L2-original.yaml)
#   2-single      + 2s hole and the single-spectator satellites 3d+-, 3p+- fed by 2s Coster-Kronig
#                 (config/base/Cu-seed-mono-SASE.yaml)
#   3-double      + double-3d satellites fed by the 3p satellites' super-Coster-Kronig decay
#                 (config/base/Cu-seed-mono-SASE-double-satellite.yaml)
#   4-middlemen   + the middleman pool (every atom stays in the model and keeps absorbing) and the metal
#                 Coster-Kronig feeds (config/base/Cu-seed-mono-SASE-middlemen.yaml)
#   5-electrons   + free electrons and electron-impact ionisation: non-thermal 24-level ladder, delta
#                 electrons, Bote-Salvat cross sections (config/base/Cu-seed-mono-SASE-middlemen-eii.yaml)
#
# Final experiments, each on top of step 5:
#   E1-deph       collisional dephasing (optical + Raman) at the valence collision rate, k = 1: electrons
#                 scramble the phase of the Kalpha coherence and the 2p3/2 sublevels (metal picture, upper bound)
#   E2-core       electron-impact ionisation of the M shell of atoms that already carry a core hole (atomic
#                 picture): base holes -> single -> double satellites, with a third spectator kept in the
#                 double block (the gap family lost those atoms to the non-resonant pool), + Raman dephasing
#   E3-exchange   2p-3d exchange in the open-shell satellites mixes the 2p3/2 sublevels at any fluence:
#                 Raman dephasing of 2 fs^-1 (~1.3 eV splitting) on the satellite blocks only
#
# Grid: 25 photon energies = the measured points from 8000 to 8070 eV (slide-9 scatter bins), 1/5/20/30 uJ,
# 5x5, 10 reps. 8 variants x 100 configs = 800 configs, 8 per array task; the model steps come first in the
# manifest, so they are scheduled first.
#
# Run from the cluster login node (after pulling), then submit with the sbatch command printed at the end.
# Data: data/final_sweep_mono_<jobid>/<variant>/.

set -euo pipefail
cd "$(dirname "$0")/.."

OUT=config/generated/final_sweep_mono
BASES=config/generated/final_bases
E_SEED=(1 5 20 30)
ENERGIES=(8000 8005 8010 8015 8019 8022 8025 8027 8028 8029 8032 8035 8038 8041 8044 8046 8047 8048 8049 8050
          8052 8055 8060 8065 8070)
GRID="xgrid=5 ygrid=5"
CONFIGS_PER_TASK=8   # must match CONFIGS_PER_TASK in submit_pathway_sweep_mono.sh

FIXES=(flag=read_field_after_last_plane:true flag=satellite_detuning_sign_fix:true)
EII=(eii=n_groups:24 eii=E_top_eV:7950 eii=E_bottom_eV:16.5 eii=anchor_birth_energies:true
     eii=secondary_spectrum:true eii=M_shell_scale:0.0 eii=spatial_factor:0.5 eii=cross_section:bote_salvat)
DEPH='eii=dephasing:{optical: 1.0, raman: 1.0}'
RAMAN='eii=dephasing:{optical: 0.0, raman: 1.0}'
# spectator on a core-holed atom: 3d5/2 : 3d3/2 = 6 : 4 and 3p3/2 : 3p1/2 = 4 : 2 on the bare hole; a second 3d
# hole by the electrons left in each j-shell; anything beyond two spectators stays in the double block that
# is nearest in line position (a 3p hole super-Coster-Kronigs to 3d^-2 in ~0.25 fs); 3s hits -> middlemen
CORE='eii=core_hole_EII:{scale: 1.0,
      base: {3d: {3d+: 0.6, 3d-: 0.4}, 3p: {3p+: 0.667, 3p-: 0.333}},
      3d+: {3d: {3d+3d+: 0.556, 3d-3d+: 0.444}, 3p: {3d+3d+: 0.556, 3d-3d+: 0.444}},
      3d-: {3d: {3d-3d+: 0.667, 3d-3d-: 0.333}, 3p: {3d-3d+: 0.667, 3d-3d-: 0.333}},
      3p+: {3d: {3d+3d+: 0.333, 3d-3d+: 0.534, 3d-3d-: 0.133}, 3p: {3d+3d+: 0.333, 3d-3d+: 0.534, 3d-3d-: 0.133}},
      3p-: {3d: {3d+3d+: 0.333, 3d-3d+: 0.534, 3d-3d-: 0.133}, 3p: {3d+3d+: 0.333, 3d-3d+: 0.534, 3d-3d-: 0.133}},
      3d+3d+: {3d: {3d+3d+: 1.0}, 3p: {3d+3d+: 1.0}},
      3d-3d+: {3d: {3d-3d+: 1.0}, 3p: {3d-3d+: 1.0}},
      3d-3d-: {3d: {3d-3d-: 1.0}, 3p: {3d-3d-: 1.0}}}'

mkdir -p "$OUT" "$BASES"
: > "$OUT/manifest.txt"

gen() {  # gen <variant> <base yaml> [derive_config transforms...]
    local name=$1 base=$2
    shift 2
    echo "=== $name ==="
    python scripts/derive_config.py --base "$base" --out "$BASES/$name.yaml" --apply "$@" > /dev/null
    python scripts/run_mono_sweep.py --yaml "$BASES/$name.yaml" --rep-end 1 --check-only | grep -E "extensions|check"
    python scripts/generate_mono_sweep_configs.py \
        --base-yaml "$BASES/$name.yaml" --out-dir "$OUT/$name" \
        --e-seed "${E_SEED[@]}" --energy "${ENERGIES[@]}" --set $GRID > /dev/null
    sed "s|^|$name |" "$OUT/$name/manifest.txt" >> "$OUT/manifest.txt"
}

gen 1-bare      config/base/Cu-seed-mono-SASE-L2-original.yaml       "${FIXES[@]}"
gen 2-single    config/base/Cu-seed-mono-SASE.yaml                   "${FIXES[@]}"
gen 3-double    config/base/Cu-seed-mono-SASE-double-satellite.yaml  "${FIXES[@]}"
gen 4-middlemen config/base/Cu-seed-mono-SASE-middlemen.yaml         "${FIXES[@]}"
gen 5-electrons config/base/Cu-seed-mono-SASE-middlemen-eii.yaml     "${FIXES[@]}" "${EII[@]}"
gen E1-deph     config/base/Cu-seed-mono-SASE-middlemen-eii.yaml     "${FIXES[@]}" "${EII[@]}" "$DEPH"
gen E2-core     config/base/Cu-seed-mono-SASE-middlemen-eii.yaml     "${FIXES[@]}" "${EII[@]}" "$CORE" "$RAMAN"
gen E3-exchange config/base/Cu-seed-mono-SASE-middlemen-eii.yaml     "${FIXES[@]}" "${EII[@]}" \
                                                                     flag=sublevel_raman_dephasing_satellite_fs_inv:2.0

n=$(( $(wc -l < "$OUT/manifest.txt") ))
echo
echo "$n configs -> $OUT/manifest.txt ($CONFIGS_PER_TASK per array task, 10 reps each)"
echo
echo "submit with:"
echo "  sbatch --array=0-$(( (n + CONFIGS_PER_TASK - 1) / CONFIGS_PER_TASK - 1 )) scripts/submit_pathway_sweep_mono.sh $OUT"
