#!/bin/bash
# Batch B1: the double-L-hole absorbers (config/base/Cu-seed-*-middlemen-eii-dLL.yaml, from
# xatom/double_L_hole_parameters.py) against the reference R, on the blue side of Kalpha1 where they
# resonate: 2s^-1 2p^-1 at 8054/8075 eV (~6 eV wide), 2p^-2 at 8063/8083 eV. The seeded data show
# absorption there switching on above ~10-20 uJ (T(8060) 0.42 -> 0.32 by 50 uJ), which the model has
# no state for. The 2p^-2 route that matters when pumping off-resonance is sequential (a 2p-hole atom
# loses a second 2p electron), so it needs the high pulse energies: estimate A(8063) ~ 0.02 at 20 uJ,
# ~0.2 at 80 uJ.
#
# Mono: 10 photon energies (line centre + blue side) x 5/20/50/80 uJ, 10 reps, both variants -- R on
# the same grid so the difference is attributable. SASE: dLL at 2/9/20/40 uJ (the 40 uJ measurement has
# the 8055-8070 shoulder). Same submit scripts as the pathway/bracket sweeps.

set -euo pipefail
cd "$(dirname "$0")/.."

SASE_OUT=config/generated/b1_sweep_sase
MONO_OUT=config/generated/b1_sweep_mono
SASE_E_SEED=(2 9 20 40)
MONO_E_SEED=(5 20 50 80)
MONO_ENERGIES=(8040 8044 8048 8052 8055 8060 8065 8070 8080 8090)
MONO_GRID="xgrid=5 ygrid=5"
MONO_CONFIGS_PER_TASK=8  # must match CONFIGS_PER_TASK in submit_pathway_sweep_mono.sh

mkdir -p "$SASE_OUT" "$MONO_OUT"
: > "$SASE_OUT/manifest.txt"
: > "$MONO_OUT/manifest.txt"

gen() {  # gen <variant> <SASE base or -> <mono base or ->
    local name=$1 sase=$2 mono=$3
    echo "=== $name ==="
    if [[ "$sase" != "-" ]]; then
        python scripts/generate_intensity_sweep_configs.py \
            --base-yaml "$sase" --out-dir "$SASE_OUT/$name" --e-seed "${SASE_E_SEED[@]}" > /dev/null
        sed "s|^|$name |" "$SASE_OUT/$name/manifest.txt" >> "$SASE_OUT/manifest.txt"
    fi
    if [[ "$mono" != "-" ]]; then
        python scripts/generate_mono_sweep_configs.py \
            --base-yaml "$mono" --out-dir "$MONO_OUT/$name" \
            --e-seed "${MONO_E_SEED[@]}" --energy "${MONO_ENERGIES[@]}" --set $MONO_GRID > /dev/null
        sed "s|^|$name |" "$MONO_OUT/$name/manifest.txt" >> "$MONO_OUT/manifest.txt"
    fi
}

gen R   -                                              config/base/Cu-seed-mono-SASE-middlemen-eii.yaml
gen dLL config/base/Cu-seed-SASE-middlemen-eii-dLL.yaml config/base/Cu-seed-mono-SASE-middlemen-eii-dLL.yaml

n_sase=$(( $(wc -l < "$SASE_OUT/manifest.txt") ))
n_mono=$(( $(wc -l < "$MONO_OUT/manifest.txt") ))
echo
echo "SASE: $n_sase configs -> $SASE_OUT/manifest.txt (one array task each)"
echo "mono: $n_mono configs -> $MONO_OUT/manifest.txt ($MONO_CONFIGS_PER_TASK per array task)"
echo
echo "submit with:"
echo "  sbatch --array=0-$(( n_sase - 1 )) scripts/submit_pathway_sweep_sase.sh $SASE_OUT"
echo "  sbatch --array=0-$(( (n_mono + MONO_CONFIGS_PER_TASK - 1) / MONO_CONFIGS_PER_TASK - 1 )) scripts/submit_pathway_sweep_mono.sh $MONO_OUT"
