#!/bin/bash
# Generate the SASE intensity sweep and the mono E_seed x energy sweep for 5 variants of the
# double-satellite model with the 2026-09-15 pathway extensions (docs/middlemen-implementation-plan.md,
# docs/eii-free-electrons-implementation-plan.md). Reuses generate_intensity_sweep_configs.py /
# generate_mono_sweep_configs.py (--base-yaml/--out-dir/--set), the same pattern as
# generate_production_sweeps.sh.
#
# Each variant adds one thing to the one before it, so the sweeps read as a ladder. In brackets: T
# from one MB shot at 20 uJ / 8048 eV / seed 0 (mono pulse, tgrid 3000, 7x7, 2026-09-15), with the
# digitised experiment at 0.278:
#
#   dsat            feed-fixed double-satellite model, no extensions (the reference)       (0.363)
#   mid             + middleman pool, L2 -> L3M45 CK feed, 2s -> bare-2p CK feeds          (0.340)
#   mid-eii         + L-shell electron-impact ionisation, spatial_factor 0.5               (0.334)
#   mid-eii-mixsat  + 2p3/2 sublevel mixing 2 fs^-1 in the satellite blocks only           (not run)
#   mid-eii-mix     + the same mixing in the base block too                                (0.320)
#
# mixsat is the physically motivated case (the 2p-3d exchange that mixes the dark m=+-3/2 sublevels
# needs an open 3d shell, theory doc Part VI sec 1.5). mix is the upper bound. M-shell EII is left
# off (M_shell_scale 0) everywhere: the middleman pool does not track how many 3d holes an atom
# has, and a 20 uJ test at 0.5 x BCF already made ~10 M-shell ionisations per atom.
# At 200 uJ (SASE, 2 shots, 2026-09-15) the dsat baseline lost 37% of its population (T 0.42) while
# mid-eii-mix kept the trace at 1 +/- 0.003 (T 0.29); that step-size error is smaller at <= 80 uJ.
#
# Grids. SASE: the base configs' grid unchanged (tgrid 600, zgrid 30, xgrid=ygrid 3, from the
# round-2 convergence sweeps), at E_seed 2-80 uJ (5 and 20 uJ shared with the mono sweep). Mono:
# the mono configs' grid (tgrid 12000, zgrid 15) with xgrid=ygrid cut from 11 to 5, at
# generate_mono_sweep_configs.py's default 5 E_seed (the experiment's 0.1-30 uJ) x 52 energies.
#
# Each family also gets one flat manifest ("<variant> <yaml>" per line) that the submit scripts index.
# Run once from the cluster login node (the manifests hold absolute paths), then submit with the
# two sbatch commands printed at the end.

set -euo pipefail
cd "$(dirname "$0")/.."

SASE_OUT=config/generated/pathway_sweep_sase
MONO_OUT=config/generated/pathway_sweep_mono
SASE_E_SEED=(2 5 9 20 40 60 80)
MONO_GRID="xgrid=5 ygrid=5"
MONO_CONFIGS_PER_TASK=8  # must match CONFIGS_PER_TASK in submit_pathway_sweep_mono.sh

mkdir -p "$SASE_OUT" "$MONO_OUT"
: > "$SASE_OUT/manifest.txt"
: > "$MONO_OUT/manifest.txt"

gen() {  # gen <variant> <SASE base> <mono base> [KEY=VALUE ...overrides for both families]
    local name=$1 sase=$2 mono=$3
    shift 3
    echo "=== $name ==="
    python scripts/generate_intensity_sweep_configs.py \
        --base-yaml "config/base/${sase}.yaml" --out-dir "$SASE_OUT/$name" \
        --e-seed "${SASE_E_SEED[@]}" --set "$@" > /dev/null
    python scripts/generate_mono_sweep_configs.py \
        --base-yaml "config/base/${mono}.yaml" --out-dir "$MONO_OUT/$name" \
        --set $MONO_GRID "$@" > /dev/null
    sed "s|^|$name |" "$SASE_OUT/$name/manifest.txt" >> "$SASE_OUT/manifest.txt"
    sed "s|^|$name |" "$MONO_OUT/$name/manifest.txt" >> "$MONO_OUT/manifest.txt"
}

gen dsat           Cu-seed-SASE-double-satellite Cu-seed-mono-SASE-double-satellite
gen mid            Cu-seed-SASE-middlemen        Cu-seed-mono-SASE-middlemen
gen mid-eii        Cu-seed-SASE-middlemen-eii    Cu-seed-mono-SASE-middlemen-eii
gen mid-eii-mixsat Cu-seed-SASE-middlemen-eii    Cu-seed-mono-SASE-middlemen-eii \
    L3_sublevel_mixing_satellite_fs_inv=2.0
gen mid-eii-mix    Cu-seed-SASE-middlemen-eii    Cu-seed-mono-SASE-middlemen-eii \
    L3_sublevel_mixing_fs_inv=2.0 L3_sublevel_mixing_satellite_fs_inv=2.0

n_sase=$(( $(wc -l < "$SASE_OUT/manifest.txt") ))
n_mono=$(( $(wc -l < "$MONO_OUT/manifest.txt") ))
echo
echo "SASE: $n_sase configs -> $SASE_OUT/manifest.txt (one array task each)"
echo "mono: $n_mono configs -> $MONO_OUT/manifest.txt ($MONO_CONFIGS_PER_TASK per array task)"
echo
echo "submit with:"
echo "  sbatch --array=0-$(( n_sase - 1 )) scripts/submit_pathway_sweep_sase.sh"
echo "  sbatch --array=0-$(( (n_mono + MONO_CONFIGS_PER_TASK - 1) / MONO_CONFIGS_PER_TASK - 1 )) scripts/submit_pathway_sweep_mono.sh"
