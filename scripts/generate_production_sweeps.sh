#!/bin/bash
# Generate per-config E_seed (and, for mono, energy) sweep manifests for each of the 7
# 2026-09-03-XATOM-recompute production configs, reusing the existing
# generate_intensity_sweep_configs.py / generate_mono_sweep_configs.py generators unmodified
# (both already accept --base-yaml/--out-dir) -- just pointed at each config with its own
# --out-dir, so the 5 SASE-family and 2 mono-family sweeps' manifests/generated YAMLs never
# collide with each other or with the pre-existing Cu-seed-SASE.yaml/Cu-seed-mono-SASE.yaml sweeps
# (which still live at the top level of config/generated/transmittance_vs_intensity/ and
# config/generated/mono_transmittance_vs_intensity/ respectively -- these write into per-config
# subdirectories instead).
#
# Run this once (locally or on a login node) before submitting
# submit_production_sweep_sase.sh / submit_production_sweep_mono.sh.

set -euo pipefail
cd "$(dirname "$0")/.."

SASE_CONFIGS=(
    Cu-seed-SASE-no-2s
    Cu-seed-SASE
    Cu-seed-SASE-satellite-no-L2
    Cu-seed-SASE-satellite
    Cu-seed-SASE-double-satellite
)

MONO_CONFIGS=(
    Cu-seed-mono-SASE-original
    Cu-seed-mono-SASE-double-satellite
)

for name in "${SASE_CONFIGS[@]}"; do
    echo "=== $name ==="
    python scripts/generate_intensity_sweep_configs.py \
        --base-yaml "config/base/${name}.yaml" \
        --out-dir "config/generated/transmittance_vs_intensity/${name}"
done

for name in "${MONO_CONFIGS[@]}"; do
    echo "=== $name ==="
    python scripts/generate_mono_sweep_configs.py \
        --base-yaml "config/base/${name}.yaml" \
        --out-dir "config/generated/mono_transmittance_vs_intensity/${name}"
done
