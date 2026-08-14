#!/usr/bin/env python3
"""Generate one YAML config per (E_seed, target energy) pair for the mono-transmittance-vs-intensity sweep.

Mirrors scripts/generate_intensity_sweep_configs.py, but the sweep has two physical
parameters instead of one: E_seed_uJ and the monochromator's absolute target
photon energy (monochromator_target_energy_eV). Rather than overriding
monochromator_target_energy_eV as a Python attribute at run time, this
flattens the E_seed x energy grid and writes one fully self-contained config
per pair (matching notebooks/mono-transmittance-vs-intensity.ipynb cells
4/5), so the SLURM array manifest below is still just a flat list of YAML
paths, same as the 1D sweeps.

Run this once, locally or on a login node, before submitting the array job:

    python scripts/generate_mono_sweep_configs.py
"""

import argparse
import os

import yaml

DEFAULT_E_SEED_VALUES = [200, 150, 100, 70, 40, 20, 10, 5, 1, 0.1]
# Offsets from the Cu Kalpha1 line (eV) used to build the default absolute
# energy grid in main() below (anchored to --base-yaml's hwKalpha1N), when
# --energy isn't given explicitly.
DEFAULT_DENERGY_OFFSETS = [-15, -12, -9, -6, -5, -4, -3, -2, -1, -0.5,
                           0, 0.5, 1, 2, 3, 4, 5, 6, 9, 12, 15]

# Must match CHUNKS_PER_CONFIG / ARRAY_THROTTLE in submit_mono_sweep.sh --
# used below only to print the matching sbatch --array bound.
CHUNKS_PER_CONFIG = 2
ARRAY_THROTTLE = 50


def yaml_modify_seed_energy_and_target_energy(input_yaml_path, output_yaml_path, new_seed_energy, target_energy_eV):
    with open(input_yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    yaml_data["E_seed_uJ"] = new_seed_energy
    # yaml.safe_dump can't represent numpy scalars (e.g. if target_energy_eV
    # came from hwKalpha1N + a numpy array of offsets), so cast explicitly.
    yaml_data["monochromator_target_energy_eV"] = float(target_energy_eV)

    with open(output_yaml_path, "w") as f:
        yaml.safe_dump(yaml_data, f)

    return output_yaml_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-yaml", default="config/base/Cu-seed-mono-SASE.yaml")
    parser.add_argument("--out-dir", default="config/generated/mono_transmittance_vs_intensity")
    parser.add_argument("--e-seed", type=float, nargs="+", default=DEFAULT_E_SEED_VALUES,
                         help="E_seed_uJ values to sweep over")
    parser.add_argument("--energy", type=float, nargs="+", default=None,
                         help="Absolute monochromator_target_energy_eV values (eV) to sweep over "
                              "(default: --base-yaml's hwKalpha1N +/- 15 eV in 3 eV steps)")
    args = parser.parse_args()

    with open(args.base_yaml, "r") as f:
        hwKalpha1N = yaml.safe_load(f)["hwKalpha1N"]

    energy_values = args.energy if args.energy is not None else [hwKalpha1N + d for d in DEFAULT_DENERGY_OFFSETS]

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.txt")

    with open(manifest_path, "w") as manifest:
        for e_seed in args.e_seed:
            for target_energy_eV in energy_values:
                out_path = os.path.join(
                    args.out_dir, f"Cu-seed-mono-SASE_{e_seed:.2f}uJ_{target_energy_eV:.2f}eV.yaml"
                )
                yaml_modify_seed_energy_and_target_energy(args.base_yaml, out_path, e_seed, target_energy_eV)
                manifest.write(os.path.abspath(out_path) + "\n")
                print(f"wrote {out_path}")

    n = len(args.e_seed) * len(energy_values)
    total_tasks = n * CHUNKS_PER_CONFIG
    print(f"manifest: {manifest_path}  ({n} configs)")
    print(f"\nsubmit with:\n  sbatch --array=0-{total_tasks - 1}%{ARRAY_THROTTLE} scripts/submit_mono_sweep.sh")


if __name__ == "__main__":
    main()
