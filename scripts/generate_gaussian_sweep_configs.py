#!/usr/bin/env python3
"""Generate one YAML config per (E_seed, target energy) pair for the fast Gaussian-pulse
transmittance-vs-photon-energy sweep.

Mirrors scripts/generate_mono_sweep_configs.py, but the base config
(config/base/Cu-seed-mono-gaussian.yaml) seeds a deterministic coherent Gaussian pulse
(tools.Gaussian_pulse_aniso_seed) instead of a SASE realization filtered through a
simulated Si(111) DCM -- no repetition averaging needed, so this sweep is meant to run
locally in the foreground via scripts/run_gaussian_sweep.py rather than through SLURM.

Run this once before running the sweep:

    python scripts/generate_gaussian_sweep_configs.py
"""

import argparse
import os

import yaml

DEFAULT_E_SEED_VALUES = [0.1, 1, 5, 10, 20, 30, 40, 50]
# Offsets from the Cu Kalpha1 line (eV) used to build the default absolute energy grid in
# main() below (anchored to --base-yaml's hwKalpha1N), when --energy isn't given explicitly.
DEFAULT_DENERGY_OFFSETS = [-15, -12, -10, -8, -6, -5, -4, -3, -2.5, -2, -1.5, -1, -0.5,
                           0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15]


def yaml_modify_seed_energy_and_target_energy(input_yaml_path, output_yaml_path, new_seed_energy, target_energy_eV):
    with open(input_yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    yaml_data["E_seed_uJ"] = new_seed_energy
    yaml_data["monochromator_target_energy_eV"] = float(target_energy_eV)

    with open(output_yaml_path, "w") as f:
        yaml.safe_dump(yaml_data, f)

    return output_yaml_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-yaml", default="config/base/Cu-seed-mono-gaussian.yaml")
    parser.add_argument("--out-dir", default="config/generated/gaussian_transmittance_vs_intensity")
    parser.add_argument("--e-seed", type=float, nargs="+", default=DEFAULT_E_SEED_VALUES,
                         help="E_seed_uJ values to sweep over")
    parser.add_argument("--energy", type=float, nargs="+", default=None,
                         help="Absolute monochromator_target_energy_eV values (eV) to sweep over "
                              "(default: --base-yaml's hwKalpha1N +/- 15 eV, denser near center)")
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
                    args.out_dir, f"Cu-seed-mono-gaussian_{e_seed:.2f}uJ_{target_energy_eV:.2f}eV.yaml"
                )
                yaml_modify_seed_energy_and_target_energy(args.base_yaml, out_path, e_seed, target_energy_eV)
                manifest.write(os.path.abspath(out_path) + "\n")
                print(f"wrote {out_path}")

    n = len(args.e_seed) * len(energy_values)
    print(f"manifest: {manifest_path}  ({n} configs)")
    print("\nrun with:\n  python scripts/run_gaussian_sweep.py --manifest " + manifest_path
          + " --data-path data/gaussian_sweep")


if __name__ == "__main__":
    main()
