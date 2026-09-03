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
import math
import os

import yaml

import numpy as np

DEFAULT_E_SEED_VALUES = [0.1, 1, 5, 20, 30] #[200, 150, 100, 70, 40, 20, 10, 5, 1, 0.1]
# Offsets from the Cu Kalpha1 line (eV) used to build the default absolute
# energy grid in main() below (anchored to --base-yaml's hwKalpha1N), when
# --energy isn't given explicitly. The fine region's 1 eV step is ~2-2.5
# samples per natural linewidth (Kalpha1: GammaKeVN+GammaL3eVN = 2.10 eV
# FWHM; Kalpha2: GammaKeVN+GammaL2eVN = 2.53 eV) -- coarser than the
# monochromator sweep's original 0.5 eV step, but still resolves the
# Kalpha1/Kalpha2 doublet shoulder before even counting the monochromator's
# own bandwidth broadening the probed feature further.
DEFAULT_DENERGY_VALUES = np.concatenate([np.arange(8000, 8015, 5), np.arange(8015, 8060, 1.0), np.arange(8060, 8100, 10)])
# [-15, -12, -9, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 9, 12, 15]

# Must match CONFIGS_PER_TASK / ARRAY_THROTTLE in submit_mono_sweep.sh --
# used below only to print the matching sbatch --array bound. Each array
# task fans out CONFIGS_PER_TASK configs as concurrent background processes
# (see submit_mono_sweep.sh), so the total task count is ceil(n_configs /
# CONFIGS_PER_TASK) rather than one task per config.
CONFIGS_PER_TASK = 8
# ARRAY_THROTTLE = 50


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

    # with open(args.base_yaml, "r") as f:
    #     hwKalpha1N = yaml.safe_load(f)["hwKalpha1N"]

    energy_values = args.energy if args.energy is not None else [d for d in DEFAULT_DENERGY_VALUES]

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
    total_tasks = math.ceil(n / CONFIGS_PER_TASK)
    print(f"manifest: {manifest_path}  ({n} configs, {CONFIGS_PER_TASK} configs/array task)")
    print(f"\nsubmit with:\n  sbatch --array=0-{total_tasks - 1} scripts/submit_mono_sweep.sh")


if __name__ == "__main__":
    main()
