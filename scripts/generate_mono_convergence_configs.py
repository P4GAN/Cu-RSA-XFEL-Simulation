#!/usr/bin/env python3
"""Generate one YAML config per (grid value, E_seed, target energy) for a mono-pulse
numerical-convergence check -- the mono counterpart of generate_tgrid_sweep_configs.py /
generate_xygrid_sweep_configs.py, since those vary only tgrid/xygrid and inherit the
base config's single E_seed_uJ and don't touch monochromator_target_energy_eV at all.

Only tgrid and xygrid are supported (--grid-axis): the z-grid convergence question is
not specific to the mono pulse shape (z-marching/Fresnel propagation don't care how the
seed's t/x/y profile was generated), so it doesn't need a separate mono check -- reuse
generate_zgrid_sweep_configs.py's conclusion.

Unlike generate_mono_sweep_configs.py's full transmittance-vs-intensity scan, this is
meant for a small, targeted set of energies (e.g. on-resonance + one wing point) and
E_seed values (e.g. lowest + highest of interest) -- pass them explicitly.

Run this once, locally or on a login node, before submitting the array job:

    python scripts/generate_mono_convergence_configs.py --grid-axis tgrid \\
        --grid-values 3000 6000 12000 24000 \\
        --e-seed 2 60 --energy 8045.91 8025.91
"""

import argparse
import os

import yaml

# Must match CONFIGS_PER_TASK in submit_mono_convergence_sweep.sh -- used below only to
# print the matching sbatch --array bound.
CONFIGS_PER_TASK = 4


def yaml_modify(input_yaml_path, output_yaml_path, grid_key, grid_value, e_seed, target_energy_eV):
    with open(input_yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    if grid_key == "xygrid":
        yaml_data["xgrid"] = grid_value
        yaml_data["ygrid"] = grid_value
    else:
        yaml_data[grid_key] = grid_value

    yaml_data["E_seed_uJ"] = e_seed
    yaml_data["monochromator_target_energy_eV"] = float(target_energy_eV)

    with open(output_yaml_path, "w") as f:
        yaml.safe_dump(yaml_data, f)

    return output_yaml_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-yaml", default="config/base/Cu-seed-mono-SASE-double-satellite.yaml")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--grid-axis", required=True, choices=["tgrid", "xygrid"])
    parser.add_argument("--grid-values", type=int, nargs="+", required=True)
    parser.add_argument("--e-seed", type=float, nargs="+", required=True, help="E_seed_uJ values to sweep over")
    parser.add_argument("--energy", type=float, nargs="+", required=True,
                         help="Absolute monochromator_target_energy_eV values (eV)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.txt")

    n = 0
    with open(manifest_path, "w") as manifest:
        for grid_value in args.grid_values:
            for e_seed in args.e_seed:
                for target_energy_eV in args.energy:
                    tag = f"{args.grid_axis}{grid_value}_{e_seed:.2f}uJ_{target_energy_eV:.2f}eV"
                    out_path = os.path.join(args.out_dir, f"Cu-seed-mono_{tag}.yaml")
                    yaml_modify(args.base_yaml, out_path, args.grid_axis, grid_value, e_seed, target_energy_eV)
                    manifest.write(os.path.abspath(out_path) + "\n")
                    print(f"wrote {out_path}")
                    n += 1

    total_tasks = -(-n // CONFIGS_PER_TASK)  # ceil
    print(f"\nmanifest: {manifest_path}  ({n} configs, {CONFIGS_PER_TASK} configs/array task)")
    print(f"CAUTION: cost scales steeply with {args.grid_axis} -- time the largest config locally "
          f"before submitting the full array.")
    print(f"\nsubmit with:\n  sbatch --array=0-{total_tasks - 1} scripts/submit_mono_convergence_sweep.sh {args.out_dir}")


if __name__ == "__main__":
    main()
