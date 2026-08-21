#!/usr/bin/env python3
"""Generate one YAML config per E_seed value for the transmittance-vs-intensity sweep.

Writes into a dedicated subfolder of config/generated/ (rather than the shared
top-level folder) and records the paths in a manifest.txt, in order. The SLURM
array script indexes into that manifest directly, so array task <-> config
mapping is explicit and doesn't depend on filename globbing (config/generated/
also holds files from other sweeps, e.g. seed duration, whose names can collide
with a glob pattern).

Run this once, locally or on a login node, before submitting the array job:

    python scripts/generate_intensity_sweep_configs.py
"""

import argparse
import os

import yaml

DEFAULT_E_SEED_VALUES = [40, 30, 20, 9, 5, 2, 1, 0.1]

# Must match CHUNKS_PER_CONFIG in submit_intensity_sweep.sh -- used below only
# to print the matching sbatch --array bound.
CHUNKS_PER_CONFIG = 10


def yaml_modify_seed_energy(input_yaml_path, output_yaml_path, new_seed_energy):
    with open(input_yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    yaml_data["E_seed_uJ"] = new_seed_energy

    with open(output_yaml_path, "w") as f:
        yaml.safe_dump(yaml_data, f)

    return output_yaml_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-yaml", default="config/base/Cu-seed-SASE.yaml")
    parser.add_argument("--out-dir", default="config/generated/transmittance_vs_intensity")
    parser.add_argument("--e-seed", type=float, nargs="+", default=DEFAULT_E_SEED_VALUES,
                         help="E_seed_uJ values to sweep over")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.txt")

    with open(manifest_path, "w") as manifest:
        for e_seed in args.e_seed:
            out_path = os.path.join(args.out_dir, f"Cu-seed-SASE_{e_seed:.2f}uJ.yaml")
            yaml_modify_seed_energy(args.base_yaml, out_path, e_seed)
            manifest.write(os.path.abspath(out_path) + "\n")
            print(f"wrote {out_path}")

    n = len(args.e_seed)
    total_tasks = n * CHUNKS_PER_CONFIG
    print(f"manifest: {manifest_path}  ({n} configs)")
    print(f"\nsubmit with:\n  sbatch --array=0-{total_tasks - 1} scripts/submit_intensity_sweep.sh")


if __name__ == "__main__":
    main()
