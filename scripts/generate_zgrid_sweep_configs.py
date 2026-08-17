#!/usr/bin/env python3
"""Generate one YAML config per zgrid value for the zgrid numerical-convergence sweep.

Checks that a regular SASE pulse at E_seed_uJ=40 (the config/base/Cu-seed-SASE.yaml
default) has converged with respect to the propagation (z) grid. zgrid is
varied while tgrid=800 and xgrid=ygrid=8 are held fixed (not the base
config's tgrid=3200 -- these are the "everything else" defaults used across
all three grid sweeps, see generate_tgrid_sweep_configs.py /
generate_xygrid_sweep_configs.py) and the physical domain (tmax, xmax, ymax,
zmax) is left untouched, so a larger zgrid means finer z resolution over the
same window.

Mirrors scripts/generate_intensity_sweep_configs.py: writes into a dedicated
subfolder of config/generated/ and records every generated path, in order,
in one manifest.txt.

Run this once, locally or on a login node, before submitting the array job:

    python scripts/generate_zgrid_sweep_configs.py
"""

import argparse
import os

import yaml

DEFAULT_ZGRID_VALUES = [5, 10, 15, 20, 30, 40, 50]

# Grid axes held fixed while zgrid is swept (shared across all three sweeps).
FIXED_TGRID = 800
FIXED_XYGRID = 8

# Must match CHUNKS_PER_CONFIG in submit_zgrid_sweep.sh -- used below only to
# print the matching sbatch --array bound.
CHUNKS_PER_CONFIG = 2


def yaml_modify_grid(input_yaml_path, output_yaml_path, zgrid):
    with open(input_yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    yaml_data["tgrid"] = FIXED_TGRID
    yaml_data["xgrid"] = FIXED_XYGRID
    yaml_data["ygrid"] = FIXED_XYGRID
    yaml_data["zgrid"] = zgrid

    with open(output_yaml_path, "w") as f:
        yaml.safe_dump(yaml_data, f)

    return output_yaml_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-yaml", default="config/base/Cu-seed-SASE.yaml")
    parser.add_argument("--out-dir", default="config/generated/convergence_zgrid")
    parser.add_argument("--zgrid", type=int, nargs="+", default=DEFAULT_ZGRID_VALUES,
                         help="zgrid values to sweep over")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.txt")

    with open(manifest_path, "w") as manifest:
        for zgrid in args.zgrid:
            out_path = os.path.join(args.out_dir, f"Cu-seed-SASE_zgrid{zgrid}.yaml")
            yaml_modify_grid(args.base_yaml, out_path, zgrid)
            manifest.write(os.path.abspath(out_path) + "\n")
            print(f"wrote {out_path}")

    n = len(args.zgrid)
    total_tasks = n * CHUNKS_PER_CONFIG
    print(f"manifest: {manifest_path}  ({n} configs, tgrid={FIXED_TGRID}, xgrid=ygrid={FIXED_XYGRID} fixed)")
    print(f"\nCAUTION: cost scales with zgrid -- zgrid={max(args.zgrid)} can be substantially slower "
          f"than zgrid={min(args.zgrid)}. Time the largest config locally before submitting the full "
          f"array, and adjust --time/--mem in submit_zgrid_sweep.sh accordingly.")
    print(f"\nsubmit with:\n  sbatch --array=0-{total_tasks - 1} scripts/submit_zgrid_sweep.sh")


if __name__ == "__main__":
    main()
