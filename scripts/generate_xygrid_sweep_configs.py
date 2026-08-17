#!/usr/bin/env python3
"""Generate one YAML config per xgrid=ygrid value for the transverse-grid convergence sweep.

Checks that a regular SASE pulse at E_seed_uJ=40 (the config/base/Cu-seed-SASE.yaml
default) has converged with respect to the transverse grid. xgrid and ygrid
are varied together (kept square) while tgrid=800 and zgrid=15 are held
fixed (not the base config's tgrid=3200 -- these are the "everything else"
defaults used across all three grid sweeps, see generate_tgrid_sweep_configs.py
/ generate_zgrid_sweep_configs.py) and the physical domain (tmax, xmax, ymax,
zmax) is left untouched, so a larger xgrid/ygrid means finer transverse
resolution over the same window.

Mirrors scripts/generate_intensity_sweep_configs.py: writes into a dedicated
subfolder of config/generated/ and records every generated path, in order,
in one manifest.txt.

Run this once, locally or on a login node, before submitting the array job:

    python scripts/generate_xygrid_sweep_configs.py
"""

import argparse
import os

import yaml

DEFAULT_XYGRID_VALUES = [2, 4, 8, 16, 32]

# Grid axes held fixed while xgrid/ygrid are swept (shared across all three sweeps).
FIXED_TGRID = 800
FIXED_ZGRID = 15

# Must match CHUNKS_PER_CONFIG in submit_xygrid_sweep.sh -- used below only
# to print the matching sbatch --array bound.
CHUNKS_PER_CONFIG = 2


def yaml_modify_grid(input_yaml_path, output_yaml_path, xygrid):
    with open(input_yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    yaml_data["tgrid"] = FIXED_TGRID
    yaml_data["xgrid"] = xygrid
    yaml_data["ygrid"] = xygrid
    yaml_data["zgrid"] = FIXED_ZGRID

    with open(output_yaml_path, "w") as f:
        yaml.safe_dump(yaml_data, f)

    return output_yaml_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-yaml", default="config/base/Cu-seed-SASE.yaml")
    parser.add_argument("--out-dir", default="config/generated/convergence_xygrid")
    parser.add_argument("--xygrid", type=int, nargs="+", default=DEFAULT_XYGRID_VALUES,
                         help="xgrid=ygrid values to sweep over, together")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.txt")

    with open(manifest_path, "w") as manifest:
        for xygrid in args.xygrid:
            out_path = os.path.join(args.out_dir, f"Cu-seed-SASE_xygrid{xygrid}.yaml")
            yaml_modify_grid(args.base_yaml, out_path, xygrid)
            manifest.write(os.path.abspath(out_path) + "\n")
            print(f"wrote {out_path}")

    n = len(args.xygrid)
    total_tasks = n * CHUNKS_PER_CONFIG
    print(f"manifest: {manifest_path}  ({n} configs, tgrid={FIXED_TGRID}, zgrid={FIXED_ZGRID} fixed)")
    print(f"\nCAUTION: cost scales steeply with xgrid/ygrid -- xygrid={max(args.xygrid)} can be orders "
          f"of magnitude slower than xygrid={min(args.xygrid)}. Time the largest config locally before "
          f"submitting the full array, and adjust --time/--mem in submit_xygrid_sweep.sh accordingly.")
    print(f"\nsubmit with:\n  sbatch --array=0-{total_tasks - 1} scripts/submit_xygrid_sweep.sh")


if __name__ == "__main__":
    main()
