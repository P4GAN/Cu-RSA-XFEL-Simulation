#!/usr/bin/env python3
"""Generate one YAML config per seed spot-size scale for the transmittance-vs-spotsize sweep.

Scales the base config's seed_width_FWHM_x/y (180/220 nm) by each factor in
DEFAULT_SCALE_VALUES, together, so the elliptical spot shape is preserved
while its overall size doubles/halves. xmax/ymax scale by the same factor,
keeping the domain-padding-to-spot-size ratio constant. xgrid/ygrid scale by
the factor too, but are floored at the base config's value (8): the base
grid is already coarse (see scripts/generate_xygrid_sweep_configs.py's
convergence sweep, which treats xgrid=ygrid=2 as a deliberately
under-resolved extreme -- np.linspace(-xmax, xmax, 2) samples only the two
domain edges and misses the spot entirely), so shrinking configs keep xgrid
fixed at 8 (resolution relative to the smaller spot only improves) while
growing configs scale it up to keep pace, matching xygrid=16/32 from that
same convergence sweep at scale=2/4.

Unlike scripts/generate_xygrid_sweep_configs.py (numerical convergence check,
tgrid/zgrid pinned to cheap stand-in values), this sweep leaves tgrid/zgrid at
the base config's production values, since spot size is a genuine physical
parameter, not just a grid-refinement check.

Mirrors scripts/generate_intensity_sweep_configs.py: writes into a dedicated
subfolder of config/generated/ and records every generated path, in order,
in one manifest.txt.

Run this once, locally or on a login node, before submitting the array job:

    python scripts/generate_spotsize_sweep_configs.py
"""

import argparse
import os

import yaml

# 2 halvings, the base size, and 2 doublings.
DEFAULT_SCALE_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]

# Must match CHUNKS_PER_CONFIG in submit_spotsize_sweep.sh -- used below only
# to print the matching sbatch --array bound.
CHUNKS_PER_CONFIG = 4


def yaml_modify_spotsize(input_yaml_path, output_yaml_path, scale):
    with open(input_yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    base_fwhm_x = yaml_data["seed_width_FWHM_x"]
    base_fwhm_y = yaml_data["seed_width_FWHM_y"]
    base_xmax = yaml_data["xmax"]
    base_ymax = yaml_data["ymax"]

    yaml_data["seed_width_FWHM_x"] = base_fwhm_x * scale
    yaml_data["seed_width_FWHM_y"] = base_fwhm_y * scale
    yaml_data["xmax"] = base_xmax * scale
    yaml_data["ymax"] = base_ymax * scale
    yaml_data["xgrid"] = 64
    yaml_data["ygrid"] = 64

    with open(output_yaml_path, "w") as f:
        yaml.safe_dump(yaml_data, f)

    return output_yaml_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-yaml", default="config/base/Cu-seed-SASE.yaml")
    parser.add_argument("--out-dir", default="config/generated/transmittance_vs_spotsize")
    parser.add_argument("--scale", type=float, nargs="+", default=DEFAULT_SCALE_VALUES,
                         help="Factors to scale seed_width_FWHM_x/y (and xmax/ymax/xgrid/ygrid) by")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.txt")

    with open(manifest_path, "w") as manifest:
        for scale in args.scale:
            out_path = os.path.join(args.out_dir, f"Cu-seed-SASE_spotsize{scale:.2f}x.yaml")
            yaml_modify_spotsize(args.base_yaml, out_path, scale)
            manifest.write(os.path.abspath(out_path) + "\n")
            print(f"wrote {out_path}")

    n = len(args.scale)
    total_tasks = n * CHUNKS_PER_CONFIG
    print(f"manifest: {manifest_path}  ({n} configs)")
    print(f"\nCAUTION: cost scales steeply with xgrid*ygrid -- scale={max(args.scale)} can be orders "
          f"of magnitude slower than scale={min(args.scale)}, on top of tgrid staying at the base "
          f"config's production value throughout. Time the largest config locally before submitting "
          f"the full array, and adjust --time/--mem/NREP in submit_spotsize_sweep.sh accordingly.")
    print(f"\nsubmit with:\n  sbatch --array=0-{total_tasks - 1} scripts/submit_spotsize_sweep.sh")


if __name__ == "__main__":
    main()
