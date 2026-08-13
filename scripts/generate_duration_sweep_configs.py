#!/usr/bin/env python3
"""Generate one YAML config per seed duration for the transmittance-vs-duration sweep.

Mirrors scripts/generate_intensity_sweep_configs.py, but modifies seed_duration_FWHM_t
(and widens tmax to fit it, matching notebooks/transmittance-vs-duration.ipynb
cell 4) instead of E_seed_uJ. Writes into its own subfolder + manifest.txt so
array-index -> config lookup doesn't depend on filename globbing.

Run this once, on Maxwell, before submitting the array job:

    python scripts/generate_duration_sweep_configs.py
"""

import argparse
import os

import yaml

# Cu 2p core-hole lifetime is ~1 fs (Ka1 linewidth ~2.5 eV FWHM -> tau ~ hbar/Gamma
# ~0.3-1 fs), so this scan spans seed durations from well below (0.1 fs) to well
# above (6 fs) that timescale -- see notebooks/transmittance-vs-duration.ipynb cell 5.
DEFAULT_DURATION_VALUES = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 4, 6, 10]


def yaml_modify_seed_duration(input_yaml_path, output_yaml_path, new_seed_duration):
    with open(input_yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    yaml_data["seed_duration_FWHM_t"] = new_seed_duration

    with open(output_yaml_path, "w") as f:
        yaml.safe_dump(yaml_data, f)

    return output_yaml_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-yaml", default="config/base/Cu-seed-SASE.yaml")
    parser.add_argument("--out-dir", default="config/generated/transmittance_vs_duration")
    parser.add_argument("--duration", type=float, nargs="+", default=DEFAULT_DURATION_VALUES,
                         help="seed_duration_FWHM_t values (fs) to sweep over")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.txt")

    with open(manifest_path, "w") as manifest:
        for duration in args.duration:
            out_path = os.path.join(args.out_dir, f"Cu-seed-SASE_{duration:.2f}fs.yaml")
            yaml_modify_seed_duration(args.base_yaml, out_path, duration)
            manifest.write(os.path.abspath(out_path) + "\n")
            print(f"wrote {out_path}")

    print(f"manifest: {manifest_path}  ({len(args.duration)} configs)")


if __name__ == "__main__":
    main()
