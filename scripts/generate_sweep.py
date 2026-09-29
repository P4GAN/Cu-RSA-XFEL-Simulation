#!/usr/bin/env python3
"""
Write one config per sweep point for one variant of a sweep family, and add them to the family's manifest.

A family is a folder (e.g. config/generated/final_sweep_mono/) with one sub-folder of configs per variant
and a manifest.txt of '<variant> <config path>' lines, which scripts/submit_sweep.sh works through. The
sweep points are every pulse energy (--e-seed) and, for a self-seeded (pulse: sase_dcm) config, every
monochromator photon energy (--energy).

    python scripts/generate_sweep.py --family config/generated/final_sweep_mono --variant 5-electrons \\
        --base config/mono/5-electrons.yaml --e-seed 1 5 20 30 --energy 8000 8048 --set zgrid=30

The generate_*_sweep.sh scripts call this once per variant and print the sbatch command.
"""

import argparse
import os

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", required=True, help="family folder, e.g. config/generated/final_sweep_mono")
    parser.add_argument("--variant", required=True, help="variant name (a sub-folder of the family)")
    parser.add_argument("--base", required=True, help="base config")
    parser.add_argument("--e-seed", type=float, nargs="+", required=True, help="pulse energies (uJ)")
    parser.add_argument("--energy", type=float, nargs="+", default=None,
                        help="monochromator photon energies (eV), for a sase_dcm config")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
                        help="config overrides for this variant, VALUE parsed as YAML (e.g. xgrid=7)")
    parser.add_argument("--reset", action="store_true", help="start a new manifest instead of adding to it")
    args = parser.parse_args()

    with open(args.base) as f:
        base = yaml.safe_load(f)
    for pair in args.overrides:
        key, sep, value = pair.partition("=")
        if not sep or key not in base:
            raise SystemExit(f"--set {pair!r}: expected KEY=VALUE with KEY a key of {args.base}")
        base[key] = yaml.safe_load(value)

    mono = base["pulse"] == "sase_dcm"
    if mono != (args.energy is not None):
        raise SystemExit("--energy is required for a sase_dcm config and not allowed for any other pulse")

    out_dir = os.path.join(args.family, args.variant)
    os.makedirs(out_dir, exist_ok=True)
    manifest = os.path.join(args.family, "manifest.txt")
    lines = []
    for e_seed in args.e_seed:
        for energy in (args.energy if mono else [None]):
            cfg = dict(base, E_seed_uJ=e_seed)
            name = f"{e_seed:.2f}uJ"
            if mono:
                cfg["monochromator_target_energy_eV"] = energy
                name += f"_{energy:.2f}eV"
            path = os.path.join(out_dir, name + ".yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            lines.append(f"{args.variant} {os.path.abspath(path)}\n")

    with open(manifest, "w" if args.reset else "a") as f:
        f.writelines(lines)
    with open(manifest) as f:
        total = sum(1 for _ in f)
    print(f"{args.variant}: {len(lines)} configs in {out_dir}; {manifest} now lists {total}")


if __name__ == "__main__":
    main()
