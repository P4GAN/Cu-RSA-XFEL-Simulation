#!/usr/bin/env python3
"""Generate one YAML config per (E_seed, target energy) pair for the Gaussian-pulse (or, with a DCM base config, monochromatised SASE)
transmittance-vs-photon-energy sweep.

Mirrors scripts/generate_mono_sweep_configs.py, but the base config seeds a deterministic coherent
Gaussian pulse (tools.Gaussian_pulse_aniso_seed) instead of a SASE realization filtered through a
simulated Si(111) DCM -- no repetition averaging needed. Each config is one single-process run:
on the cluster, scripts/submit_gaussian_sweep.sh runs CONFIGS_PER_TASK of them side by side per
array task (the sbatch command is printed at the end); locally, scripts/run_gaussian_sweep.py
--manifest runs them in a process pool.

Photon energies: --energy (absolute, eV), or --energy-preset mono_exp for the 28 photon energies of
the self-seeded 20 um measurement (Results_RSA_seeded.pdf slides 5-6), each + --energy-shift eV
(the measured lines sit ~1.8 eV below hwKalpha1N/hwKalpha2N; +1.8 samples the simulated lines
where the measurement sampled the measured ones). Default: hwKalpha1N +- 15 eV.

--set key=value overrides any top-level key of the base config (value parsed as YAML), e.g.
--set additional_dephasing=0 resonant_source_scale=1 for a nominal-parameter control sweep.

Examples:
    python scripts/generate_gaussian_sweep_configs.py
    python scripts/generate_gaussian_sweep_configs.py --base-yaml config/base/Cu-L2-mono-gaussian-fit.yaml \\
        --out-dir config/generated/l2fit_mono_gaussian --energy-preset mono_exp --energy-shift 1.8 \\
        --e-seed 0.001 0.1 1 5 10 20 30 40 50
"""

import argparse
import math
import os

import yaml

DEFAULT_E_SEED_VALUES = [0.1, 1, 5, 10, 20, 30, 40, 50]
# Offsets from the Cu Kalpha1 line (eV) used to build the default absolute energy grid in
# main() below (anchored to --base-yaml's hwKalpha1N), when --energy isn't given explicitly.
DEFAULT_DENERGY_OFFSETS = [-15, -12, -10, -8, -6, -5, -4, -3, -2.5, -2, -1.5, -1, -0.5,
                           0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15]
# Photon energies of the self-seeded single-shot scans (Results_RSA_seeded.pdf slides 5 and 6).
ENERGY_PRESETS = {
    "mono_exp": [8000, 8005, 8010, 8015, 8019, 8022, 8025, 8027, 8028, 8029, 8032, 8035,
                 8038, 8041, 8044, 8046, 8047, 8048, 8049, 8050, 8052, 8055, 8060, 8065, 8070,
                 8080, 8090, 8100],
}
CONFIGS_PER_TASK = 40   # must match scripts/submit_gaussian_sweep.sh


def write_config(base, output_yaml_path, new_seed_energy, target_energy_eV, overrides):
    yaml_data = dict(base)
    yaml_data.update(overrides)
    yaml_data["E_seed_uJ"] = new_seed_energy
    yaml_data["monochromator_target_energy_eV"] = float(target_energy_eV)
    with open(output_yaml_path, "w") as f:
        yaml.safe_dump(yaml_data, f)
    return output_yaml_path


def parse_overrides(items):
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        out[key] = yaml.safe_load(value)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-yaml", default="config/base/Cu-seed-mono-gaussian.yaml")
    parser.add_argument("--out-dir", default="config/generated/gaussian_transmittance_vs_intensity")
    parser.add_argument("--e-seed", type=float, nargs="+", default=DEFAULT_E_SEED_VALUES,
                        help="E_seed_uJ values to sweep over")
    parser.add_argument("--energy", type=float, nargs="+", default=None,
                        help="Absolute monochromator_target_energy_eV values (eV) to sweep over "
                             "(default: --base-yaml's hwKalpha1N +/- 15 eV, denser near center)")
    parser.add_argument("--energy-preset", choices=sorted(ENERGY_PRESETS), default=None,
                        help="named photon-energy list instead of --energy")
    parser.add_argument("--energy-shift", type=float, default=0.0, help="added to every photon energy (eV)")
    parser.add_argument("--nrep", type=int, default=10, help="repetitions per config (DCM/SASE bases only)")
    parser.add_argument("--set", nargs="+", default=None, metavar="KEY=VALUE",
                        help="override top-level keys of the base config")
    args = parser.parse_args()

    with open(args.base_yaml, "r") as f:
        base = yaml.safe_load(f)
    overrides = parse_overrides(args.set)

    if args.energy is not None:
        energy_values = list(args.energy)
    elif args.energy_preset is not None:
        energy_values = list(ENERGY_PRESETS[args.energy_preset])
    else:
        energy_values = [base["hwKalpha1N"] + d for d in DEFAULT_DENERGY_OFFSETS]
    energy_values = [round(e + args.energy_shift, 4) for e in energy_values]

    stem = os.path.splitext(os.path.basename(args.base_yaml))[0]
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.txt")

    with open(manifest_path, "w") as manifest:
        for e_seed in args.e_seed:
            for target_energy_eV in energy_values:
                out_path = os.path.join(args.out_dir, f"{stem}_{e_seed:g}uJ_{target_energy_eV:.2f}eV.yaml")
                write_config(base, out_path, e_seed, target_energy_eV, overrides)
                manifest.write(os.path.abspath(out_path) + "\n")

    n = len(args.e_seed) * len(energy_values)
    tag = os.path.basename(os.path.normpath(args.out_dir))
    print(f"wrote {n} configs ({len(args.e_seed)} pulse energies x {len(energy_values)} photon energies)"
          + (f", overrides {overrides}" if overrides else ""))
    print(f"manifest: {manifest_path}")
    if base.get("seed_pulse_format") == "Ocelot_SASE_seed_111_dcm_pstxy":
        per_task = 40 // args.nrep
        print(f"\nsubmit with ({args.nrep} reps/config, {per_task} configs/task):\n  mkdir -p logs && MANIFEST="
              + manifest_path + f" NREP={args.nrep} CONFIGS_PER_TASK={per_task} DATA_TAG={tag}"
              + f" sbatch --array=0-{math.ceil(n / per_task) - 1} scripts/submit_mono_sweep.sh")
    else:
        print("\nsubmit with:\n  mkdir -p logs && MANIFEST=" + manifest_path + f" DATA_TAG={tag}"
              + f" sbatch --array=0-{math.ceil(n / CONFIGS_PER_TASK) - 1} scripts/submit_gaussian_sweep.sh")

if __name__ == "__main__":
    main()
