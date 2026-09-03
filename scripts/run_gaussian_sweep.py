#!/usr/bin/env python3
"""Run the fast Gaussian-pulse transmittance-vs-photon-energy sweep.

Counterpart of scripts/run_mono_sweep.py, but each (E_seed, target energy) config is a
single deterministic run (tools.Gaussian_pulse_aniso_seed -- no SASE stochastic seed, so
no repetition averaging is needed). Because of that, this is meant to run locally in the
foreground rather than through SLURM: it parallelizes across the (E_seed, target energy)
grid instead of across repetitions of one config, one process per config.

Output layout matches the SASE sweep scripts (runs_seed_<E>_uJ__energy_<target>_eV/*.npz,
with n_reps=1), so tools.data_from_folder(..., group_keys=('E_seed_uJ',
'monochromator_target_energy_eV')) reads it identically to a mono-SASE sweep's output.

Example:
    python scripts/generate_gaussian_sweep_configs.py
    python scripts/run_gaussian_sweep.py \\
        --manifest config/generated/gaussian_transmittance_vs_intensity/manifest.txt \\
        --data-path data/gaussian_sweep
"""

import os  # noqa: E402  (must come before numpy loads)

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
import concurrent.futures as cf  # noqa: E402

import numpy as np  # noqa: E402

from XLO_sim.XLO_sim import XLO_sim  # noqa: E402
from XLO_sim import tools  # noqa: E402

TPAD = 1000
YPAD = 64


def run_one(yaml_path, data_path):
    t0 = time.perf_counter()

    X = XLO_sim(yaml_path)
    X.keep_z_history = False
    seed_field = tools.Gaussian_pulse_aniso_seed(X)
    X.configure(seed_field)
    X.run_3D()

    out = tools.compute_run_outputs(X, TPAD, YPAD)
    acc = tools.accumulate_run_outputs([out])

    target_energy_eV = X.monochromator_target_energy_eV
    run_path = os.path.join(data_path, f"runs_seed_{X.E_seed_uJ:.1f}_uJ__energy_{target_energy_eV:.2f}_eV")
    os.makedirs(run_path, exist_ok=True)
    shutil.copy2(yaml_path, run_path)

    out_stem = f"run_at_seed_{X.E_seed_uJ:.1f}_uJ__energy_{target_energy_eV:.2f}_eV__reps_0-1"
    out_path = os.path.join(run_path, out_stem + ".npz")
    np.savez(out_path, **acc)

    print(f"{os.path.basename(yaml_path)} done ({tools.format_duration(time.perf_counter() - t0)})", flush=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", help="Path to a manifest.txt of YAML config paths (one per line)")
    group.add_argument("--yaml", help="Path to a single generated config YAML (runs just that one point)")
    parser.add_argument("--data-path", required=True, help="Top-level output directory")
    parser.add_argument("--nproc", type=int, default=None,
                         help="Worker processes for --manifest mode (default: cores available to this job)")
    args = parser.parse_args()

    os.makedirs(args.data_path, exist_ok=True)

    if args.yaml:
        run_one(args.yaml, args.data_path)
        return

    with open(args.manifest) as f:
        yaml_paths = [line.strip() for line in f if line.strip()]

    nproc = args.nproc or len(os.sched_getaffinity(0))
    print(f"Running {len(yaml_paths)} configs on {nproc} processes -> {args.data_path}", flush=True)

    with cf.ProcessPoolExecutor(max_workers=nproc) as ex:
        futures = {ex.submit(run_one, p, args.data_path): p for p in yaml_paths}
        n_done = 0
        for fut in cf.as_completed(futures):
            fut.result()  # re-raise any worker exception here
            n_done += 1
            print(f"[{n_done}/{len(yaml_paths)}] complete", flush=True)


if __name__ == "__main__":
    main()
