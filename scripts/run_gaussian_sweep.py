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


def check_one(yaml_path):
    """Build the config and the seed without running the solver (cheap, safe locally): photon
    count, peak fluence, cold transmission from the ground-state cross sections, the fit knobs, and
    the seed's spectral centroid on the same axis as the saved spectra (hwKalpha1N + womega_ar),
    which must equal monochromator_target_energy_eV."""
    X = XLO_sim(yaml_path)
    seed = tools.Gaussian_pulse_aniso_seed(X)
    X.Omega_pstxyz = seed[..., None]
    womega, I_int, _ = tools.SF_spectrum_w(X, 0, YPAD, TPAD)
    I_w = np.real(I_int)
    centroid = X.hwKalpha1N + float(np.sum(womega * I_w) / np.sum(I_w))
    J_txy = np.real(np.einsum("stxy,stxy->txy", seed[0], seed[1])) / X.flux_factor
    F_xy = J_txy.sum(axis=0) * X.dt
    N = F_xy.sum() * X.dx * X.dy
    sig_g = X.S_ground_Fi.sum(axis=1).mean()
    T_cold = np.exp(-X.n * sig_g * X.zmax)
    print(f"{os.path.basename(yaml_path)}: nlevel {X.nlevel}, 2s {X.use_2s_pathway}, L2 {X.use_L2_pathway}, "
          f"satellites {len(X.satellite_channel_params)}, additional_dephasing {X.additional_dephasing} fs^-1, "
          f"sigma1 2p3/2p1/other {X.sigma1_Ka1_2p3:.4g}/{X.sigma1_Ka1_2p1:.4g}/{X.sigma1_Ka1_other:.4g}\n"
          f"  photons {N:.4e} (E/hw {X.E_seed_uJ * 1e-6 / (X.hwKalpha1N * 1.602176634e-19):.4e}), "
          f"peak fluence {F_xy.max():.4e} /nm^2, sigma_g {sig_g:.4e} nm^2 -> T_cold(L) {T_cold:.4f}\n"
          f"  target {X.monochromator_target_energy_eV:.2f} eV, seed spectral centroid {centroid:.2f} eV, "
          f"dt {X.dt:.4f} fs, dx {X.dx:.1f} nm", flush=True)
    return abs(centroid - X.monochromator_target_energy_eV)


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
    run_path = os.path.join(data_path, f"runs_seed_{X.E_seed_uJ:g}_uJ__energy_{target_energy_eV:.2f}_eV")
    os.makedirs(run_path, exist_ok=True)
    shutil.copy2(yaml_path, run_path)

    out_stem = f"run_at_seed_{X.E_seed_uJ:g}_uJ__energy_{target_energy_eV:.2f}_eV__reps_0-1"
    out_path = os.path.join(run_path, out_stem + ".npz")
    np.savez(out_path, **acc)

    print(f"{os.path.basename(yaml_path)} done ({tools.format_duration(time.perf_counter() - t0)})", flush=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", help="Path to a manifest.txt of YAML config paths (one per line)")
    group.add_argument("--yaml", help="Path to a single generated config YAML (runs just that one point)")
    parser.add_argument("--data-path", default=None, help="Top-level output directory (required unless --check-only)")
    parser.add_argument("--check-only", action="store_true",
                         help="build config + seed and report them, no simulation (for --manifest: first, middle, last)")
    parser.add_argument("--nproc", type=int, default=None,
                         help="Worker processes for --manifest mode (default: cores available to this job)")
    args = parser.parse_args()

    if args.check_only:
        if args.yaml:
            paths = [args.yaml]
        else:
            with open(args.manifest) as f:
                all_paths = [line.strip() for line in f if line.strip()]
            paths = [all_paths[0], all_paths[len(all_paths) // 2], all_paths[-1]]
        worst = max(check_one(p) for p in paths)
        print(f"max |seed centroid - target| = {worst:.3f} eV")
        return
    if args.data_path is None:
        parser.error("--data-path is required unless --check-only")
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
