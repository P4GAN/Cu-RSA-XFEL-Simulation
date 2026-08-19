#!/usr/bin/env python3
"""Run a chunk of SASE repetitions for one (E_seed, target energy) config and save accumulated (sum/sumsq) results.

Batch-job counterpart of notebooks/mono-transmittance-vs-intensity.ipynb (cell 7),
generalized to take a config path and a [rep_start, rep_end) range from the
command line so it can be driven by a SLURM array task. See
scripts/run_intensity_sweep.py for the broadband-SASE equivalent -- the
two share their FFT/SF_spectrum_w post-processing via XLO_sim/tools.py, but
this one seeds through the 111 DCM monochromator response
(tools.Ocelot_SASE_seed_111_dcm_pstxy) instead of the bare SASE seed. The
absolute target photon energy is baked into the config's
monochromator_target_energy_eV by generate_mono_sweep_configs.py rather than
overridden at run time.

Output layout: one run_at_seed_..._energy_..._reps_<start>-<end>.npz per
array-task chunk (not per repetition -- see tools.accumulate_run_outputs),
all in the same runs_seed_<E>_uJ__energy_<E_target>_eV/ folder regardless of
which array task produced them. tools.data_from_folder() combines chunks'
sum/sumsq/n_reps losslessly, so it doesn't matter how NREP was split across
array tasks.

Example:
    python scripts/run_mono_sweep.py \\
        --yaml config/generated/mono_transmittance_vs_intensity/Cu-seed-mono-SASE_40.00uJ_8041.91eV.yaml \\
        --rep-start 0 --rep-end 20 --nproc 40 --data-path data/mono_sweep_2026-08-12
"""

import os  # noqa: E402  (must come before numpy loads)

# Set single-thread BLAS/OMP env vars before numpy loads. Each multiprocessing
# worker below gets its own NumPy; without this, each one spins up its own
# OpenBLAS/MKL thread pool and oversubscribes the cores the Pool already
# split across processes.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import shutil  # noqa: E402

import numpy as np  # noqa: E402  (must come after the env vars above)

from XLO_sim.XLO_sim import XLO_sim  # noqa: E402
from XLO_sim import tools  # noqa: E402

TPAD = 1000
YPAD = 64


def run_simulation(yaml_path, rep):
    print(f"repetition {rep + 1}", flush=True)

    X = XLO_sim(yaml_path)
    X.random_seed = rep
    seed_field = tools.Ocelot_SASE_seed_111_dcm_pstxy(X)
    X.configure(seed_field)
    X.run_3D()

    return tools.compute_run_outputs(X, TPAD, YPAD)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", required=True, help="Path to the generated config YAML for this (E_seed, target energy) pair")
    parser.add_argument("--rep-start", type=int, default=0, help="First repetition index (inclusive)")
    parser.add_argument("--rep-end", type=int, required=True, help="Last repetition index (exclusive)")
    parser.add_argument("--nproc", type=int, default=None,
                         help="Worker processes (default: cores actually available to this job)")
    parser.add_argument("--data-path", required=True,
                         help="Top-level output directory (shared across array tasks for the same config)")
    args = parser.parse_args()

    nproc = args.nproc or len(os.sched_getaffinity(0))

    X = XLO_sim(args.yaml)
    target_energy_eV = X.monochromator_target_energy_eV
    run_path = os.path.join(args.data_path, f"runs_seed_{X.E_seed_uJ:.1f}_uJ__energy_{target_energy_eV:.2f}_eV")
    os.makedirs(run_path, exist_ok=True)
    shutil.copy2(args.yaml, run_path)

    reps = list(range(args.rep_start, args.rep_end))
    print(f"Running {len(reps)} repetitions ({args.rep_start}-{args.rep_end}) for {args.yaml} "
          f"on {nproc} processes -> {run_path}", flush=True)

    # 'fork' is Linux's default anyway, set explicitly to match the notebook's
    # behavior and keep this runnable interactively on macOS during testing.
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=nproc) as pool:
        results = pool.starmap(run_simulation, [(args.yaml, rep) for rep in reps])

    acc = tools.accumulate_run_outputs(results)
    date_string = np.datetime_as_string(np.datetime64('now'))
    np.savez_compressed(
        os.path.join(
            run_path,
            f"run_at_seed_{X.E_seed_uJ:.1f}_uJ__energy_{target_energy_eV:.2f}_eV"
            f"__reps_{args.rep_start}-{args.rep_end}_{date_string}.npz",
        ),
        **acc,
    )


if __name__ == "__main__":
    main()
