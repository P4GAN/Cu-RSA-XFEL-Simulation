#!/usr/bin/env python3
"""Run a chunk of SASE repetitions for one E_seed config and save per-repetition results.

This is the batch-job counterpart of the multiprocessing.Pool loop in
notebooks/transmittance-vs-intensity.ipynb (cell 7): same run_simulation/SF_spectrum_w/
fft_field_t_y_to_w_thy logic, generalized to take a config path and a [rep_start, rep_end)
range from the command line so it can be driven by a SLURM array task instead
of one JupyterHub kernel looping over all E_seed values sequentially.

Output layout matches the notebook's expectations: one run_at_seed_..._repetition_N.npz
per repetition, all in the same runs_seed_<E>_uJ/ folder regardless of which
array task produced them, so the existing data_from_folder() aggregation in the
notebook works unchanged.

Example:
    python scripts/run_transmittance_sweep.py \\
        --yaml config/generated/transmittance_vs_intensity/Cu-seed-SASE_30.00uJ.yaml \\
        --rep-start 0 --rep-end 300 --nproc 40 --data-path data/sweep_2026-08-07
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


def run_simulation(yaml_path, run_path, rep):
    print(f"repetition {rep + 1}", flush=True)

    X = XLO_sim(yaml_path)
    X.random_seed = rep
    seed_field = tools.Ocelot_SASE_seed_pstxy(X)
    X.configure(seed_field)
    X.run_3D()

    womega_ar, I_int_thy_w_0, I_thy0_w_0 = tools.SF_spectrum_w(X, 0, YPAD, TPAD)
    womega_ar, I_int_thy_w_last, I_thy0_w_last = tools.SF_spectrum_w(X, -1, YPAD, TPAD)

    date_string = np.datetime_as_string(np.datetime64('now'))
    np.savez_compressed(
        os.path.join(run_path, f"run_at_seed_{X.E_seed_uJ:.1f}_uJ__repetition_{rep + 1}_{date_string}.npz"),
        womega_ar=womega_ar,
        I_int_thy_w_0=I_int_thy_w_0,
        I_thy0_w_0=I_thy0_w_0,
        I_int_thy_w_last=I_int_thy_w_last,
        I_thy0_w_last=I_thy0_w_last,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", required=True, help="Path to the generated config YAML for this E_seed value")
    parser.add_argument("--rep-start", type=int, default=0, help="First repetition index (inclusive)")
    parser.add_argument("--rep-end", type=int, required=True, help="Last repetition index (exclusive)")
    parser.add_argument("--nproc", type=int, default=None,
                         help="Worker processes (default: cores actually available to this job)")
    parser.add_argument("--data-path", required=True,
                         help="Top-level output directory (shared across array tasks for the same E_seed)")
    args = parser.parse_args()

    nproc = args.nproc or len(os.sched_getaffinity(0))

    X = XLO_sim(args.yaml)
    run_path = os.path.join(args.data_path, f"runs_seed_{X.E_seed_uJ:.1f}_uJ")
    os.makedirs(run_path, exist_ok=True)
    shutil.copy2(args.yaml, run_path)

    reps = list(range(args.rep_start, args.rep_end))
    print(f"Running {len(reps)} repetitions ({args.rep_start}-{args.rep_end}) for {args.yaml} "
          f"on {nproc} processes -> {run_path}", flush=True)

    # 'fork' is Linux's default anyway, set explicitly to match the notebook's
    # behavior and keep this runnable interactively on macOS during testing.
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=nproc) as pool:
        pool.starmap(run_simulation, [(args.yaml, run_path, rep) for rep in reps])


if __name__ == "__main__":
    main()
