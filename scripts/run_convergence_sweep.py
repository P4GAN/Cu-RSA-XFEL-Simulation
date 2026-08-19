#!/usr/bin/env python3
"""Run a chunk of SASE repetitions for one grid-convergence config and save accumulated (sum/sumsq) results.

Batch-job counterpart of scripts/run_intensity_sweep.py, shared by all three
numerical-convergence sweeps (scripts/generate_tgrid_sweep_configs.py,
generate_xygrid_sweep_configs.py, generate_zgrid_sweep_configs.py):
E_seed_uJ is fixed at the base config's value (40 uJ) throughout, and what
varies between configs is tgrid, xgrid/ygrid, or zgrid instead. Same
run_simulation/SF_spectrum_w post-processing as the intensity sweep, so
outputs from both are directly comparable.

Repetition index is used as X.random_seed (as in the other sweep scripts),
so the same rep across different grid settings starts from the same noise
draw index -- useful for eyeballing whether a given SASE shot's spectrum/
population traces are stable as the grid is refined, on top of the
aggregate statistics.

Output layout: one run_at_tgridT_xgridX_ygridY_zgridZ__reps_<start>-<end>.npz
per array-task chunk (not per repetition -- see tools.accumulate_run_outputs),
all in the same runs_tgridT_xgridX_ygridY_zgridZ/ folder regardless of which
array task produced them. tools.data_from_folder() combines chunks'
sum/sumsq/n_reps losslessly, so it doesn't matter how NREP was split across
array tasks.

Note: memory and runtime both grow steeply with grid size (tgrid up to
20000, xgrid=ygrid up to 32, zgrid up to 50 in the default sweep) --
consider a smaller --nproc than the intensity sweep at the largest grid
settings so worker memory usage doesn't exceed the node.

Example:
    python scripts/run_convergence_sweep.py \\
        --yaml config/generated/convergence_tgrid/Cu-seed-SASE_tgrid6400.yaml \\
        --rep-start 0 --rep-end 20 --nproc 40 --data-path data/convergence_tgrid_2026-08-17
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


def grid_tag(X):
    return f"tgrid{X.tgrid}_xgrid{X.xgrid}_ygrid{X.ygrid}_zgrid{X.zgrid}"


def run_simulation(yaml_path, rep):
    print(f"repetition {rep + 1}", flush=True)

    X = XLO_sim(yaml_path)
    X.random_seed = rep
    seed_field = tools.Ocelot_SASE_seed_pstxy(X)
    X.configure(seed_field)
    X.run_3D()

    return tools.compute_run_outputs(X, TPAD, YPAD)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", required=True, help="Path to the generated config YAML for this grid setting")
    parser.add_argument("--rep-start", type=int, default=0, help="First repetition index (inclusive)")
    parser.add_argument("--rep-end", type=int, required=True, help="Last repetition index (exclusive)")
    parser.add_argument("--nproc", type=int, default=None,
                         help="Worker processes (default: cores actually available to this job)")
    parser.add_argument("--data-path", required=True,
                         help="Top-level output directory (shared across array tasks for the same grid setting)")
    args = parser.parse_args()

    nproc = args.nproc or len(os.sched_getaffinity(0))

    X = XLO_sim(args.yaml)
    run_path = os.path.join(args.data_path, f"runs_{grid_tag(X)}")
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
        os.path.join(run_path, f"run_at_{grid_tag(X)}__reps_{args.rep_start}-{args.rep_end}_{date_string}.npz"),
        **acc,
    )


if __name__ == "__main__":
    main()
