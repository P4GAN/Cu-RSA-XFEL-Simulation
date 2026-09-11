#!/usr/bin/env python3
"""Run a chunk of SASE repetitions for one E_seed config and save accumulated (sum/sumsq) results.

This is the batch-job counterpart of the multiprocessing.Pool loop in
notebooks/transmittance-vs-intensity.ipynb (cell 7): same run_simulation/SF_spectrum_w/
fft_field_t_y_to_w_thy logic, generalized to take a config path and a [rep_start, rep_end)
range from the command line so it can be driven by a SLURM array task instead
of one JupyterHub kernel looping over all E_seed values sequentially.

Output layout: one run_at_seed_..._reps_<start>-<end>.npz per array-task
chunk (not per repetition -- see tools.accumulate_run_outputs), all in the
same runs_seed_<E>_uJ/ folder regardless of which array task produced them.
tools.data_from_folder() combines chunks' sum/sumsq/n_reps losslessly, so it
doesn't matter how NREP was split across array tasks.

Example:
    python scripts/run_intensity_sweep.py \\
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
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

# Always run THIS checkout's XLO_sim. `python scripts/<this>.py` puts scripts/ (not the repo root)
# first on sys.path, so without this the import falls through to whatever XLO_sim is
# pip-installed -- possibly a stale non-editable copy or another clone. That silently ran the full
# Maxwell-Bloch model in three sweeps that requested linear_resonant_response/use_rate_equations
# (jobs 24533426, 24535882, 24541257), since XLO_sim setattrs unknown config keys without complaint.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

import XLO_sim as XLO_sim_pkg  # noqa: E402
from XLO_sim import Model  # noqa: E402
from XLO_sim.XLO_sim import XLO_sim  # noqa: E402
from XLO_sim import tools  # noqa: E402

TPAD = 1000
YPAD = 64


def verify_code(X):
    """Fail fast (before any repetition) unless the imported XLO_sim is this checkout's and, when
    the config asks for rate equations, the compiled kernel actually honours the flag. Returns a
    provenance string to log and save next to the outputs."""
    pkg_dir = os.path.dirname(os.path.realpath(XLO_sim_pkg.__file__))
    if pkg_dir != os.path.join(REPO_ROOT, "XLO_sim"):
        sys.exit(f"imported XLO_sim from {pkg_dir}, not {REPO_ROOT}/XLO_sim -- refusing to run")

    use_re = bool(X.config.get("use_rate_equations", False))
    if use_re:
        if not hasattr(Model, "physical_rho"):
            sys.exit(f"{Model.__file__} has no rate-equation support but the config sets use_rate_equations")
        # Functional check through the real call path: evaluate the base block's RHS on a test state
        # with an L3-K coherence (pair (0,4) is Kalpha-coupled) with the flag on and off; the two must
        # differ, or the kernel that will run tonight is not the rate-equation one.
        n = X.nlevel
        rho = np.zeros((n, n, 1, 1), dtype=complex)
        rho[0, 0] = 0.5
        rho[0, 4] = rho[4, 0] = 0.1
        Omega = np.ones((2, 2, 1, 1), dtype=complex)
        params = [X, Omega, np.zeros((1, 1), dtype=complex), np.zeros((1, 1), dtype=complex),
                  np.zeros((1, 1)), np.zeros((1, 1))]
        d_re = Model.MB_nlevel_regular(0.0, rho, params)
        X.use_rate_equations = False
        try:
            d_mb = Model.MB_nlevel_regular(0.0, rho, params)
        finally:
            X.use_rate_equations = True
        if np.allclose(d_re, d_mb):
            sys.exit("use_rate_equations is set but the kernel gives the Maxwell-Bloch RHS -- refusing to run")

    def git(*cmd):
        try:
            return subprocess.run(["git", "-C", REPO_ROOT, *cmd], capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception as e:
            return f"unavailable ({e})"

    dirty = git("status", "--porcelain", "--", "XLO_sim")
    return (f"XLO_sim: {pkg_dir}\n"
            f"git commit: {git('rev-parse', 'HEAD')}{' (XLO_sim has uncommitted changes)' if dirty else ''}\n"
            f"use_rate_equations: {use_re} (kernel check {'passed' if use_re else 'n/a'})\n")


def run_simulation(yaml_path, rep):
    t0 = time.perf_counter()

    X = XLO_sim(yaml_path)
    X.random_seed = rep
    # This is a batch/statistics job: compute_run_outputs only ever reads the
    # z=0/z=-1 planes, so skip storing the full z history (tens of GB/worker
    # at production grid sizes -- see Sample._evaluate_n_level_3D_lean).
    X.keep_z_history = False
    seed_field = tools.Ocelot_SASE_seed_pstxy(X)
    X.configure(seed_field)
    X.run_3D()

    out = tools.compute_run_outputs(X, TPAD, YPAD)
    print(f"repetition {rep + 1} done ({tools.format_duration(time.perf_counter() - t0)}, "
          f"worker peak mem {tools.peak_memory_gb():.2f} GB)", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", required=True, help="Path to the generated config YAML for this E_seed value")
    parser.add_argument("--rep-start", type=int, default=0, help="First repetition index (inclusive)")
    parser.add_argument("--rep-end", type=int, required=True, help="Last repetition index (exclusive)")
    parser.add_argument("--nproc", type=int, default=None,
                         help="Worker processes (default: cores actually available to this job)")
    parser.add_argument("--data-path", default=None,
                         help="Top-level output directory (shared across array tasks for the same E_seed)")
    parser.add_argument("--check-only", action="store_true",
                        help="Run the code/flag checks (verify_code) and exit without simulating")
    args = parser.parse_args()

    X = XLO_sim(args.yaml)
    provenance = verify_code(X)
    print(provenance, end="", flush=True)
    if args.check_only:
        print("check passed", flush=True)
        return
    if args.data_path is None:
        parser.error("--data-path is required unless --check-only")

    nproc = args.nproc or len(os.sched_getaffinity(0))

    run_path = os.path.join(args.data_path, f"runs_seed_{X.E_seed_uJ:.1f}_uJ")
    os.makedirs(run_path, exist_ok=True)
    shutil.copy2(args.yaml, run_path)

    reps = list(range(args.rep_start, args.rep_end))
    print(f"Running {len(reps)} repetitions ({args.rep_start}-{args.rep_end}) for {args.yaml} "
          f"on {nproc} processes -> {run_path}", flush=True)

    output_stem = f"run_at_seed_{X.E_seed_uJ:.1f}_uJ__reps_{args.rep_start}-{args.rep_end}"
    with open(os.path.join(run_path, f"{output_stem}.provenance.txt"), "w") as f:
        f.write(provenance)
    final_path = tools.run_sweep_chunk(run_simulation, args.yaml, reps, run_path, output_stem, nproc)
    print(f"Saved {final_path}", flush=True)


if __name__ == "__main__":
    main()
