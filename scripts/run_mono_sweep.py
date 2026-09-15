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
invocation (not per repetition -- see tools.accumulate_run_outputs), in a
runs_seed_<E>_uJ__energy_<E_target>_eV/ folder. tools.data_from_folder()
combines any chunk files found there losslessly, so it doesn't matter how
many times this was invoked for the same config.

submit_mono_sweep.sh runs 8 of these concurrently per SLURM array task (one
per config, --nproc 5 each) rather than one config using the whole node --
NREP=5 is too few repetitions to keep a 40-core node busy on its own.

Example:
    python scripts/run_mono_sweep.py \\
        --yaml config/generated/mono_transmittance_vs_intensity/Cu-seed-mono-SASE_40.00uJ_8041.91eV.yaml \\
        --rep-start 0 --rep-end 5 --nproc 5 --data-path data/mono_sweep_2026-08-12
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
import sys  # noqa: E402
import time  # noqa: E402

# Always run THIS checkout's XLO_sim (see scripts/run_intensity_sweep.py): `python scripts/<this>.py`
# puts scripts/ first on sys.path, so the import could otherwise fall through to a stale pip install.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO_ROOT)

from XLO_sim.XLO_sim import XLO_sim  # noqa: E402
from XLO_sim import tools  # noqa: E402

TPAD = 1000
YPAD = 64


def run_simulation(yaml_path, rep):
    t0 = time.perf_counter()

    X = XLO_sim(yaml_path)
    X.random_seed = rep
    # This is a batch/statistics job: compute_run_outputs only ever reads the
    # z=0/z=-1 planes, so skip storing the full z history (tens of GB/worker
    # at production grid sizes -- see Sample._evaluate_n_level_3D_lean).
    X.keep_z_history = False
    seed_field = tools.Ocelot_SASE_seed_111_dcm_pstxy(X)
    X.configure(seed_field)
    X.run_3D()

    out = tools.compute_run_outputs(X, TPAD, YPAD)
    print(f"repetition {rep + 1} done ({tools.format_duration(time.perf_counter() - t0)}, "
          f"worker peak mem {tools.peak_memory_gb():.2f} GB)", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", required=True, help="Path to the generated config YAML for this (E_seed, target energy) pair")
    parser.add_argument("--rep-start", type=int, default=0, help="First repetition index (inclusive)")
    parser.add_argument("--rep-end", type=int, required=True, help="Last repetition index (exclusive)")
    parser.add_argument("--nproc", type=int, default=None,
                         help="Worker processes (default: cores actually available to this job)")
    parser.add_argument("--data-path", default=None,
                         help="Top-level output directory (shared across array tasks for the same config)")
    parser.add_argument("--check-only", action="store_true",
                        help="Run the code/flag checks (tools.verify_code) and exit without simulating")
    args = parser.parse_args()

    X = XLO_sim(args.yaml)
    provenance = tools.verify_code(X, REPO_ROOT)
    print(provenance, end="", flush=True)
    if args.check_only:
        print("check passed", flush=True)
        return
    if args.data_path is None:
        parser.error("--data-path is required unless --check-only")

    nproc = args.nproc or len(os.sched_getaffinity(0))

    target_energy_eV = X.monochromator_target_energy_eV
    run_path = os.path.join(args.data_path, f"runs_seed_{X.E_seed_uJ:.1f}_uJ__energy_{target_energy_eV:.2f}_eV")
    os.makedirs(run_path, exist_ok=True)
    shutil.copy2(args.yaml, run_path)

    reps = list(range(args.rep_start, args.rep_end))
    print(f"Running {len(reps)} repetitions ({args.rep_start}-{args.rep_end}) for {args.yaml} "
          f"on {nproc} processes -> {run_path}", flush=True)

    output_stem = (f"run_at_seed_{X.E_seed_uJ:.1f}_uJ__energy_{target_energy_eV:.2f}_eV"
                    f"__reps_{args.rep_start}-{args.rep_end}")
    with open(os.path.join(run_path, f"{output_stem}.provenance.txt"), "w") as f:
        f.write(provenance)
    final_path = tools.run_sweep_chunk(run_simulation, args.yaml, reps, run_path, output_stem, nproc)
    print(f"Saved {final_path}", flush=True)


if __name__ == "__main__":
    main()
