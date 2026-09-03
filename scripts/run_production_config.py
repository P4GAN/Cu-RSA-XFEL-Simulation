#!/usr/bin/env python3
"""Run a chunk of noise repetitions for ONE standalone config and save accumulated (sum/sumsq) results.

Generic counterpart of run_intensity_sweep.py/run_mono_sweep.py for a *fixed set of distinct
config/base/*.yaml files* (different physics models at the same nominal seed energy) rather than a
parameter sweep over one base config. The two existing scripts name their output folder after the
physical parameter being swept (runs_seed_<E>_uJ[__energy_<E>_eV]) -- fine when that parameter
differs across configs, but our 7 target configs all share E_seed_uJ: 40, so that scheme would
collide every one of them into the same folder. This script instead keys the output folder off the
config file's own name (its stem), which is unique per config by construction.

Seed-field dispatch is generic (getattr(tools, X.seed_pulse_format)(X)) instead of hardcoding
Ocelot_SASE_seed_pstxy vs Ocelot_SASE_seed_111_dcm_pstxy -- config/base/*.yaml's seed_pulse_format
string is always exactly the tools.py function name (verified against tools.py's def list), so this
one script runs any of the SASE or monochromator configs unmodified.

Example:
    python scripts/run_production_config.py \\
        --yaml config/base/Cu-seed-SASE-double-satellite.yaml \\
        --rep-start 0 --rep-end 200 --nproc 40 --data-path data/production_sase_2026-09-03
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
import time  # noqa: E402
from pathlib import Path  # noqa: E402

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
    seed_field = getattr(tools, X.seed_pulse_format)(X)
    X.configure(seed_field)
    X.run_3D()

    out = tools.compute_run_outputs(X, TPAD, YPAD)
    print(f"repetition {rep + 1} done ({tools.format_duration(time.perf_counter() - t0)}, "
          f"worker peak mem {tools.peak_memory_gb():.2f} GB)", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", required=True, help="Path to a config/base/*.yaml (used as-is, not a generated variant)")
    parser.add_argument("--rep-start", type=int, default=0, help="First repetition index (inclusive)")
    parser.add_argument("--rep-end", type=int, required=True, help="Last repetition index (exclusive)")
    parser.add_argument("--nproc", type=int, default=None,
                         help="Worker processes (default: cores actually available to this job)")
    parser.add_argument("--data-path", required=True,
                         help="Top-level output directory (shared across array tasks for this submission)")
    args = parser.parse_args()

    nproc = args.nproc or len(os.sched_getaffinity(0))

    X = XLO_sim(args.yaml)
    config_name = Path(args.yaml).stem
    run_path = os.path.join(args.data_path, config_name)
    os.makedirs(run_path, exist_ok=True)
    shutil.copy2(args.yaml, run_path)

    reps = list(range(args.rep_start, args.rep_end))
    print(f"Running {len(reps)} repetitions ({args.rep_start}-{args.rep_end}) for {args.yaml} "
          f"on {nproc} processes -> {run_path}", flush=True)

    output_stem = f"run_{config_name}__reps_{args.rep_start}-{args.rep_end}"
    final_path = tools.run_sweep_chunk(run_simulation, args.yaml, reps, run_path, output_stem, nproc)
    print(f"Saved {final_path}", flush=True)


if __name__ == "__main__":
    main()
