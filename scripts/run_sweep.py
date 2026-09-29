#!/usr/bin/env python3
"""
Run shots [rep-start, rep-end) of one config on a pool of processes and save their accumulated outputs.

Shot `rep` uses SASE random seed `rep`, so any split of a sweep point's shots over array tasks gives the same
shots. The outputs go to <data-path>/runs_seed_<E>_uJ[__energy_<E>_eV]/: a copy of the config, a provenance
file (package location and git commit), and one .npz of summed outputs per call
(xraymb_sim.analysis.accumulate_run_outputs), which analysis.data_from_folder combines.

    python scripts/run_sweep.py --yaml config/generated/final_sweep_mono/5-electrons/20.00uJ_8048.00eV.yaml \\
        --rep-start 0 --rep-end 10 --nproc 5 --data-path data/final_sweep_mono_test/5-electrons
"""

import os

# one BLAS thread per worker process; must be set before numpy is imported
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

# import this checkout's package, not whichever copy happens to be installed
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO_ROOT)

from xraymb_sim import Simulation  # noqa: E402
from xraymb_sim.analysis import compute_run_outputs  # noqa: E402
from xraymb_sim.sweep import format_duration, peak_memory_gb, provenance, run_sweep_chunk  # noqa: E402


def run_shot(yaml_path, rep):
    t0 = time.perf_counter()
    X = Simulation(yaml_path)
    X.random_seed = rep
    X.keep_z_history = False
    X.configure()
    X.run()
    out = compute_run_outputs(X)
    print(f"repetition {rep + 1} done ({format_duration(time.perf_counter() - t0)}, "
          f"worker peak memory {peak_memory_gb():.2f} GB)", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yaml", required=True, help="config of one sweep point")
    parser.add_argument("--rep-start", type=int, default=0, help="first shot (inclusive)")
    parser.add_argument("--rep-end", type=int, required=True, help="last shot (exclusive)")
    parser.add_argument("--nproc", type=int, default=None, help="worker processes (default: the cores available)")
    parser.add_argument("--data-path", required=True, help="output folder of this variant")
    args = parser.parse_args()

    X = Simulation(args.yaml)
    prov = provenance(REPO_ROOT)
    print(prov, end="", flush=True)
    nproc = args.nproc or len(os.sched_getaffinity(0))

    point = f"seed_{X.E_seed_uJ:.1f}_uJ"
    if X.pulse == "sase_dcm":
        point += f"__energy_{X.monochromator_target_energy_eV:.2f}_eV"
    run_path = os.path.join(args.data_path, f"runs_{point}")
    os.makedirs(run_path, exist_ok=True)
    shutil.copy2(args.yaml, run_path)

    output_stem = f"run_at_{point}__reps_{args.rep_start}-{args.rep_end}"
    with open(os.path.join(run_path, f"{output_stem}.provenance.txt"), "w") as f:
        f.write(prov)
    reps = list(range(args.rep_start, args.rep_end))
    print(f"running {len(reps)} shots of {args.yaml} on {nproc} processes -> {run_path}", flush=True)
    final_path = run_sweep_chunk(run_shot, args.yaml, reps, run_path, output_stem, nproc)
    print(f"saved {final_path}", flush=True)


if __name__ == "__main__":
    main()
