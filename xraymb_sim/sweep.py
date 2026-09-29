"""Running many shots of one config in parallel on a cluster node (scripts/run_sweep.py)."""

import multiprocessing as mp
import os
import resource
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .analysis import accumulate_run_outputs


def format_duration(seconds):
    """e.g. '1h 02m 05.3s', '2m 05.3s' or '45.3s'."""
    hours, rem = divmod(float(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{int(hours)}h {int(minutes):02d}m {secs:04.1f}s"
    if minutes:
        return f"{int(minutes)}m {secs:04.1f}s"
    return f"{secs:.1f}s"


def peak_memory_gb(who=resource.RUSAGE_SELF):
    """Peak resident memory of this process (or of its finished children) in GiB."""
    ru_maxrss = resource.getrusage(who).ru_maxrss
    kb = ru_maxrss / 1024 if sys.platform == "darwin" else ru_maxrss   # bytes on macOS, KiB on Linux
    return kb / (1024 ** 2)


def provenance(repo_root):
    """
    The package location and git commit this run uses, to save next to its outputs. Refuses to run if the
    imported package is not the one in repo_root: a stale installed copy would otherwise run silently.
    """
    pkg_dir = os.path.dirname(os.path.realpath(__file__))
    if pkg_dir != os.path.join(os.path.realpath(repo_root), "xraymb_sim"):
        sys.exit(f"imported xraymb_sim from {pkg_dir}, not {repo_root}/xraymb_sim -- refusing to run")

    def git(*cmd):
        try:
            return subprocess.run(["git", "-C", repo_root, *cmd], capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception as e:
            return f"unavailable ({e})"

    dirty = git("status", "--porcelain", "--", "xraymb_sim")
    return (f"xraymb_sim: {pkg_dir}\n"
            f"git commit: {git('rev-parse', 'HEAD')}{' (xraymb_sim has uncommitted changes)' if dirty else ''}\n")


def run_sweep_chunk(run_simulation, yaml_path, reps, run_path, output_stem, nproc, checkpoint_every=None):
    """
    Run run_simulation(yaml_path, rep) for every rep on a pool of nproc processes and save the accumulated
    outputs to '<run_path>/<output_stem>_<timestamp>.npz'. Returns that path.

    Built for batch jobs that can lose a worker (e.g. killed for running out of memory): the process pool
    then raises instead of hanging until the wall-clock limit, and every checkpoint_every finished shots
    (default nproc) the shots so far are saved to '<output_stem>.partial.npz', which analysis.data_from_folder
    picks up if the job never finishes. The partial file is removed once the final file is written.
    """
    checkpoint_every = checkpoint_every or max(1, nproc)
    partial_path = os.path.join(run_path, f"{output_stem}.partial.npz")

    t0 = time.perf_counter()
    results = []
    executor = ProcessPoolExecutor(max_workers=nproc, mp_context=mp.get_context("fork"))
    futures = {executor.submit(run_simulation, yaml_path, rep): rep for rep in reps}
    try:
        for i, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if i % checkpoint_every == 0 and i < len(reps):
                elapsed = time.perf_counter() - t0
                np.savez_compressed(partial_path, **accumulate_run_outputs(results))
                print(f"checkpoint: {len(results)}/{len(reps)} repetitions done ({format_duration(elapsed)} elapsed, "
                      f"{format_duration(elapsed / len(results))}/rep) -> {partial_path}", flush=True)
    except Exception as e:
        print(f"{type(e).__name__} after {len(results)}/{len(reps)} repetitions "
              f"({format_duration(time.perf_counter() - t0)}) -- a worker died or raised:", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        if results:
            np.savez_compressed(partial_path, **accumulate_run_outputs(results))
            print(f"saved {len(results)} finished repetitions to {partial_path}", flush=True)
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown(wait=True)

    elapsed = time.perf_counter() - t0
    print(f"chunk finished: {len(results)} repetitions in {format_duration(elapsed)} "
          f"({format_duration(elapsed / max(1, len(results)))}/rep, {nproc} processes, "
          f"peak worker memory {peak_memory_gb(resource.RUSAGE_CHILDREN):.2f} GB)", flush=True)

    final_path = os.path.join(run_path, f"{output_stem}_{np.datetime_as_string(np.datetime64('now'))}.npz")
    np.savez_compressed(final_path, **accumulate_run_outputs(results))
    if os.path.exists(partial_path):
        os.remove(partial_path)
    return final_path
