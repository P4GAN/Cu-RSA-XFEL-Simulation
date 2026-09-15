#!/usr/bin/env python3
"""Run ONE SASE shot and stream its full (t, x, y, z) history to HDF5, for movies and 3D renders.

Same physics as the sweeps (Sample._evaluate_n_level_3D_lean), with an XLO_sim/movie.py
MovieRecorder attached. The recorder keeps a single-precision reduction of every grid point
(field, populations by manifold, base-block Kalpha coherences, free electrons) plus the full density
matrices at the centre pixel. See the XLO_sim/movie.py docstring for the file layout. Planes are
written as they finish, so a killed run still leaves every completed z-plane readable
(`planes_written` attribute).

Example:
    python scripts/run_movie.py --check-only   # size estimate + code/flag checks, no simulation
    python scripts/run_movie.py --out data/movie/movie.h5
"""

import os  # noqa: E402  (must come before numpy loads)

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

# Always run THIS checkout's XLO_sim (see run_intensity_sweep.py).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from XLO_sim.XLO_sim import XLO_sim  # noqa: E402
from XLO_sim import tools, movie  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yaml", default=os.path.join(REPO_ROOT, "config/base/Cu-seed-SASE-movie.yaml"))
    parser.add_argument("--out", default=None, help="Output .h5 path (the config and provenance are copied next to it)")
    parser.add_argument("--seed", type=int, default=None, help="SASE random seed (default: the config's random_seed)")
    parser.add_argument("--max-gb", type=float, default=20.0, help="Refuse to start if the uncompressed estimate exceeds this")
    parser.add_argument("--check-only", action="store_true", help="Print the size estimate and run the code checks, then exit")
    args = parser.parse_args()

    X = XLO_sim(args.yaml)
    provenance = tools.verify_code(X, REPO_ROOT)
    print(provenance, end="", flush=True)
    if args.seed is not None:
        X.random_seed = args.seed
    X.keep_z_history = False

    size_gb = movie.estimate_size_gb(X)
    print(f"grid t x y z = {X.tgrid} x {X.xgrid} x {X.ygrid} x {X.zgrid}, "
          f"{len(X.satellite_channel_params)} satellite blocks, seed {X.random_seed}: "
          f"{size_gb:.1f} GB uncompressed (limit {args.max_gb:g} GB)", flush=True)
    if size_gb > args.max_gb:
        sys.exit("estimated output exceeds --max-gb -- shrink the grid")
    if args.check_only:
        print("check passed", flush=True)
        return
    if args.out is None:
        parser.error("--out is required unless --check-only")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(args.yaml, out_dir)
    stem = os.path.splitext(args.out)[0]
    with open(f"{stem}.provenance.txt", "w") as f:
        f.write(provenance)

    t0 = time.perf_counter()
    seed_field = getattr(tools, X.seed_pulse_format)(X)
    with open(args.yaml) as f:
        config_text = f.read()
    attrs = {"config_yaml": config_text, "config_path": os.path.abspath(args.yaml), "provenance": provenance,
             "random_seed": -1 if X.random_seed is None else X.random_seed, "E_seed_uJ": X.E_seed_uJ}
    with movie.MovieRecorder(X, args.out, attrs=attrs) as recorder:
        X.movie_recorder = recorder
        X.configure(seed_field)
        X.run_3D()
        # Pulse energy in and out, as a sanity line in the log.
        dA = X.dx * X.dy * X.dt
        E_in = recorder.f["flux"][0].sum() * dA
        E_out = np.sum(np.real(recorder.f["field_exit"][0] * recorder.f["field_exit"][1])) / X.flux_factor * dA
        recorder.f.attrs["energy_transmission"] = E_out / E_in
        recorder.f.attrs["runtime_s"] = time.perf_counter() - t0

    print(f"Saved {args.out}: {os.path.getsize(args.out) / 1e9:.2f} GB on disk, energy transmission "
          f"{E_out / E_in:.3f}, {tools.format_duration(time.perf_counter() - t0)}, "
          f"peak mem {tools.peak_memory_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
