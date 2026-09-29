#!/usr/bin/env python3
"""Run one shot and stream its full (t, x, y, z) history to HDF5, for the movies and 3D renders
(scripts/render_movie.py, render_movie_3d.py, render_level_diagram.py).

The recorder (scripts/movie.py, which describes the file layout) keeps a single-precision reduction of every
grid point plus the full density matrices at the centre pixel. Planes are written as they finish, so a killed
run still leaves every completed z plane readable.

    python scripts/run_movie.py --check-only   # size estimate only, no simulation
    python scripts/run_movie.py --out data/movie/movie.h5
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

import movie  # noqa: E402
from xraymb_sim import Simulation  # noqa: E402
from xraymb_sim.sweep import format_duration, peak_memory_gb, provenance  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yaml", default=os.path.join(REPO_ROOT, "config/movie.yaml"))
    parser.add_argument("--out", default=None, help="Output .h5 path (the config and provenance are copied next to it)")
    parser.add_argument("--seed", type=int, default=None, help="SASE random seed (default: the config's random_seed)")
    parser.add_argument("--max-gb", type=float, default=20.0, help="Refuse to start if the uncompressed estimate exceeds this")
    parser.add_argument("--check-only", action="store_true", help="Print the size estimate, then exit")
    args = parser.parse_args()

    X = Simulation(args.yaml)
    prov = provenance(REPO_ROOT)
    print(prov, end="", flush=True)
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
        f.write(prov)

    t0 = time.perf_counter()
    with open(args.yaml) as f:
        config_text = f.read()
    attrs = {"config_yaml": config_text, "config_path": os.path.abspath(args.yaml), "provenance": prov,
             "random_seed": -1 if X.random_seed is None else X.random_seed, "E_seed_uJ": X.E_seed_uJ}
    with movie.MovieRecorder(X, args.out, attrs=attrs) as recorder:
        X.recorder = recorder
        X.configure()
        X.run()
        # pulse energy in and out, as a sanity check in the log
        dA = X.dx * X.dy * X.dt
        E_in = recorder.f["flux"][0].sum() * dA
        E_out = np.sum(np.real(recorder.f["field_exit"][0] * recorder.f["field_exit"][1])) / X.flux_factor * dA
        recorder.f.attrs["energy_transmission"] = E_out / E_in
        recorder.f.attrs["runtime_s"] = time.perf_counter() - t0

    print(f"Saved {args.out}: {os.path.getsize(args.out) / 1e9:.2f} GB on disk, energy transmission "
          f"{E_out / E_in:.3f}, {format_duration(time.perf_counter() - t0)}, "
          f"peak memory {peak_memory_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
