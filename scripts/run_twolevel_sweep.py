#!/usr/bin/env python3
"""Two-level (g, l, u + auxiliary x) Maxwell-Bloch runs for the analytic-vs-numerical RSA study.

Runs XLO_sim/twolevel.py for one beam (mono = transform-limited photon-energy scan, sase =
spectrally resolved chaotic pulses) and one mode (mb = Maxwell-Bloch, re = rate equations) over a
list of pulse energies, and writes one <beam>_<mode>_<E>uJ.npz per energy into --data-path.
Every run reads out all the foil thicknesses in grid.z_record_um plus the entrance plane
(L -> 0), so a single run covers thin and thick foils. Analyse with
scripts/plot_twolevel_vs_analytic.py.

The kernel self-checks (closed-form steady states, photon bookkeeping; a few seconds) run
before every sweep, so a stale or broken checkout fails fast instead of producing data.
--check-only additionally runs tiny end-to-end normalisation checks and exits.

Examples:
    python scripts/run_twolevel_sweep.py --beam mono --mode mb --check-only
    python scripts/run_twolevel_sweep.py --beam sase --mode re --data-path data/twolevel_test \\
        --E-uJ 1 10 --set sase.n_shots=40 sase.n_blocks=4
"""

import os  # noqa: E402  (thread settings must precede numpy/numba imports)

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
# numba's own thread pool does the parallel work; workqueue ignores the OMP_* settings above.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)   # never fall through to a stale pip-installed XLO_sim

import numba  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from XLO_sim import twolevel as tl  # noqa: E402


def available_cores():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "config/base/Cu-2level-analytic.yaml"))
    parser.add_argument("--beam", choices=tl.BEAMS, required=True)
    parser.add_argument("--mode", choices=tl.MODES, required=True)
    parser.add_argument("--E-uJ", type=float, nargs="+", default=None,
                        help="pulse energies (default: the config's pulse_energies_uJ)")
    parser.add_argument("--chunk", type=int, default=0, help="run energies[chunk::nchunks]")
    parser.add_argument("--nchunks", type=int, default=1)
    parser.add_argument("--data-path", default=None, help="output directory (required unless --check-only)")
    parser.add_argument("--nthreads", type=int, default=None, help="numba threads (default: available cores)")
    parser.add_argument("--set", nargs="*", default=[], metavar="section.key=value",
                        help="config overrides, e.g. sase.n_shots=100 grid.zmax_um=2")
    parser.add_argument("--check-only", action="store_true", help="run the self-checks and exit")
    parser.add_argument("--overwrite", action="store_true", help="recompute energies whose output exists")
    args = parser.parse_args()

    nthreads = args.nthreads or available_cores()
    numba.set_num_threads(min(nthreads, numba.config.NUMBA_NUM_THREADS))
    cfg = tl.load_config(args.config, args.set)
    print(f"{args.beam} {args.mode}: config {args.config} {' '.join(args.set)}; "
          f"{numba.get_num_threads()} numba threads ({os.environ['NUMBA_THREADING_LAYER']})", flush=True)

    t0 = time.perf_counter()
    tl.self_check(cfg, log=lambda s: print(s, flush=True))
    if args.check_only:
        tl.pipeline_check(cfg, log=lambda s: print(s, flush=True))
        print(f"checks passed ({time.perf_counter() - t0:.1f} s)", flush=True)
        return
    print(f"self-check passed ({time.perf_counter() - t0:.1f} s)", flush=True)
    if args.data_path is None:
        parser.error("--data-path is required unless --check-only")

    energies = list(args.E_uJ if args.E_uJ is not None else cfg["pulse_energies_uJ"])
    energies = energies[args.chunk::args.nchunks]
    os.makedirs(args.data_path, exist_ok=True)
    prov = tl.provenance(REPO_ROOT)
    print(f"code {prov['module_path']} sha1 {prov['code_sha1'][:12]}, git {prov['git_head'][:12]}"
          f"{' (dirty)' if prov['git_dirty'] else ''}; energies {energies}", flush=True)

    run = tl.run_mono if args.beam == "mono" else tl.run_sase
    for E_uJ in energies:
        out_path = os.path.join(args.data_path, f"{args.beam}_{args.mode}_{E_uJ:g}uJ.npz")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"{out_path} exists, skipping", flush=True)
            continue
        t1 = time.perf_counter()
        res = run(cfg, E_uJ, args.mode, log=lambda s: print(s, flush=True))
        res.update(config_yaml=yaml.safe_dump(cfg, sort_keys=False), overrides=" ".join(args.set),
                   argv=" ".join(sys.argv), wall_s=time.perf_counter() - t1, **prov)
        tmp_path = out_path[:-4] + ".partial.npz"
        np.savez_compressed(tmp_path, **res)
        os.replace(tmp_path, out_path)
        print(f"saved {out_path} ({tl_format(time.perf_counter() - t1)})", flush=True)
    print(f"done ({tl_format(time.perf_counter() - t0)})", flush=True)


def tl_format(seconds):
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:04.1f}s" if minutes else f"{secs:.1f}s"


if __name__ == "__main__":
    main()
