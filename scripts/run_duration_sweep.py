#!/usr/bin/env python3
"""Run a chunk of SASE repetitions for one seed-duration config and save per-repetition results.

Batch-job counterpart of notebooks/transmittance-vs-duration.ipynb (cell 7),
generalized to take a config path and a [rep_start, rep_end) range from the
command line so it can be driven by a SLURM array task. See
scripts/run_transmittance_sweep.py for the E_seed-sweep equivalent -- the two
share their FFT/SF_spectrum_w post-processing via XLO_sim/tools.py.

Output layout matches the notebook's expectations: one
run_at_duration_..._repetition_N.npz per repetition, all in the same
runs_duration_<fs>_fs/ folder regardless of which array task produced them,
so the existing data_from_folder() aggregation in the notebook works
unchanged.

Example:
    python scripts/run_duration_sweep.py \\
        --yaml config/generated/transmittance_vs_duration/Cu-seed-SASE_2.00fs.yaml \\
        --rep-start 0 --rep-end 100 --nproc 40 --data-path data/duration_sweep_2026-08-07
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

    P_pstxyz = np.array([ np.einsum('ijs,jitxyz->stxyz', X.Tijs_minus, X.rho_ijtxyz) , np.einsum('ijtxyz,jis->stxyz', X.rho_ijtxyz, X.Tijs_plus) ])
    
    I_t_0 = np.einsum('stxy,stxy->t', X.Omega_pstxyz[0, :, :, :, :, 0], X.Omega_pstxyz[1, :, :, :, :, 0])
    I_t_last = np.einsum('stxy,stxy->t', X.Omega_pstxyz[0, :, :, :, :, -1], X.Omega_pstxyz[1, :, :, :, :, -1])
    rho_ee_t_last = np.einsum('ijs, jkt, kis-> t', X.Tijs_minus, X.rho_ijtxyz[:,:,:,int(X.xgrid/2),int(X.ygrid/2),-1], X.Tijs_plus)
    rho_gg_t_last = np.einsum('ijs, jkt, kis-> t', X.Tijs_plus, X.rho_ijtxyz[:,:,:,int(X.xgrid/2),int(X.ygrid/2),-1], X.Tijs_minus)
    rho_eg_t_last = P_pstxyz[0,0,:,int(X.xgrid/2),int(X.ygrid/2),-1]
    rho_ground_t_last = X.rho_0_3D[:,int(X.xgrid/2),int(X.ygrid/2),-1]
    t_axis = X.t

    date_string = np.datetime_as_string(np.datetime64('now'))
    np.savez_compressed(
        os.path.join(run_path, f"run_at_duration_{X.seed_duration_FWHM_t:.2f}_fs__repetition_{rep + 1}_{date_string}.npz"),
        womega_ar=womega_ar,
        I_int_thy_w_0=I_int_thy_w_0,
        I_int_thy_w_last=I_int_thy_w_last,
        I_t_0=I_t_0,
        I_t_last=I_t_last,
        rho_ee_t_last=rho_ee_t_last,
        rho_gg_t_last=rho_gg_t_last,
        rho_eg_t_last=rho_eg_t_last,
        rho_ground_t_last=rho_ground_t_last,
        t_axis=t_axis
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", required=True, help="Path to the generated config YAML for this duration value")
    parser.add_argument("--rep-start", type=int, default=0, help="First repetition index (inclusive)")
    parser.add_argument("--rep-end", type=int, required=True, help="Last repetition index (exclusive)")
    parser.add_argument("--nproc", type=int, default=None,
                         help="Worker processes (default: cores actually available to this job)")
    parser.add_argument("--data-path", required=True,
                         help="Top-level output directory (shared across array tasks for the same duration)")
    args = parser.parse_args()

    nproc = args.nproc or len(os.sched_getaffinity(0))

    X = XLO_sim(args.yaml)
    run_path = os.path.join(args.data_path, f"runs_duration_{X.seed_duration_FWHM_t:.2f}_fs")
    os.makedirs(run_path, exist_ok=True)
    shutil.copy2(args.yaml, run_path)

    reps = list(range(args.rep_start, args.rep_end))
    print(f"Running {len(reps)} repetitions ({args.rep_start}-{args.rep_end}) for {args.yaml} "
          f"on {nproc} processes -> {run_path}", flush=True)

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=nproc) as pool:
        pool.starmap(run_simulation, [(args.yaml, run_path, rep) for rep in reps])


if __name__ == "__main__":
    main()
