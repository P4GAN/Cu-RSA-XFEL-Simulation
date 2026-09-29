"""Run a batch of shots of one (sweep-generated) config and save where every atom and electron is, at every
z plane, vs time: ground, "other", 2s holes, the middleman pool, the base block and each satellite block
per manifold, and each free-electron energy group. Recorder and channel layout: XLO_sim/population_budget.py.

Unlike scripts/run_population_record.py (per-shot traces of the original L3/K model, centre pixel only,
for the analytic comparison) this runner is for the extended models: it records every extension's
populations, averages over shots (mean and standard deviation), and keeps both the centre pixel and the
incident-fluence-weighted beam average. It also saves the shot-summed transmitted spectra, so each run
carries its own transmittance.

Output .npz (plus the config and provenance next to it):
  names (q,)                           channel names
  centre_mean, centre_std (z, q, ts)   centre pixel; std over shots
  beam_mean, beam_std     (z, q, ts)   incident-fluence-weighted (x, y) average
  t (ts,), z (z,), dz, depth_nm (z,)   sampled times (fs); plane positions; depth = max(iz-1, 0)*dz
  weights_xy (x, y)                    beam weights (shot mean)
  trace_centre_max_dev                 max |sum of populations - 1| over (z, t), worst shot (0 without middlemen means nothing)
  satellite_channel_names, E_edges_eV, E_centres_eV, eii_rate_table (rows 2p3/2, 2p1/2, 2s, M; fs^-1 per
    electron per atom, x spatial_factor, M row x M_shell_scale), eii_rates_by_subshell (per eii_subshells,
    unscaled n sigma v), eii_secondary_matrix, eii_birth_names/_groups, eii_fixed_energy
  womega_ar, I_int_thy_w_0, I_int_thy_w_last (shot sums), T_integrated (ratio of the sums; the mono T)
  n_shots, reps, stride, E_seed_uJ, target_energy_eV, seed_pulse_format, config_yaml, provenance

Example (scripts/generate_population_budget.sh builds the configs and prints the cluster commands):
    python scripts/run_population_budget.py --yaml config/generated/population_budget_mono/R/Cu-seed-mono-SASE_20.00uJ_8048.00eV.yaml \\
        --rep-start 0 --rep-end 10 --out data/population_budget/R/mono_20uJ_8048eV.npz
"""

import os  # noqa: E402  (must come before numpy loads)

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO_ROOT)

from XLO_sim.XLO_sim import XLO_sim  # noqa: E402
from XLO_sim import tools  # noqa: E402
from XLO_sim.population_budget import PopulationBudgetRecorder, incident_fluence_weights  # noqa: E402

TPAD = 1000  # as run_intensity_sweep.py / run_mono_sweep.py
YPAD = 64
SAMPLE_FS = 0.04  # default time resolution of the record


def default_stride(X):
    return max(1, int(round(SAMPLE_FS / X.dt)))


def run_shot(args):
    yaml_path, rep, stride = args
    t0 = time.perf_counter()
    X = XLO_sim(yaml_path)
    X.random_seed = rep
    X.keep_z_history = False
    seed_field = getattr(tools, X.seed_pulse_format)(X)
    weights = incident_fluence_weights(X, seed_field)
    rec = PopulationBudgetRecorder(X, weights, stride)
    X.movie_recorder = rec
    X.configure(seed_field)
    X.run_3D()
    out = tools.compute_run_outputs(X, TPAD, YPAD)
    dev = float(np.max(np.abs(rec.trace('centre') - 1.0))) if X.use_middlemen else 0.0
    print(f"shot {rep} done ({tools.format_duration(time.perf_counter() - t0)}, worker peak mem "
          f"{tools.peak_memory_gb():.2f} GB, trace deviation {dev:.1e})", flush=True)
    return (rec.centre.astype(np.float32), rec.beam.astype(np.float32), weights, dev,
            np.real(out["I_int_thy_w_0"]), np.real(out["I_int_thy_w_last"]), out["womega_ar"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yaml", required=True, help="Config (e.g. one written by a sweep generator)")
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-end", type=int, default=10)
    parser.add_argument("--nproc", type=int, default=None, help="Worker processes (default: cores available)")
    parser.add_argument("--stride", type=int, default=None,
                        help=f"Record every N-th time step (default: about {SAMPLE_FS} fs)")
    parser.add_argument("--out", default=None, help="Output .npz path")
    parser.add_argument("--check-only", action="store_true", help="Load the config, run the checks, exit")
    args = parser.parse_args()

    X = XLO_sim(args.yaml)
    provenance = tools.verify_code(X, REPO_ROOT)
    print(provenance, end="", flush=True)
    stride = args.stride or default_stride(X)
    rec = PopulationBudgetRecorder(X, np.ones((X.xgrid, X.ygrid)) / (X.xgrid * X.ygrid), stride)
    print(f"{args.yaml}: grid t x y z = {X.tgrid} x {X.xgrid} x {X.ygrid} x {X.zgrid}, E_seed {X.E_seed_uJ} uJ, "
          f"{len(X.satellite_channel_params)} satellite blocks, middlemen {bool(X.use_middlemen)}, "
          f"EII {bool(X.use_eii)}; recording {len(rec.names)} channels every {stride} steps "
          f"({stride * X.dt:.3f} fs, {rec.centre.size * 16 / 1e6:.0f} MB of accumulators per worker)", flush=True)
    if args.check_only:
        print("check passed", flush=True)
        return
    if args.out is None:
        parser.error("--out is required unless --check-only")

    stem = os.path.splitext(os.path.abspath(args.out))[0]
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    shutil.copy2(args.yaml, f"{stem}.yaml")
    with open(f"{stem}.provenance.txt", "w") as f:
        f.write(provenance)

    nproc = args.nproc or len(os.sched_getaffinity(0))
    reps = list(range(args.rep_start, args.rep_end))
    print(f"running {len(reps)} shots on {nproc} processes -> {args.out}", flush=True)
    t0 = time.perf_counter()
    with Pool(nproc) as pool:
        res = pool.map(run_shot, [(args.yaml, rep, stride) for rep in reps], chunksize=1)

    centre, beam, weights, devs, I0, Ilast, womega = zip(*res)
    centre = np.stack(centre).astype(float)
    beam = np.stack(beam).astype(float)
    I0_sum, Ilast_sum = np.sum(I0, axis=0), np.sum(Ilast, axis=0)
    np.savez_compressed(
        args.out,
        names=np.array(rec.names), t=rec.t, z=np.asarray(X.z, float), dz=float(X.dz),
        depth_nm=np.maximum(np.arange(X.zgrid) - 1, 0) * float(X.dz),
        centre_mean=centre.mean(axis=0).astype(np.float32), centre_std=centre.std(axis=0).astype(np.float32),
        beam_mean=beam.mean(axis=0).astype(np.float32), beam_std=beam.std(axis=0).astype(np.float32),
        weights_xy=np.mean(weights, axis=0), trace_centre_max_dev=float(max(devs)),
        satellite_channel_names=np.array([chan.name for chan in X.satellite_channel_params]),
        E_edges_eV=np.asarray(X.eii_ladder["E_edges"]) if X.use_eii else np.zeros(0),
        E_centres_eV=np.asarray(X.eii_ladder["E_centres"]) if X.use_eii else np.zeros(0),
        eii_fixed_energy=bool(X.use_eii and X.eii_ladder["fixed_energy"]),
        eii_birth_names=np.array(sorted(X.eii_birth)) if X.use_eii else np.zeros(0, dtype=str),
        eii_birth_groups=np.array([X.eii_birth[k] for k in sorted(X.eii_birth)]) if X.use_eii else np.zeros(0, int),
        eii_rate_table=np.asarray(X.eii_rate_table) if X.use_eii else np.zeros((4, 0)),
        eii_rates_by_subshell=(np.stack([X.eii_ladder["rates_fs"][s] for s in sorted(X.eii_ladder["rates_fs"])])
                               if X.use_eii else np.zeros((0, 0))),
        eii_subshells=np.array(sorted(X.eii_ladder["rates_fs"])) if X.use_eii else np.zeros(0, dtype=str),
        eii_secondary_matrix=(X.eii_secondary_matrix if X.use_eii and X.eii_secondary_matrix is not None
                              else np.zeros((0, 0))),
        womega_ar=np.asarray(womega[0], float), I_int_thy_w_0=I0_sum, I_int_thy_w_last=Ilast_sum,
        T_integrated=float(Ilast_sum.sum() / I0_sum.sum()),
        n_shots=len(reps), reps=np.asarray(reps), stride=stride, dt=float(X.dt), E_seed_uJ=float(X.E_seed_uJ),
        target_energy_eV=float(getattr(X, "monochromator_target_energy_eV", np.nan)),
        seed_pulse_format=X.seed_pulse_format, config_yaml=open(args.yaml).read(), provenance=provenance)
    print(f"saved {args.out} ({tools.format_duration(time.perf_counter() - t0)}; worst trace deviation "
          f"{max(devs):.1e})", flush=True)


if __name__ == "__main__":
    main()
