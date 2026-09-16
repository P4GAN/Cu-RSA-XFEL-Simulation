"""Run a batch of shots of the original (pre-extension) model and save the centre-pixel populations
and flux of EVERY z plane vs time, per shot -- the data for comparing populations with the analytic
two-level RSA model (../RSA-derivation, attempt 1).

The sweeps (tools.compute_run_outputs) only keep the exit plane, and their rho_K/rho_l3 outputs are
Tijs-weighted rather than populations. This runner instead attaches a read-only recorder to the lean
path (the same X.movie_recorder hook XLO_sim/movie.py uses, so the numerics are unchanged) and keeps,
per shot and per z plane, at the centre pixel:
  flux    (shot, z, t)          photons nm^-2 fs^-1 of the field that drove that plane
  ground, other, 2s (shot, z, t)
  base    (shot, z, nlevel, t)  raw diagonal of the base block (levels 0-3 = 2p3/2 sublevels, 4-5 = 1s)
Plane iz is driven by the field after max(iz-1, 0) absorption steps (the solver's readout convention;
see Sample._evaluate_n_level_3D_lean), so its effective field depth is max(iz-1, 0) * dz.

By default it refuses configs with any level-structure extension switched on (2s, L2, satellites,
middlemen, EII, sublevel mixing), since the analytic comparison assumes the plain L3/K model;
--allow-extensions lifts that.

Example (see scripts/submit_population_record.sh for the cluster job):
    python scripts/run_population_record.py --yaml config/base/Cu-seed-SASE-no-2s.yaml \\
        --set E_seed_uJ=40 --rep-start 0 --rep-end 40 --out data/population_record/sase_40uJ.npz
"""

import os  # noqa: E402  (must come before numpy loads)

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO_ROOT)

from XLO_sim.XLO_sim import XLO_sim  # noqa: E402
from XLO_sim import tools  # noqa: E402

EXTENSION_FLAGS = ("use_2s_pathway", "use_L2_pathway", "satellite_channels", "double_satellite_channels",
                   "use_middlemen", "use_eii", "L3_sublevel_mixing_fs_inv", "L3_sublevel_mixing_satellite_fs_inv",
                   "L2_CK_feed", "GammaA_L1_to_L2eVN", "use_rate_equations")


class PopulationRecorder:
    """Centre-pixel flux and populations of every z plane. Only reads the solver state."""

    def __init__(self, X):
        nt, nz = X.tgrid, X.zgrid
        self.X = X
        self.cx, self.cy = int(X.xgrid / 2), int(X.ygrid / 2)  # same centre pixel as the lean path
        self.diag = np.arange(X.nlevel)
        self.flux = np.zeros((nz, nt), np.float32)
        self.ground = np.zeros((nz, nt), np.float32)
        self.other = np.zeros((nz, nt), np.float32)
        self.pop_2s = np.zeros((nz, nt), np.float32)
        self.base = np.zeros((nz, X.nlevel, nt), np.float32)
        self.iz = 0

    def record_step(self, it, Omega_it, rho_ijxy, rho_sat_ijxy, rho_ground_xy, rho_other_xy, rho_2s_xy,
                    rho_mid_xy, rho_e_gxy):
        cx, cy = self.cx, self.cy
        O = Omega_it[:, :, cx, cy]
        self.flux[self.iz, it] = np.real(O[0, 0] * O[1, 0] + O[0, 1] * O[1, 1]) / self.X.flux_factor
        self.ground[self.iz, it] = np.real(rho_ground_xy[cx, cy])
        self.other[self.iz, it] = np.real(rho_other_xy[cx, cy])
        self.pop_2s[self.iz, it] = np.real(rho_2s_xy[cx, cy])
        self.base[self.iz, :, it] = np.real(rho_ijxy[self.diag, self.diag, cx, cy])

    def end_plane(self, iz, Omega_pstxy):
        self.iz = iz + 1


def parse_overrides(pairs, base):
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or key not in base:
            raise SystemExit(f"--set {pair!r}: expected KEY=VALUE with KEY a top-level key of the base config")
        out[key] = yaml.safe_load(value)
    return out


def run_shot(args):
    yaml_path, rep = args
    t0 = time.perf_counter()
    X = XLO_sim(yaml_path)
    X.random_seed = rep
    X.keep_z_history = False
    seed_field = getattr(tools, X.seed_pulse_format)(X)
    rec = PopulationRecorder(X)
    X.movie_recorder = rec
    X.configure(seed_field)
    X.run_3D()
    print(f"shot {rep} done ({tools.format_duration(time.perf_counter() - t0)}, "
          f"worker peak mem {tools.peak_memory_gb():.2f} GB)", flush=True)
    return rec.flux, rec.ground, rec.other, rec.pop_2s, rec.base


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yaml", required=True, help="Base config")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Top-level config overrides, VALUE parsed as YAML (e.g. E_seed_uJ=40)")
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-end", type=int, default=40)
    parser.add_argument("--nproc", type=int, default=None, help="Worker processes (default: cores available)")
    parser.add_argument("--out", default=None, help="Output .npz path (derived config + provenance saved next to it)")
    parser.add_argument("--allow-extensions", action="store_true",
                        help="Run even if the config switches on a level-structure extension")
    parser.add_argument("--check-only", action="store_true", help="Load the config, run the checks, exit")
    args = parser.parse_args()

    with open(args.yaml) as f:
        cfg = yaml.safe_load(f)
    cfg.update(parse_overrides(args.overrides, cfg))
    active = [k for k in EXTENSION_FLAGS if cfg.get(k)]
    if active and not args.allow_extensions:
        sys.exit(f"config switches on {active}; this runner is for the original L3/K model "
                 "(pass --allow-extensions to run anyway)")

    if args.out is None and not args.check_only:
        parser.error("--out is required unless --check-only")
    out_dir = os.path.dirname(os.path.abspath(args.out)) if args.out else None
    stem = os.path.splitext(os.path.abspath(args.out))[0] if args.out else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        derived_yaml = f"{stem}.yaml"
    else:
        import tempfile
        derived_yaml = tempfile.mkstemp(suffix=".yaml")[1]
    with open(derived_yaml, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    X = XLO_sim(derived_yaml)
    provenance = tools.verify_code(X, REPO_ROOT)
    print(provenance, end="", flush=True)
    print(f"{args.yaml} {args.overrides}: grid t x y z = {X.tgrid} x {X.xgrid} x {X.ygrid} x {X.zgrid}, "
          f"E_seed {X.E_seed_uJ} uJ, seed format {X.seed_pulse_format}, "
          f"{len(X.satellite_channel_params)} satellite blocks, nlevel {X.nlevel}", flush=True)
    if args.check_only:
        if not out_dir:
            os.remove(derived_yaml)
        print("check passed", flush=True)
        return

    with open(f"{stem}.provenance.txt", "w") as f:
        f.write(provenance)
    nproc = args.nproc or len(os.sched_getaffinity(0))
    reps = list(range(args.rep_start, args.rep_end))
    print(f"running {len(reps)} shots on {nproc} processes -> {args.out}", flush=True)
    t0 = time.perf_counter()
    with Pool(nproc) as pool:
        res = pool.map(run_shot, [(derived_yaml, rep) for rep in reps], chunksize=1)

    flux, ground, other, pop_2s, base = (np.stack(a) for a in zip(*res))
    np.savez_compressed(
        args.out, flux=flux, ground=ground, other=other, pop_2s=pop_2s, base=base,
        t=np.asarray(X.t, float), z=np.asarray(X.z, float), dz=float(X.dz), reps=np.asarray(reps),
        ei_L3=np.asarray(X.ei_L3), ei_K=np.asarray(X.ei_K), E_seed_uJ=float(X.E_seed_uJ),
        target_energy_eV=float(getattr(X, "monochromator_target_energy_eV", np.nan)),
        seed_pulse_format=X.seed_pulse_format, config_yaml=open(derived_yaml).read(), provenance=provenance)
    print(f"saved {args.out} ({tools.format_duration(time.perf_counter() - t0)})", flush=True)


if __name__ == "__main__":
    main()
