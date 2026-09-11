"""Rate-equation vs full Maxwell-Bloch comparison for the SASE double-satellite model.

Isolates the coherent part of the resonant Kalpha dynamics: the rate-equation run
(use_rate_equations: true, see Model._MB_nlevel_regular_core / Model.physical_rho) has the same
hole kinetics and still saturates (stimulated absorption/emission at the Lorentzian-filtered
rate), but its coherences are adiabatically eliminated -- no Rabi oscillations or coherent
transients.

Data:
  MB  2/9/40/60 uJ : data/production_sweep_sase_24437070/Cu-seed-SASE-double-satellite
  MB  0.12/0.5 uJ  : data/linear_response_sweep_sase_24533426/Cu-seed-SASE-double-satellite
                     (valid full-MB runs -- that job's *-linear folder is NOT, see below)
  RE  all          : newest data/rate_eq_sweep_sase_<jobid>/Cu-seed-SASE-double-satellite-rate-eq,
                     or --rate-eq-dir
  experiment       : SASE pulse centred ~8045 eV (transmittance_2_9_40uJ.csv,
                     transmittance_bottom_extra_uJ.csv) -- same centre as the simulated seed.

A run from code that predates use_rate_equations silently ignores the config key and produces
the full model (job 24533426 did this with the earlier linear-response flag), and nothing in the
run outputs distinguishes the two -- submit_rate_eq_sweep.sh's preflight is what guards this.

Energy axis: 8045 + womega_ar, the same calibration notebooks/plot-simulation-vs-experiment.ipynb
uses for the slide figures (it sets hwKalpha1N = 8045 there), so these plots line up with the deck.

Writes figs/rate_eq_vs_mb_spectra.{png,pdf} and figs/rate_eq_vs_mb_dip_scaling.{png,pdf}.
Run from the repo root:  python scripts/plot_rate_eq_vs_mb.py [--rate-eq-dir DIR]
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
FIGS = os.path.join(REPO, "figs")

MB_DIRS = [os.path.join(DATA, "production_sweep_sase_24437070", "Cu-seed-SASE-double-satellite"),
           os.path.join(DATA, "linear_response_sweep_sase_24533426", "Cu-seed-SASE-double-satellite")]
RE_CONFIG_NAME = "Cu-seed-SASE-double-satellite-rate-eq"

ENERGY_OFFSET_EV = 8045.0
KA1_WINDOW = (8040.0, 8052.0)     # where the Kalpha1 dip minimum is searched for
SIM_COLD_WINDOW = (8005.0, 8015.0)  # off-resonant wing used as the sim's cold-foil reference
EXP_COLD_WINDOW = (8030.0, 8065.0)

# Pulse energy at which the resonance-filtered peak Rabi frequency equals the Kalpha1 coherence
# width Gamma_coh = (Gamma_K + Gamma_L3)/2 = 1.05 eV, at the entrance-face centre pixel, using the
# strongest Clebsch-Gordan factor 1/sqrt(3) and averaging 8 SASE shots of this config's seed
# (Omega_eff/Gamma = 0.75 at 2 uJ, 1.58 at 9 uJ, scaling as sqrt(E)).
E_RABI_EQUALS_GAMMA_UJ = 3.6

MB_COLOUR, RE_COLOUR, EXP_COLOUR = "#eda100", "#2a78d6", "#23262f"
INK_MUTED = "#6b6e76"
MB_LABEL, RE_LABEL = "Maxwell–Bloch", "Rate equations"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "lines.linewidth": 2.0, "figure.dpi": 120, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "axes.edgecolor": "#b9bbc0",
    "axes.spines.top": False, "axes.spines.right": False,
})


def rate_eq_dirs():
    """Every data/rate_eq_sweep_sase_<jobid>/<RE config> folder, newest job first -- jobs are
    merged (e.g. 24549384's 0.12-60 uJ plus a later high-fluence extension)."""
    candidates = sorted(glob.glob(os.path.join(DATA, "rate_eq_sweep_sase_*", RE_CONFIG_NAME)), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no data/rate_eq_sweep_sase_*/{RE_CONFIG_NAME} -- pass --rate-eq-dir")
    return candidates


def check_rate_eq_provenance(re_dir):
    """Every RE output must have been written by a run whose kernel check passed
    (run_intensity_sweep.verify_code). The outputs themselves can't tell RE from MB apart, and
    jobs 24533426/24535882/24541257 silently ran the full model."""
    for npz in glob.glob(os.path.join(re_dir, "runs_seed_*_uJ", "*.npz")):
        # <output_stem>.provenance.txt sits next to <output_stem>[_<timestamp>|.partial].npz
        prov = [p for p in glob.glob(os.path.join(os.path.dirname(npz), "*.provenance.txt"))
                if os.path.basename(npz).startswith(os.path.basename(p)[:-len(".provenance.txt")])]
        if not prov or "use_rate_equations: True (kernel check passed)" not in open(prov[0]).read():
            raise RuntimeError(f"{npz}: no provenance confirming the rate-equation kernel ran -- "
                               "treat as full Maxwell-Bloch output, not rate equations")


def load_sweeps(dirs):
    """{E_seed_uJ: (energy_eV, T, n_reps)} over every runs_seed_* folder in dirs. Every job runs
    seeds 0..n-1, so the same E_seed in two jobs repeats the same SASE shots rather than adding
    new ones: the first dir listed (newest job) wins instead of being summed."""
    out = {}
    for d in dirs:
        for run_dir in glob.glob(os.path.join(d, "runs_seed_*_uJ")):
            yaml_files = glob.glob(os.path.join(run_dir, "*.yaml"))
            # folder names round E_seed to 1 decimal (0.12 -> runs_seed_0.1_uJ); the YAML is exact
            e_seed = float(yaml.safe_load(open(yaml_files[0]))["E_seed_uJ"])
            npz_files = glob.glob(os.path.join(run_dir, "*.npz"))
            if not npz_files:
                print(f"warning: {os.path.relpath(run_dir, REPO)} has no output yet -- skipped")
                continue
            if e_seed in out:
                print(f"warning: {e_seed:g} uJ also in {os.path.relpath(run_dir, REPO)} -- using the newer job's")
                continue
            acc, n = {}, 0
            for f in npz_files:
                z = np.load(f)
                n += int(z["n_reps"])
                w = z["womega_ar"]
                for k in ("I_int_thy_w_last_sum", "I_int_thy_w_0_sum"):
                    acc[k] = acc.get(k, 0) + z[k]
            T = acc["I_int_thy_w_last_sum"] / acc["I_int_thy_w_0_sum"]
            out[e_seed] = (ENERGY_OFFSET_EV + w, T, n)
    return dict(sorted(out.items()))


def load_experiment():
    """{E_seed_uJ: (energy_eV, T, T_err)} plus the cold-foil curve, SASE centred ~8045 eV."""
    a = pd.read_csv(os.path.join(DATA, "transmittance_2_9_40uJ.csv"))
    b = pd.read_csv(os.path.join(DATA, "transmittance_bottom_extra_uJ.csv"))
    exp = {}
    for df, labels in ((a, ["40", "9", "2"]), (b, ["0.5", "0.12"])):
        for lab in labels:
            exp[float(lab)] = (df["Photon_energy_eV"].values, df[f"{lab}uJ_transmittance"].values,
                               df[f"{lab}uJ_error"].values)
    cold = (b["Photon_energy_eV"].values, b["Cold_transmittance"].values)
    return dict(sorted(exp.items())), cold


def smooth_ev(E, y, half_width_eV=0.25):
    """Boxcar in photon energy, so sim (0.09 eV) and experiment (0.1-0.4 eV) grids smooth alike."""
    return np.array([y[np.abs(E - e) <= half_width_eV].mean() for e in E])


def ka1_absorbance(E, T, T_cold, T_err=None):
    """A = -ln(T_min / T_cold) at the Kalpha1 dip, and its 1-sigma error from T_err at the minimum."""
    m = (E >= KA1_WINDOW[0]) & (E <= KA1_WINDOW[1])
    Ts = smooth_ev(E[m], T[m])
    i = np.argmin(Ts)
    A = -np.log(Ts[i] / T_cold)
    dA = np.nan if T_err is None else T_err[m][i] / Ts[i]
    return A, dA


def in_window(E, window):
    return (E >= window[0]) & (E <= window[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rate-eq-dir", default=None,
                        help=f"one folder holding rate-equation runs_seed_*_uJ/ (default: every "
                             f"data/rate_eq_sweep_sase_*/{RE_CONFIG_NAME}, merged)")
    args = parser.parse_args()
    re_dirs = [args.rate_eq_dir] if args.rate_eq_dir else rate_eq_dirs()
    for d in re_dirs:
        check_rate_eq_provenance(d)
    print(f"rate-equation data: {', '.join(os.path.relpath(d, REPO) for d in re_dirs)}")

    # Prefer the full-MB runs submitted alongside the RE runs (same seeds and rep count, so the
    # ratio compares identical SASE shots); fall back to the older production/low-fluence runs.
    same_job_mb = [os.path.join(os.path.dirname(d), "Cu-seed-SASE-double-satellite") for d in re_dirs]
    mb_dirs = [d for d in same_job_mb if glob.glob(os.path.join(d, "runs_seed_*_uJ"))] or MB_DIRS
    print(f"Maxwell-Bloch data: {', '.join(os.path.relpath(d, REPO) for d in mb_dirs)}")

    mb = load_sweeps(mb_dirs)
    re = load_sweeps(re_dirs)
    exp, (E_cold, T_cold_curve) = load_experiment()

    # References: each simulated run's own off-resonant wing (flat photoionization continuum), since
    # at >~100 uJ ground-state depletion lifts the continuum itself and a fixed cold reference would
    # fold that into the dip; experiment = median of the measured cold-foil curve (its own wing is
    # inside the broad Kalpha1 dip at 40 uJ, and its continuum shift is small).
    T_cold_exp = np.median(T_cold_curve[in_window(E_cold, EXP_COLD_WINDOW)])

    rows = []
    for e in sorted(set(mb) | set(re) | set(exp)):
        row = {"E_uJ": e}
        for name, src in (("MB", mb), ("RE", re)):
            if e in src:
                E, T, n = src[e]
                wing = np.median(T[in_window(E, SIM_COLD_WINDOW)])
                row[name], _ = ka1_absorbance(E, T, wing)
                row[f"{name}_wing"] = wing
                row[f"{name}_n"] = n
        if e in exp:
            E, T, T_err = exp[e]
            row["exp"], row["exp_err"] = ka1_absorbance(E, T, T_cold_exp, T_err)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("E_uJ")
    table["MB/RE"] = table["MB"] / table["RE"]
    pd.set_option("display.float_format", "{:.3f}".format)
    print(f"experiment cold-foil reference T = {T_cold_exp:.3f}; sim uses each run's own wing "
          f"({SIM_COLD_WINDOW[0]:g}-{SIM_COLD_WINDOW[1]:g} eV, *_wing columns)\n")
    print("Kalpha1 dip absorbance A = -ln(T_min / T_ref):")
    print(table.to_string())

    plot_spectra(mb, re, exp)
    plot_dip_scaling(table)


def plot_spectra(mb, re, exp):
    energies = [2.0, 9.0, 40.0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, e in zip(axes, energies):
        E, T, _ = mb[e]
        ax.plot(E, T, color=MB_COLOUR, lw=2.4, label=MB_LABEL)
        E, T, _ = re[e]
        ax.plot(E, T, color=RE_COLOUR, lw=2.0, ls="--", label=RE_LABEL)
        Ee, Te, dTe = exp[e]
        ax.errorbar(Ee, Te, yerr=dTe, fmt="o", ms=3.5, mfc="white", mec=EXP_COLOUR, ecolor=EXP_COLOUR,
                    elinewidth=0.8, capsize=0, color=EXP_COLOUR, label="Experiment (centred ~8045 eV)",
                    zorder=1, alpha=0.8)
        ax.set_title(f"{e:g} µJ")
        ax.set_xlim(8000, 8070)
        ax.set_xlabel("Photon energy (eV)")
    axes[0].set_ylabel("Transmittance")
    axes[0].set_ylim(0.0, 0.46)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "rate_eq_vs_mb_spectra")


def plot_dip_scaling(table):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.5), gridspec_kw={"width_ratios": [1.35, 1]})

    for a in (ax, ax2):
        a.axvspan(E_RABI_EQUALS_GAMMA_UJ, 200, color="#8c8f96", alpha=0.10, lw=0, zorder=0)
        a.axvline(E_RABI_EQUALS_GAMMA_UJ, color=INK_MUTED, lw=1.0, ls=":", zorder=0)
        a.set_xscale("log")
        a.set_xlim(0.08, 90)
        a.set_xticks([0.12, 0.5, 2, 9, 40, 60])
        a.set_xticklabels(["0.12", "0.5", "2", "9", "40", "60"])
        a.minorticks_off()
        a.set_xlabel("Pulse energy (µJ)")

    # --- left: absorbance vs pulse energy ---
    re = table.dropna(subset=["RE"])
    mb = table.dropna(subset=["MB"])
    ex = table.dropna(subset=["exp"])
    ax.plot(re.index, re["RE"], ls="--", marker="s", ms=8, color=RE_COLOUR, mec="white", mew=1.5,
            label=RE_LABEL, zorder=3)
    ax.plot(mb.index, mb["MB"], ls="-", marker="o", ms=8, color=MB_COLOUR, mec="white", mew=1.5,
            lw=2.4, label=MB_LABEL, zorder=4)
    ax.errorbar(ex.index, ex["exp"], yerr=ex["exp_err"], fmt="D", ms=7, color=EXP_COLOUR, mfc=EXP_COLOUR,
                mec="white", mew=1.2, elinewidth=1.2, capsize=0, label="Experiment", zorder=5)
    ax.set_yscale("log")
    ax.set_ylabel(r"K$\alpha_1$ dip absorbance  $-\ln(T_\mathrm{min}/T_\mathrm{cold})$")
    ax.set_title(r"K$\alpha_1$ dip vs pulse energy")
    ax.legend(loc="upper left", frameon=False)
    ax.text(E_RABI_EQUALS_GAMMA_UJ * 1.12, 0.975, r"$\Omega_\mathrm{eff} > \Gamma$  (Rabi regime)",
            transform=ax.get_xaxis_transform(), va="top", ha="left", fontsize=9.5, color=INK_MUTED)

    # direct labels at the right end of each simulated series
    for df, key in ((re, "RE"), (mb, "MB")):
        ax.annotate(f"{df[key].iloc[-1]:.2f}", (df.index[-1], df[key].iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9.5, color="#23262f")

    # --- right: MB / RE ratio ---
    both = table.dropna(subset=["MB", "RE"])
    ax2.axhline(1.0, color=INK_MUTED, lw=1.0)
    ax2.plot(both.index, both["MB/RE"], marker="o", ms=8, color=MB_COLOUR, mec="white", mew=1.5, lw=2.4)
    for e, r in both["MB/RE"].items():
        ax2.annotate(f"{r:.2f}", (e, r), xytext=(0, 10), textcoords="offset points", ha="center",
                     fontsize=9.5, color="#23262f")
    lo, hi = both["MB/RE"].min(), both["MB/RE"].max()
    pad = max(0.1, 0.25 * (hi - lo))
    ax2.set_ylim(min(lo, 1.0) - pad, max(hi, 1.0) + pad)
    ax2.set_ylabel(r"$A_\mathrm{MB}\,/\,A_\mathrm{RE}$")
    ax2.set_title("Effect of coherence on the dip")

    fig.tight_layout()
    save(fig, "rate_eq_vs_mb_dip_scaling")


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{stem}.{ext}"), dpi=200)
    print(f"wrote figs/{stem}.png/.pdf")


if __name__ == "__main__":
    main()
