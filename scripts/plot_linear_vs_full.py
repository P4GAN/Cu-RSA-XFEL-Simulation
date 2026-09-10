"""Linear-response vs full Maxwell-Bloch comparison for the SASE double-satellite model.

Isolates the Rabi nonlinearity of the resonant Kalpha coupling: the linear run
(linear_resonant_response: true, see Model._MB_nlevel_regular_core) keeps the same
photoionization/Auger hole kinetics but never lets the field move population, so it
has no saturation, power broadening or stimulated emission.

Data:
  full   2/9/40/60 uJ : data/production_sweep_sase_24437070/Cu-seed-SASE-double-satellite
  full   0.12/0.5 uJ  : data/linear_response_sweep_sase_24533426/Cu-seed-SASE-double-satellite
  linear all          : data/linear_response_sweep_sase_24533426/Cu-seed-SASE-double-satellite-linear
  experiment          : SASE pulse centred ~8045 eV (transmittance_2_9_40uJ.csv,
                        transmittance_bottom_extra_uJ.csv) -- same centre as the simulated seed.

Energy axis: 8045 + womega_ar, the same calibration notebooks/plot-simulation-vs-experiment.ipynb
uses for the slide figures (it sets hwKalpha1N = 8045 there), so these plots line up with the deck.

Writes figs/linear_vs_full_spectra.{png,pdf} and figs/linear_vs_full_dip_scaling.{png,pdf}.
Run from the repo root:  python scripts/plot_linear_vs_full.py
"""

import os
import glob

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
FIGS = os.path.join(REPO, "figs")

FULL_DIRS = [os.path.join(DATA, "production_sweep_sase_24437070", "Cu-seed-SASE-double-satellite"),
             os.path.join(DATA, "linear_response_sweep_sase_24533426", "Cu-seed-SASE-double-satellite")]
LINEAR_DIRS = [os.path.join(DATA, "linear_response_sweep_sase_24533426", "Cu-seed-SASE-double-satellite-linear")]

ENERGY_OFFSET_EV = 8045.0
KA1_WINDOW = (8040.0, 8052.0)     # where the Kalpha1 dip minimum is searched for
SIM_COLD_WINDOW = (8005.0, 8015.0)  # off-resonant wing used as the sim's cold-foil reference
EXP_COLD_WINDOW = (8030.0, 8065.0)

# Pulse energy at which the resonance-filtered peak Rabi frequency equals the Kalpha1 coherence
# width Gamma_coh = (Gamma_K + Gamma_L3)/2 = 1.05 eV, at the entrance-face centre pixel, using the
# strongest Clebsch-Gordan factor 1/sqrt(3) and averaging 8 SASE shots of this config's seed
# (Omega_eff/Gamma = 0.75 at 2 uJ, 1.58 at 9 uJ, scaling as sqrt(E)).
E_RABI_EQUALS_GAMMA_UJ = 3.6

FULL_COLOUR, LINEAR_COLOUR, EXP_COLOUR = "#eda100", "#2a78d6", "#23262f"
INK_MUTED = "#6b6e76"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "lines.linewidth": 2.0, "figure.dpi": 120, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "axes.edgecolor": "#b9bbc0",
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_sweeps(dirs, expect_linear=False):
    """{E_seed_uJ: (energy_eV, T, n_reps)} over every runs_seed_* folder in dirs.

    With expect_linear, checks the 1s-hole (K) population is identically zero -- the signature of
    linear_resonant_response actually being active. A run from code that predates the flag
    silently ignores the config key and produces the full model instead (job 24533426 did).
    """
    out = {}
    for d in dirs:
        for run_dir in glob.glob(os.path.join(d, "runs_seed_*_uJ")):
            yaml_files = glob.glob(os.path.join(run_dir, "*.yaml"))
            # folder names round E_seed to 1 decimal (0.12 -> runs_seed_0.1_uJ); the YAML is exact
            e_seed = float(yaml.safe_load(open(yaml_files[0]))["E_seed_uJ"])
            acc, n = {}, 0
            for f in glob.glob(os.path.join(run_dir, "*.npz")):
                z = np.load(f)
                n += int(z["n_reps"])
                w = z["womega_ar"]
                for k in ("I_int_thy_w_last_sum", "I_int_thy_w_0_sum", "rho_ee_t_last_sum"):
                    acc[k] = acc.get(k, 0) + z[k]
            if expect_linear and np.any(acc["rho_ee_t_last_sum"] != 0):
                raise RuntimeError(f"{run_dir}: nonzero 1s-hole population, so linear_resonant_response "
                                   "was not active in this run (stale XLO_sim code on the cluster?)")
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
    full = load_sweeps(FULL_DIRS)
    linear = load_sweeps(LINEAR_DIRS, expect_linear=True)
    exp, (E_cold, T_cold_curve) = load_experiment()

    # Cold-foil references: sim = off-resonant wing of the lowest-fluence linear run (flat
    # photoionization only); experiment = median of the measured cold-foil curve.
    E0, T0, _ = linear[min(linear)]
    T_cold_sim = np.median(T0[in_window(E0, SIM_COLD_WINDOW)])
    T_cold_exp = np.median(T_cold_curve[in_window(E_cold, EXP_COLD_WINDOW)])

    rows = []
    for e in sorted(set(full) | set(linear) | set(exp)):
        row = {"E_uJ": e}
        for name, src in (("full", full), ("linear", linear)):
            if e in src:
                E, T, n = src[e]
                row[name], _ = ka1_absorbance(E, T, T_cold_sim)
                row[f"{name}_n"] = n
        if e in exp:
            E, T, T_err = exp[e]
            row["exp"], row["exp_err"] = ka1_absorbance(E, T, T_cold_exp, T_err)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("E_uJ")
    table["full/linear"] = table["full"] / table["linear"]
    pd.set_option("display.float_format", "{:.3f}".format)
    print(f"cold-foil reference: sim T = {T_cold_sim:.3f}, experiment T = {T_cold_exp:.3f}\n")
    print("Kalpha1 dip absorbance A = -ln(T_min / T_cold):")
    print(table.to_string())

    plot_spectra(full, linear, exp)
    plot_dip_scaling(table)


def plot_spectra(full, linear, exp):
    energies = [2.0, 9.0, 40.0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, e in zip(axes, energies):
        E, T, _ = full[e]
        ax.plot(E, T, color=FULL_COLOUR, lw=2.4, label="Full Maxwell–Bloch")
        E, T, _ = linear[e]
        ax.plot(E, T, color=LINEAR_COLOUR, lw=2.0, ls="--", label="Linear response (no Rabi)")
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
    save(fig, "linear_vs_full_spectra")


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
    lin = table.dropna(subset=["linear"])
    ful = table.dropna(subset=["full"])
    ex = table.dropna(subset=["exp"])
    ax.plot(lin.index, lin["linear"], ls="--", marker="s", ms=8, color=LINEAR_COLOUR, mec="white", mew=1.5,
            label="Linear response (no Rabi)", zorder=3)
    ax.plot(ful.index, ful["full"], ls="-", marker="o", ms=8, color=FULL_COLOUR, mec="white", mew=1.5,
            lw=2.4, label="Full Maxwell–Bloch", zorder=4)
    ax.errorbar(ex.index, ex["exp"], yerr=ex["exp_err"], fmt="D", ms=7, color=EXP_COLOUR, mfc=EXP_COLOUR,
                mec="white", mew=1.2, elinewidth=1.2, capsize=0, label="Experiment", zorder=5)
    ax.set_yscale("log")
    ax.set_ylabel(r"K$\alpha_1$ dip absorbance  $-\ln(T_\mathrm{min}/T_\mathrm{cold})$")
    ax.set_title(r"K$\alpha_1$ dip vs pulse energy")
    ax.legend(loc="upper left", frameon=False)
    ax.text(E_RABI_EQUALS_GAMMA_UJ * 1.12, 0.975, r"$\Omega_\mathrm{eff} > \Gamma$  (Rabi regime)",
            transform=ax.get_xaxis_transform(), va="top", ha="left", fontsize=9.5, color=INK_MUTED)

    # direct labels at the right end of each simulated series
    for df, key, colour in ((lin, "linear", LINEAR_COLOUR), (ful, "full", FULL_COLOUR)):
        ax.annotate(f"{df[key].iloc[-1]:.2f}", (df.index[-1], df[key].iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9.5, color="#23262f")

    # --- right: suppression factor ---
    both = table.dropna(subset=["full", "linear"])
    ax2.axhline(1.0, color=INK_MUTED, lw=1.0)
    ax2.plot(both.index, both["full/linear"], marker="o", ms=8, color=FULL_COLOUR, mec="white", mew=1.5, lw=2.4)
    for e, r in both["full/linear"].items():
        ax2.annotate(f"{r:.2f}", (e, r), xytext=(0, 10), textcoords="offset points", ha="center",
                     fontsize=9.5, color="#23262f")
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel(r"$A_\mathrm{full}\,/\,A_\mathrm{linear}$")
    ax2.set_title("Dip suppressed by Rabi saturation")

    fig.tight_layout()
    save(fig, "linear_vs_full_dip_scaling")


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{stem}.{ext}"), dpi=200)
    print(f"wrote figs/{stem}.png/.pdf")


if __name__ == "__main__":
    main()
