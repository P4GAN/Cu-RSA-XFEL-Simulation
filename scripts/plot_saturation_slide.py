"""Slide figures for the saturation of the RSA dip, full (double-satellite + L2) SASE model, 0.12-60 uJ.

  figs/saturation_slide_dip.png          transmittance spectra (all pulse energies) + Kalpha dip absorbance
                                         vs pulse energy with the saturable-absorber fit
  figs/saturation_slide_populations.png  fluence scaling of every tracked manifold (the figure from
                                         plot_double_satellite_diagnostics.py, plus the 0.12 / 0.5 uJ runs)
  figs/saturation_absorbance_scales.png  the dip-absorbance panel on log-log, linear and log-x axes, for
                                         choosing ABSORBANCE_SCALE (not meant for the slides)

The two slide figures are drawn at their true size on the slide (Beamer 16:9, PaloAlto sidebar:
\\textwidth = 381.8 pt = 5.28 in), so the fonts below are the fonts the audience sees.

Saturable-absorber fit  A(E) = a E / (1 + E / E_sat): the resonant absorbance is (number of absorbers) x
(absorption per absorber). The 2p3/2 holes are made by photoionization, so their number is ~ E; the
absorption per hole follows two-level saturation, 1 / (1 + s) with saturation parameter s = I / I_sat ~ E.
a is the low-fluence slope, E_sat the pulse energy at which s = 1. It is a single-intensity (steady-state,
flat-top) model -- it ignores the Gaussian beam, SASE spikes, ground-state depletion and power broadening --
so it is used as a two-number summary of the curve, not a first-principles prediction.

Data, metric definitions and fits are shared with plot_saturation.py.
Run from the repo root:  python scripts/plot_saturation_slide.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator

from plot_saturation import (FIGS, SIM_DIRS, ENERGY_OFFSET_EV, KA1_WINDOW, KA2_WINDOW, EXP_NOISE_FLOOR_UJ,
                             KA1_COLOUR, KA2_COLOUR, EXP_COLOUR,
                             load_sweeps, load_experiment, sim_spectrum, smooth_ev, dip_absorbance,
                             exp_absorbance, fit_saturable, saturable)

SLIDE_WIDTH_IN = 381.79 / 72.27

# Axes for the dip-absorbance panel on the slide: "semilogx" (log E, linear A), "loglog" or "linear".
ABSORBANCE_SCALE = "linear"

# Manifold colours/markers exactly as plot_double_satellite_diagnostics.py assigns them.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
E_TICKS = [0.12, 0.5, 2, 9, 40, 60]
FIT_LABEL = r"fit  $aE\,/\,(1+E/E_\mathrm{sat})$"

SLIDE_RC = {
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8, "legend.fontsize": 6.6, "lines.linewidth": 1.3, "lines.markersize": 4,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
}


def energy_colours(n):
    """viridis, dark (high energy, deepest dip) to light (low energy); the top 8% is dropped for contrast."""
    return plt.cm.viridis(np.linspace(0.92, 0.0, n))


def fixed_log_ticks(axis, values):
    axis.set_major_locator(FixedLocator(values))
    axis.set_major_formatter(FixedFormatter([f"{v:g}" for v in values]))
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


# =============================================================================
# data
# =============================================================================
def dip_data(sim, E_sim):
    # T_cold = off-resonant (8005-8015 eV) transmittance of the weakest pulse, i.e. the cold foil -- the same
    # reference the experiment uses and plot_rate_eq_vs_mb.py uses for the simulation.
    T_cold = sim_spectrum(sim[E_sim[0]])[2]
    A1, A2 = [], []
    for e in E_sim:
        E, T, _ = sim_spectrum(sim[e])
        A = -np.log(smooth_ev(E, T) / T_cold)
        A1.append(dip_absorbance(E, A, KA1_WINDOW)[0])
        A2.append(dip_absorbance(E, A, KA2_WINDOW)[0])
    A1, A2 = np.array(A1), np.array(A2)

    exp, T_cold = load_experiment()
    table = pd.DataFrame({e: exp_absorbance(E, T, dT, T_cold) for e, (E, T, dT) in exp.items()},
                         index=["A", "dA"]).T.sort_index()
    # 0.12 uJ: the expected dip (dT ~ 0.005) is below the measurement noise, so the minimum of the
    # smoothed curve is a noise dip -- dropped from the plot and the fit.
    table = table[table.index > EXP_NOISE_FLOOR_UJ]
    return {"A1": A1, "A2": A2, "exp": table,
            "fit1": fit_saturable(E_sim, A1)[0], "fit2": fit_saturable(E_sim, A2)[0],
            "fit_exp": fit_saturable(table.index.values, table["A"].values)[0]}


# =============================================================================
# panels
# =============================================================================
def panel_spectra(ax, sim, E_sim):
    for e, colour in zip(E_sim, energy_colours(len(E_sim))):
        run = sim[e]
        E = ENERGY_OFFSET_EV + run["womega_ar"]
        ax.plot(E, run["I_int_thy_w_last"] / run["I_int_thy_w_0"], color=colour, lw=1.0, alpha=0.85,
                label=f"{e:g} µJ")
    ax.set_xlim(8010, 8062)
    ax.set_ylim(0.15, 0.425)
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Transmittance")
    # between the two dips, where every curve is at T > 0.37
    ax.legend(loc="lower center", ncol=1, frameon=False, handlelength=1.2, handletextpad=0.4,
              labelspacing=0.2, bbox_to_anchor=(0.515, -0.01))


def panel_absorbance(ax, E_sim, d, scale, legend=True, ms=4.5):
    E_fine = np.geomspace(0.1, 70, 300) if scale != "linear" else np.linspace(0, 65, 300)
    for fit, colour in ((d["fit1"], KA1_COLOUR), (d["fit2"], KA2_COLOUR), (d["fit_exp"], EXP_COLOUR)):
        ax.plot(E_fine, saturable(E_fine, *fit), color=colour, lw=1.0, alpha=0.55, zorder=2)
    ax.plot(E_sim, d["A1"], ls="none", marker="o", ms=ms, color=KA1_COLOUR, mec="white", mew=0.7, zorder=4,
            label=r"K$\alpha_1$, simulation")
    ax.plot(E_sim, d["A2"], ls="none", marker="s", ms=ms * 0.93, color=KA2_COLOUR, mec="white", mew=0.7,
            zorder=4, label=r"K$\alpha_2$, simulation")
    ax.errorbar(d["exp"].index, d["exp"]["A"], yerr=d["exp"]["dA"], fmt="D", ms=ms * 0.85, color=EXP_COLOUR,
                mec="white", mew=0.6, elinewidth=0.8, capsize=0, zorder=5, label=r"K$\alpha_1$, experiment")
    ax.plot([], [], color="0.45", lw=1.0, label=FIT_LABEL)

    if scale == "loglog":
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.09, 80)
        ax.set_ylim(0.005, 1.5)
    elif scale == "semilogx":
        ax.set_xscale("log")
        ax.set_xlim(0.09, 80)
        ax.set_ylim(0, 0.9)
    else:
        ax.set_xlim(0, 64)
        ax.set_ylim(0, 0.9)
    if scale != "linear":
        fixed_log_ticks(ax.xaxis, E_TICKS)
        labels = ax.get_xticklabels()   # 40 and 60 are close on a log axis: push the labels apart
        labels[-2].set_horizontalalignment("right")
        labels[-1].set_horizontalalignment("left")
    ax.set_xlabel("Pulse energy (µJ)")
    ax.set_ylabel(r"$-\ln(T_\mathrm{dip}/T_\mathrm{cold})$")
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index(k) for k in (r"K$\alpha_1$, simulation", r"K$\alpha_2$, simulation",
                                           r"K$\alpha_1$, experiment", FIT_LABEL)]
        # lower right is empty on linear axes (every curve is above A = 0.45 beyond 25 uJ); upper left on log x
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  loc="lower right" if scale == "linear" else "upper left", frameon=False,
                  handlelength=1.5, labelspacing=0.3)


def manifold_series(sim, E_sim):
    """The series of plot_double_satellite_diagnostics.py figure 3, same labels, colours and styles."""
    names = [str(n) for n in sim[E_sim[-1]]["satellite_channel_names"]]
    dt = sim[E_sim[-1]]["t_axis"][1] - sim[E_sim[-1]]["t_axis"][0]

    def integrated(key, row=None):
        return [(sim[e][key] if row is None else sim[e][key][row]).sum() * dt for e in E_sim]

    series = [("base $1s^{-1}$ (K)", integrated("rho_ee_t_last"), "#e34948", "-", "D", 2.6),
              ("base $2p_{3/2}^{-1}$ (L3)", integrated("rho_l3_t_last"), "#111111", "-", "o", 2.6),
              ("base $2p_{1/2}^{-1}$ (L2)", integrated("rho_l2_t_last"), "#111111", "--", "o", 1.8)]
    for i, name in enumerate(names):
        series.append((f"${name}$", integrated("rho_l3_t_last_sat", i), PALETTE[i % len(PALETTE)],
                       "-" if len(name) <= 3 else "--", "s" if len(name) <= 3 else "^", 1.4))
    return series


def plot_populations(sim, E_sim):
    series = manifold_series(sim, E_sim)
    fig, ax = plt.subplots(figsize=(SLIDE_WIDTH_IN, 2.32))
    slopes = {}
    for label, values, colour, ls, marker, lw in series:
        slope = np.polyfit(np.log(E_sim), np.log(values), 1)[0]
        slopes[label] = slope
        # line widths scaled from the original 8.2 in figure to this 5.3 in one
        ax.plot(E_sim, values, color=colour, ls=ls, marker=marker, lw=0.62 * lw, ms=3.2,
                label=f"{label}  ($p={slope:.2f}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    fixed_log_ticks(ax.xaxis, E_TICKS)
    labels = ax.get_xticklabels()
    labels[-2].set_horizontalalignment("right")
    labels[-1].set_horizontalalignment("left")
    ax.set_xlabel("Pulse energy (µJ)")
    ax.set_ylabel("Time-integrated dipole-weighted\npopulation (arb.)")
    ax.set_title(r"Fluence scaling of each hole-state population, fitted as $\propto E^{\,p}$")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=6.6, labelspacing=0.45)
    fig.tight_layout()
    save(fig, "saturation_slide_populations")
    return slopes


def plot_dip(sim, E_sim, d):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(SLIDE_WIDTH_IN, 2.2), gridspec_kw={"width_ratios": [1, 1.2]})
    panel_spectra(ax1, sim, E_sim)
    panel_absorbance(ax2, E_sim, d, ABSORBANCE_SCALE)
    fig.tight_layout(w_pad=1.2)
    save(fig, "saturation_slide_dip")


def plot_scale_comparison(E_sim, d):
    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "legend.fontsize": 8.5, "savefig.bbox": "tight",
                         "axes.grid": True, "grid.alpha": 0.3}):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
        for ax, scale, title in zip(axes, ("loglog", "linear", "semilogx"),
                                    ("log–log (previous slide)", "linear–linear", "log E, linear A")):
            panel_absorbance(ax, E_sim, d, scale, legend=(scale == "linear"), ms=6.5)
            ax.set_title(title)
        fig.tight_layout()
        save(fig, "saturation_absorbance_scales")


def save(fig, stem):
    path = os.path.join(FIGS, f"{stem}.png")
    fig.savefig(path, dpi=400)
    plt.close(fig)
    print(f"wrote {os.path.relpath(path)}")


def main():
    sim = load_sweeps(SIM_DIRS)
    E_sim = np.array(list(sim))
    d = dip_data(sim, E_sim)

    with plt.rc_context(SLIDE_RC):
        plot_dip(sim, E_sim, d)
        slopes = plot_populations(sim, E_sim)
    plot_scale_comparison(E_sim, d)

    for name, (a, E_sat) in (("sim Ka1", d["fit1"]), ("sim Ka2", d["fit2"]), ("exp Ka1", d["fit_exp"])):
        print(f"{name}: a = {a:.4f} /uJ, E_sat = {E_sat:.1f} uJ, plateau a*E_sat = {a * E_sat:.2f}")
    print("fluence exponents p (fit over all pulse energies):")
    for label, p in slopes.items():
        print(f"  {label:28s} {p:.2f}")


if __name__ == "__main__":
    main()
