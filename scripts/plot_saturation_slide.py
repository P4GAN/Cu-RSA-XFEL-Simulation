"""Three-panel saturation slide figure for the full (double-satellite + L2) SASE model.

  left   transmittance spectra at every pulse energy, overlaid
  middle resonant dip absorbance vs pulse energy (Kalpha1/Kalpha2 simulation, Kalpha1 experiment)
  right  fluence scaling of every tracked manifold (as plot_double_satellite_diagnostics.py, now
         including the 0.12 and 0.5 uJ runs)

Data, metric definitions and fits are shared with plot_saturation.py. Sized to sit beside a text
sidebar on a 16:9 slide (~10 in wide). Writes figs/saturation_slide.png.
Run from the repo root:  python scripts/plot_saturation_slide.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_saturation import (FIGS, SIM_DIRS, ENERGY_OFFSET_EV, KA1_WINDOW, KA2_WINDOW, EXP_NOISE_FLOOR_UJ,
                             E_RAMP, KA1_COLOUR, KA2_COLOUR, EXP_COLOUR, INK, INK_MUTED,
                             load_sweeps, load_experiment, resonant_absorbance, dip_absorbance,
                             exp_absorbance, fit_saturable, saturable)

# Manifold colours, in the order plot_double_satellite_diagnostics.py assigns them, so this panel
# matches the existing slide figure.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
E_TICKS = [0.12, 0.5, 2, 9, 40, 60]
LEGEND_BELOW = dict(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)

plt.rcParams.update({
    "font.size": 10.5, "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 8.8, "lines.linewidth": 1.8, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "axes.edgecolor": "#b9bbc0",
    "axes.spines.top": False, "axes.spines.right": False,
})


def log_energy_axis(ax, lim=(0.09, 80)):
    ax.set_xscale("log")
    ax.set_xlim(*lim)
    ax.set_xticks(E_TICKS)
    ax.set_xticklabels([f"{v:g}" for v in E_TICKS])
    # 40 and 60 sit ~0.18 decades apart: push their labels away from each other
    labels = ax.get_xticklabels()
    labels[-2].set_horizontalalignment("right")
    labels[-1].set_horizontalalignment("left")
    ax.minorticks_off()
    ax.set_xlabel("Pulse energy (µJ)")


def panel_spectra(ax, sim, E_sim):
    for e, colour in zip(E_sim, E_RAMP):
        run = sim[e]
        E = ENERGY_OFFSET_EV + run["womega_ar"]
        ax.plot(E, run["I_int_thy_w_last"] / run["I_int_thy_w_0"], color=colour, lw=1.5, label=f"{e:g} µJ")
    for centre, name in ((8045.4, r"K$\alpha_1$"), (8023.5, r"K$\alpha_2$")):
        ax.text(centre, 0.438, name, ha="center", va="top", fontsize=10, color=INK)
    ax.set_xlim(8010, 8062)
    ax.set_ylim(0.18, 0.445)
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel(r"Transmittance $T(\omega)$")
    ax.legend(title="Pulse energy", title_fontsize=8.8, **LEGEND_BELOW, ncol=3, columnspacing=1.0,
              handlelength=1.4)


def panel_absorbance(ax, sim, E_sim):
    A1, A2 = [], []
    for e in E_sim:
        E, A = resonant_absorbance(sim[e])
        A1.append(dip_absorbance(E, A, KA1_WINDOW)[0])
        A2.append(dip_absorbance(E, A, KA2_WINDOW)[0])
    A1, A2 = np.array(A1), np.array(A2)
    fit1, _ = fit_saturable(E_sim, A1)
    fit2, _ = fit_saturable(E_sim, A2)

    exp, T_cold = load_experiment()
    table = pd.DataFrame({e: exp_absorbance(E, T, dT, T_cold) for e, (E, T, dT) in exp.items()},
                         index=["A", "dA"]).T.sort_index()
    fitted = table[table.index > EXP_NOISE_FLOOR_UJ]
    floor = table[table.index <= EXP_NOISE_FLOOR_UJ]
    fit_exp, _ = fit_saturable(fitted.index.values, fitted["A"].values)

    E_fine = np.geomspace(0.09, 80, 300)
    ax.plot(E_fine, fit1[0] * E_fine, color=INK_MUTED, lw=1.0, ls=(0, (2, 2)), zorder=1,
            label=r"$\propto E$ (no saturation)")
    for fit, colour in ((fit1, KA1_COLOUR), (fit2, KA2_COLOUR), (fit_exp, EXP_COLOUR)):
        ax.plot(E_fine, saturable(E_fine, *fit), color=colour, lw=1.2, alpha=0.6, zorder=2)

    ax.plot(E_sim, A1, ls="none", marker="o", ms=7, color=KA1_COLOUR, mec="white", mew=1.2, zorder=4,
            label=r"K$\alpha_1$, simulation")
    ax.plot(E_sim, A2, ls="none", marker="s", ms=6.5, color=KA2_COLOUR, mec="white", mew=1.2, zorder=4,
            label=r"K$\alpha_2$, simulation")
    ax.errorbar(fitted.index, fitted["A"], yerr=fitted["dA"], fmt="D", ms=6, color=EXP_COLOUR, mec="white",
                mew=1.0, elinewidth=1.0, capsize=0, zorder=5, label=r"K$\alpha_1$, experiment")
    ax.errorbar(floor.index, floor["A"], yerr=floor["dA"], fmt="D", ms=6, mfc="white", mec=EXP_COLOUR,
                color=EXP_COLOUR, elinewidth=1.0, capsize=0, zorder=5)
    ax.annotate("noise\nfloor", (floor.index[0], floor["A"].iloc[0]), xytext=(0, 13), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, color=INK_MUTED, linespacing=0.95)

    ax.set_yscale("log")
    ax.set_ylim(0.004, 2.5)
    ax.set_ylabel(r"Dip absorbance  $-\ln(T_\mathrm{dip}/T_\mathrm{wing})$")
    log_energy_axis(ax)
    ax.plot([], [], color=INK_MUTED, lw=1.2, alpha=0.8, label="saturable fit")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(k) for k in (r"K$\alpha_1$, simulation", r"K$\alpha_2$, simulation",
                                       r"K$\alpha_1$, experiment", "saturable fit",
                                       r"$\propto E$ (no saturation)")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], **LEGEND_BELOW, handlelength=1.8)
    return fit1, fit2, fit_exp


def panel_manifolds(ax, sim, E_sim):
    names = [str(n) for n in sim[E_sim[-1]]["satellite_channel_names"]]
    dt = sim[E_sim[-1]]["t_axis"][1] - sim[E_sim[-1]]["t_axis"][0]

    def integrated(key, row=None):
        return np.array([(sim[e][key] if row is None else sim[e][key][row]).sum() * dt for e in E_sim])

    series = [(r"$1s^{-1}$ (K)", integrated("rho_ee_t_last"), "#e34948", "-", "D", 2.4),
              (r"$2p_{3/2}^{-1}$ (L3)", integrated("rho_l3_t_last"), "#111111", "-", "o", 2.4),
              (r"$2p_{1/2}^{-1}$ (L2)", integrated("rho_l2_t_last"), "#111111", "--", "o", 1.6)]
    for i, name in enumerate(names):
        series.append((f"${name}$", integrated("rho_l3_t_last_sat", i), PALETTE[i % len(PALETTE)],
                       "-" if len(name) <= 3 else "--", "s" if len(name) <= 3 else "^", 1.2))

    exponents = {}
    for label, values, colour, ls, marker, lw in series:
        local = np.diff(np.log(values)) / np.diff(np.log(E_sim))
        exponents[label] = (np.polyfit(np.log(E_sim), np.log(values), 1)[0], local)
        # K is sequential (hole creation, then resonant pumping), so its exponent runs from ~2 at low
        # fluence down as the Kalpha1 transition saturates; a single fitted p would hide that.
        p_text = (rf"$p$: {local[0]:.1f}$\to${local[-1]:.1f}" if local[0] - local[-1] > 0.3
                  else rf"$p$ = {exponents[label][0]:.2f}")
        ax.plot(E_sim, values, color=colour, ls=ls, marker=marker, lw=lw, ms=4.2, label=f"{label}  {p_text}")

    guide = np.array([0.12, 9.0])
    ax.plot(guide, 1.5e-4 * guide / 0.12, color=INK_MUTED, lw=1.0, ls=(0, (2, 2)), zorder=0)
    ax.text(0.5, 1.5e-4 * 0.5 / 0.12 * 1.6, r"$\propto E$", color=INK_MUTED, fontsize=9.5,
            ha="right", va="bottom")

    ax.set_yscale("log")
    ax.set_ylabel("Time-integrated population (arb.)")
    log_energy_axis(ax)
    ax.legend(**LEGEND_BELOW, ncol=2, fontsize=8.2,
              columnspacing=0.8, handlelength=2.0, handletextpad=0.5, labelspacing=0.35)
    return exponents


def main():
    sim = load_sweeps(SIM_DIRS)
    E_sim = np.array(list(sim))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.2, 3.9))
    panel_spectra(ax1, sim, E_sim)
    fit1, fit2, fit_exp = panel_absorbance(ax2, sim, E_sim)
    exponents = panel_manifolds(ax3, sim, E_sim)
    fig.tight_layout(w_pad=1.6)

    path = os.path.join(FIGS, "saturation_slide.png")
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"wrote {os.path.relpath(path)}")

    print(f"\nsaturable fit E_sat: sim Ka1 {fit1[1]:.1f} uJ, sim Ka2 {fit2[1]:.1f} uJ, exp Ka1 {fit_exp[1]:.1f} uJ")
    print("fluence exponents (fit over all energies; local slopes between neighbouring energies):")
    for label, (p, local) in exponents.items():
        print(f"  {label:24s} p = {p:.2f}   local {np.round(local, 2)}")


if __name__ == "__main__":
    main()
