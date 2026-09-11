"""Saturation of the reverse-saturable-absorption (RSA) dip, and why it happens.

Full model (double satellite + L2), SASE seed, 0.12-60 uJ:
  2/9/40/60 uJ : data/production_sweep_sase_24437070/Cu-seed-SASE-double-satellite
  0.12/0.5 uJ  : data/linear_response_sweep_sase_24533426/Cu-seed-SASE-double-satellite
                 (full-MB runs; that job's *-linear folder is NOT linear, see plot_rate_eq_vs_mb.py)
Experiment: SASE pulse centred ~8045 eV (transmittance_2_9_40uJ.csv, transmittance_bottom_extra_uJ.csv).

Figures written to figs/:
  saturation_dip_vs_energy   dip absorbance vs pulse energy, saturable fit, per-uJ efficiency
  saturation_spectral        absorbance spectrum per uJ: collapse at low E, burnt-out, broadened core at high E
  saturation_mechanism       1s-hole population catching up with the 2p3/2-hole population (bleaching)
  saturation_cycle_diagram   level scheme of the absorb / stimulated-emission / Auger cycle

Definitions
  A(w)   = -ln(T(w) / T_wing), T_wing = median T over 8005-8015 eV at the same pulse energy, so A is
           the resonant part only (the photoionization background is divided out).
  A_dip  = max of A(w) (0.25 eV boxcar) in the Kalpha1 (8040-8052 eV) / Kalpha2 (8018-8030 eV) window.
  bleach = (2/3) rho_K / rho_2p3/2 from the exit-face, beam-centre traces. rho_l3_t_last and rho_ee_t_last
           are dipole-weighted (Tr T+ rho T-); each 1s sublevel has 2/3 of its dipole weight on 2p3/2 and 1/3 on
           2p1/2 (Tijs_plus), so (2/3) rho_K is the 1s population seen by Kalpha1: net Kalpha1 absorption
           ~ rho_2p3/2 - (2/3) rho_K, and bleach = 1 is a transparent transition.

Energy axis: 8045 + womega_ar, the calibration used by the slide notebooks and plot_rate_eq_vs_mb.py.
Run from the repo root:  python scripts/plot_saturation.py
"""

import os
import glob

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import curve_fit

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
FIGS = os.path.join(REPO, "figs")

SIM_DIRS = [os.path.join(DATA, "production_sweep_sase_24437070", "Cu-seed-SASE-double-satellite"),
            os.path.join(DATA, "linear_response_sweep_sase_24533426", "Cu-seed-SASE-double-satellite")]

ENERGY_OFFSET_EV = 8045.0
KA1_WINDOW = (8040.0, 8052.0)
KA2_WINDOW = (8018.0, 8030.0)
WING_WINDOW = (8005.0, 8015.0)
EXP_COLD_WINDOW = (8030.0, 8065.0)
EXP_NOISE_FLOOR_UJ = 0.12   # dip at 0.12 uJ is below the measurement noise; shown, not fitted

# Pulse energy at which the resonance-filtered peak Rabi frequency equals the Kalpha1 coherence width
# (Gamma_K + Gamma_L3)/2 = 1.05 eV at the entrance-face centre pixel -- see plot_rate_eq_vs_mb.py.
E_RABI_EQUALS_GAMMA_UJ = 3.6

# Ordinal blue ramp for pulse energy (light = weak, dark = strong), validated with the dataviz
# validator (--ordinal, light surface).
E_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0b2a55"]
KA1_COLOUR, KA2_COLOUR, K_COLOUR = "#2a78d6", "#eb6834", "#1baf7a"
EXP_COLOUR, INK, INK_MUTED, PULSE_FILL = "#23262f", "#23262f", "#6b6e76", "#e4e4e1"

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5,
    "lines.linewidth": 2.0, "figure.dpi": 120, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "axes.edgecolor": "#b9bbc0",
    "axes.spines.top": False, "axes.spines.right": False,
})


# =============================================================================
# loading
# =============================================================================
def load_sweeps(dirs):
    """{E_seed_uJ: element-wise means over every chunk} for every runs_seed_* folder in dirs."""
    out = {}
    for d in dirs:
        for run_dir in glob.glob(os.path.join(d, "runs_seed_*_uJ")):
            # folder names round E_seed to 1 decimal (0.12 -> runs_seed_0.1_uJ); the YAML is exact
            cfg = yaml.safe_load(open(glob.glob(os.path.join(run_dir, "*.yaml"))[0]))
            totals, axes, n_reps = {}, {}, 0
            for path in glob.glob(os.path.join(run_dir, "*.npz")):
                z = np.load(path, allow_pickle=True)
                n_reps += int(z["n_reps"])
                for key in z.files:
                    if key.endswith("_sum"):
                        stem = key[:-4]
                        s, c = totals.get(stem, (0, 0))
                        totals[stem] = (s + z[key], c + z[f"{stem}_count"])
                    elif not (key.endswith("_sumsq") or key.endswith("_count")):
                        axes[key] = z[key]
            run = {stem: np.real(s / np.maximum(c, 1)) for stem, (s, c) in totals.items()}
            run.update(axes)
            run["n_reps"] = n_reps
            out[float(cfg["E_seed_uJ"])] = run
    return dict(sorted(out.items()))


def load_experiment():
    """{E_seed_uJ: (energy_eV, T, T_err)} and the cold-foil T, SASE centred ~8045 eV."""
    a = pd.read_csv(os.path.join(DATA, "transmittance_2_9_40uJ.csv"))
    b = pd.read_csv(os.path.join(DATA, "transmittance_bottom_extra_uJ.csv"))
    exp = {}
    for df, labels in ((a, ["40", "9", "2"]), (b, ["0.5", "0.12"])):
        for lab in labels:
            exp[float(lab)] = (df["Photon_energy_eV"].values, df[f"{lab}uJ_transmittance"].values,
                               df[f"{lab}uJ_error"].values)
    cold = b["Cold_transmittance"].values[in_window(b["Photon_energy_eV"].values, EXP_COLD_WINDOW)]
    return dict(sorted(exp.items())), float(np.median(cold))


# =============================================================================
# metrics
# =============================================================================
def in_window(E, window):
    return (E >= window[0]) & (E <= window[1])


def smooth_ev(E, y, half_width_eV=0.25):
    """Boxcar in photon energy, so sim (0.09 eV) and experiment (0.1-0.4 eV) grids smooth alike."""
    return np.array([y[np.abs(E - e) <= half_width_eV].mean() for e in E])


def sim_spectrum(run):
    E = ENERGY_OFFSET_EV + run["womega_ar"]
    T = run["I_int_thy_w_last"] / run["I_int_thy_w_0"]
    return E, T, float(np.median(T[in_window(E, WING_WINDOW)]))


def resonant_absorbance(run, half_width_eV=0.25):
    E, T, T_wing = sim_spectrum(run)
    return E, -np.log(smooth_ev(E, T, half_width_eV) / T_wing)


def dip_absorbance(E, A, window):
    m = in_window(E, window)
    i = np.argmax(A[m])
    return float(A[m][i]), float(E[m][i])


def dip_fwhm(E, A, window):
    m = in_window(E, window)
    peak = A[m].max()
    above = E[m][A[m] >= peak / 2]
    return float(above.max() - above.min())


def exp_absorbance(E, T, T_err, T_cold, window=KA1_WINDOW):
    m = in_window(E, window)
    Ts = smooth_ev(E[m], T[m])
    i = np.argmin(Ts)
    return float(-np.log(Ts[i] / T_cold)), float(T_err[m][i] / Ts[i])


def saturable(E, a, E_sat):
    """Homogeneous saturation of a fluence-created absorber: linear a*E below E_sat, a*E_sat above."""
    return a * E / (1.0 + E / E_sat)


def fit_saturable(E, A):
    p, cov = curve_fit(lambda lE, a, Es: np.log(saturable(np.exp(lE), a, Es)),
                       np.log(E), np.log(A), p0=[A[0] / E[0], 5.0])
    return p, np.sqrt(np.diag(cov))


def exit_face_traces(run):
    t = run["t_axis"]
    l3 = run["rho_l3_t_last"]
    k_seen = (2.0 / 3.0) * run["rho_ee_t_last"]
    return t, l3, k_seen


def bleach_fractions(run):
    """(peak, 2p3/2-weighted pulse average) of (2/3) rho_K / rho_2p3/2 at the exit face."""
    t, l3, k_seen = exit_face_traces(run)
    live = l3 > 0.1 * l3.max()
    ratio = k_seen[live] / l3[live]
    return float(ratio.max()), float((ratio * l3[live]).sum() / l3[live].sum())


# =============================================================================
# figures
# =============================================================================
def log_energy_axis(ax, ticks, lim=(0.085, 90)):
    ax.set_xscale("log")
    ax.set_xlim(*lim)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:g}" for v in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Pulse energy (µJ)")


def rabi_marker(ax, label=True):
    ax.axvline(E_RABI_EQUALS_GAMMA_UJ, color=INK_MUTED, lw=1.0, ls=":", zorder=0)
    if label:
        ax.text(E_RABI_EQUALS_GAMMA_UJ * 0.9, 0.975, r"$\Omega_\mathrm{Rabi}=\Gamma$",
                transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=10, color=INK_MUTED)


def plot_dip_vs_energy(E_sim, A1, A2, fit1, fit2, exp_table, fit_exp, l3_per_uJ):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0), gridspec_kw={"width_ratios": [1.25, 1]})
    E_fine = np.geomspace(0.085, 90, 300)
    ticks = [0.12, 0.5, 2, 9, 40, 60]

    # --- left: absorbance vs pulse energy ---
    for (a, Es), A, colour, marker, name in ((fit1, A1, KA1_COLOUR, "o", r"K$\alpha_1$"),
                                            (fit2, A2, KA2_COLOUR, "s", r"K$\alpha_2$")):
        ax.plot(E_fine, a * E_fine, color=colour, lw=1.2, ls=(0, (2, 2)), alpha=0.7, zorder=1)
        ax.plot(E_fine, saturable(E_fine, a, Es), color=colour, lw=1.6, alpha=0.9, zorder=2)
        ax.plot(E_sim, A, ls="none", marker=marker, ms=8.5, color=colour, mec="white", mew=1.4, zorder=4,
                label=f"{name} dip, simulation")
        ax.plot([Es], [saturable(Es, a, Es)], marker="|", ms=16, mew=2.0, color=colour, zorder=3)
        ax.annotate(rf"$E_\mathrm{{sat}}$ = {Es:.0f} µJ", (Es, saturable(Es, a, Es)), xytext=(8, -14),
                    textcoords="offset points", fontsize=10, color=INK)

    fitted = exp_table[exp_table.index > EXP_NOISE_FLOOR_UJ]
    floor = exp_table[exp_table.index <= EXP_NOISE_FLOOR_UJ]
    ax.errorbar(fitted.index, fitted["A"], yerr=fitted["dA"], fmt="D", ms=7, color=EXP_COLOUR,
                mec="white", mew=1.1, elinewidth=1.1, capsize=0, zorder=5,
                label=r"K$\alpha_1$ dip, experiment")
    ax.errorbar(floor.index, floor["A"], yerr=floor["dA"], fmt="D", ms=7, mfc="white", mec=EXP_COLOUR,
                color=EXP_COLOUR, elinewidth=1.1, capsize=0, zorder=5)
    ax.plot(E_fine, saturable(E_fine, *fit_exp), color=EXP_COLOUR, lw=1.1, alpha=0.55, zorder=1)

    ax.plot([], [], color=INK_MUTED, lw=1.6, label=r"fit  $aE\,/\,(1+E/E_\mathrm{sat})$")
    ax.plot([], [], color=INK_MUTED, lw=1.2, ls=(0, (2, 2)), label=r"$\propto E$  (no saturation)")
    ax.annotate(rf"exp. $E_\mathrm{{sat}}$ = {fit_exp[1]:.0f} µJ", (40, saturable(40, *fit_exp)), xytext=(-10, 12),
                textcoords="offset points", ha="center", fontsize=10, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(0.004, 3.5)
    ax.set_ylabel(r"Resonant dip absorbance  $-\ln(T_\mathrm{dip}/T_\mathrm{wing})$")
    ax.set_title("RSA dip grows linearly, then saturates")
    log_energy_axis(ax, ticks)
    rabi_marker(ax)
    ax.legend(loc="upper left", frameon=False)

    # --- right: per-uJ efficiency, normalised to the linear-regime slope ---
    ax2.axhline(1.0, color=INK_MUTED, lw=0.9, zorder=0)
    ax2.plot(E_sim, l3_per_uJ / l3_per_uJ[0], color=INK_MUTED, ls="--", marker="^", ms=7.5,
             mec="white", mew=1.2, lw=1.6, label=r"$2p_{3/2}$ holes per µJ (absorber supply)")
    for (a, Es), A, colour, marker, name in ((fit1, A1, KA1_COLOUR, "o", r"K$\alpha_1$"),
                                            (fit2, A2, KA2_COLOUR, "s", r"K$\alpha_2$")):
        ax2.plot(E_fine, 1.0 / (1.0 + E_fine / Es), color=colour, lw=1.4, alpha=0.8)
        ax2.plot(E_sim, A / E_sim / a, ls="none", marker=marker, ms=8.5, color=colour, mec="white", mew=1.4,
                 label=f"{name} absorbance per µJ")
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel("Per µJ, relative to low-fluence limit")
    ax2.set_title("Absorbers keep coming; each absorbs less")
    log_energy_axis(ax2, ticks)
    rabi_marker(ax2, label=False)
    ax2.legend(loc="lower left", frameon=False)
    ax2.annotate(f"{100 * l3_per_uJ[-1] / l3_per_uJ[0]:.0f}%", (E_sim[-1], l3_per_uJ[-1] / l3_per_uJ[0]),
                 xytext=(7, 0), textcoords="offset points", va="center", fontsize=10, color=INK)
    ax2.annotate(f"{100 * A1[-1] / E_sim[-1] / fit1[0]:.0f}%", (E_sim[-1], A1[-1] / E_sim[-1] / fit1[0]),
                 xytext=(7, 0), textcoords="offset points", va="center", fontsize=10, color=INK)

    fig.tight_layout()
    save(fig, "saturation_dip_vs_energy")


def plot_spectral(sim, E_sim):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.9), gridspec_kw={"width_ratios": [1.55, 1]})

    run = sim[E_sim[-1]]
    E_axis = ENERGY_OFFSET_EV + run["womega_ar"]
    spec = run["I_int_thy_w_0"]
    for e, colour in zip(E_sim, E_RAMP):
        E, A = resonant_absorbance(sim[e], half_width_eV=0.35)
        m = in_window(E, (8010, 8060))
        ax.plot(E[m], A[m] / e, color=colour, lw=1.9, label=f"{e:g} µJ")
    ymax = 0.118
    ax.fill_between(E_axis, 0, 0.55 * ymax * spec / spec.max(), color=PULSE_FILL, zorder=0, lw=0,
                    label="incident spectrum (arb.)")
    for centre, name in ((8045.4, r"K$\alpha_1$"), (8023.5, r"K$\alpha_2$")):
        ax.text(centre, ymax * 0.97, name, ha="center", va="top", fontsize=11, color=INK)
    ax.set_xlim(8012, 8058)
    ax.set_ylim(-0.004, ymax)
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel(r"Resonant absorbance per µJ,  $A(\omega)/E$")
    ax.set_title("Linear regime: curves collapse. Saturated: the core is burnt out")
    ax.legend(loc="upper center", frameon=False, ncol=1, fontsize=10, bbox_to_anchor=(0.46, 1.0))

    # --- right: Kalpha1 dip shape, peak-normalised (power broadening) ---
    for e, colour in zip(E_sim, E_RAMP):
        E, A = resonant_absorbance(sim[e], half_width_eV=0.35)
        peak, centre = dip_absorbance(E, A, KA1_WINDOW)
        m = in_window(E, (8036, 8056))
        width = dip_fwhm(E, A, KA1_WINDOW)
        ax2.plot(E[m], A[m] / peak, color=colour, lw=1.9, label=f"{e:g} µJ: FWHM {width:.1f} eV")
    ax2.axhline(0.5, color=INK_MUTED, lw=0.8, ls=":", zorder=0)
    ax2.set_xlim(8036, 8056)
    ax2.set_xticks([8040, 8045, 8050, 8055])
    ax2.set_xticklabels(["8040", "8045", "8050", "8055"])
    ax2.set_ylim(-0.05, 1.08)
    ax2.set_xlabel("Photon energy (eV)")
    ax2.set_ylabel("Absorbance / dip peak")
    ax2.set_title(r"K$\alpha_1$ dip broadens as its core saturates")
    ax2.legend(loc="upper right", frameon=False, fontsize=9.5)

    fig.tight_layout()
    save(fig, "saturation_spectral")


def plot_mechanism(sim, E_sim, bleach):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7), gridspec_kw={"width_ratios": [1, 1, 1.15]})

    for ax, e in zip(axes[:2], (E_sim[0], E_sim[-1])):
        run = sim[e]
        t, l3, k_seen = exit_face_traces(run)
        pulse = np.abs(run["I_t_0"])
        ax.fill_between(t, 0, pulse / pulse.max() * 1.0, color=PULSE_FILL, lw=0, zorder=0,
                        label="incident pulse (arb.)")
        ax.plot(t, l3 / l3.max(), color=KA1_COLOUR, lw=2.4, label=r"$2p_{3/2}$ holes (absorbers)")
        ax.plot(t, k_seen / l3.max(), color=K_COLOUR, lw=2.4,
                label=r"$1s$ holes, $\frac{2}{3}\rho_K$ (re-emitters)")
        peak, _ = bleach[e]
        ax.set_xlim(4, 19)
        ax.set_ylim(0, 1.12)
        ax.set_xlabel("Time (fs)")
        ax.set_title(f"{e:g} µJ: " + (r"$1s$ level stays empty" if peak < 0.1 else r"$1s$ level fills up"))
        ax.text(0.97, 0.97, f"peak ratio {peak:.2g}", transform=ax.transAxes, ha="right", va="top",
                fontsize=10.5, color=INK)
    axes[0].set_ylabel(r"Population / peak $2p_{3/2}$ (exit face, beam centre)")
    handles, labels = axes[0].get_legend_handles_labels()

    ax = axes[2]
    peaks = np.array([bleach[e][0] for e in E_sim])
    means = np.array([bleach[e][1] for e in E_sim])
    ax.axhline(1.0, color=INK_MUTED, lw=0.9, ls="--", zorder=0)
    ax.text(0.1, 0.97, "fully bleached (transparent)", fontsize=10, color=INK_MUTED, va="top")
    ax.plot(E_sim, peaks, marker="o", ms=8.5, color=K_COLOUR, mec="white", mew=1.4, label="peak during pulse")
    ax.plot(E_sim, means, marker="o", ms=8.5, color=K_COLOUR, mfc="white", mew=1.6, ls="--", lw=1.6,
            label="pulse average")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(r"Bleaching  $\frac{2}{3}\rho_K\,/\,\rho_{2p_{3/2}}$")
    ax.set_title(r"K$\alpha_1$ transition bleaches with fluence")
    log_energy_axis(ax, [0.12, 0.5, 2, 9, 40, 60])
    rabi_marker(ax)
    ax.legend(loc="center left", frameon=False)

    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.legend(handles, labels, loc="lower left", ncol=3, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.04, 0.0))
    save(fig, "saturation_mechanism")


def plot_cycle_diagram():
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    def level(x0, x1, y, text, colour, sub=None):
        ax.plot([x0, x1], [y, y], color=colour, lw=4.0, solid_capstyle="round")
        ax.text(x0 - 0.15, y, text, ha="right", va="center", fontsize=12.5, color=INK)
        if sub:
            ax.text(x0 - 0.15, y - 0.32, sub, ha="right", va="center", fontsize=10, color=INK_MUTED)

    def arrow(p0, p1, colour, style="-|>", lw=2.0, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16, color=colour, lw=lw,
                                     linestyle=ls, connectionstyle=f"arc3,rad={rad}"))

    level(2.5, 4.3, 0.7, "neutral Cu", INK_MUTED, r"$1s^2\,2p^6$")
    level(2.5, 4.3, 2.3, r"$2p_{3/2}$ hole", KA1_COLOUR, "the RSA absorber")
    level(2.5, 4.3, 5.1, r"$1s$ hole", K_COLOUR, r"$\tau_K$ = 0.44 fs")

    # creation of the absorber
    arrow((3.0, 0.75), (3.0, 2.25), INK_MUTED)
    ax.text(2.9, 1.5, "photoionization\n" + r"$\propto$ fluence", ha="right", va="center", fontsize=10.5,
            color=INK)

    # resonant absorption / stimulated emission
    arrow((3.55, 2.35), (3.55, 5.05), KA1_COLOUR, lw=2.6)
    arrow((3.95, 5.05), (3.95, 2.35), K_COLOUR, lw=2.6)
    ax.text(3.45, 3.95, r"K$\alpha_1$ absorption" + "\nrate $W\\propto I$", ha="right", va="center",
            fontsize=10.5, color=INK)
    ax.text(4.08, 3.55, "stimulated emission,\nsame rate $W$:\nphoton returned\nto the beam",
            ha="left", va="center", fontsize=10.5, color=INK)

    # losses from the 1s hole
    arrow((4.35, 5.1), (6.1, 5.1), INK_MUTED)
    ax.text(6.2, 5.1, "KLL Auger (60%) or K$\\alpha$ fluorescence\ninto $4\\pi$ (40%): photon lost from beam",
            ha="left", va="center", fontsize=10.5, color=INK)
    # loss from the 2p hole
    arrow((4.35, 2.3), (6.1, 2.3), INK_MUTED)
    ax.text(6.2, 2.3, r"$L_3$ Auger, $\tau$ = 1.1 fs:" + "\nabsorber lost",
            ha="left", va="center", fontsize=10.5, color=INK)

    ax.text(0.15, 5.85, "Why the RSA dip saturates", fontsize=14.5, color=INK, va="top", weight="bold")
    ax.text(6.2, 4.5,
            r"$W \ll \Gamma$:  $1s$ level empty, every excitation" + "\n"
            + r"is absorbed, $A \propto$ (number of holes) $\propto E$" + "\n\n"
            + r"$W \gtrsim \Gamma$:  $1s$ fills to $\sim 2p_{3/2}$, stimulated" + "\n"
            + "emission cancels absorption; each hole\n"
            + r"absorbs only $\approx 1$ photon before it decays," + "\n"
            + r"so absorbed / incident $\to$ const.",
            ha="left", va="top", fontsize=10.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f3", ec="#d5d5d1"))

    save(fig, "saturation_cycle_diagram")


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{stem}.{ext}"), dpi=200)
    plt.close(fig)
    print(f"wrote figs/{stem}.png/.pdf")


# =============================================================================
# main
# =============================================================================
def main():
    os.makedirs(FIGS, exist_ok=True)
    sim = load_sweeps(SIM_DIRS)
    E_sim = np.array(list(sim))

    A1, A2, W1, W2 = [], [], [], []
    for e in E_sim:
        E, A = resonant_absorbance(sim[e])
        A1.append(dip_absorbance(E, A, KA1_WINDOW)[0])
        A2.append(dip_absorbance(E, A, KA2_WINDOW)[0])
        W1.append(dip_fwhm(E, A, KA1_WINDOW))
        W2.append(dip_fwhm(E, A, KA2_WINDOW))
    A1, A2 = np.array(A1), np.array(A2)
    fit1, err1 = fit_saturable(E_sim, A1)
    fit2, err2 = fit_saturable(E_sim, A2)

    exp, T_cold = load_experiment()
    exp_rows = {e: exp_absorbance(E, T, dT, T_cold) for e, (E, T, dT) in exp.items()}
    exp_table = pd.DataFrame(exp_rows, index=["A", "dA"]).T.sort_index()
    fitted = exp_table[exp_table.index > EXP_NOISE_FLOOR_UJ]
    fit_exp, err_exp = fit_saturable(fitted.index.values, fitted["A"].values)

    dt = sim[E_sim[0]]["t_axis"][1] - sim[E_sim[0]]["t_axis"][0]
    l3_per_uJ = np.array([sim[e]["rho_l3_t_last"].sum() * dt / e for e in E_sim])
    bleach = {e: bleach_fractions(sim[e]) for e in E_sim}

    plot_dip_vs_energy(E_sim, A1, A2, fit1, fit2, exp_table, fit_exp, l3_per_uJ)
    plot_spectral(sim, E_sim)
    plot_mechanism(sim, E_sim, bleach)
    plot_cycle_diagram()

    print(f"\nrepetitions: { {e: sim[e]['n_reps'] for e in E_sim} }")
    print("\n  E_uJ   A_Ka1   A_Ka2   A_Ka1/E  FWHM_Ka1  FWHM_Ka2  2p3/2/uJ(rel)  bleach_peak  bleach_avg")
    for i, e in enumerate(E_sim):
        print(f"  {e:5g}  {A1[i]:.4f}  {A2[i]:.4f}  {A1[i] / e:.4f}   {W1[i]:5.2f}     {W2[i]:5.2f}     "
              f"{l3_per_uJ[i] / l3_per_uJ[0]:.3f}          {bleach[e][0]:.3f}        {bleach[e][1]:.3f}")
    print(f"\nsaturable fit A = a E / (1 + E/E_sat):")
    for name, p, dp in (("sim Ka1", fit1, err1), ("sim Ka2", fit2, err2), ("exp Ka1", fit_exp, err_exp)):
        print(f"  {name}: a = {p[0]:.4f} +- {dp[0]:.4f} /uJ,  E_sat = {p[1]:.2f} +- {dp[1]:.2f} uJ,  "
              f"A_sat = a E_sat = {p[0] * p[1]:.3f}")
    print(f"\nexperiment (cold T = {T_cold:.3f}; {EXP_NOISE_FLOOR_UJ} uJ excluded from fit):")
    print(exp_table.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
