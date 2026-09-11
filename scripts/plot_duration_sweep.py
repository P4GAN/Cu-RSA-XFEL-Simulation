"""Analysis plots for a transmittance-vs-seed-duration sweep (run_duration_sweep.py output).

Reads every runs_duration_<fs>_fs/ chunk file of one sweep and writes into figs/<sweep name>/:

  1. duration_summary       energy transmission, resonantly created 1s holes, Kalpha1 dip
                            depth and width, each vs seed duration
  2. duration_spectra       transmittance vs absolute photon energy (E_Kalpha1 + womega), one
                            line per duration
  3. duration_time_domain   time-resolved T(t) through the pulse, and the resonant
                            free-induction-decay tail left behind by the shortest pulses
  4. duration_populations   exit-face 2p3/2-, 2p1/2- and 1s-hole populations (main line vs
                            spectator satellites) through each pulse, one row of panels
  5. duration_slide_*       2 and 4 at their true size on the Beamer slide (copy the PDFs into
                            ../Presentation/ for the "Dependence on pulse duration" bonus slide)

Durations in DROP_FROM_SPECTRA_AND_POPULATIONS (0.1 fs) are left out of 2, 4 and 5.

Run from the repo root:  python scripts/plot_duration_sweep.py [--data data/duration_sweep_<id>]

Conventions / caveats baked in here:
- womega_ar is the detuning from the shared Kalpha1 rotating frame (XLO_sim.Delta_ij), so
  0 eV is Kalpha1 = hwKalpha1N; the SASE seed envelope is centred there too (seed_center_E
  only sets ocelot's xlamds). Line positions below are placed on that axis.
- The stored z=-1 field is the state after the second-to-last z step (the documented
  off-by-one reproduced by Sample._evaluate_n_level_3D_lean), i.e. after (zgrid-2) absorbing
  steps. The "cold" reference transmission (no core holes) uses that same path length, so
  T/T_cold isolates the core-hole (RSA) contribution.
- rho_ground/2s/other_t_last are read at the transverse corner in lean mode (see
  plot_double_satellite_diagnostics.py), so they are not used. Block traces (rho_ee = 1s
  hole, rho_l3 = 2p3/2 hole) are at the centre pixel of the last z plane.
- Error bars are the standard error over the independent 75-repetition chunks.
"""

import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FixedFormatter, MaxNLocator, NullFormatter, NullLocator

from XLO_sim.XLO_sim import XLO_sim

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MAIN, SAT = "#2a78d6", "#eb6834"                     # categorical slots 1, 2
# Ordinal blue ramp, one step per duration (light = short, dark = long); validated with the
# dataviz skill's validate_palette.js --ordinal.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0b2a55"]
# Hole-type colours for the population figure (categorical slots 1-3; pure green #008300 fails the
# protan CVD check against orange, this aqua-green passes).
HOLE_2P3, HOLE_2P1, HOLE_1S = "#2a78d6", "#1baf7a", "#eb6834"

# Durations left out of the transmittance-spectra and population figures.
DROP_FROM_SPECTRA_AND_POPULATIONS = (0.1,)

# Slide versions are drawn at their true size on the Beamer 16:9 PaloAlto slide (\textwidth = 5.28 in),
# with the same fonts as plot_saturation_slide.py, so they can be included at width=\linewidth.
SLIDE_WIDTH_IN = 381.79 / 72.27
SLIDE_RC = {
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7.3,
    "ytick.labelsize": 7.3, "legend.fontsize": 7.3, "legend.title_fontsize": 7.3, "lines.linewidth": 1.1,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "grid.linewidth": 0.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "savefig.dpi": 400,
}

plt.rcParams.update({
    "font.size": 10.5, "axes.titlesize": 11, "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
    "lines.linewidth": 1.6, "figure.dpi": 130, "savefig.bbox": "tight", "savefig.dpi": 200,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "axes.titlelocation": "left", "axes.titleweight": "bold",
})

HBAR_EV_FS = 0.6582119569509067


def load_sweep(sweep_dir):
    """{duration: {"chunks": [mean dicts], "mean": pooled mean dict, "n_reps": int, "yaml": path}}."""
    out = {}
    for runs in sorted(glob.glob(os.path.join(sweep_dir, "runs_duration_*_fs"))):
        files = sorted(glob.glob(os.path.join(runs, "*.npz")))
        if any(f.endswith(".partial.npz") for f in files):
            print(f"note: {runs} contains .partial.npz checkpoints (incomplete chunks)")
        raw = [dict(np.load(f)) for f in files]
        keys = [k[:-4] for k in raw[0] if k.endswith("_sum")]
        axes = {k: v for k, v in raw[0].items() if not k.endswith(("_sum", "_sumsq", "_count")) and k != "n_reps"}
        chunks = [{**axes, **{k: r[f"{k}_sum"] / r[f"{k}_count"] for k in keys}} for r in raw]
        pooled = {**axes, **{k: sum(r[f"{k}_sum"] for r in raw) / sum(r[f"{k}_count"] for r in raw) for k in keys}}
        yaml_path = glob.glob(os.path.join(runs, "*.yaml"))[0]
        duration = float(os.path.basename(runs).split("_")[2])
        out[duration] = {"chunks": chunks, "mean": pooled, "n_reps": int(sum(r["n_reps"] for r in raw)),
                         "yaml": yaml_path}
    return dict(sorted(out.items()))


def sim_constants(yaml_path):
    X = XLO_sim(yaml_path)
    kappa = X.n * X.S_ground_Fi[1].sum() + X.n * X.sigma_compound_Ka1   # seed is in the s=1 component
    n_abs_steps = X.zgrid - 2                                          # see module docstring
    chans = X.satellite_channels   # already includes double_satellite_channels, in rho_*_sat row order
    names = [c["name"] for c in chans]
    # Upper-manifold (1s-hole) feeds between satellite blocks, in eV: feed_upper[dst, src]. These
    # convert an existing 1s hole into another block's 1s hole, so they must not be counted twice.
    feed_upper = np.zeros((len(chans), len(chans)))
    for k, c in enumerate(chans):
        for f in c.get("feed_from", []):
            if f.get("manifold", "lower") == "upper":
                feed_upper[k, names.index(f["channel"])] += f["Gamma_feed_eV"]
    return {
        "T_cold": float(np.exp(-kappa * X.dz * n_abs_steps)),
        "thickness_um": X.dz * n_abs_steps / 1e3,
        "Gamma_K": X.GammaKeVN, "Gamma_L3": X.GammaL3eVN,
        "Gamma_K_sat": np.array([c["Gamma_K_eV"] for c in chans]),
        "feed_upper": feed_upper,
        "Ka2": -(X.hwKalpha1N - X.hwKalpha2N),
        "E_Ka1": X.hwKalpha1N,
        "sat_lines": [(c["name"], c["detuning_eV"]) for c in chans],
        "E_seed_uJ": X.E_seed_uJ,
        # Effective area of the Gaussian seed spot, cm^2 (FWHMs are in nm).
        "spot_area_cm2": np.pi * X.seed_width_FWHM_x * X.seed_width_FWHM_y / (4 * np.log(2)) * 1e-14,
    }


def input_intensity_W_cm2(a, C):
    """Shot-averaged on-axis input intensity I(t) in W/cm^2: the pulse energy distributed over the
    mean temporal profile (I_t_0, spatially integrated) and the Gaussian spot's effective area."""
    t, P = a["t_axis"], np.real(a["I_t_0"])
    return C["E_seed_uJ"] * 1e-6 * P / np.trapz(P, t * 1e-15) / C["spot_area_cm2"]


def sci_unicode(x):
    """2.64e20 -> '2.6×10²⁰' (plain text: far more compact than mathtext at slide font sizes)."""
    exp = int(np.floor(np.log10(x)))
    return f"{x / 10 ** exp:.1f}×10" + str(exp).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


def metrics(a, C):
    """Scalar observables for one (chunk- or pool-) mean dict."""
    t, w = a["t_axis"], a["womega_ar"]
    I0, IL = np.real(a["I_t_0"]), np.real(a["I_t_last"])
    T = np.trapz(IL, t) / np.trapz(I0, t)

    Tw = a["I_int_thy_w_last"] / a["I_int_thy_w_0"]
    dip = 1 - Tw / C["T_cold"]
    m = (w > -12) & (w < 12)                     # Kalpha1 dip only (Kalpha2 sits near -20 eV)
    wm, dm = w[m], dip[m]
    above = wm[dm >= dm.max() / 2]

    # 8 keV is below the K edge, so (up to the ~1% photoionization-of-a-1s-hole-atom route) every
    # 1s hole is made by resonantly absorbing a Kalpha photon on a 2p hole; each decays at Gamma_K,
    # so Gamma_K * int(rho_1s dt) counts net resonant absorption events per atom. Satellite-to-
    # satellite 1s-hole feeds (3p -> 3d3d "upper") are subtracted so converted holes count once.
    K_main = C["Gamma_K"] / HBAR_EV_FS * np.trapz(np.real(a["rho_ee_t_last"]), t)
    int_K_sat = np.trapz(np.real(a["rho_ee_t_last_sat"]), t, axis=1)
    K_sat = (np.sum(C["Gamma_K_sat"] * int_K_sat) - np.sum(C["feed_upper"] @ int_K_sat)) / HBAR_EV_FS
    return {"T": T, "dip_depth": dm.max(), "dip_fwhm": above[-1] - above[0],
            "K_main": K_main, "K_sat": K_sat, "K_total": K_main + K_sat}


def with_sem(entry, C):
    per_chunk = [metrics(c, C) for c in entry["chunks"]]
    pooled = metrics(entry["mean"], C)
    n = len(per_chunk)
    sem = {k: np.std([m[k] for m in per_chunk], ddof=1) / np.sqrt(n) for k in pooled}
    return pooled, sem


def log_duration_axis(ax, durations):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(durations))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{d:g}" for d in durations]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(min(durations) / 1.4, max(durations) * 1.4)


def lifetime_markers(ax, C, label=True):
    for name, gamma, ha in (("τ$_{1s}$", C["Gamma_K"], "right"), ("τ$_{2p}$", C["Gamma_L3"], "left")):
        tau = HBAR_EV_FS / gamma
        ax.axvline(tau, color=MUTED, lw=0.8, zorder=0)
        if label:
            ax.text(tau * (0.95 if ha == "right" else 1.05), 0.5, f"{name} = {tau:.2f} fs", ha=ha,
                    transform=ax.get_xaxis_transform(), color=INK_2, fontsize=8.5, va="center")


def save(fig, out_dir, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"))
    print("wrote", os.path.join(out_dir, stem + ".{png,pdf}"))


def fig_summary(S, C, out_dir):
    d = np.array(list(S))
    M = {k: np.array([S[x]["metrics"][k] for x in d]) for k in S[d[0]]["metrics"]}
    E = {k: np.array([S[x]["sem"][k] for x in d]) for k in S[d[0]]["sem"]}

    fig, axs = plt.subplots(2, 2, figsize=(10, 7.4), sharex=True)
    ax = axs[0, 0]
    ax.axhline(C["T_cold"], color=INK_2, lw=0.9)
    ax.text(d[0] / 1.3, C["T_cold"] - 0.0012, "cold sample (no core holes)", color=INK_2, fontsize=8.5, va="top")
    ax.errorbar(d, M["T"], yerr=E["T"], color=MAIN, marker="o", ms=6, capsize=0, mec="white", mew=1.2)
    i = int(np.argmin(M["T"]))
    ax.annotate(f"min T = {M['T'][i]:.4f}", (d[i], M["T"][i]), xytext=(0, -16), textcoords="offset points",
                ha="center", color=INK, fontsize=9)
    ax.set_ylabel("energy transmission  $E_{out}/E_{in}$")
    ax.set_title("a  Total transmission")
    ax.set_ylim(0.34, 0.415)

    ax = axs[0, 1]
    for key, col, lab in (("K_total", INK, "total"), ("K_sat", SAT, "spectator satellites"), ("K_main", MAIN, "main line")):
        ax.errorbar(d, M[key], yerr=E[key], color=col, marker="o", ms=6, mec="white", mew=1.2, label=lab)
        ax.text(d[-1] * 1.12, M[key][-1], lab, color=INK_2, fontsize=8.5, va="center")
    ax.set_ylabel("net 1s holes created per atom (exit face)")
    ax.set_title("b  Resonant Kα absorption events  (Γ$_{1s}$∫ρ$_{1s}$dt)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")

    ax = axs[1, 0]
    ax.errorbar(d, M["dip_depth"], yerr=E["dip_depth"], color=MAIN, marker="o", ms=6, mec="white", mew=1.2)
    ax.set_ylabel("max  $1 - T(\\omega)/T_{cold}$")
    ax.set_title("c  Kα1 dip depth")
    ax.set_ylim(0, None)

    ax = axs[1, 1]
    ax.errorbar(d, M["dip_fwhm"], yerr=E["dip_fwhm"], color=MAIN, marker="o", ms=6, mec="white", mew=1.2)
    ax.set_ylabel("dip FWHM (eV)")
    ax.set_title("d  Kα1 dip width")
    ax.set_ylim(0, None)

    for ax in axs.flat:
        log_duration_axis(ax, list(d))
        lifetime_markers(ax, C, label=ax is axs[0, 0])
    for ax in axs[1]:
        ax.set_xlabel("seed duration, FWHM (fs)")
    n_reps = S[d[0]]["n_reps"]
    fig.suptitle(f"Cu RSA vs SASE seed duration  ·  {C['E_seed_uJ']:g} µJ, {n_reps} SASE shots per point, "
                 f"error bars = SEM over chunks (mostly smaller than markers)", x=0.01, ha="left",
                 fontsize=10, color=INK_2)
    fig.tight_layout()
    save(fig, out_dir, "duration_summary")


def fig_spectra(S, C, out_dir, slide=False):
    """Clean single-panel T(omega) vs absolute photon energy, one viridis line per duration.

    viridis runs light (short) -> dark (long, deepest dip), the same dark-means-deepest-dip
    convention as plot_saturation_slide.py; the top of the map is dropped for contrast on white.
    """
    colours = plt.get_cmap("viridis")(np.linspace(0.85, 0.0, len(S)))
    with plt.rc_context(SLIDE_RC if slide else {}):
        fig, ax = plt.subplots(figsize=(SLIDE_WIDTH_IN, 1.42) if slide else (7.5, 4.6))
        for col, (dur, e) in zip(colours, S.items()):
            a = e["mean"]
            w, I0, IL = a["womega_ar"], a["I_int_thy_w_0"], a["I_int_thy_w_last"]
            ok = I0 > 0.03 * I0.max()          # don't plot a ratio where there is ~no input light
            ax.plot(C["E_Ka1"] + w[ok], IL[ok] / I0[ok], color=col, lw=1.2 if slide else 1.7, label=f"{dur:g} fs")
        ax.set_xlim(C["E_Ka1"] - 30, C["E_Ka1"] + 13)
        ax.set_xlabel("Photon energy (eV)")
        ax.set_ylabel("Transmittance")
        ax.ticklabel_format(axis="x", useOffset=False)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        if slide:
            ax.legend(title="Pulse FWHM", loc="center left", bbox_to_anchor=(1.01, 0.5), labelspacing=0.35)
        else:
            ax.legend(title="Pulse duration (FWHM)", ncol=2, loc="lower center", bbox_to_anchor=(0.45, 0.0))
        fig.tight_layout()
        save(fig, out_dir, "duration_slide_spectra" if slide else "duration_spectra")


def fig_time_domain(S, C, out_dir):
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(11, 4.4))
    for col, (dur, e) in zip(RAMP, S.items()):
        a = e["mean"]
        t, I0, IL = a["t_axis"], np.real(a["I_t_0"]), np.real(a["I_t_last"])
        tc = np.sum(t * I0) / np.sum(I0)
        ok = I0 > 0.01 * I0.max()
        ax.plot(((t - tc) / dur)[ok], IL[ok] / I0[ok], color=col, label=f"{dur:g} fs")
    ax.axhline(C["T_cold"], color=INK_2, lw=0.9)
    ax.text(-0.45, C["T_cold"] + 0.002, "cold sample", color=INK_2, fontsize=8.5)
    ax.set_xlabel("(t − t$_c$) / FWHM")
    ax.set_ylabel("$I_{out}(t) / I_{in}(t)$")
    ax.set_title("a  Transmission through the pulse")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(0.30, 0.43)
    ax.legend(title="seed FWHM", ncol=2, loc="lower left")

    short = [x for x in S if x <= 0.5]
    first = True
    for col, (dur, e) in zip(RAMP, S.items()):
        if dur not in short:
            continue
        a = e["mean"]
        t, I0, IL = a["t_axis"], np.real(a["I_t_0"]), np.real(a["I_t_last"])
        tc = np.sum(t * I0) / np.sum(I0)
        axr.semilogy(t - tc, I0 / I0.max(), color=col, lw=1.0, alpha=0.55,
                     label="inputs (thin)" if first else None)
        axr.semilogy(t - tc, IL / I0.max(), color=col, label=f"output, {dur:g} fs")
        first = False
    # Reference slope: coherence (FID) intensity decay at (Gamma_1s + Gamma_2p)/hbar.
    rate = (C["Gamma_K"] + C["Gamma_L3"]) / HBAR_EV_FS
    a = S[short[0]]["mean"]
    t, I0, IL = a["t_axis"], np.real(a["I_t_0"]), np.real(a["I_t_last"])
    tc = np.sum(t * I0) / np.sum(I0)
    t_ref = np.linspace(0.8, 4.5, 50)
    anchor = np.interp(1.0, t - tc, IL / I0.max())
    axr.semilogy(t_ref, 4 * anchor * np.exp(-rate * (t_ref - 1.0)), color=INK, lw=1.0)
    axr.text(2.6, 4 * anchor * np.exp(-rate * 1.6) * 3, f"∝ exp[−(Γ$_{{1s}}$+Γ$_{{2p}}$)t/ħ]\n(1/e = {1 / rate:.2f} fs)",
             color=INK, fontsize=8.5)
    beat = 4.135667 / abs(C["Ka2"])   # h / (Kalpha1 - Kalpha2 splitting), fs
    axr.text(0.35, 3e-9, f"ripple: Kα1–Kα2 quantum beat\n(h/ΔE ≈ {beat:.2f} fs)", color=INK_2, fontsize=8.5)
    axr.set_ylim(1e-10, 2)
    axr.set_xlim(-0.8, 5)
    axr.set_xlabel("t − t$_c$  (fs)")
    axr.set_ylabel("intensity / input peak")
    axr.set_title("b  Resonant tail after short pulses")
    axr.legend(loc="upper right")
    fig.tight_layout()
    save(fig, out_dir, "duration_time_domain")


def fig_populations(S, C, out_dir, slide=False):
    """Exit-face core-hole populations in one row, one panel per duration.

    Colour = hole type (2p3/2, 2p1/2, 1s), line style = main line (solid) vs spectator satellites
    summed over all satellite blocks (dashed). Populations are shown in units of 1e-3. The grey
    fill is the shot-averaged input intensity on ONE scale shared by every panel (all pulses carry
    the same energy, so shorter = more intense), with its peak value printed under the duration.
    """
    dash = (0, (3.5, 1.8)) if slide else (0, (5, 2.5))
    lw = 0.8 if slide else 1.4
    with plt.rc_context(SLIDE_RC if slide else {}):
        fig, axs = plt.subplots(1, len(S), figsize=(SLIDE_WIDTH_IN, 1.6) if slide else (15, 3.9), sharey=True)
        ymax = 0
        pulses = {}
        for ax, (dur, e) in zip(axs, S.items()):
            a = e["mean"]
            t, I0 = a["t_axis"], np.real(a["I_t_0"])
            tc = np.sum(t * I0) / np.sum(I0)
            x = t - tc
            series = []
            for key, col in (("rho_l3", HOLE_2P3), ("rho_l2", HOLE_2P1), ("rho_ee", HOLE_1S)):
                series.append((1e3 * np.real(a[f"{key}_t_last"]), col, "-"))
                series.append((1e3 * np.real(a[f"{key}_t_last_sat"]).sum(0), col, dash))
            for y, col, ls in series:
                ax.plot(x, y, color=col, lw=lw, ls=ls)
                ymax = max(ymax, y.max())
            pulses[dur] = (x, input_intensity_W_cm2(a, C))
            half = max(1.6 * dur, 0.5)
            ax.set_xlim(-half, half + 2.5)
            ax.xaxis.set_major_locator(MaxNLocator(3 if slide else 5))
        I_scale = max(I.max() for _, I in pulses.values())
        for ax, (dur, (x, I)) in zip(axs, pulses.items()):
            ax.fill_between(x, 0, 0.95 * ymax * I / I_scale, color=GRID, zorder=0, lw=0)
            ax.text(0.97, 0.97, f"{dur:g} fs", transform=ax.transAxes, ha="right", va="top",
                    fontweight="bold", color=INK, fontsize=None if slide else 11)
            ax.text(0.97, 0.83, f"{sci_unicode(I.max())} W/cm²", transform=ax.transAxes, ha="right",
                    va="top", color=INK_2, fontsize=6.4 if slide else 9)
        axs[0].set_ylim(0, 1.45 * ymax)
        axs[0].yaxis.set_major_locator(MaxNLocator(4, integer=True))
        axs[0].set_ylabel("Population ($10^{-3}$)")
        axs[len(axs) // 2].set_xlabel("Time relative to pulse centre (fs)")

        handles = [Line2D([], [], color=HOLE_2P3, lw=1.3 * lw, label="2p$_{3/2}$"),
                   Line2D([], [], color=HOLE_2P1, lw=1.3 * lw, label="2p$_{1/2}$"),
                   Line2D([], [], color=HOLE_1S, lw=1.3 * lw, label="1s"),
                   Line2D([], [], color=INK_2, lw=1.3 * lw, label="main line"),
                   Line2D([], [], color=INK_2, lw=1.3 * lw, ls=dash, label="satellites (Σ)"),
                   Patch(color=GRID, label="input intensity")]
        fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.0),
                   handlelength=1.8 if slide else 2.6, columnspacing=1.1 if slide else 2.0,
                   handletextpad=0.5 if slide else 0.8, fontsize=None if slide else 9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.86 if slide else 0.92), w_pad=0.6 if slide else 1.5)
        save(fig, out_dir, "duration_slide_populations" if slide else "duration_populations")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join(REPO, "data", "duration_sweep_24530244"))
    parser.add_argument("--out", default=None, help="output folder (default figs/<sweep name>)")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(REPO, "figs", os.path.basename(os.path.normpath(args.data)))
    os.makedirs(out_dir, exist_ok=True)

    S = load_sweep(args.data)
    C = sim_constants(next(iter(S.values()))["yaml"])
    print(f"T_cold = {C['T_cold']:.4f} over {C['thickness_um']:.2f} um (z=-1 field plane)")
    for dur, e in S.items():
        e["metrics"], e["sem"] = with_sem(e, C)
        m, s = e["metrics"], e["sem"]
        print(f"{dur:5g} fs  n={e['n_reps']}  T={m['T']:.4f}±{s['T']:.4f}  dOD={np.log(C['T_cold'] / m['T']):.3f}  "
              f"dip {m['dip_depth']:.3f}/{m['dip_fwhm']:.1f} eV  1s holes: main {m['K_main']:.4f} sat {m['K_sat']:.4f}")

    fig_summary(S, C, out_dir)
    fig_time_domain(S, C, out_dir)
    S_trim = {d: e for d, e in S.items() if d not in DROP_FROM_SPECTRA_AND_POPULATIONS}
    for slide in (False, True):
        fig_spectra(S_trim, C, out_dir, slide=slide)
        fig_populations(S_trim, C, out_dir, slide=slide)


if __name__ == "__main__":
    main()
