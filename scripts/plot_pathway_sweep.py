"""Part VI/VII pathway-extension sweeps: model variants against experiment.

Reads the two sweeps written by scripts/submit_pathway_sweep_{mono,sase}.sh (see
scripts/generate_pathway_sweeps.sh for what each variant contains) and writes into figs/:

  pathway_mono_spectra     T(E) at 0.1-30 uJ, self-seeded/mono beam, 5 variants + experiment
  pathway_sase_spectra     T(E) at 2-80 uJ, SASE beam, 5 variants + experiment
  pathway_wing_and_budget  the non-resonant wing vs pulse energy (the bleaching artefact and
                           its fix), with the population budget that causes it
  pathway_gap              resonant absorbance A = ln(T_wing/T) at the line and integrated
                           over the band, model vs experiment vs pulse energy

Definitions
  T(E)      mono: one run per monochromator setting, T = sum(I_int_thy_w_last)/sum(I_int_thy_w_0),
            on the absolute monochromator_target_energy_eV axis.
            SASE: one run per pulse energy, T(E) = I_int_thy_w_last(E)/I_int_thy_w_0(E) on the
            hwKalpha1N + womega_ar axis. womega_ar is the detuning from the shared Kalpha1
            rotating frame, so 0 is Kalpha1 -- cross-checked here by aligning the model's own SASE
            and mono absorbance shapes, which agree at an offset of 8047.75 +- 0.05 eV. NOTE:
            plot_saturation.py / plot_rate_eq_vs_mb.py use 8045 instead, which slides the model
            2.9 eV down and happens to hide a real disagreement: the measured SASE dip peaks at
            8044.9-8045.7 eV, about 2.5 eV below the model's Kalpha1.
  T_wing    non-resonant background: mean T over 8000-8012 eV (mono) or 8060-8070 eV (SASE).
  A(E)      = ln(T_wing/T(E)), the resonant part only, with the photoionisation background
            divided out. A_line = max of A over 8040-8054 eV (2 eV boxcar on the noisy
            experimental curves); area = integral of A over the band.
  leak      1 - min(total_population_t_last): the fraction of atoms that has left every tracked
            population, at the exit-face centre pixel. Zero once middlemen are on.

Experiment: self-seeded/mono panel from data/transmittance_seeded_uJ.csv (hand-digitised, a
qualitative depth/width target only); SASE panel from data/transmittance_2_9_40uJ.csv and
data/transmittance_bottom_extra_uJ.csv (vector-extracted, precise -- see the memory note on
which reference goes with which beam type).

Run from the repo root:  python scripts/plot_pathway_sweep.py
"""

import os
import re
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
FIGS = os.path.join(REPO, "figs")

MONO_DIR = os.path.join(DATA, "pathway_sweep_mono_24689983")
SASE_DIR = os.path.join(DATA, "pathway_sweep_sase_24689982")

KA1_EV, KA2_EV = 8047.91, 8027.98
MONO_WING = (8000.0, 8012.0)
SASE_WING = (8060.0, 8070.0)
LINE_WINDOW = (8040.0, 8054.0)
MONO_BAND = (8015.0, 8060.0)
SASE_BAND = (8025.0, 8070.0)
SASE_VALID = (8022.0, 8072.0)   # outside this the SASE seed carries too little spectral power

# dsat (the model before the extensions) against the cumulative ladder of extensions: one
# ordinal blue ramp for the ladder, one contrasting hue for the "before", ink for experiment.
# Ramp steps are from plot_saturation.py's validated light-surface palette.
VARIANTS = ["dsat", "mid", "mid-eii", "mid-eii-mixsat", "mid-eii-mix"]
STYLE = {
    "dsat":           dict(color="#eb6834", ls="-",  label="dsat (before: no extensions)"),
    "mid":            dict(color="#86b6ef", ls="-",  label="+ middlemen, CK feeds"),
    "mid-eii":        dict(color="#2a78d6", ls="-",  label="+ L-shell EII"),
    "mid-eii-mixsat": dict(color="#1c5cab", ls="--", label="+ mixing 2/fs (satellites)"),
    "mid-eii-mix":    dict(color="#0b2a55", ls="-",  label="+ mixing 2/fs (all)"),
}
EXP_COLOUR, INK, INK_MUTED, GRIDC = "#23262f", "#23262f", "#6b6e76", "#d8d8d4"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": INK_MUTED, "axes.linewidth": 0.8,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 130, "savefig.bbox": "tight",
})


# --------------------------------------------------------------------------- loading

def _npz_mean(run_dir, key):
    """Repetition-averaged output `key`, summed over the chunk files of one run folder."""
    out = None
    for path in glob.glob(os.path.join(run_dir, "*.npz")):
        with np.load(path) as d:
            if key + "_sum" not in d.files:
                return None
            x = np.real(d[key + "_sum"]) / np.maximum(d[key + "_count"], 1)
        out = x if out is None else out + x
    return out


DIAG_KEYS = ("rho_mid_t_last", "n_e_total_t_last", "total_population_t_last",
             "rho_ground_t_last", "base_pop_t_last", "sat_pop_t_last")


def _diagnostics(run_dir):
    """Exit-face, centre-pixel scalars at the end of the pulse (plus the trace minimum)."""
    out = {}
    for key in DIAG_KEYS:
        x = _npz_mean(run_dir, key)
        if x is None:
            continue
        out[key] = float(np.atleast_1d(x)[-1])
    trace = _npz_mean(run_dir, "total_population_t_last")
    out["leak"] = float(1.0 - np.min(trace)) if trace is not None else np.nan
    return out


def load_mono(data_dir=MONO_DIR):
    """{variant: {E_seed: (energies, T, diagnostics at the line centre)}}"""
    out = {}
    for variant in VARIANTS:
        per_seed = {}
        for run in glob.glob(os.path.join(data_dir, variant, "runs_seed_*")):
            m = re.search(r"runs_seed_([\d.]+)_uJ__energy_([\d.]+)_eV", os.path.basename(run))
            if not m or not glob.glob(os.path.join(run, "*.npz")):
                continue
            e_seed, energy = float(m.group(1)), float(m.group(2))
            I_last, I_0 = _npz_mean(run, "I_int_thy_w_last"), _npz_mean(run, "I_int_thy_w_0")
            per_seed.setdefault(e_seed, {})[energy] = (I_last.sum() / I_0.sum(), run)
        curves = {}
        for e_seed, by_energy in sorted(per_seed.items()):
            energies = np.array(sorted(by_energy))
            T = np.array([by_energy[e][0] for e in energies])
            at_line = by_energy[energies[np.argmin(np.abs(energies - KA1_EV))]][1]
            curves[e_seed] = (energies, T, _diagnostics(at_line))
        out[variant] = curves
    return out


def load_sase(data_dir=SASE_DIR):
    """{variant: {E_seed: (energies, T, diagnostics)}}, restricted to SASE_VALID."""
    out = {}
    for variant in VARIANTS:
        curves = {}
        for run in glob.glob(os.path.join(data_dir, variant, "runs_seed_*")):
            npz = glob.glob(os.path.join(run, "*.npz"))
            if not npz:
                continue
            e_seed = float(re.search(r"runs_seed_([\d.]+)_uJ", os.path.basename(run)).group(1))
            I_last, I_0 = _npz_mean(run, "I_int_thy_w_last"), _npz_mean(run, "I_int_thy_w_0")
            with np.load(npz[0]) as d:
                energies = KA1_EV + d["womega_ar"]
            keep = (energies >= SASE_VALID[0]) & (energies <= SASE_VALID[1])
            curves[e_seed] = (energies[keep], (I_last / np.maximum(I_0, 1e-300))[keep],
                              _diagnostics(run))
        out[variant] = dict(sorted(curves.items()))
    return out


def load_experiment():
    """(mono {uJ: (E, T)}, sase {uJ: (E, T, err)}, sase cold (E, T))."""
    mono_csv = pd.read_csv(os.path.join(DATA, "transmittance_seeded_uJ.csv"), comment="#")
    E_mono = mono_csv.Photon_energy_eV.values
    mono = {float(c.replace("uJ_transmittance", "")): (E_mono, mono_csv[c].values)
            for c in mono_csv.columns if c.endswith("uJ_transmittance")}

    main = pd.read_csv(os.path.join(DATA, "transmittance_2_9_40uJ.csv"))
    extra = pd.read_csv(os.path.join(DATA, "transmittance_bottom_extra_uJ.csv"))
    sase = {}
    for df in (main, extra):
        E = df.Photon_energy_eV.values
        for c in df.columns:
            if c.endswith("uJ_transmittance"):
                uJ = float(c.replace("uJ_transmittance", ""))
                sase[uJ] = (E, df[c].values, df[c.replace("_transmittance", "_error")].values)
    cold = (extra.Photon_energy_eV.values, extra.Cold_transmittance.values)
    return mono, dict(sorted(sase.items())), cold


# --------------------------------------------------------------------------- metrics

def boxcar(E, y, width_eV):
    n = max(1, int(round(width_eV / np.mean(np.diff(E)))))
    return np.convolve(y, np.ones(n) / n, mode="same")


def wing_T(E, T, window):
    m = (E >= window[0]) & (E <= window[1])
    return float(np.mean(T[m])) if m.any() else np.nan


def absorbance(E, T, T_wing, smooth_eV=0.0):
    A = np.log(T_wing / T)
    return boxcar(E, A, smooth_eV) if smooth_eV else A


def line_depth(E, T, T_wing, smooth_eV=0.0):
    A = absorbance(E, T, T_wing, smooth_eV)
    m = (E >= LINE_WINDOW[0]) & (E <= LINE_WINDOW[1])
    return float(A[m].max())


def band_area(E, T, T_wing, band, smooth_eV=0.0):
    A = absorbance(E, T, T_wing, smooth_eV)
    m = (E >= band[0]) & (E <= band[1])
    return float(np.trapz(A[m], E[m]))


# --------------------------------------------------------------------------- figures

def _line_markers(ax, label=True):
    for E_line, name in [(KA1_EV, r"K$\alpha_1$"), (KA2_EV, r"K$\alpha_2$")]:
        ax.axvline(E_line, color=INK_MUTED, ls=":", lw=0.7, zorder=0)
        if label:
            ax.annotate(name, (E_line, 0.03), xycoords=("data", "axes fraction"),
                        ha="right", va="bottom", fontsize=7, color=INK_MUTED, rotation=90)


def fig_spectra(sim, exp, beam, out_name):
    """T(E) panels, one per pulse energy, every variant plus the experiment."""
    if beam == "mono":
        energies = [0.1, 1.0, 5.0, 20.0, 30.0]
        xlim, exp_style = (8000, 8090), dict(marker="s", ms=2.6, lw=0.9, ls="--")
    else:
        energies = [2.0, 9.0, 40.0, 80.0]
        xlim, exp_style = (8022, 8072), dict(marker="", lw=1.1, ls="--")

    fig, axes = plt.subplots(1, len(energies), figsize=(3.05 * len(energies), 3.1),
                             sharey=True, constrained_layout=True)
    for ax, uJ in zip(axes, energies):
        for variant in VARIANTS:
            if uJ not in sim[variant]:
                continue
            E, T = sim[variant][uJ][0], sim[variant][uJ][1]
            ax.plot(E, T, lw=1.3, **{k: v for k, v in STYLE[variant].items() if k != "label"},
                    label=STYLE[variant]["label"])
        if uJ in exp:
            e = exp[uJ]
            ax.plot(e[0], e[1], color=EXP_COLOUR, label="experiment", zorder=5, **exp_style)
            if len(e) > 2:
                ax.fill_between(e[0], e[1] - e[2], e[1] + e[2], color=EXP_COLOUR, alpha=0.18,
                                lw=0, zorder=4)
        _line_markers(ax, label=(ax is axes[0]))
        ax.set_xlim(*xlim)
        ax.set_title(f"{uJ:g} µJ")
        ax.set_xlabel("Photon energy (eV)")
    axes[0].set_ylabel("Transmittance")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False)
    beam_name = "self-seeded (monochromatic) beam" if beam == "mono" else "SASE beam"
    fig.suptitle(f"Transmittance of 20 µm Cu, {beam_name}", fontsize=10.5)
    fig.savefig(os.path.join(FIGS, out_name + ".pdf"))
    fig.savefig(os.path.join(FIGS, out_name + ".png"))
    plt.close(fig)


def fig_wing_and_budget(mono, sase, exp_mono, exp_sase, exp_cold):
    """Why the wings stopped rising: the leak, and where the population goes instead."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)

    # (a) non-resonant wing vs pulse energy
    ax = axes[0]
    for variant in ["dsat", "mid", "mid-eii-mix"]:
        for curves, window, marker, dash in [(mono[variant], MONO_WING, "o", (None, None)),
                                             (sase[variant], SASE_WING, "^", (2, 1.5))]:
            uJ = np.array(sorted(curves))
            Tw = np.array([wing_T(*curves[u][:2], window) for u in uJ])
            style = dict(STYLE[variant])
            style.pop("label")
            style["ls"] = "-" if dash[0] is None else (0, dash)
            ax.plot(uJ, Tw, marker=marker, ms=3.4, lw=1.3, **style)
    for uJ, (E, T) in exp_mono.items():
        ax.plot(uJ, wing_T(E, T, MONO_WING), "s", ms=4, mfc="none", color=EXP_COLOUR, zorder=5)
    for uJ, e in exp_sase.items():
        ax.plot(uJ, wing_T(e[0], e[1], SASE_WING), "^", ms=4, color=EXP_COLOUR, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Pulse energy (µJ)")
    ax.set_ylabel("Non-resonant wing transmittance")
    ax.set_title("(a) The wing stops bleaching")
    ax.set_ylim(0.342, 0.435)
    ax.annotate("dsat: atoms leave the model\n→ wing brightens", (0.16, 0.425), fontsize=7.5,
                color="#eb6834", ha="left")
    ax.annotate("with middlemen they keep absorbing\n→ wing darkens, as measured",
                (2.2, 0.4135), fontsize=7.5, color="#0b2a55", ha="left")
    handles = [plt.Line2D([], [], color=STYLE[v]["color"], lw=1.3, label=STYLE[v]["label"])
               for v in ["dsat", "mid", "mid-eii-mix"]]
    handles += [plt.Line2D([], [], color=INK_MUTED, lw=1.3, marker="o", ms=3.4, label="mono (model)"),
                plt.Line2D([], [], color=INK_MUTED, lw=1.3, ls=(0, (2, 1.5)), marker="^", ms=3.4,
                           label="SASE (model)"),
                plt.Line2D([], [], color=EXP_COLOUR, lw=0, marker="s", ms=4, mfc="none",
                           label="mono (exp.)"),
                plt.Line2D([], [], color=EXP_COLOUR, lw=0, marker="^", ms=4, label="SASE (exp.)")]
    ax.legend(handles=handles, frameon=False, fontsize=6.5, loc="lower left", ncol=1,
              labelspacing=0.3, handlelength=1.6)

    # (b) population budget, mono
    ax = axes[1]
    uJ = np.array(sorted(mono["dsat"]))
    leak = np.array([mono["dsat"][u][2]["leak"] for u in uJ])
    mid = np.array([mono["mid-eii-mix"][u][2]["rho_mid_t_last"] for u in uJ])
    leak_fixed = np.array([mono["mid-eii-mix"][u][2]["leak"] for u in uJ])
    ax.plot(uJ, leak, "o-", ms=3.4, lw=1.3, color="#eb6834", label="dsat: population lost")
    ax.plot(uJ, mid, "o-", ms=3.4, lw=1.3, color="#0b2a55", label="mid-eii-mix: middleman pool")
    ax.plot(uJ, leak_fixed, "o-", ms=3.4, lw=1.3, color=INK_MUTED, label="mid-eii-mix: lost")
    ax.set_xscale("log")
    ax.set_xlabel("Pulse energy (µJ)")
    ax.set_ylabel("Fraction of atoms")
    ax.set_title("(b) Where those atoms went (mono)")
    ax.legend(frameon=False, loc="upper left")

    # (c) free electrons
    ax = axes[2]
    for curves, label, ls in [(mono["mid-eii-mix"], "mono", "-"), (sase["mid-eii-mix"], "SASE", "--")]:
        uJ = np.array(sorted(curves))
        n_e = np.array([curves[u][2]["n_e_total_t_last"] for u in uJ])
        ground = np.array([curves[u][2]["rho_ground_t_last"] for u in uJ])
        ax.plot(uJ, n_e, ls, marker="o", ms=3.4, lw=1.3, color="#1baf7a",
                label=f"free electrons/atom, {label}")
        ax.plot(uJ, 1 - ground, ls, marker="o", ms=3.4, lw=1.3, color="#0b2a55",
                label=f"ionised fraction, {label}")
    ax.set_xscale("log")
    ax.set_xlabel("Pulse energy (µJ)")
    ax.set_ylabel("Per atom")
    ax.set_title("(c) Ionisation at the exit face")
    ax.legend(frameon=False, loc="upper left")

    fig.suptitle("Population bookkeeping: exit-face centre pixel, end of pulse", fontsize=10.5)
    fig.savefig(os.path.join(FIGS, "pathway_wing_and_budget.pdf"))
    fig.savefig(os.path.join(FIGS, "pathway_wing_and_budget.png"))
    plt.close(fig)


def fig_gap(mono, sase, exp_mono, exp_sase):
    """What is left: resonant depth and integrated resonant absorption vs pulse energy."""
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), constrained_layout=True)

    for row, (beam, sim, exp, window, band, smooth) in enumerate([
            ("Self-seeded (mono)", mono, exp_mono, MONO_WING, MONO_BAND, 0.0),
            ("SASE", sase, exp_sase, SASE_WING, SASE_BAND, 2.0)]):
        for col, (metric, ylabel) in enumerate([
                (lambda E, T, Tw, s: line_depth(E, T, Tw, s), r"Peak resonant absorbance  $\ln(T_{wing}/T)$"),
                (lambda E, T, Tw, s: band_area(E, T, Tw, band, s), "Integrated absorbance (eV)")]):
            ax = axes[row][col]
            for variant in VARIANTS:
                uJ = np.array(sorted(sim[variant]))
                y = [metric(*sim[variant][u][:2], wing_T(*sim[variant][u][:2], window), 0.0)
                     for u in uJ]
                ax.plot(uJ, y, marker="o", ms=3.2, lw=1.3, **STYLE[variant])
            uJ_e, y_e = [], []
            for u, e in sorted(exp.items()):
                E, T = e[0], e[1]
                value = metric(E, T, wing_T(E, T, window), smooth)
                # the digitised 0.5 uJ SASE series has a sloping baseline, so its own wing window
                # sits below the band and the area comes out negative: not a measurement of
                # anything, drop it rather than plot it
                if col == 1 and value < 0:
                    continue
                uJ_e.append(u)
                y_e.append(value)
            ax.plot(uJ_e, y_e, "s--", ms=4, lw=1.1, color=EXP_COLOUR, label="experiment")
            ax.set_xscale("log")
            ax.set_xlabel("Pulse energy (µJ)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{beam}")
    axes[0][0].legend(frameon=False, loc="upper left", fontsize=7.5)
    fig.suptitle("The remaining gap is entirely in the resonant part", fontsize=10.5)
    fig.savefig(os.path.join(FIGS, "pathway_gap.pdf"))
    fig.savefig(os.path.join(FIGS, "pathway_gap.png"))
    plt.close(fig)


def fig_lineshape(mono, sase, exp_mono, exp_sase):
    """The resonant line itself: the model is too weak AND too narrow, by separate factors."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), constrained_layout=True)
    cases = [("Self-seeded, 20 µJ", mono, exp_mono, 20.0, MONO_WING, MONO_BAND, 0.0, (8015, 8065)),
             ("SASE, 9 µJ", sase, exp_sase, 9.0, SASE_WING, SASE_BAND, 2.0, (8025, 8065))]
    for ax, (title, sim, exp, uJ, window, band, smooth, xlim) in zip(axes, cases):
        E, T = sim["mid-eii-mix"][uJ][:2]
        A = absorbance(E, T, wing_T(E, T, window))
        e = exp[uJ]
        Ee, Te = e[0], e[1]
        Ae = absorbance(Ee, Te, wing_T(Ee, Te, window), smooth)
        scale = band_area(Ee, Te, wing_T(Ee, Te, window), band, smooth) / band_area(E, T, wing_T(E, T, window), band)

        ax.plot(Ee, Ae, color=EXP_COLOUR, lw=1.3, label="experiment")
        ax.plot(E, A, color="#0b2a55", lw=1.4, label="model (all extensions)")
        ax.plot(E, A * scale, color="#0b2a55", lw=1.1, ls=":",
                label=f"model × {scale:.1f} (same area)")
        ax.axhline(0, color=INK_MUTED, lw=0.6)
        _line_markers(ax)
        ax.set_xlim(*xlim)
        ax.set_xlabel("Photon energy (eV)")
        ax.set_ylabel(r"Resonant absorbance $\ln(T_{wing}/T)$")
        ax.set_title(title)
        ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    axes[1].annotate("measured line sits ~2.5 eV below\nthe model's Kα₁, and is 2× wider",
                     (8051, 0.72), fontsize=7.5, color=EXP_COLOUR)
    fig.suptitle("Even rescaled to the same strength, the modelled resonance is too narrow",
                 fontsize=10.5)
    fig.savefig(os.path.join(FIGS, "pathway_lineshape.pdf"))
    fig.savefig(os.path.join(FIGS, "pathway_lineshape.png"))
    plt.close(fig)


def print_table(mono, sase, exp_mono, exp_sase):
    print("\n--- mono: T_wing / peak absorbance A_line ---")
    header = f"{'uJ':>6} " + " ".join(f"{v:>16}" for v in VARIANTS) + f" {'experiment':>16}"
    print(header)
    for u in sorted(mono["dsat"]):
        cells = []
        for v in VARIANTS:
            E, T = mono[v][u][:2]
            Tw = wing_T(E, T, MONO_WING)
            cells.append(f"{Tw:.4f}/{line_depth(E, T, Tw):.3f}")
        if u in exp_mono:
            E, T = exp_mono[u]
            Tw = wing_T(E, T, MONO_WING)
            cells.append(f"{Tw:.4f}/{line_depth(E, T, Tw):.3f}")
        print(f"{u:>6g} " + " ".join(f"{c:>16}" for c in cells))

    print("\n--- SASE: T_wing / peak absorbance A_line (2 eV boxcar on experiment) ---")
    print(header)
    for u in sorted(sase["dsat"]):
        cells = []
        for v in VARIANTS:
            E, T = sase[v][u][:2]
            Tw = wing_T(E, T, SASE_WING)
            cells.append(f"{Tw:.4f}/{line_depth(E, T, Tw):.3f}")
        if u in exp_sase:
            E, T = exp_sase[u][0], exp_sase[u][1]
            Tw = wing_T(E, T, SASE_WING)
            cells.append(f"{Tw:.4f}/{line_depth(E, T, Tw, 2.0):.3f}")
        print(f"{u:>6g} " + " ".join(f"{c:>16}" for c in cells))
    for u, e in sorted(exp_sase.items()):
        if u in sase["dsat"]:
            continue
        E, T = e[0], e[1]
        Tw = wing_T(E, T, SASE_WING)
        print(f"{u:>6g} " + " " * (17 * len(VARIANTS)) + f"{Tw:.4f}/{line_depth(E, T, Tw, 2.0):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mono-data", default=MONO_DIR)
    parser.add_argument("--sase-data", default=SASE_DIR)
    args = parser.parse_args()

    os.makedirs(FIGS, exist_ok=True)
    mono = load_mono(args.mono_data)
    sase = load_sase(args.sase_data)
    exp_mono, exp_sase, exp_cold = load_experiment()

    fig_spectra(mono, exp_mono, "mono", "pathway_mono_spectra")
    fig_spectra(sase, exp_sase, "sase", "pathway_sase_spectra")
    fig_wing_and_budget(mono, sase, exp_mono, exp_sase, exp_cold)
    fig_gap(mono, sase, exp_mono, exp_sase)
    fig_lineshape(mono, sase, exp_mono, exp_sase)
    print_table(mono, sase, exp_mono, exp_sase)
    print(f"\nwrote figures to {FIGS}")
