"""FWHM of the Kalpha1 and Kalpha2 resonances: model (no middlemen / EII) against experiment.

Model: the `dsat` variant of the 2026-09-15 pathway sweeps -- the feed-fixed double-satellite model
with L2, before the Part VI/VII extensions (no middlemen, no CK feeds, no EII, no mixing).
  SASE  data/pathway_sweep_sase_24689982/dsat   200 reps, xgrid=ygrid 3 (see caveats)
  mono  data/pathway_sweep_mono_24689983/dsat   10 reps, xgrid=ygrid 5, 1 eV steps 8015-8060 eV
Loading, the energy axes (SASE: hwKalpha1N + womega_ar) and T_wing are those of
plot_pathway_sweep.py.

Experiment:
  SASE Kalpha1  bottom panel (pulse centred ~8045 eV): transmittance_2_9_40uJ.csv + ..._bottom_extra_uJ.csv
  SASE Kalpha2  top panel (pulse centred ~8020 eV):    transmittance_top_all_uJ.csv
                (the bottom panel's pulse has no power at Kalpha2, and the top panel stops at 8050 eV,
                i.e. halfway up the high side of Kalpha1)
  mono          transmittance_seeded_uJ.csv (hand-digitised by eye, 2-10 eV point spacing: widths
                from it are good to a few eV at best)

FWHM definition. A(E) = ln(T_wing/T(E)), the resonant absorbance with the flat photoionisation
background divided out (T_wing = mean T over 8000-8012 eV, or 8060-8070 eV for the SASE bottom
panel). For each line: peak = max of A inside the line window; walk outward from the peak to the
first point where A drops below half the peak, and interpolate linearly. A side is NOT resolved if
the walk reaches the valley between the two lines (the valley stays above half maximum: the line
merges into its neighbour) or the edge of the usable data first. The width is then only a lower bound (open markers), measured to
that boundary. Noisy SASE experimental curves get a 2 eV boxcar first (adds ~0.3 eV to a 6 eV line
in quadrature -- negligible here).

Run from the repo root:  python scripts/plot_fwhm_comparison.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

import plot_pathway_sweep as P

P.VARIANTS = ["dsat"]
FIGS, DATA = P.FIGS, P.DATA

LINES = {"Ka1": dict(window=(8038.0, 8056.0), label=r"K$\alpha_1$"),
         "Ka2": dict(window=(8018.0, 8034.0), label=r"K$\alpha_2$")}
WING_LOW = (8000.0, 8012.0)
SMOOTH_EV = 2.0

MODEL_C, EXP_C = "#eb6834", P.EXP_COLOUR
# Experimental peaks shallower than this are not measured: the point-to-point scatter of A is
# ~0.02-0.04 on the SASE curves and ~0.01-0.02 on the by-eye mono digitisation. This drops the
# SASE 0.12/0.5 uJ series and mono 0.1 uJ.
MIN_PEAK = {"SASE": 0.10, "mono": 0.05}


# --------------------------------------------------------------------------- measurement

def fwhm(E, A, line, valid):
    """Half-maximum crossings lo/hi, width, peak position/height, and whether both sides resolved."""
    in_valid = (E >= valid[0]) & (E <= valid[1])

    def peak_index(name):
        w0, w1 = LINES[name]["window"]
        idx = np.where((E >= w0) & (E <= w1) & in_valid)[0]
        return idx[np.argmax(A[idx])] if len(idx) else None

    i0, i_other = peak_index(line), peak_index("Ka2" if line == "Ka1" else "Ka1")
    half = A[i0] / 2
    # Stop at the valley between the two lines (or the data edge on the outer side).
    valid_idx = np.where(in_valid)[0]
    if i_other is not None:
        a, b = sorted((i0, i_other))
        valley = a + int(np.argmin(A[a:b + 1]))
    lo_stop = valid_idx[0] if line == "Ka2" or i_other is None else valley
    hi_stop = valid_idx[-1] if line == "Ka1" or i_other is None else valley

    def walk(step, stop):
        i = i0
        while i != stop:
            j = i + step
            if A[j] < half:
                return E[i] + (half - A[i]) * (E[j] - E[i]) / (A[j] - A[i]), True
            i = j
        return E[stop], False

    lo, ok_lo = walk(-1, lo_stop)
    hi, ok_hi = walk(+1, hi_stop)
    return dict(lo=lo, hi=hi, width=hi - lo, E_peak=E[i0], A_max=A[i0], half=half,
                resolved=ok_lo and ok_hi)


def trimmed_absorbance(E, T, wing, smooth_eV):
    """A(E) and the valid range, dropping the boxcar's edge half-window."""
    A = P.absorbance(E, T, P.wing_T(E, T, wing), smooth_eV)
    pad = smooth_eV / 2 + 0.5 if smooth_eV else 0.0
    return A, (E[0] + pad, E[-1] - pad)


# --------------------------------------------------------------------------- loading

def load_exp_sase():
    """{uJ: {'Ka1': (E, A, valid), 'Ka2': (E, A, valid)}} from the two SASE panels."""
    out = {}
    for fname in ["transmittance_2_9_40uJ.csv", "transmittance_bottom_extra_uJ.csv"]:
        df = pd.read_csv(os.path.join(DATA, fname))
        E = df.Photon_energy_eV.values
        for c in df.columns:
            if c.endswith("uJ_transmittance"):
                out.setdefault(float(c[:-len("uJ_transmittance")]), {})["Ka1"] = (
                    E, *trimmed_absorbance(E, df[c].values, P.SASE_WING, SMOOTH_EV))
    df = pd.read_csv(os.path.join(DATA, "transmittance_top_all_uJ.csv"))
    E = df.Photon_energy_eV.values
    for c in df.columns:
        if c.endswith("uJ_transmittance"):
            out.setdefault(float(c[:-len("uJ_transmittance")]), {})["Ka2"] = (
                E, *trimmed_absorbance(E, df[c].values, WING_LOW, SMOOTH_EV))
    return dict(sorted(out.items()))


def measure_all():
    rows = []
    sase = P.load_sase()["dsat"]
    for uJ, (E, T, _) in sase.items():
        A = P.absorbance(E, T, P.wing_T(E, T, P.SASE_WING))
        for line in LINES:
            rows.append(dict(beam="SASE", source="model", uJ=uJ, line=line,
                             **fwhm(E, A, line, P.SASE_VALID)))
    exp_sase = load_exp_sase()
    for uJ, panels in exp_sase.items():
        for line, (E, A, valid) in panels.items():
            rows.append(dict(beam="SASE", source="exp", uJ=uJ, line=line,
                             **fwhm(E, A, line, valid)))

    mono = P.load_mono()["dsat"]
    for uJ, (E, T, _) in mono.items():
        A = P.absorbance(E, T, P.wing_T(E, T, P.MONO_WING))
        for line in LINES:
            rows.append(dict(beam="mono", source="model", uJ=uJ, line=line,
                             **fwhm(E, A, line, (E[0], E[-1]))))
    exp_mono, _, _ = P.load_experiment()
    for uJ, (E, T) in exp_mono.items():
        A = P.absorbance(E, T, P.wing_T(E, T, P.MONO_WING))
        for line in LINES:
            rows.append(dict(beam="mono", source="exp", uJ=uJ, line=line,
                             **fwhm(E, A, line, (E[0], E[-1]))))
    df = pd.DataFrame(rows)
    df["measurable"] = (df.source == "model") | (df.A_max >= df.beam.map(MIN_PEAK))
    return df, sase, exp_sase, mono, exp_mono


# --------------------------------------------------------------------------- figure

def _fwhm_bar(ax, r, color, below=False, dy=0.0):
    ls = "-" if r["resolved"] else (0, (2, 1.5))
    y = r["half"] + dy
    ax.plot([r["lo"], r["hi"]], [y, y], color=color, lw=1.6, ls=ls, zorder=6,
            solid_capstyle="butt")
    for x in (r["lo"], r["hi"]):
        ax.plot([x, x], [y - 0.012, y + 0.012], color=color, lw=1.2, zorder=6)
    ax.annotate(f"{r['width']:.1f}" + ("" if r["resolved"] else "+"), ((r["lo"] + r["hi"]) / 2, y),
                xytext=(0, -4 if below else 3), textcoords="offset points", ha="center",
                va="top" if below else "bottom",
                fontsize=7, color=P.INK, zorder=7)


def figure(df, sase, exp_sase, mono, exp_mono, sase_uJ=9.0, mono_uJ=5.0):
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.55, 1, 1]))

    # --- spectra with FWHM bars
    for row, (beam, uJ) in enumerate([("SASE", sase_uJ), ("mono", mono_uJ)]):
        ax = axes[row][0]
        if beam == "SASE":
            E, T, _ = sase[uJ]
            ax.plot(E, P.absorbance(E, T, P.wing_T(E, T, P.SASE_WING)), color=MODEL_C, lw=1.4,
                    label="model (dsat)")
            for line, ls, lab in [("Ka1", "-", "experiment, 8045 eV panel"),
                                  ("Ka2", (0, (3, 1.5)), "experiment, 8020 eV panel")]:
                Ee, Ae, valid = exp_sase[uJ][line]
                m = (Ee >= valid[0]) & (Ee <= valid[1])
                ax.plot(Ee[m], Ae[m], color=EXP_C, lw=1.1, ls=ls, label=lab)
            ax.set_xlim(8015, 8070)
            title = f"SASE, {uJ:g} µJ (model: 3×3 grid, 200 shots)"
        else:
            E, T, _ = mono[uJ]
            ax.plot(E, P.absorbance(E, T, P.wing_T(E, T, P.MONO_WING)), color=MODEL_C, lw=1.4,
                    marker="o", ms=2.2, label="model (dsat)")
            Ee, Te = exp_mono[uJ]
            ax.plot(Ee, P.absorbance(Ee, Te, P.wing_T(Ee, Te, P.MONO_WING)), color=EXP_C,
                    lw=1.1, ls="--", marker="s", ms=3, label="experiment (digitised by eye)")
            ax.set_xlim(8005, 8075)
            title = f"Self-seeded (mono), {uJ:g} µJ"
        sel = df[(df.beam == beam) & (df.uJ == uJ) & df.measurable]
        for _, r in sel.iterrows():
            _fwhm_bar(ax, r, MODEL_C if r.source == "model" else EXP_C, below=r.source == "model")
        ax.axhline(0, color=P.INK_MUTED, lw=0.6)
        P._line_markers(ax)
        ax.set_title(title)
        ax.set_xlabel("Photon energy (eV)")
        ax.set_ylabel(r"Resonant absorbance $\ln(T_{wing}/T)$")
        ax.legend(frameon=False, loc="upper left", fontsize=7.5)

    # --- FWHM vs pulse energy
    for row, beam in enumerate(["SASE", "mono"]):
        for col, line in enumerate(["Ka1", "Ka2"], start=1):
            ax = axes[row][col]
            for source, color, marker, lab in [("model", MODEL_C, "o", "model (dsat)"),
                                               ("exp", EXP_C, "s", "experiment")]:
                sel = df[(df.beam == beam) & (df.line == line) & (df.source == source)
                         & df.measurable].sort_values("uJ")
                if sel.empty:
                    continue
                ok = sel[sel.resolved]
                lb = sel[~sel.resolved]
                ax.plot(ok.uJ, ok.width, color=color, lw=1.2, ls="-" if source == "model" else "--",
                        zorder=3)
                ax.plot(ok.uJ, ok.width, marker, color=color, ms=5, lw=0, zorder=4, label=lab)
                ax.plot(lb.uJ, lb.width, marker, color=color, mfc="white", ms=5, lw=0, zorder=4,
                        label=f"{lab}: lower bound")
                for _, r in lb.iterrows():
                    ax.annotate("", (r.uJ, r.width * 1.12), (r.uJ, r.width * 1.01),
                                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8,
                                                mutation_scale=7))
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.1, 0.3, 1, 3, 10, 30, 100]))
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.set_xlim((1.5, 110) if beam == "SASE" else (0.07, 70))
            ax.set_ylim(0, 34)
            ax.set_xlabel("Pulse energy (µJ)")
            ax.set_ylabel("FWHM (eV)")
            ax.set_title(f"{'SASE' if beam == 'SASE' else 'Self-seeded (mono)'}: "
                         f"{LINES[line]['label']} FWHM")
    for ax in (axes[0][1], axes[1][1]):
        ax.legend(frameon=False, loc="upper left", fontsize=7)

    fig.suptitle("Kα₁ and Kα₂ resonance widths, model without middlemen/EII vs experiment "
                 "(open = merged with the neighbouring line or truncated: lower bound)", fontsize=10.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fwhm_comparison.{ext}"))
    plt.close(fig)


if __name__ == "__main__":
    df, sase, exp_sase, mono, exp_mono = measure_all()
    pd.set_option("display.width", 160)
    print(df[["beam", "line", "source", "uJ", "E_peak", "A_max", "lo", "hi", "width",
              "resolved", "measurable"]]
          .sort_values(["beam", "line", "uJ", "source"]).round(3).to_string(index=False))
    figure(df, sase, exp_sase, mono, exp_mono)
    print(f"\nwrote {os.path.join(FIGS, 'fwhm_comparison.png')}")
