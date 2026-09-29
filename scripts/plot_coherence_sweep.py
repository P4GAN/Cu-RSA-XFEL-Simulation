"""Coherence levers (scripts/generate_coherence_sweeps.sh) on the reference model R against experiment.

Reads, from data/:
  coherence_sweep_mono_<id>/<variant>/   self-seeded/mono beam, 1/5/20/30 uJ x 15 photon energies, 5x5, 10 reps
  coherence_sweep_sase_<id>/<variant>/   SASE beam, 2/9/20/40 uJ, 5x5, 200 reps
with variants R (middlemen + CK + EII), raman10 (sublevel_raman_dephasing_fs_inv = 10: removes the 2p3/2
dark state), broad25 (additional_dephasing = 2.5 fs^-1: +3.3 eV homogeneous FWHM) and both. All four share
the random seeds, so variant differences are paired and free of SASE shot noise to first order.
Writes into figs/:
  coherence_mono_absorbance   resonant absorbance A(E) per 20 um, every variant vs experiment
  coherence_sase_absorbance   the same for the SASE beam (spectrally resolved)
  coherence_summary           line depth, width and area vs pulse energy for both beams

Definitions as scripts/plot_bracket_sweep.py (mono) and scripts/plot_pathway_sweep.py (SASE), except
that the mono line depths are the peak of A in a window (Kalpha1 8040-8054 eV, Kalpha2 8022-8032 eV)
on each curve's own grid, because the measured lines sit ~2 eV below the model's (8045.9/8026.4 vs
8047.9/8028.0 eV). Mono band area: trapezoid over 8012-8072 eV on the sweep's 15-point grid, for model
and experiment alike.

Mono experiment: the slide-9 pulse-energy bins of the single-shot scatter panels (slides 5-6 of
docs/Results_RSA_seeded.pdf), digitised by ../RSA-derivation-bloch/exp_mono_scatter.py; it supersedes
the by-eye data/transmittance_seeded_uJ.csv, which is used only if the scatter file is missing.
SASE experiment: the vector-extracted data/transmittance_2_9_40uJ.csv (2 eV boxcar).

Run from the repo root:  python scripts/plot_coherence_sweep.py [--mono DIR] [--sase DIR]
"""

import os
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_pathway_sweep as pps
import plot_bracket_sweep as pbs

REPO, DATA, FIGS = pps.REPO, pps.DATA, pps.FIGS

MONO_DIR = os.path.join(DATA, "coherence_sweep_mono_24748573")
SASE_DIR = os.path.join(DATA, "coherence_sweep_sase_24748571")
EXP_MONO_SCATTER = os.path.normpath(os.path.join(REPO, "..", "RSA-derivation-bloch", "data",
                                                 "exp_mono_scatter_slide9bins.csv"))

VARIANTS = ["R", "raman10", "broad25", "raman10-broad25"]
COLOUR = dict(zip(VARIANTS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))  # validated categorical slots 1-4
MARKER = dict(zip(VARIANTS, ["o", "s", "^", "D"]))
LABEL = {
    "R": "R: middlemen + CK + EII",
    "raman10": "+ Raman dephasing 10/fs (no dark state)",
    "broad25": "+ broadening 2.5/fs (+3.3 eV FWHM)",
    "raman10-broad25": "+ both",
}
INK, INK_2, MUTED = pbs.INK, pbs.INK_2, pbs.MUTED

MONO_E_SEED = [1.0, 5.0, 20.0, 30.0]
SASE_E_SEED = [2.0, 9.0, 20.0, 40.0]
KA1_WINDOW = (8040.0, 8054.0)
KA2_WINDOW = (8022.0, 8032.0)
AREA_BAND = pbs.AREA_BAND


# --------------------------------------------------------------------------- loading

def exp_mono(path=EXP_MONO_SCATTER):
    """{uJ: (E, T, T_err)} for the self-seeded beam; the nominal uJ is the centre of the slide-9 bin."""
    if not os.path.exists(path):
        print(f"WARNING: {path} missing, falling back to the by-eye transmittance_seeded_uJ.csv")
        return {u: (E, T, np.zeros_like(T)) for u, (E, T) in pbs.exp_mono_curves().items()}
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(line for line in fh if not line.startswith("#")):
            uJ = round(0.5 * (float(r["E_lo"]) + float(r["E_hi"])), 3)
            rows.setdefault(uJ, []).append((float(r["E_ph"]), float(r["T"]), float(r["T_err"])))
    return {u: tuple(np.array(sorted(v)).T) for u, v in sorted(rows.items())}


# --------------------------------------------------------------------------- metrics

def peak(E, A, window):
    m = (E >= window[0]) & (E <= window[1])
    return float(A[m].max())


def mono_exp_absorbance(exp, uJ):
    E, T, err = exp[uJ]
    T_wing = float(np.mean(np.interp([8000.0, 8005.0], E, T)))
    return E, np.log(T_wing / T), err / T


def mono_metrics(fam, exp):
    """{observable: {variant or 'experiment': [value per MONO_E_SEED]}}"""
    grid = sorted(fam["R"][20.0])
    curves = {v: [pbs.absorbance_model(fam, v, u) for u in MONO_E_SEED] for v in VARIANTS}
    curves["experiment"] = [mono_exp_absorbance(exp, u)[:2] for u in MONO_E_SEED]
    out = {"Kα1 depth": {}, "Kα2 depth": {}, "band area": {}}
    for key, cs in curves.items():
        out["Kα1 depth"][key] = [peak(E, A, KA1_WINDOW) for E, A in cs]
        out["Kα2 depth"][key] = [peak(E, A, KA2_WINDOW) for E, A in cs]
        out["band area"][key] = [pbs.area(E, A, grid=grid) for E, A in cs]
    return out


def sase_metrics(sase, exp_sase):
    out = {k: {} for k in ("depth", "fwhm", "area")}
    for v in VARIANTS:
        ms = [pbs.sase_metrics(*sase[v][u]) for u in SASE_E_SEED]
        for k in out:
            out[k][v] = [m[k] for m in ms]
    exp_uJ = sorted(u for u in exp_sase if u >= 1.0)  # the 0.12/0.5 uJ lines are too weak for a width
    ms = [pbs.sase_metrics(exp_sase[u][0], exp_sase[u][1], 2.0) for u in exp_uJ]
    for k in out:
        out[k]["experiment"] = [m[k] for m in ms]
    return out, exp_uJ


# --------------------------------------------------------------------------- figures

def _save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=150)
    fig.savefig(os.path.join(FIGS, name + ".pdf"))
    plt.close(fig)


def _variant_line(ax, x, y, v, ms=3.2):
    ax.plot(x, y, color=COLOUR[v], marker=MARKER[v], ms=ms, lw=1.9 if v == "R" else 1.4,
            label=LABEL[v], zorder=4 if v == "R" else 3)


def _legend_below(fig, axes):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False, fontsize=8)


def fig_mono_absorbance(fam, exp):
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.7), sharex=True, sharey=True, layout="constrained")
    for ax, uJ in zip(axes, MONO_E_SEED):
        for v in VARIANTS:
            _variant_line(ax, *pbs.absorbance_model(fam, v, uJ), v, ms=2.8)
        Ee, Ae, sA = mono_exp_absorbance(exp, uJ)
        ax.errorbar(Ee, Ae, yerr=sA, color=INK, ls="--", marker="s", ms=2.6, lw=1.0, elinewidth=0.7,
                    capsize=0, label="experiment (scatter-panel bins)", zorder=5)
        pbs._line_markers(ax)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xlim(8000, 8072)
        ax.set_ylim(-0.02, 0.43)
        ax.set_title(f"{uJ:g} µJ", fontsize=9.5)
        ax.set_xlabel("Photon energy (eV)")
    axes[0].set_ylabel(r"$\ln(T_{wing}/T)$ per 20 µm")
    _legend_below(fig, axes)
    fig.suptitle("Self-seeded beam: Raman dephasing deepens Kα1 only (by ~45% when saturated); broadening "
                 "trades peak for wings. Neither fills the band between the lines", fontsize=10.5)
    _save(fig, "coherence_mono_absorbance")


def fig_sase_absorbance(sase, exp_sase):
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.7), sharex=True, sharey=True, layout="constrained")
    for ax, uJ in zip(axes, SASE_E_SEED):
        for v in VARIANTS:
            E, T = sase[v][uJ]
            ax.plot(E, pps.absorbance(E, T, pps.wing_T(E, T, pps.SASE_WING)), color=COLOUR[v],
                    lw=1.9 if v == "R" else 1.4, label=LABEL[v], zorder=4 if v == "R" else 3)
        if uJ in exp_sase:
            Ee, Te = exp_sase[uJ][0], exp_sase[uJ][1]
            ax.plot(Ee, pps.absorbance(Ee, Te, pps.wing_T(Ee, Te, pps.SASE_WING), 2.0), color=INK, ls="--",
                    lw=1.1, label="experiment (2 eV boxcar)", zorder=5)
        else:
            ax.annotate("no measurement at this energy", (0.97, 0.95), xycoords="axes fraction", ha="right",
                        va="top", fontsize=7.5, color=INK_2)
        pbs._line_markers(ax)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xlim(8024, 8070)
        ax.set_title(f"{uJ:g} µJ", fontsize=9.5)
        ax.set_xlabel("Photon energy (eV)")
    axes[0].set_ylabel(r"$\ln(T_{wing}/T)$, $T_{wing}$ = 8060-8070 eV")
    handles, labels = [], []
    for ax in axes:  # the experiment handle lives in the 2 uJ panel
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False, fontsize=8)
    fig.suptitle("SASE beam (5x5 grid): broadening matches the measured 2-9 µJ width but halves the peak; "
                 "the measured line holds ~5x the area at 2 µJ", fontsize=10.5)
    _save(fig, "coherence_sase_absorbance")


def fig_summary(mono, sase, exp_sase_uJ):
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.8), layout="constrained")
    panels = [
        (axes[0][0], mono["Kα1 depth"], MONO_E_SEED, MONO_E_SEED, "Mono Kα1 depth, per 20 µm",
         "Raman dephasing lifts the plateau 0.17 → 0.25"),
        (axes[0][1], mono["Kα2 depth"], MONO_E_SEED, MONO_E_SEED, "Mono Kα2 depth, per 20 µm",
         "Kα2 (2p1/2, no dark state) is untouched by it"),
        (axes[0][2], mono["band area"], MONO_E_SEED, MONO_E_SEED, "Mono area 8012-8072 eV (eV)",
         "Both levers together: +45%, still 2.1-2.3x short"),
        (axes[1][0], sase["depth"], SASE_E_SEED, exp_sase_uJ, "SASE peak depth",
         "Model onset too late; Raman overshoots at 40 µJ"),
        (axes[1][1], sase["fwhm"], SASE_E_SEED, exp_sase_uJ, "SASE FWHM (eV)",
         "Broadening fits 2-9 µJ, not the growth to 14 eV"),
        (axes[1][2], sase["area"], SASE_E_SEED, exp_sase_uJ, "SASE area 8025-8070 eV (eV)",
         "Area is 2.3-5x short in every variant"),
    ]
    for ax, data, x_model, x_exp, ylabel, title in panels:
        for v in VARIANTS:
            _variant_line(ax, x_model, data[v], v, ms=4)
        ax.plot(x_exp, data["experiment"], color=INK, ls="--", marker="s", ms=4.5, lw=1.1,
                label="experiment", zorder=5)
        ax.set_xscale("log")
        ax.set_xticks(x_model if len(x_model) >= len(x_exp) else x_exp)
        ax.set_xticklabels([f"{u:g}" for u in ax.get_xticks()])
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9.5)
    for ax in axes[1]:
        ax.set_xlabel("Pulse energy (µJ)")
    _legend_below(fig, axes[0])
    fig.suptitle("What the two coherence levers buy: Raman dephasing moves the saturated Kα1 depth, broadening "
                 "moves the width, and neither supplies the missing area", fontsize=10.5)
    _save(fig, "coherence_summary")


# --------------------------------------------------------------------------- report

def print_report(mono, sase, exp_sase_uJ):
    print("\n=== mono: model variants | experiment ===")
    for name, data in mono.items():
        print(f"\n{name}")
        print(f"{'uJ':>5} " + "".join(f"{v:>17}" for v in VARIANTS) + f"{'exp':>9}")
        for i, u in enumerate(MONO_E_SEED):
            print(f"{u:5g} " + "".join(f"{data[v][i]:17.3f}" for v in VARIANTS) + f"{data['experiment'][i]:9.3f}")
    print("\n=== mono: share of the gap to experiment closed, relative to R (%) ===")
    for name, data in mono.items():
        cells = []
        for v in VARIANTS[1:]:
            fr = [(data[v][i] - data["R"][i]) / (data["experiment"][i] - data["R"][i]) for i in range(len(MONO_E_SEED))]
            cells.append(f"{v}: " + "/".join(f"{100 * f:+.0f}" for f in fr))
        print(f"{name:10s} " + "   ".join(cells) + f"   (at {'/'.join(f'{u:g}' for u in MONO_E_SEED)} uJ)")
    print("\n=== SASE: depth / FWHM (eV) / area (eV) ===")
    for i, u in enumerate(SASE_E_SEED):
        row = [f"{v} {sase['depth'][v][i]:.3f}/{sase['fwhm'][v][i]:.1f}/{sase['area'][v][i]:.2f}" for v in VARIANTS]
        if u in exp_sase_uJ:
            j = exp_sase_uJ.index(u)
            row.append(f"exp {sase['depth']['experiment'][j]:.3f}/{sase['fwhm']['experiment'][j]:.1f}/"
                       f"{sase['area']['experiment'][j]:.2f}")
        print(f"  {u:4g} uJ: " + "   ".join(row))
    print("\n=== ratio to R (depth, area): the Raman lever tracks saturation ===")
    for name, data, uJs in [("mono Kα1", mono["Kα1 depth"], MONO_E_SEED), ("mono area", mono["band area"], MONO_E_SEED),
                            ("SASE depth", sase["depth"], SASE_E_SEED), ("SASE area", sase["area"], SASE_E_SEED)]:
        print(f"{name:11s} " + "   ".join(f"{v}: " + "/".join(f"{data[v][i] / data['R'][i]:.2f}" for i in range(len(uJs)))
                                         for v in VARIANTS[1:]) + f"   (at {'/'.join(f'{u:g}' for u in uJs)} uJ)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mono", default=MONO_DIR)
    parser.add_argument("--sase", default=SASE_DIR)
    parser.add_argument("--exp-mono", default=EXP_MONO_SCATTER)
    args = parser.parse_args()

    os.makedirs(FIGS, exist_ok=True)
    fam = pbs.load_mono_family(args.mono)
    sase = {v: pbs.load_sase_variant(os.path.join(args.sase, v)) for v in VARIANTS}
    missing = [v for v in VARIANTS if v not in fam or set(fam[v]) != set(MONO_E_SEED)
               or set(sase[v]) != set(SASE_E_SEED)]
    if missing:
        raise SystemExit(f"incomplete sweep data for variants {missing}")
    exp = exp_mono(args.exp_mono)
    _, exp_sase, _ = pps.load_experiment()

    mono = mono_metrics(fam, exp)
    sase_m, exp_sase_uJ = sase_metrics(sase, exp_sase)
    fig_mono_absorbance(fam, exp)
    fig_sase_absorbance(sase, exp_sase)
    fig_summary(mono, sase_m, exp_sase_uJ)
    print_report(mono, sase_m, exp_sase_uJ)
    print(f"\nwrote figures to {FIGS}")
