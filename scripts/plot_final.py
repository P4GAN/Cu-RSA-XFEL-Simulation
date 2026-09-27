"""Presentation figures for the final family (scripts/generate_final_sweeps.sh): self-seeded absorbance spectra
of the model built up step by step, and of the final experiments, against the measured spectra.

Reads data/final_sweep_mono_<id>/<variant>/ (default: the newest such folder) with variants
  1-bare, 2-single, 3-double, 4-middlemen, 5-electrons   the model steps (docs/final-model-progression.md)
  E1-deph, E2-core, E3-exchange                          final experiments on top of step 5
and writes into figs/ (PNG for slides, PDF alongside):
  final_steps                 2x2: 1/5/20/30 uJ, every model step + experiment
  final_buildup_20uJ_<k>      one panel at 20 uJ per step k: steps 1..k (k highlighted) + experiment, for a
                              slide-by-slide build-up
  final_experiments           2x2: step 5 and the final experiments + experiment
Variants that are not (yet) in the folder are left out, so a partially finished run can be plotted.

Absorbance A(E) = ln(T_wing / T(E)): T_wing = T(8000 eV) for the model (a true 20 um Cu foil, every absorption
step counted) and the mean of 8000/8005 eV for the experiment (the slide-9 scatter-panel bins of
../RSA-derivation-bloch/data/exp_mono_scatter_slide9bins.csv, error bars = bin standard error). The model runs on
the measured photon energies.

Run from the repo root:  python scripts/plot_final.py [--data DIR]
"""

import os
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_pathway_sweep as pps
import plot_bracket_sweep as pbs
import plot_coherence_sweep as pcs
import plot_population_budget as ppb

DATA, FIGS = pps.DATA, pps.FIGS
STEPS = ["1-bare", "2-single", "3-double", "4-middlemen", "5-electrons"]
EXPERIMENTS = ["E1-deph", "E2-core", "E3-exchange"]
# validated categorical slots 1-8 in fixed order: a variant keeps its colour in every figure
COLOUR = dict(zip(STEPS + EXPERIMENTS, ppb.SLOTS[:8]))
MARKER = dict(zip(STEPS + EXPERIMENTS, ["o", "^", "D", "v", "P", "h", "X", "s"]))
LABEL = {
    "1-bare": "1  2p and 1s holes (Kα1, Kα2)",
    "2-single": "2  + 2s holes, single satellites",
    "3-double": "3  + double satellites",
    "4-middlemen": "4  + middlemen, metal Coster–Kronig",
    "5-electrons": "5  + free electrons (EII)",
    "E1-deph": "E1  + collisional dephasing",
    "E2-core": "E2  + EII of core-holed ions",
    "E3-exchange": "E3  + 2p–3d exchange mixing",
}
EXP_LABEL = "experiment"
INK, INK_2, MUTED, GRIDC = ppb.INK, ppb.INK_2, ppb.MUTED, ppb.GRIDC
UJ = [1.0, 5.0, 20.0, 30.0]
XLIM = (8000, 8070)
FOOTNOTE = ("Model: 20 µm Cu foil, self-seeded pulse, 10 shots per point, on the measured photon energies. "
            "Experiment: scatter-panel bins, ±1 standard error.")

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12.5, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 10.5, "axes.linewidth": 1.0, "lines.linewidth": 2.2})


def newest_family():
    dirs = sorted(glob.glob(os.path.join(DATA, "final_sweep_mono_*")), key=os.path.getmtime)
    return dirs[-1] if dirs else None


def _save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=200)
    fig.savefig(os.path.join(FIGS, name + ".pdf"))
    plt.close(fig)


def _model(ax, fam, v, uJ, ms=4.5, lw=2.2, faded=False, zorder=3):
    E, A = pbs.absorbance_model(fam, v, uJ)
    if faded:
        ax.plot(E, A, color=MUTED, lw=1.2, alpha=0.8, zorder=2)
    else:
        ax.plot(E, A, color=COLOUR[v], marker=MARKER[v], ms=ms, lw=lw, label=LABEL[v], zorder=zorder)


def _experiment(ax, exp, uJ, ms=4.5):
    E, A, sA = pcs.mono_exp_absorbance(exp, uJ)
    keep = E <= XLIM[1]
    ax.errorbar(E[keep], A[keep], yerr=sA[keep], color=INK, ls="--", marker="s", ms=ms, lw=1.4, elinewidth=1.0,
                capsize=0, label=EXP_LABEL, zorder=5)


def _frame(ax, uJ, ylim):
    for E_line, name in ((pps.KA2_EV, "Kα2"), (pps.KA1_EV, "Kα1")):
        ax.axvline(E_line, color=MUTED, ls=":", lw=1.0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*ylim)
    ax.set_title(f"{uJ:g} µJ")
    ax.grid(True, color=GRIDC, lw=0.7)


def _ylim(fam, exp, variants, uJs):
    top = max(max(pbs.absorbance_model(fam, v, u)[1].max() for v in variants for u in uJs),
              max(pcs.mono_exp_absorbance(exp, u)[1].max() for u in uJs))
    return (-0.03, 1.08 * top)


def _grid(fam, exp, variants, name, title):
    ylim = _ylim(fam, exp, variants, UJ)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), sharex=True, sharey=True, layout="constrained")
    for ax, uJ in zip(axes.flat, UJ):
        for v in variants:
            _model(ax, fam, v, uJ)
        _experiment(ax, exp, uJ)
        _frame(ax, uJ, ylim)
    for ax in axes[-1]:
        ax.set_xlabel("Photon energy (eV)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Absorbance  $\ln(T_{wing}/T)$")
    handles, labels = axes[0][0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, title=FOOTNOTE,
                     title_fontsize=9)
    leg.get_title().set_color(INK_2)
    fig.suptitle(title, fontsize=14)
    _save(fig, name)


def fig_buildup(fam, exp, uJ=20.0):
    steps = [v for v in STEPS if v in fam]
    ylim = _ylim(fam, exp, steps, [uJ])
    for k, v in enumerate(steps, start=1):
        fig, ax = plt.subplots(figsize=(8.6, 5.6), layout="constrained")
        for earlier in steps[:k - 1]:
            _model(ax, fam, earlier, uJ, faded=True)
        _model(ax, fam, v, uJ, ms=5.5, lw=2.8, zorder=4)
        _experiment(ax, exp, uJ, ms=5)
        _frame(ax, uJ, ylim)
        ax.set_title(f"{LABEL[v]}   ({uJ:g} µJ; earlier steps in grey)", fontsize=13)
        ax.set_xlabel("Photon energy (eV)")
        ax.set_ylabel(r"Absorbance  $\ln(T_{wing}/T)$")
        ax.legend(frameon=False, loc="upper left")
        _save(fig, f"final_buildup_{uJ:g}uJ_{k}")


def print_report(fam, exp):
    print("\npeak absorbance in Kalpha1 (8040-8054) / Kalpha2 (8022-8032) windows and area 8012-8072 eV, 1/5/20/30 uJ")
    grid = sorted(fam[next(iter(fam))][20.0])
    for v in [v for v in STEPS + EXPERIMENTS if v in fam]:
        cs = [pbs.absorbance_model(fam, v, u) for u in UJ]
        print(f"  {v:12s} Ka1 " + " ".join(f"{pcs.peak(E, A, pcs.KA1_WINDOW):.3f}" for E, A in cs)
              + "   Ka2 " + " ".join(f"{pcs.peak(E, A, pcs.KA2_WINDOW):.3f}" for E, A in cs)
              + "   area " + " ".join(f"{pbs.area(E, A, grid=grid):.2f}" for E, A in cs))
    cs = [pcs.mono_exp_absorbance(exp, u)[:2] for u in UJ]
    print(f"  {'experiment':12s} Ka1 " + " ".join(f"{pcs.peak(E, A, pcs.KA1_WINDOW):.3f}" for E, A in cs)
          + "   Ka2 " + " ".join(f"{pcs.peak(E, A, pcs.KA2_WINDOW):.3f}" for E, A in cs)
          + "   area " + " ".join(f"{pbs.area(E, A, grid=grid):.2f}" for E, A in cs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None, help="final_sweep_mono_<id> folder (default: the newest)")
    parser.add_argument("--exp-mono", default=pcs.EXP_MONO_SCATTER)
    args = parser.parse_args()
    root = args.data or newest_family()
    if not root:
        raise SystemExit("no data/final_sweep_mono_* folder; pass --data")
    fam = {v: d for v, d in pbs.load_mono_family(root).items() if v in STEPS + EXPERIMENTS and set(d) >= set(UJ)}
    missing = [v for v in STEPS + EXPERIMENTS if v not in fam]
    if missing:
        print(f"not plotted (missing or incomplete): {missing}")
    exp = pcs.exp_mono(args.exp_mono)
    os.makedirs(FIGS, exist_ok=True)
    steps = [v for v in STEPS if v in fam]
    if steps:
        _grid(fam, exp, steps, "final_steps", "The model step by step against the measured self-seeded absorbance")
        fig_buildup(fam, exp)
    exps = [v for v in ["5-electrons"] + EXPERIMENTS if v in fam]
    if len(exps) > 1:
        _grid(fam, exp, exps, "final_experiments", "Final experiments on the full model (step 5)")
    print_report(fam, exp)
    print(f"\nread {root}; wrote figures to {FIGS}")
