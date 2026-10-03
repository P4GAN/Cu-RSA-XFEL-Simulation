"""Figures of the final sweep (scripts/generate_final_sweep.sh): the self-seeded absorbance spectrum of the model,
built up step by step, against the measured one.

Reads data/final_sweep_mono_<job id>/<step>/ (default: the newest such folder) and writes into figs/:
  final_steps              2x2 panels at 1/5/20/30 uJ, every model step and the experiment
  final_buildup_20uJ_<k>   one panel at 20 uJ per step k: steps 1..k (k highlighted) and the experiment
It also prints the peak absorbance in the Kalpha1 and Kalpha2 windows and the area under each spectrum.
Steps that have not finished are left out, so a sweep can be plotted while it runs.

Absorbance A(E) = ln(T_wing / T(E)), with T_wing the transmittance at 8000 eV (model) or the mean of 8000 and
8005 eV (experiment), so the cold, non-resonant absorption drops out.

Run from the repo root:  python scripts/plot_final.py [--data DIR] [--exp CSV]
"""

import argparse
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA = os.path.join(REPO, "data")
FIGS = os.path.join(REPO, "figs")
EXP_CSV = os.path.join(DATA, "experiment", "cu_20um_self_seeded_transmittance.csv")

STEPS = ["1-bare", "2-single", "3-double", "4-middlemen", "5-electrons"]
LABEL = {
    "1-bare": "1  2p and 1s holes (Kα1, Kα2)",
    "2-single": "2  + 2s holes, single satellites",
    "3-double": "3  + double satellites",
    "4-middlemen": "4  + middlemen, metal Coster–Kronig",
    "5-electrons": "5  + free electrons (EII)",
}
COLOUR = dict(zip(STEPS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MARKER = dict(zip(STEPS, ["o", "^", "D", "v", "P"]))
INK, INK_2, MUTED, GRIDC = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

UJ = [1.0, 5.0, 20.0, 30.0]
N_ENERGIES = 25                       # photon energies per pulse energy in generate_final_sweep.sh
XLIM = (8000, 8070)
KA1_EV, KA2_EV = 8047.91, 8027.98
KA1_WINDOW = (8040.0, 8054.0)
KA2_WINDOW = (8022.0, 8032.0)
AREA_BAND = (8012.0, 8072.0)
FOOTNOTE = ("Model: 20 µm Cu foil, self-seeded pulse, 10 shots per point, on the measured photon energies. "
            "Experiment: median of the single shots in each pulse-energy bin, ±1 standard error.")

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12.5, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 10.5, "axes.linewidth": 1.0, "lines.linewidth": 2.2,
                     "grid.color": GRIDC, "axes.edgecolor": MUTED, "xtick.color": INK_2, "ytick.color": INK_2})


# ------------------------------------------------------------------------------------------------ loading

def mean_output(run_dir, key):
    """Shot-averaged output `key` over the chunk files of one run folder (sums and counts pooled). A
    '.partial.npz' checkpoint is used only when the folder has no finished chunk, and unreadable files
    (copied while being written) are skipped."""
    paths = glob.glob(os.path.join(run_dir, "*.npz"))
    final = [p for p in paths if not p.endswith(".partial.npz")]
    total = count = None
    for path in final or paths:
        try:
            with np.load(path) as d:
                if key + "_sum" not in d.files:
                    continue
                s, c = np.real(d[key + "_sum"]), d[key + "_count"]
        except Exception:
            continue
        total, count = (s, c) if total is None else (total + s, count + c)
    return None if total is None else total / np.maximum(count, 1)


def load_family(root):
    """{step: {E_seed: {photon energy: T}}}, T the transmitted over the incident pulse energy."""
    out = {}
    for vdir in sorted(glob.glob(os.path.join(root, "*"))):
        for run in glob.glob(os.path.join(vdir, "runs_seed_*")):
            m = re.search(r"runs_seed_([\d.]+)_uJ__energy_([\d.]+)_eV", os.path.basename(run))
            if not m:
                continue
            I_out, I_in = mean_output(run, "I_int_thy_w_last"), mean_output(run, "I_int_thy_w_0")
            if I_out is None or I_in is None:
                continue
            by_uJ = out.setdefault(os.path.basename(vdir), {}).setdefault(float(m.group(1)), {})
            by_uJ[float(m.group(2))] = float(I_out.sum() / I_in.sum())
    return out


def load_experiment(path):
    """{pulse-energy bin centre (uJ): (E, T, T_err)}."""
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(line for line in fh if not line.startswith("#")):
            uJ = round(0.5 * (float(r["E_lo"]) + float(r["E_hi"])), 3)
            rows.setdefault(uJ, []).append((float(r["E_ph"]), float(r["T"]), float(r["T_err"])))
    return {u: tuple(np.array(sorted(v)).T) for u, v in sorted(rows.items())}


def model_absorbance(fam, step, uJ):
    by_E = fam[step][uJ]
    E = np.array(sorted(by_E))
    T = np.array([by_E[e] for e in E])
    return E, np.log(by_E[8000.0] / T)


def experiment_absorbance(exp, uJ):
    E, T, err = exp[uJ]
    T_wing = float(np.mean(np.interp([8000.0, 8005.0], E, T)))
    return E, np.log(T_wing / T), err / T


def peak(E, A, window):
    m = (E >= window[0]) & (E <= window[1])
    return float(A[m].max())


def area(E, A, grid):
    """Trapezoid area over AREA_BAND after resampling onto the model's photon energies, so that model and
    experiment are integrated with the same quadrature."""
    grid = np.asarray([g for g in grid if AREA_BAND[0] <= g <= AREA_BAND[1]])
    return float(np.trapz(np.interp(grid, E, A), grid))


# ------------------------------------------------------------------------------------------------ figures

def _save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=200)
    fig.savefig(os.path.join(FIGS, name + ".pdf"))
    plt.close(fig)


def _model(ax, fam, step, uJ, ms=4.5, lw=2.2, faded=False, zorder=3):
    E, A = model_absorbance(fam, step, uJ)
    if faded:
        ax.plot(E, A, color=MUTED, lw=1.2, alpha=0.8, zorder=2)
    else:
        ax.plot(E, A, color=COLOUR[step], marker=MARKER[step], ms=ms, lw=lw, label=LABEL[step], zorder=zorder)


def _experiment(ax, exp, uJ, ms=4.5):
    E, A, sA = experiment_absorbance(exp, uJ)
    keep = E <= XLIM[1]
    ax.errorbar(E[keep], A[keep], yerr=sA[keep], color=INK, ls="--", marker="s", ms=ms, lw=1.4, elinewidth=1.0,
                capsize=0, label="experiment", zorder=5)


def _frame(ax, uJ, ylim):
    for E_line in (KA2_EV, KA1_EV):
        ax.axvline(E_line, color=MUTED, ls=":", lw=1.0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*ylim)
    ax.set_title(f"{uJ:g} µJ")
    ax.grid(True, color=GRIDC, lw=0.7)


def _ylim(fam, exp, steps, uJs):
    top = max(max(model_absorbance(fam, s, u)[1].max() for s in steps for u in uJs),
              max(experiment_absorbance(exp, u)[1].max() for u in uJs))
    return (-0.03, 1.08 * top)


def fig_steps(fam, exp, steps):
    ylim = _ylim(fam, exp, steps, UJ)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), sharex=True, sharey=True, layout="constrained")
    for ax, uJ in zip(axes.flat, UJ):
        for step in steps:
            _model(ax, fam, step, uJ)
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
    fig.suptitle("The model step by step against the measured self-seeded absorbance", fontsize=14)
    _save(fig, "final_steps")


def fig_buildup(fam, exp, steps, uJ=20.0):
    ylim = _ylim(fam, exp, steps, [uJ])
    for k, step in enumerate(steps, start=1):
        fig, ax = plt.subplots(figsize=(8.6, 5.6), layout="constrained")
        for earlier in steps[:k - 1]:
            _model(ax, fam, earlier, uJ, faded=True)
        _model(ax, fam, step, uJ, ms=5.5, lw=2.8, zorder=4)
        _experiment(ax, exp, uJ, ms=5)
        _frame(ax, uJ, ylim)
        ax.set_title(f"{LABEL[step]}   ({uJ:g} µJ; earlier steps in grey)", fontsize=13)
        ax.set_xlabel("Photon energy (eV)")
        ax.set_ylabel(r"Absorbance  $\ln(T_{wing}/T)$")
        ax.legend(frameon=False, loc="upper left")
        _save(fig, f"final_buildup_{uJ:g}uJ_{k}")


def print_report(fam, exp, steps):
    print("\npeak absorbance in Kalpha1 (8040-8054 eV) and Kalpha2 (8022-8032 eV), and area over 8012-8072 eV,"
          " at 1/5/20/30 uJ")
    grid = sorted(fam[steps[0]][20.0])
    curves = {s: [model_absorbance(fam, s, u) for u in UJ] for s in steps}
    curves["experiment"] = [experiment_absorbance(exp, u)[:2] for u in UJ]
    for name, cs in curves.items():
        print(f"  {name:12s} Ka1 " + " ".join(f"{peak(E, A, KA1_WINDOW):.3f}" for E, A in cs)
              + "   Ka2 " + " ".join(f"{peak(E, A, KA2_WINDOW):.3f}" for E, A in cs)
              + "   area " + " ".join(f"{area(E, A, grid):.2f}" for E, A in cs))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None, help="final_sweep_mono_<job id> folder (default: the newest)")
    parser.add_argument("--exp", default=EXP_CSV, help="measured transmittance CSV")
    args = parser.parse_args()

    root = args.data
    if root is None:
        found = sorted(glob.glob(os.path.join(DATA, "final_sweep_mono_*")), key=os.path.getmtime)
        if not found:
            raise SystemExit("no data/final_sweep_mono_* folder; pass --data")
        root = found[-1]
    loaded = load_family(root)
    fam = {s: d for s, d in loaded.items() if s in STEPS and all(len(d.get(u, {})) == N_ENERGIES for u in UJ)}
    for s in STEPS:
        if s not in fam:
            done = sum(len(loaded.get(s, {}).get(u, {})) for u in UJ)
            print(f"not plotted: {s} ({done}/{len(UJ) * N_ENERGIES} points finished)")
    steps = [s for s in STEPS if s in fam]
    if not steps:
        raise SystemExit(f"no complete step in {root}")

    exp = load_experiment(args.exp)
    os.makedirs(FIGS, exist_ok=True)
    fig_steps(fam, exp, steps)
    fig_buildup(fam, exp, steps)
    print_report(fam, exp, steps)
    print(f"\nread {root}; wrote figures to {FIGS}")


if __name__ == "__main__":
    main()
