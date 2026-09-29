"""Can more electron-impact ionisation reach the measurement? Plots the eii_scale family
(scripts/generate_eii_scale_sweeps.sh): step 5 of the final family with the L-shell EII cross sections x2, x4,
x8, x16, against the experiment. Scale x1 is 5-electrons of the final family.

Reads (default: the newest of each)
  data/eii_scale_sweep_mono_<id>/x<k>/     self-seeded spectra, 25 measured energies x 1/5/20/30 uJ
  data/final_sweep_mono_<id>/5-electrons/  scale 1
  data/eii_scale_budget_mono_<id>/x<k>/    population budget at 8048 eV (optional), with scale 1 from
                                           data/gap_budget_mono_24933285/c-bs (the same physics)
and writes into figs/ (PNG + PDF):
  eii_scale_spectra   2x2: absorbance at 1/5/20/30 uJ for every scale, against the experiment
  eii_scale_summary   model / experiment for the Kalpha1 peak, the Kalpha2 peak and the band area vs scale; the
                      20 uJ spectra normalised to their Kalpha1 peak (does the shape change?)
  eii_scale_budget    (with budget data) direct 2p3/2 holes from EII vs photoionisation, during the pulse and
                      in the whole window, vs scale
Definitions as scripts/plot_final.py. Variants not (yet) complete are left out.

Run from the repo root:  python scripts/plot_eii_scale.py [--data DIR] [--final DIR] [--budget DIR]
"""

import os
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

import plot_final as pf
import plot_bracket_sweep as pbs
import plot_coherence_sweep as pcs
import plot_eii_diagnostics as ped
import plot_population_budget as ppb

DATA, FIGS = pf.DATA, pf.FIGS
SCALES = [1, 2, 4, 8, 16]
# ordered parameter -> one hue, light to dark; scale 1 (the model as it is) in neutral grey
COLOUR = dict(zip(SCALES, [ppb.MUTED] + ppb.RAMP))
MARKER = dict(zip(SCALES, ["o", "^", "D", "v", "s"]))
LABEL = {1: "×1  step 5 (Bote–Salvat)", 2: "×2  physical upper bound", 4: "×4", 8: "×8", 16: "×16  energy ceiling"}
UJ_COLOUR = dict(zip(pf.UJ, ppb.SLOTS[:4]))
INK, INK_2, MUTED, GRIDC = pf.INK, pf.INK_2, pf.MUTED, pf.GRIDC
REF_BUDGET = os.path.join(DATA, "gap_budget_mono_24933285", "c-bs")

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12.5, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 10.5, "axes.linewidth": 1.0, "lines.linewidth": 2.2,
                     "savefig.bbox": "standard"})


def newest(pattern):
    dirs = sorted(glob.glob(os.path.join(DATA, pattern)), key=os.path.getmtime)
    return dirs[-1] if dirs else None


def load(scale_root, final_root):
    """{scale: {uJ: {energy: T}}} for the complete variants."""
    out = {}
    fam = pbs.load_mono_family(final_root)
    if "5-electrons" in fam:
        out[1] = fam["5-electrons"]
    for name, d in pbs.load_mono_family(scale_root).items():
        if name.startswith("x") and name[1:].isdigit():
            out[int(name[1:])] = d
    complete = {k: d for k, d in out.items() if all(len(d.get(u, {})) == pf.N_ENERGIES for u in pf.UJ)}
    for k in SCALES:
        if k not in complete:
            done = sum(len(out.get(k, {}).get(u, {})) for u in pf.UJ)
            print(f"not plotted: x{k} ({done}/{len(pf.UJ) * pf.N_ENERGIES} points finished)")
    return {k: complete[k] for k in SCALES if k in complete}


def _A(fam, k, uJ):
    return pbs.absorbance_model({k: fam[k]}, k, uJ)


def metrics(fam, exp):
    grid = sorted(fam[next(iter(fam))][20.0])
    out = {}
    for k in fam:
        for u in pf.UJ:
            E, A = _A(fam, k, u)
            out[k, u] = (pcs.peak(E, A, pcs.KA1_WINDOW), pcs.peak(E, A, pcs.KA2_WINDOW), pbs.area(E, A, grid=grid))
    for u in pf.UJ:
        E, A, _ = pcs.mono_exp_absorbance(exp, u)
        out["exp", u] = (pcs.peak(E, A, pcs.KA1_WINDOW), pcs.peak(E, A, pcs.KA2_WINDOW), pbs.area(E, A, grid=grid))
    return out


def fig_spectra(fam, exp):
    top = max(max(_A(fam, k, u)[1].max() for k in fam for u in pf.UJ),
              max(pcs.mono_exp_absorbance(exp, u)[1].max() for u in pf.UJ))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), sharex=True, sharey=True, layout="constrained")
    for ax, u in zip(axes.flat, pf.UJ):
        for k in fam:
            E, A = _A(fam, k, u)
            ax.plot(E, A, color=COLOUR[k], marker=MARKER[k], ms=4.5, label=LABEL[k], zorder=3)
        pf._experiment(ax, exp, u)
        pf._frame(ax, u, (-0.03, 1.08 * top))
    for ax in axes[-1]:
        ax.set_xlabel("Photon energy (eV)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Absorbance  $\ln(T_{wing}/T)$")
    handles, labels = axes[0][0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False,
                     title="L-shell electron-impact cross sections scaled by ×k; every other ingredient as step 5.\n"
                           + pf.FOOTNOTE, title_fontsize=9)
    leg.get_title().set_color(INK_2)
    fig.suptitle("Scaling up electron-impact ionisation, against the measured self-seeded absorbance", fontsize=14)
    pf._save(fig, "eii_scale_spectra")


def _scale_axis(ax, scales):
    ax.set_xscale("log", base=2)
    ax.set_xticks(scales)
    ax.set_xticklabels([f"×{k}" for k in scales])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.axvspan(0.8, 2.0, color=GRIDC, alpha=0.6, lw=0)
    ax.set_xlim(0.8, 20)


def fig_summary(fam, exp, m):
    scales = list(fam)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.0), layout="constrained")
    for ax, (i, name) in zip(axes.flat[:3], enumerate(["Kα1 peak", "Kα2 peak", "band area 8012–8072 eV"])):
        for u in pf.UJ:
            ax.plot(scales, [m[k, u][i] / m["exp", u][i] for k in scales], color=UJ_COLOUR[u], marker="o", ms=6,
                    label=f"{u:g} µJ")
        ax.axhline(1.0, color=INK, lw=1.1, ls="--")
        ax.text(19, 1.0, "experiment", ha="right", va="bottom", fontsize=10)
        _scale_axis(ax, scales)
        ax.set_ylim(0, max(1.15, max(m[k, u][i] / m["exp", u][i] for k in scales for u in pf.UJ) * 1.08))
        ax.set_xlabel("L-shell EII cross section scale")
        ax.set_ylabel("model / experiment")
        ax.set_title(f"({'abc'[i]}) {name}")
    axes[0, 0].text(1.41, 0.04, "physically\ndefensible", ha="center", va="bottom", fontsize=9.5, color=INK_2)
    axes[0, 0].legend(frameon=False, loc="lower right", title="pulse energy", ncol=2)

    ax = axes[1, 1]
    u = 20.0
    for k in [s for s in (1, max(scales)) if s in fam]:
        E, A = _A(fam, k, u)
        ax.plot(E, A / pcs.peak(E, A, pcs.KA1_WINDOW), color=COLOUR[k], marker=MARKER[k], ms=4.5, label=LABEL[k])
    E, A, sA = pcs.mono_exp_absorbance(exp, u)
    keep = E <= pf.XLIM[1]
    p = pcs.peak(E, A, pcs.KA1_WINDOW)
    ax.errorbar(E[keep], A[keep] / p, yerr=sA[keep] / p, color=INK, ls="--", marker="s", ms=4.5, lw=1.4,
                elinewidth=1.0, capsize=0, label=pf.EXP_LABEL, zorder=5)
    for E_line in (pf.pps.KA2_EV, pf.pps.KA1_EV):
        ax.axvline(E_line, color=MUTED, ls=":", lw=1.0, zorder=0)
    ax.set_xlim(*pf.XLIM)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("absorbance / its Kα1 peak")
    ax.set_title(f"(d) Shape at {u:g} µJ: each curve divided by its Kα1 peak")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    for ax in axes.flat:
        ax.grid(True, color=GRIDC, lw=0.7)
    fig.suptitle("How far does scaling the L-shell electron-impact cross section get? (grey band: ×1–×2, "
                 "the physically defensible range)", fontsize=13.5)
    pf._save(fig, "eii_scale_summary")


def fig_budget(budget):
    """budget: {scale: StepRun at 8048 eV for each uJ}."""
    scales = sorted(budget)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained")
    ax = axes[0]
    u = 20.0
    for in_pulse, ls, mfc, tag in ((True, "-", None, "during the pulse FWHM"), (False, "--", "white", "whole window")):
        for src, c, mk in (("photo", ped.C_PHOTO, "o"), ("eii", ped.C_EII, "s")):
            vals = []
            for k in scales:
                r = budget[k][u]
                rate = r.photo_rate_L3() if src == "photo" else r.eii_rate_L3().sum(axis=0)
                vals.append(r.in_pulse(rate) if in_pulse else r.cumulative(rate)[-1])
            ax.plot(scales, vals, color=c, marker=mk, ms=6, ls=ls, mfc=mfc,
                    label=f"{'photoionisation' if src == 'photo' else 'electron impact'}, {tag}")
    _scale_axis(ax, scales)
    ax.set_yscale("log")
    ax.set_xlabel("L-shell EII cross section scale")
    ax.set_ylabel("direct 2p₃/₂ holes per atom")
    ax.set_title(f"(a) {u:g} µJ on Kα1: holes by source")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False, fontsize=10)

    ax = axes[1]
    for uu in pf.UJ:
        share = []
        for k in scales:
            r = budget[k][uu]
            p, e = r.in_pulse(r.photo_rate_L3()), r.in_pulse(r.eii_rate_L3().sum(axis=0))
            share.append(100 * e / (e + p))
        ax.plot(scales, share, color=UJ_COLOUR[uu], marker="o", ms=6, label=f"{uu:g} µJ")
    _scale_axis(ax, scales)
    ax.set_ylim(0, 100)
    ax.set_xlabel("L-shell EII cross section scale")
    ax.set_ylabel("share made by electrons (%)")
    ax.set_title("(b) EII share of the 2p₃/₂ holes made during the pulse")
    ax.legend(frameon=False, loc="upper left", title="pulse energy")
    for ax in axes:
        ax.grid(True, color=GRIDC, lw=0.7)
    fig.suptitle("Where the extra holes come from, and when (population budget, 8048 eV, beam and foil average)",
                 fontsize=13)
    pf._save(fig, "eii_scale_budget")


def load_budget(root):
    out = {}
    if os.path.isdir(REF_BUDGET):
        out[1] = ped.load(REF_BUDGET)
    if root:
        for vdir in sorted(glob.glob(os.path.join(root, "x*"))):
            name = os.path.basename(vdir)
            if name[1:].isdigit():
                runs = ped.load(vdir)
                if set(runs) >= set(pf.UJ):
                    out[int(name[1:])] = runs
    return out if len(out) > 1 else None


def print_report(fam, m):
    print("model / experiment: Kalpha1 peak | Kalpha2 peak | band area, at 1/5/20/30 uJ")
    for k in fam:
        print(f"  x{k:<3d} " + " | ".join(" ".join(f"{m[k, u][i] / m['exp', u][i]:.2f}" for u in pf.UJ)
                                        for i in range(3)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None, help="eii_scale_sweep_mono_<id> (default: the newest)")
    parser.add_argument("--final", default=None, help="final_sweep_mono_<id> for scale 1 (default: the newest)")
    parser.add_argument("--budget", default=None, help="eii_scale_budget_mono_<id> (default: the newest, optional)")
    parser.add_argument("--exp-mono", default=pcs.EXP_MONO_SCATTER)
    args = parser.parse_args()
    scale_root = args.data or newest("eii_scale_sweep_mono_*")
    final_root = args.final or newest("final_sweep_mono_*")
    if not scale_root:
        raise SystemExit("no data/eii_scale_sweep_mono_* folder; pass --data")
    fam = load(scale_root, final_root)
    if len(fam) < 2:
        raise SystemExit("fewer than two complete scales; nothing to compare")
    exp = pcs.exp_mono(args.exp_mono)
    os.makedirs(FIGS, exist_ok=True)
    m = metrics(fam, exp)
    fig_spectra(fam, exp)
    fig_summary(fam, exp, m)
    budget = load_budget(args.budget or newest("eii_scale_budget_mono_*"))
    if budget:
        fig_budget(budget)
    print_report(fam, m)
    print(f"\nread {scale_root} and {final_root}; wrote eii_scale_* to {FIGS}")
