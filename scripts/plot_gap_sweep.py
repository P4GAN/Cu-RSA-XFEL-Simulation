"""Gap family (scripts/generate_gap_sweeps.sh; docs/eii-model-evaluation.md) against experiment.

Reads, from data/:
  gap_sweep_mono_<id>/<variant>/    1/5/20/30 uJ x 15 photon energies, 5x5, 10 reps
  gap_sweep_sase_<id>/<variant>/    2/9/20/40 uJ, 5x5, 200 reps
  gap_budget_mono_<id>/<variant>/   population budget, 8000/8048 eV
  progression_sweep_mono_24911420/4a-eii-L/   the starting model (grey)
with variants a-readout, b-sign, c-bs (the corrected reference), d-deph, e-deph-raman, f-core, g-core-raman.
Writes into figs/:
  gap_fixes           what the readout fix, the satellite sign fix and Bote-Salvat change: Kalpha1 line shape,
                      line centroids vs pulse energy, and the off-resonant transmission (foil thickness)
  gap_mono_spectra    absorbance of the collision variants vs experiment
  gap_mono_summary    Kalpha1 / Kalpha2 depth and band area vs pulse energy, divided by each curve's own cold
                      absorbance (foil-thickness free; see below)
  gap_sase_spectra    SASE absorbance at 2/9/20/40 uJ
  gap_sase_summary    SASE peak depth, FWHM and area vs pulse energy
  gap_budget          where the 2p holes are (base / single / double satellites), the core-hole EII fluxes,
                      and the 1s/2p3/2 hole ratio (dark-state release)

Why normalise by the cold absorbance: with read_field_after_last_plane the model's off-resonant
transmission at 1 uJ is 0.379 (20 um at sigma = 5.86e-7 nm^2), the measured one 0.413, i.e. an effective foil
of 18.25 um. The old readout off-by-one (18.6 um effective) hid this. A(E) / A_cold compares model and
experiment independently of the foil thickness; multiply by the experiment's cold absorbance (0.885) to read
it as a measured absorbance.
Other definitions as scripts/plot_coherence_sweep.py (line depth = peak of A in 8040-8054 / 8022-8032 eV,
band area 8012-8072 eV on the sweep grid; SASE: scripts/plot_pathway_sweep.py, 2 eV boxcar on the data).
Budget time integrals are beam-weighted and foil-averaged; core-hole EII fluxes use the centre pixel (products
of two beam averages are biased when both vary across the beam).

Run from the repo root:  python scripts/plot_gap_sweep.py [--mono DIR] [--sase DIR] [--budget DIR]
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_pathway_sweep as pps
import plot_bracket_sweep as pbs
import plot_coherence_sweep as pcs
import plot_population_budget as ppb
import plot_progression as ppr

DATA, FIGS = pps.DATA, pps.FIGS
MONO_DIR = os.path.join(DATA, "gap_sweep_mono_24933261")
SASE_DIR = os.path.join(DATA, "gap_sweep_sase_24933282")
BUDGET_DIR = os.path.join(DATA, "gap_budget_mono_24933285")
START_DIR = os.path.join(DATA, "progression_sweep_mono_24911420", "4a-eii-L")

VARIANTS = ["a-readout", "b-sign", "c-bs", "d-deph", "e-deph-raman", "f-core", "g-core-raman"]
SASE_VARIANTS = ["a-readout", "c-bs", "d-deph", "f-core", "g-core-raman"]
COLLISIONS = ["c-bs", "d-deph", "e-deph-raman", "f-core", "g-core-raman"]
# validated categorical slots 1-7 in fixed order; the starting model is a grey context series
COLOUR = dict(zip(VARIANTS, ppb.SLOTS[:7]))
MARKER = dict(zip(VARIANTS, ["o", "^", "D", "v", "P", "h", "X"]))
LABEL = {
    "4a": "4a: starting model (progression)",
    "a-readout": "a: + readout fix",
    "b-sign": "b: + satellite sign fix",
    "c-bs": "c: + Bote-Salvat EII (reference)",
    "d-deph": "d: c + collisional dephasing",
    "e-deph-raman": "e: c + Raman part only",
    "f-core": "f: c + core-hole EII",
    "g-core-raman": "g: f + Raman dephasing",
}
START_COLOUR = ppb.MUTED
INK, INK_2, MUTED = ppb.INK, ppb.INK_2, ppb.MUTED
MONO_E_SEED = pcs.MONO_E_SEED
SASE_E_SEED = [2.0, 9.0, 20.0, 40.0]
DOUBLES = ["3d+3d+", "3d-3d+", "3d-3d-"]
SINGLES = ["3d+", "3d-", "3p+", "3p-"]


def _save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=150)
    fig.savefig(os.path.join(FIGS, name + ".pdf"))
    plt.close(fig)


def _line(ax, x, y, v, ms=3.0, lw=1.6, **kw):
    if v == "4a":
        ax.plot(x, y, color=START_COLOUR, ls="--", lw=1.3, label=LABEL[v], zorder=2, **kw)
    else:
        ax.plot(x, y, color=COLOUR[v], marker=MARKER[v], ms=ms, lw=lw, label=LABEL[v], zorder=3, **kw)


def _exp(ax, x, y, yerr=None, ms=3.0, label=ppr.EXP_LABEL):
    ax.errorbar(x, y, yerr=yerr, color=INK, ls="--", marker="s", ms=ms, lw=1.0, elinewidth=0.7, capsize=0,
                label=label, zorder=5)


def _legend_below(fig, axes, ncol=4):
    ppr._legend_below(fig, axes, ncol)


def _uJ_axis(ax, uJ=MONO_E_SEED):
    ax.set_xscale("log")
    ax.set_xticks(uJ)
    ax.set_xticklabels([f"{u:g}" for u in uJ])
    ax.minorticks_off()
    ax.set_xlabel("Pulse energy (µJ)")


def centroid(E, A, lo, hi):
    g = np.arange(lo, hi + 0.01, 0.25)
    a = np.clip(np.interp(g, E, A), 0, None)
    return float((g * a).sum() / a.sum())


# --------------------------------------------------------------------------- data

def load(args):
    fam = pbs.load_mono_family(args.mono)
    start = pbs.load_mono_family(os.path.dirname(START_DIR))["4a-eii-L"]
    fam["4a"] = start
    sase = {v: pbs.load_sase_variant(os.path.join(args.sase, v)) for v in SASE_VARIANTS}
    budget = {}
    for v in ("c-bs", "d-deph", "f-core"):
        vdir = os.path.join(args.budget, v)
        runs = [ppr.BudgetRun(os.path.join(vdir, f)) for f in sorted(os.listdir(vdir)) if f.endswith(".npz")]
        budget[v] = {(r.E_seed, r.energy): r for r in runs}
    missing = [v for v in VARIANTS if v not in fam or set(fam[v]) != set(MONO_E_SEED)]
    missing += [v for v in SASE_VARIANTS if set(sase[v]) != set(SASE_E_SEED)]
    if missing:
        raise SystemExit(f"incomplete data for {missing}")
    return fam, sase, budget


def cold(fam, v):
    """Off-resonant (8000 eV) absorbance at 1 uJ: the foil's cold absorbance."""
    return -np.log(fam[v][1.0][8000.0])


def exp_cold(exp):
    return -np.log(float(np.mean(np.interp([8000.0, 8005.0], *exp[1.0][:2]))))


def mono_metrics(fam, exp):
    grid = sorted(fam["c-bs"][20.0])
    out = {k: {} for k in ("Kα1 depth", "Kα2 depth", "band area", "Kα1 centroid", "Kα2 centroid", "wing T")}
    for v in ["4a"] + VARIANTS:
        cs = [pbs.absorbance_model(fam, v, u) for u in MONO_E_SEED]
        c = cold(fam, v)
        out["Kα1 depth"][v] = [pcs.peak(E, A, pcs.KA1_WINDOW) / c for E, A in cs]
        out["Kα2 depth"][v] = [pcs.peak(E, A, pcs.KA2_WINDOW) / c for E, A in cs]
        out["band area"][v] = [pbs.area(E, A, grid=grid) / c for E, A in cs]
        out["Kα1 centroid"][v] = [centroid(E, A, 8040, 8056) for E, A in cs]
        out["Kα2 centroid"][v] = [centroid(E, A, 8020, 8034) for E, A in cs]
        out["wing T"][v] = [fam[v][u][8000.0] for u in MONO_E_SEED]
    cs = [pcs.mono_exp_absorbance(exp, u)[:2] for u in MONO_E_SEED]
    c = exp_cold(exp)
    out["Kα1 depth"]["experiment"] = [pcs.peak(E, A, pcs.KA1_WINDOW) / c for E, A in cs]
    out["Kα2 depth"]["experiment"] = [pcs.peak(E, A, pcs.KA2_WINDOW) / c for E, A in cs]
    out["band area"]["experiment"] = [pbs.area(E, A, grid=grid) / c for E, A in cs]
    out["Kα1 centroid"]["experiment"] = [centroid(E, A, 8040, 8056) for E, A in cs]
    out["Kα2 centroid"]["experiment"] = [centroid(E, A, 8020, 8034) for E, A in cs]
    out["wing T"]["experiment"] = [float(np.mean(np.interp([8000.0, 8005.0], *exp[u][:2]))) for u in MONO_E_SEED]
    return out


# --------------------------------------------------------------------------- figures: mono

def fig_fixes(fam, exp, m):
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.9), layout="constrained")
    ax = axes[0]
    uJ = 5.0
    for v in ("4a", "a-readout", "b-sign", "c-bs"):
        E, A = pbs.absorbance_model(fam, v, uJ)
        _line(ax, E, A / cold(fam, v), v, ms=3.2)
    Ee, Ae, sA = pcs.mono_exp_absorbance(exp, uJ)
    _exp(ax, Ee, Ae / exp_cold(exp), sA / exp_cold(exp))
    pbs._line_markers(ax)
    ax.set_xlim(8036, 8058)
    ax.set_ylim(bottom=-0.01)
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel(r"$A / A_{cold}$")
    ax.set_title(f"Kα1 at {uJ:g} µJ: the sign fix moves it red", fontsize=9.5)
    for ax, key, title in ((axes[1], "Kα1 centroid", "Kα1 centroid, 8040-8056 eV"),
                           (axes[2], "Kα2 centroid", "Kα2 centroid, 8020-8034 eV")):
        for v in ("4a", "b-sign", "c-bs", "d-deph", "f-core"):
            _line(ax, MONO_E_SEED, m[key][v], v, ms=4)
        _exp(ax, MONO_E_SEED, m[key]["experiment"], ms=4)
        _uJ_axis(ax)
        ax.set_ylabel("eV")
        ax.set_title(title, fontsize=9.5)
    ax = axes[3]
    for v in ("4a", "a-readout", "c-bs", "d-deph", "f-core"):
        _line(ax, MONO_E_SEED, m["wing T"][v], v, ms=4)
    _exp(ax, MONO_E_SEED, m["wing T"]["experiment"], ms=4)
    _uJ_axis(ax)
    ax.set_ylabel("T at 8000 eV")
    ax.set_title("Off-resonant transmission, 8000 eV", fontsize=9.5)
    ax.annotate("measured = 18.25 µm of Cu\nat the model's cross sections", (0.97, 0.55), xycoords="axes fraction",
                ha="right", fontsize=7.5, color=INK_2)
    _legend_below(fig, axes, ncol=4)
    fig.suptitle("The two fixes. Satellites on the correct side move both lines ~0.7 eV red, toward the measured lines "
                 "(which move blue with fluence); the readout fix exposes that the foil absorbs like 18.25 µm, not 20",
                 fontsize=10.2)
    _save(fig, "gap_fixes")


def fig_mono_spectra(fam, exp):
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.9), sharex=True, sharey=True, layout="constrained")
    for ax, uJ in zip(axes, MONO_E_SEED):
        for v in ["4a"] + COLLISIONS:
            E, A = pbs.absorbance_model(fam, v, uJ)
            _line(ax, E, A / cold(fam, v), v, ms=2.6)
        Ee, Ae, sA = pcs.mono_exp_absorbance(exp, uJ)
        _exp(ax, Ee, Ae / exp_cold(exp), sA / exp_cold(exp), ms=2.4)
        pbs._line_markers(ax)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xlim(8000, 8072)
        ax.set_title(f"{uJ:g} µJ", fontsize=9.5)
        ax.set_xlabel("Photon energy (eV)")
    axes[0].set_ylabel(r"$A / A_{cold}$  (thickness free)")
    _legend_below(fig, axes, ncol=4)
    fig.suptitle("Self-seeded beam: collisions add depth at high fluence (dephasing, mostly its Raman part) but none fills "
                 "the band between the lines, which holds most of the measured absorption", fontsize=10.2)
    _save(fig, "gap_mono_spectra")


def fig_mono_summary(m):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), layout="constrained")
    for ax, key, title in ((axes[0], "Kα1 depth", "Kα1 depth / A_cold"), (axes[1], "Kα2 depth", "Kα2 depth / A_cold"),
                           (axes[2], "band area", "Area 8012-8072 eV / A_cold (eV)")):
        _line(ax, MONO_E_SEED, m[key]["4a"], "4a")
        for v in COLLISIONS:
            _line(ax, MONO_E_SEED, m[key][v], v, ms=4)
        _exp(ax, MONO_E_SEED, m[key]["experiment"], ms=4)
        _uJ_axis(ax)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(title)
    _legend_below(fig, axes, ncol=4)
    fig.suptitle("At best 55-57% of the measured saturated Kα1 depth (e) and 38-41% of the area (d); dephasing ends the "
                 "model's 20 → 30 µJ decline, but the measured line keeps growing", fontsize=10.2)
    _save(fig, "gap_mono_summary")


# --------------------------------------------------------------------------- figures: SASE

def sase_curves(sase, exp_sase):
    out = {}
    for v in SASE_VARIANTS:
        out[v] = {u: sase[v][u] for u in SASE_E_SEED}
    return out


def fig_sase_spectra(sase, exp_sase):
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.9), sharex=True, layout="constrained")
    for ax, uJ in zip(axes, SASE_E_SEED):
        for v in SASE_VARIANTS:
            E, T = sase[v][uJ]
            ax.plot(E, pps.absorbance(E, T, pps.wing_T(E, T, pps.SASE_WING)), color=COLOUR[v], lw=1.6,
                    label=LABEL[v], zorder=3)
        if uJ in exp_sase:
            Ee, Te = exp_sase[uJ][0], exp_sase[uJ][1]
            ax.plot(Ee, pps.absorbance(Ee, Te, pps.wing_T(Ee, Te, pps.SASE_WING), 2.0), color=INK, ls="--", lw=1.1,
                    label="experiment (2 eV boxcar)", zorder=5)
        else:
            ax.annotate("no measurement at this energy", (0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
                        fontsize=7.5, color=INK_2)
        pbs._line_markers(ax)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xlim(8024, 8070)
        ax.set_title(f"{uJ:g} µJ", fontsize=9.5)
        ax.set_xlabel("Photon energy (eV)")
    axes[0].set_ylabel(r"$\ln(T_{wing}/T)$, $T_{wing}$ = 8060-8070 eV")
    _legend_below(fig, axes, ncol=6)
    fig.suptitle("SASE beam: electron collisions lower and barely widen the line; the measured line is 2.5x wider at 9 µJ "
                 "and 3x at 40 µJ, and sits 2 eV to the red", fontsize=10.2)
    _save(fig, "gap_sase_spectra")


def fig_sase_summary(sase, exp_sase):
    exp_uJ = sorted(u for u in exp_sase if u >= 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), layout="constrained")
    for key, ax, ylabel in (("depth", axes[0], "peak depth"), ("fwhm", axes[1], "FWHM (eV)"),
                            ("area", axes[2], "area 8025-8070 eV (eV)")):
        for v in SASE_VARIANTS:
            _line(ax, SASE_E_SEED, [pbs.sase_metrics(*sase[v][u])[key] for u in SASE_E_SEED], v, ms=4)
        _exp(ax, exp_uJ, [pbs.sase_metrics(exp_sase[u][0], exp_sase[u][1], 2.0)[key] for u in exp_uJ], ms=4,
             label="experiment")
        _uJ_axis(ax, SASE_E_SEED)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(ylabel)
    axes[1].set_title("collisions: +0.5-0.8 eV at 40 µJ; measured +8 eV", fontsize=9.5)
    _legend_below(fig, axes, ncol=6)
    fig.suptitle("SASE line vs pulse energy: the fluence-dependent width is 10x beyond what electron collisions give, and "
                 "the area is 2.5-5x short at every pulse energy", fontsize=10.2)
    _save(fig, "gap_sase_summary")


# --------------------------------------------------------------------------- figures: budget

def budget_numbers(budget, energy=8048.0):
    out = {}
    for v, runs in budget.items():
        for u in MONO_E_SEED:
            r = runs[(u, energy)]
            dt = r.dt
            block = lambda names: sum(r.foil(f"sat/{n}/{m}") for n in names for m in ("L3", "K", "L2")).sum() * dt
            base = sum(r.foil(f"base/{m}") for m in ("L3", "K", "L2")).sum() * dt
            K = (r.foil("base/K").sum() + sum(r.foil(f"sat/{n}/K").sum() for n in r.sat_names)) * dt
            L3 = (r.foil("base/L3").sum() + sum(r.foil(f"sat/{n}/L3").sum() for n in r.sat_names)) * dt
            rec = {"base": base, "single": block(SINGLES), "double": block(DOUBLES), "K/L3": K / L3}
            if v == "f-core":
                with np.load(os.path.join(BUDGET_DIR, v, f"Cu-seed-mono-SASE_{u:.2f}uJ_{energy:.2f}eV.npz")) as d:
                    subs = list(d["eii_subshells"])
                    val = 0.5 * sum(d["eii_rates_by_subshell"][subs.index(s)] for s in ("3s", "3p", "3d"))
                n = np.stack([r.channel(g, "centre")[1:] for g in r.groups[:-1]], axis=1)
                R = np.einsum("zgt,g->zt", n, val)
                pop = lambda names, pre: sum(r.channel(f"{pre}{nm}/{m}" if pre else f"base/{m}", "centre")[1:]
                                             for nm in names for m in ("L3", "K", "L2"))
                rec["flux base"] = (R * pop([""], "")).mean(0).sum() * dt
                rec["flux singles"] = (R * pop(SINGLES, "sat/")).mean(0).sum() * dt
                rec["flux doubles"] = (R * pop(DOUBLES, "sat/")).mean(0).sum() * dt
            out[(v, u)] = rec
    return out


def fig_budget(bn):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), layout="constrained")
    ax = axes[0]
    shades = {"base": ppb.RAMP[0], "single": ppb.RAMP[2], "double": ppb.RAMP[3]}
    x = np.arange(len(MONO_E_SEED))
    c_total = np.array([sum(bn[("c-bs", u)][p] for p in ("base", "single", "double")) for u in MONO_E_SEED])
    for off, v in ((-0.2, "c-bs"), (0.2, "f-core")):
        bottom = np.zeros(len(x))
        for part in ("base", "single", "double"):
            h = np.array([bn[(v, u)][part] for u in MONO_E_SEED]) / c_total
            ax.bar(x + off, h, width=0.36, bottom=bottom, color=shades[part], edgecolor="white", linewidth=1.0,
                   label=f"{part} {'2p/1s holes' if part == 'base' else 'satellites'}" if v == "c-bs" else None)
            bottom += h
        for xi, top in zip(x + off, bottom):
            ax.annotate(v[0], (xi, top), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{u:g}" for u in MONO_E_SEED])
    ax.set_ylim(0, 1.18)
    ax.set_xlabel("Pulse energy (µJ)")
    ax.set_ylabel("core-hole population x time, / c's total")
    ax.set_title("c vs f: core-hole EII moves holes to satellites,\nand loses 30-37% of them at 20-30 µJ", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=3)
    ax = axes[1]
    for key, label, colour in (("flux base", "base holes → single satellites", ppb.RAMP[0]),
                               ("flux singles", "single → double satellites / pool", ppb.RAMP[2]),
                               ("flux doubles", "double → triple holes (pool today)", ppb.RAMP[3])):
        ax.plot(MONO_E_SEED, [bn[("f-core", u)][key] for u in MONO_E_SEED], color=colour, marker="o", ms=4, lw=1.6,
                label=label)
    _uJ_axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel("transitions per atom (beam centre)")
    ax.set_title("f: core-hole EII fluxes; triple holes are as\nfrequent as the base → single step", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax = axes[2]
    for v in ("c-bs", "d-deph", "f-core"):
        _line(ax, MONO_E_SEED, [bn[(v, u)]["K/L3"] for u in MONO_E_SEED], v, ms=4)
    _uJ_axis(ax)
    ax.set_ylabel("1s-hole / 2p$_{3/2}$-hole time")
    ax.set_title("Dephasing releases the dark state: +19-24%\nmore 1s holes per 2p hole at 20-30 µJ", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("On Kα1 (8048 eV), beam and foil average: where the core holes are, and what the collisions do to them",
                 fontsize=10.2)
    _save(fig, "gap_budget")


# --------------------------------------------------------------------------- report

def print_report(m, sase, exp_sase, bn):
    print("\n=== mono, divided by each curve's cold absorbance (1/5/20/30 uJ); ratio to experiment ===")
    for key in ("Kα1 depth", "Kα2 depth", "band area"):
        print(f"\n{key}")
        for v in ["4a"] + VARIANTS:
            vals, ex = m[key][v], m[key]["experiment"]
            print(f"  {v:13s} " + " ".join(f"{x:7.4f}" for x in vals) + "   /exp " + " ".join(f"{a / b:5.2f}" for a, b in zip(vals, ex)))
        print(f"  {'experiment':13s} " + " ".join(f"{x:7.4f}" for x in m[key]["experiment"]))
    for key in ("Kα1 centroid", "Kα2 centroid", "wing T"):
        print(f"\n{key}")
        for v in ["4a", "a-readout", "b-sign", "c-bs", "d-deph", "f-core", "experiment"]:
            print(f"  {v:13s} " + " ".join(f"{x:9.3f}" for x in m[key][v]))
    print("\n=== SASE: depth / FWHM / area ===")
    for v in SASE_VARIANTS:
        print(f"  {v:13s} " + "   ".join(
            "{depth:.3f}/{fwhm:.1f}/{area:.2f}".format(**pbs.sase_metrics(*sase[v][u])) for u in SASE_E_SEED))
    print("  experiment    " + "   ".join(f"{u:g} uJ " + "{depth:.3f}/{fwhm:.1f}/{area:.2f}".format(
        **pbs.sase_metrics(exp_sase[u][0], exp_sase[u][1], 2.0)) for u in sorted(exp_sase) if u >= 1.0))
    print("\n=== budget at 8048 eV ===")
    for (v, u), rec in sorted(bn.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        print(f"  {v:7s} {u:4g} uJ " + ", ".join(f"{k} {x:.4f}" for k, x in rec.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mono", default=MONO_DIR)
    parser.add_argument("--sase", default=SASE_DIR)
    parser.add_argument("--budget", default=BUDGET_DIR)
    parser.add_argument("--exp-mono", default=pcs.EXP_MONO_SCATTER)
    args = parser.parse_args()

    os.makedirs(FIGS, exist_ok=True)
    fam, sase, budget = load(args)
    exp = pcs.exp_mono(args.exp_mono)
    _, exp_sase, _ = pps.load_experiment()
    m = mono_metrics(fam, exp)
    bn = budget_numbers(budget)
    fig_fixes(fam, exp, m)
    fig_mono_spectra(fam, exp)
    fig_mono_summary(m)
    fig_sase_spectra(sase, exp_sase)
    fig_sase_summary(sase, exp_sase)
    fig_budget(bn)
    print_report(m, sase, exp_sase, bn)
    print(f"\nwrote figures to {FIGS}")
