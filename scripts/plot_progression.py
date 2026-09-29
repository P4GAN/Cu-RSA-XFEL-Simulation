"""Model progression (scripts/generate_progression_sweeps.sh) against the self-seeded experiment, and where
the free electrons are in the three EII variants.

Reads, from data/:
  progression_sweep_mono_<id>/<variant>/    1/5/20/30 uJ x 15 photon energies, 5x5, 10 reps
  progression_budget_mono_<id>/<variant>/   population budget of 4a-4c at 8000/8048 eV, same pulse energies
with variants
  1-L2          2p3/2 + 1s + 2p1/2 holes only, original cross sections
  2-sat         + 2s hole, single-spectator satellites, middlemen, metal CK feeds
  3-dsat        + double-3d satellites
  4a-eii-L      + electron-impact ionisation of the L shell, non-thermal 24-level ladder
  4b-eii-LM     4a + M-shell EII (neutral atoms -> middlemen), atomic rate
  4c-eii-fixed  4a's L-shell EII with fixed-energy electrons (no slowing down)
and writes into figs/:
  progression_mono_spectra     T(E) and absorbance per 20 um, stages 1 -> 4a, vs experiment
  progression_mono_summary     Kalpha1 / Kalpha2 depth, band area and wing T vs pulse energy, every variant
  progression_eii_difference   absorbance change of 4a/4b/4c relative to stage 3
  progression_electron_map     electrons per atom in each ladder level vs time (20 uJ, 8048 eV)
  progression_electron_levels  which levels carry the electrons and make the holes (20 uJ, 8048 eV)
  progression_eii_budget       EII holes, middlemen and electron numbers vs pulse energy

Definitions as scripts/plot_coherence_sweep.py: A(E) = ln(T_wing/T(E)) with T_wing = T(8000 eV) for the
model and the mean of 8000/8005 eV for the experiment (the slide-9 scatter-panel bins); line depths are
the peak of A in a window (Kalpha1 8040-8054, Kalpha2 8022-8032 eV) because the measured lines sit ~2 eV
below the model's; the band area is the trapezoid over 8012-8072 eV on the sweep's 15-point grid.
Budget quantities are beam-weighted (incident fluence) and averaged over the foil (planes 1..zgrid-1).
EII rates are rebuilt from the saved eii_rate_table: rate_g(z, t) = table[row, g] n_g(z, t) target(z, t),
with target = ground + middlemen for the 2p3/2 row (a middleman's 2p hole goes to the satellites it routes
to) and ground for the M shell. The 2p3/2 row uses the beam view: its target is ~1 everywhere, so the product
of beam averages is the beam average of the product. The M-shell row does not (4b empties the ground state
at the beam centre while the edge stays neutral, so the product of averages overstates it up to 2x); it is
taken at the centre pixel and shown as each level's share. The electron model:
docs/theory-eii-electron-ladder-explained.md.

Run from the repo root:  python scripts/plot_progression.py [--sweep DIR] [--budget DIR]
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

import plot_pathway_sweep as pps
import plot_bracket_sweep as pbs
import plot_coherence_sweep as pcs
import plot_population_budget as ppb

DATA, FIGS = pps.DATA, pps.FIGS
SWEEP_DIR = os.path.join(DATA, "progression_sweep_mono_24911420")
BUDGET_DIR = os.path.join(DATA, "progression_budget_mono_24911448")

VARIANTS = ["1-L2", "2-sat", "3-dsat", "4a-eii-L", "4b-eii-LM", "4c-eii-fixed"]
STAGES = VARIANTS[:4]
EII = VARIANTS[3:]
# validated categorical slots 1-6 in fixed order (scripts/plot_population_budget.SLOTS); a variant keeps its
# colour and marker in every figure
COLOUR = dict(zip(VARIANTS, ppb.SLOTS[:6]))
MARKER = dict(zip(VARIANTS, ["o", "^", "D", "v", "P", "h"]))
LABEL = {
    "1-L2": "1: 2p$_{3/2}$, 1s, 2p$_{1/2}$ holes only",
    "2-sat": "2: + 2s, single satellites, middlemen",
    "3-dsat": "3: + double satellites",
    "4a-eii-L": "4a: + EII, L shell",
    "4b-eii-LM": "4b: + EII, L + M shell",
    "4c-eii-fixed": "4c: + EII, fixed-energy electrons",
}
INK, INK_2, MUTED, GRIDC = ppb.INK, ppb.INK_2, ppb.MUTED, ppb.GRIDC
EXP_LABEL = "experiment (scatter-panel bins)"
MONO_E_SEED = pcs.MONO_E_SEED
BUDGET_UJ = 20.0
BUDGET_EV = 8048.0
# the model puts a satellite of config detuning_eV = d at Kalpha1 - d (see the note in fig_eii_difference)
DSAT_MODEL_EV = pps.KA1_EV + 1.7
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", ppb.RAMP[0], ppb.RAMP[1], ppb.RAMP[2], ppb.RAMP[3]])


# --------------------------------------------------------------------------- helpers

def _save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=150)
    fig.savefig(os.path.join(FIGS, name + ".pdf"))
    plt.close(fig)


def _line(ax, x, y, v, ms=3.0, **kw):
    ax.plot(x, y, color=COLOUR[v], marker=MARKER[v], ms=ms, lw=1.6, label=LABEL[v], **kw)


def _exp(ax, x, y, yerr=None, ms=3.0):
    ax.errorbar(x, y, yerr=yerr, color=INK, ls="--", marker="s", ms=ms, lw=1.0, elinewidth=0.7, capsize=0,
                label=EXP_LABEL, zorder=5)


def _legend_below(fig, axes, ncol=4):
    handles, labels = [], []
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False, fontsize=8)


def _uJ_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(MONO_E_SEED)
    ax.set_xticklabels([f"{u:g}" for u in MONO_E_SEED])
    ax.minorticks_off()
    ax.set_xlabel("Pulse energy (µJ)")


# --------------------------------------------------------------------------- sweep metrics

def sweep_metrics(fam, exp):
    """{observable: {variant or 'experiment': [value per MONO_E_SEED]}}"""
    grid = sorted(fam["3-dsat"][20.0])
    out = {"Kα1 depth": {}, "Kα2 depth": {}, "band area": {}, "wing T (8000 eV)": {}}
    for v in VARIANTS:
        cs = [pbs.absorbance_model(fam, v, u) for u in MONO_E_SEED]
        out["Kα1 depth"][v] = [pcs.peak(E, A, pcs.KA1_WINDOW) for E, A in cs]
        out["Kα2 depth"][v] = [pcs.peak(E, A, pcs.KA2_WINDOW) for E, A in cs]
        out["band area"][v] = [pbs.area(E, A, grid=grid) for E, A in cs]
        out["wing T (8000 eV)"][v] = [fam[v][u][8000.0] for u in MONO_E_SEED]
    cs = [pcs.mono_exp_absorbance(exp, u)[:2] for u in MONO_E_SEED]
    out["Kα1 depth"]["experiment"] = [pcs.peak(E, A, pcs.KA1_WINDOW) for E, A in cs]
    out["Kα2 depth"]["experiment"] = [pcs.peak(E, A, pcs.KA2_WINDOW) for E, A in cs]
    out["band area"]["experiment"] = [pbs.area(E, A, grid=grid) for E, A in cs]
    out["wing T (8000 eV)"]["experiment"] = [float(np.mean(np.interp([8000.0, 8005.0], *exp[u][:2])))
                                             for u in MONO_E_SEED]
    return out


# --------------------------------------------------------------------------- budget

class BudgetRun(ppb.Run):
    """ppb.Run plus the EII tables saved alongside (scripts/run_population_budget.py)."""

    def __init__(self, path):
        super().__init__(path)
        with np.load(path, allow_pickle=False) as d:
            self.E_centres = d["E_centres_eV"].astype(float)
            self.table = d["eii_rate_table"].astype(float)
        self.groups = [n for n in self.names if n.startswith("electrons/")]
        self.dt = float(self.t[1] - self.t[0])
        self.t0, self.t1 = self.pulse_fwhm()

    def levels(self):
        """(G, t) electrons per atom in each level (last = bin below E_bottom), beam and foil average."""
        return np.stack([self.foil(g) for g in self.groups])

    def eii_rate(self, row, view="beam"):
        """(G - 1, t) EII rate per atom from each active level into table row (0 = 2p3/2, 3 = M shell),
        per plane in the given view (see the module docstring for which view suits which row), then
        foil-averaged."""
        n = np.stack([self.channel(g, view)[1:] for g in self.groups[:-1]], axis=1)   # (z, G-1, t)
        target = self.channel("ground", view)[1:]
        if row < 3:
            target = target + self.channel("middlemen", view)[1:]
        return np.einsum("zgt,g,zt->gt", n, self.table[row, :-1], target) / n.shape[0]

    def integral(self, rate_gt, in_pulse=False):
        m = (self.t >= self.t0) & (self.t <= self.t1) if in_pulse else slice(None)
        return rate_gt[:, m].sum(axis=1) * self.dt


def load_budget(root):
    out = {}
    for v in EII:
        vdir = os.path.join(root, v)
        runs = [BudgetRun(os.path.join(vdir, f)) for f in sorted(os.listdir(vdir)) if f.endswith(".npz")]
        out[v] = {(r.E_seed, r.energy): r for r in runs}
    return out


# --------------------------------------------------------------------------- figures: spectra

def fig_spectra(fam, exp):
    fig, axes = plt.subplots(2, 4, figsize=(12.8, 6.6), sharex=True, sharey="row", layout="constrained")
    for j, uJ in enumerate(MONO_E_SEED):
        Ee, Te, sT = exp[uJ]
        Ae_E, Ae, sA = pcs.mono_exp_absorbance(exp, uJ)
        for v in STAGES:
            E, T = pbs.model_curve(fam, v, uJ)
            _line(axes[0][j], E, T, v, ms=2.6)
            _line(axes[1][j], *pbs.absorbance_model(fam, v, uJ), v, ms=2.6)
        _exp(axes[0][j], Ee, Te, sT, ms=2.4)
        _exp(axes[1][j], Ae_E, Ae, sA, ms=2.4)
        for ax in axes[:, j]:
            pbs._line_markers(ax)
            ax.set_xlim(8000, 8072)
        axes[1][j].axhline(0, color=MUTED, lw=0.6)
        axes[0][j].set_title(f"{uJ:g} µJ", fontsize=9.5)
        axes[1][j].set_xlabel("Photon energy (eV)")
    axes[0][0].set_ylabel("Transmission T")
    axes[1][0].set_ylabel(r"$\ln(T_{wing}/T)$ per 20 µm")
    axes[1][0].set_ylim(-0.02, 0.43)
    _legend_below(fig, axes, ncol=5)
    fig.suptitle("Model progression, self-seeded beam: satellites + middlemen (stage 2) double the dip and stop the "
                 "wing bleaching; double satellites and EII add ~10% each; the measured lines stay 2-3x deeper",
                 fontsize=10.5)
    _save(fig, "progression_mono_spectra")


def fig_summary(m):
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.9), layout="constrained")
    panels = [("Kα1 depth", "Kα1 depth, per 20 µm"), ("Kα2 depth", "Kα2 depth, per 20 µm"),
              ("band area", "Area 8012-8072 eV (eV)"), ("wing T (8000 eV)", "Wing transmission, 8000 eV")]
    for ax, (key, ylabel) in zip(axes, panels):
        for v in VARIANTS:
            _line(ax, MONO_E_SEED, m[key][v], v, ms=4)
        _exp(ax, MONO_E_SEED, m[key]["experiment"], ms=4)
        _uJ_axis(ax)
        ax.set_ylabel(ylabel)
        if key != "wing T (8000 eV)":
            ax.set_ylim(bottom=0)
    _legend_below(fig, axes, ncol=4)
    fig.suptitle("Stage by stage vs pulse energy: stage 1 saturates at 1/5 of the measured Kα1 depth and bleaches; "
                 "the full model (4a) reaches half the depth and a third of the area", fontsize=10.5)
    _save(fig, "progression_mono_summary")


def fig_eii_difference(fam):
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.6), sharex=True, sharey=True, layout="constrained")
    for ax, uJ in zip(axes, MONO_E_SEED):
        E, A3 = pbs.absorbance_model(fam, "3-dsat", uJ)
        for v in EII:
            _line(ax, E, pbs.absorbance_model(fam, v, uJ)[1] - A3, v, ms=2.8)
        pbs._line_markers(ax)
        ax.axvline(DSAT_MODEL_EV, color=COLOUR["4b-eii-LM"], ls=(0, (1, 2)), lw=0.9, zorder=0)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xlim(8000, 8072)
        ax.set_title(f"{uJ:g} µJ", fontsize=9.5)
        ax.set_xlabel("Photon energy (eV)")
    axes[0].set_ylabel(r"$A - A_{\rm stage\ 3}$ per 20 µm")
    axes[-1].annotate("double-3d satellite\nas the model places it\n(config: $-$1.7 eV)", (DSAT_MODEL_EV, 0.024),
                      xytext=(8054, 0.022), fontsize=7, color=INK_2, ha="left", va="bottom",
                      arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    _legend_below(fig, axes, ncol=3)
    fig.suptitle("What the three EII variants add to stage 3: L-shell EII deepens both lines by ~12% (4a), fixed-energy "
                 "electrons a little more (4c); M-shell EII (4b) moves absorption from 8032-8046 eV to 8050 and 8025 eV",
                 fontsize=10.5)
    _save(fig, "progression_eii_difference")


# --------------------------------------------------------------------------- figures: electrons

def fig_electron_map(budget):
    runs = [budget[v][(BUDGET_UJ, BUDGET_EV)] for v in EII]
    vmax = max(r.levels()[:-1].max() for r in runs)
    norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0), sharey=True, layout="constrained")
    for ax, v, r in zip(axes, EII, runs):
        n = np.maximum(r.levels()[:-1], norm.vmin * 0.5)
        t_edges = np.append(r.t - 0.5 * r.dt, r.t[-1] + 0.5 * r.dt)
        mesh = ax.pcolormesh(t_edges, r.E_edges[::-1], n[::-1], cmap=SEQ, norm=norm, shading="flat")
        ax.set_yscale("log")
        ax.axhline(952.0, color=INK_2, lw=0.7, ls="--")
        for tt in (r.t0, r.t1):
            ax.axvline(tt, color=INK_2, lw=0.6, ls=":")
        ax.set_xlim(r.t[0], r.t[-1])
        ax.set_title(LABEL[v], fontsize=9.5)
        ax.set_xlabel("Time (fs)")
        ax.grid(False)
        ax.annotate(f"below {r.E_edges[-1]:g} eV at the end: {r.levels()[-1, -1]:.2g} per atom",
                    (0.03, 0.03), xycoords="axes fraction", fontsize=7.5, color=INK)
    axes[0].annotate("2p threshold", (runs[0].t[-1], 952.0), xytext=(-4, 3), textcoords="offset points",
                     ha="right", fontsize=7, color=INK_2)
    axes[0].set_ylabel("Electron level energy (eV)")
    cb = fig.colorbar(mesh, ax=axes, shrink=0.9, pad=0.01)
    cb.set_label("electrons per atom (beam and foil average)")
    fig.suptitle(f"Free electrons by energy level, {BUDGET_UJ:g} µJ on Kα1 ({BUDGET_EV:.0f} eV): photo- and KLL "
                 "electrons enter at the top and cascade down in ~8 fs; fixed-energy electrons (4c) stay where "
                 "they are born. Dotted: pulse FWHM", fontsize=10.5)
    _save(fig, "progression_electron_map")


def fig_electron_levels(budget):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9), layout="constrained")
    ax = axes[0]
    for v in EII:
        r = budget[v][(BUDGET_UJ, BUDGET_EV)]
        it = int(np.argmin(np.abs(r.t - 0.5 * (r.t0 + r.t1))))
        ax.stairs(r.levels()[:-1, it][::-1], r.E_edges[::-1], color=COLOUR[v], lw=1.6, label=LABEL[v])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Electron energy (eV)")
    ax.set_ylabel("electrons per atom in the level")
    ax.set_title("At the pulse peak", fontsize=9.5)
    ax.annotate("4a and 4b coincide: M-shell EII\nchanges the atoms, not the electrons", (0.04, 0.96),
                xycoords="axes fraction", va="top", fontsize=7, color=INK_2)

    ax = axes[1]
    for v in ("4a-eii-L", "4c-eii-fixed"):
        r = budget[v][(BUDGET_UJ, BUDGET_EV)]
        rate = r.eii_rate(0)
        for in_pulse, ls in ((False, "--"), (True, "-")):
            h = r.integral(rate, in_pulse)
            ax.stairs(h[::-1], r.E_edges[::-1], color=COLOUR[v], lw=1.6, ls=ls,
                      label=f"{v[:2]}, {'pulse FWHM' if in_pulse else 'whole 45 fs'}")
    ax.set_xscale("log")
    ax.set_xlim(500, 9000)
    ax.set_xticks([500, 1000, 2000, 4000, 8000])
    ax.set_xticklabels(["500", "1000", "2000", "4000", "8000"])
    ax.minorticks_off()
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Electron energy (eV)")
    ax.set_ylabel("2p$_{3/2}$ holes made per atom")
    ax.set_title("Which levels make the 2p$_{3/2}$ holes (beam)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    ax = axes[2]
    for uJ, ls in ((1.0, "-"), (BUDGET_UJ, "--")):
        r = budget["4b-eii-LM"][(uJ, BUDGET_EV)]
        h = r.integral(r.eii_rate(3, view="centre"))
        ax.stairs(100 * h[::-1] / h.sum(), r.E_edges[::-1], color=COLOUR["4b-eii-LM"], lw=1.6, ls=ls,
                  label=f"{uJ:g} µJ")
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Electron energy (eV)")
    ax.set_ylabel("share of M-shell EII (%)")
    ax.set_title("4b: which levels ionise the M shell (beam centre)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"{BUDGET_UJ:g} µJ, {BUDGET_EV:.0f} eV: the 2.5-7 keV levels make 80% of the L-shell EII holes and "
                 "nothing below 1 keV makes any; the M shell is ionised all the way down the ladder", fontsize=10.5)
    _save(fig, "progression_electron_levels")


def fig_eii_budget(budget):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9), layout="constrained")
    for v in EII:
        runs = [budget[v][(u, BUDGET_EV)] for u in MONO_E_SEED]
        rates = [r.eii_rate(0) for r in runs]
        axes[0].plot(MONO_E_SEED, [r.integral(q, True).sum() for r, q in zip(runs, rates)], color=COLOUR[v],
                     marker=MARKER[v], ms=4, lw=1.6, label=LABEL[v])
        axes[0].plot(MONO_E_SEED, [r.integral(q).sum() for r, q in zip(runs, rates)], color=COLOUR[v],
                     marker=MARKER[v], ms=4, lw=1.1, ls="--")
        axes[1].plot(MONO_E_SEED, [r.foil("middlemen")[-1] for r in runs], color=COLOUR[v], marker=MARKER[v],
                     ms=4, lw=1.6, label=LABEL[v])
        axes[2].plot(MONO_E_SEED, [r.levels()[:-1].sum(axis=0).max() for r in runs], color=COLOUR[v],
                     marker=MARKER[v], ms=4, lw=1.6, label=LABEL[v])
        axes[2].plot(MONO_E_SEED, [r.levels()[:, -1].sum() for r in runs], color=COLOUR[v], marker=MARKER[v],
                     ms=4, lw=1.1, ls="--")
    axes[0].set_ylabel("2p$_{3/2}$ EII holes per atom")
    for ax in (axes[0], axes[2]):
        ax.annotate("4a and 4b coincide", (0.97, 0.04), xycoords="axes fraction", ha="right", fontsize=7,
                    color=INK_2)
    axes[0].set_title("solid: during pulse FWHM; dashed: whole 45 fs", fontsize=9)
    axes[1].set_ylabel("middleman fraction, end of window")
    axes[1].set_title("Atoms that ended as 3d$^{-n}$ ions", fontsize=9)
    axes[2].set_ylabel("electrons per atom")
    axes[2].set_yscale("log")
    axes[2].set_title("solid: above E$_{bottom}$ at peak; dashed: all, end", fontsize=9)
    for ax in axes:
        _uJ_axis(ax)
    axes[0].set_yscale("log")
    _legend_below(fig, axes, ncol=3)
    fig.suptitle(f"EII bookkeeping on Kα1 ({BUDGET_EV:.0f} eV), beam and foil average: fixed-energy electrons make "
                 "4x the 2p holes but only +9% during the pulse; M-shell EII turns most atoms into middlemen already "
                 "at 1-5 µJ", fontsize=10.5)
    _save(fig, "progression_eii_budget")


# --------------------------------------------------------------------------- report

def print_report(m, budget):
    print("\n=== mono sweep, per 20 um (1/5/20/30 uJ) ===")
    for name, data in m.items():
        print(f"\n{name}")
        for k, vals in data.items():
            print(f"  {k:13s} " + " ".join(f"{x:8.4f}" for x in vals))
    print("\n=== share of the experiment reached (model / experiment, 1/5/20/30 uJ) ===")
    for name in ("Kα1 depth", "Kα2 depth", "band area"):
        print(f"{name:10s} " + "   ".join(
            f"{v}: " + "/".join(f"{a / b:.2f}" for a, b in zip(m[name][v], m[name]["experiment"])) for v in VARIANTS))
    print(f"\n=== budget, {BUDGET_EV:.0f} eV (beam + foil average) ===")
    for v in EII:
        for u in MONO_E_SEED:
            r = budget[v][(u, BUDGET_EV)]
            L3 = r.eii_rate(0)
            line = (f"{v:13s} {u:4g} uJ  T {r.T:.4f}  2p3/2 EII holes in FWHM {r.integral(L3, True).sum():.4f} "
                    f"total {r.integral(L3).sum():.4f}  middlemen end {r.foil('middlemen')[-1]:.3f}  "
                    f"hot peak {r.levels()[:-1].sum(axis=0).max():.3g}  all end {r.levels()[:, -1].sum():.3g}")
            if r.table[3].any():
                h = r.integral(r.eii_rate(3, view="centre"))
                top = np.argsort(-h)[:3]
                line += ("  middlemen end (centre) " + f"{r.foil('middlemen', 'centre')[-1]:.3f}" + "  M-shell EII by level: "
                         + ", ".join(f"{r.E_centres[g]:.0f} eV {h[g] / h.sum():.2f}" for g in top))
            print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", default=SWEEP_DIR)
    parser.add_argument("--budget", default=BUDGET_DIR)
    parser.add_argument("--exp-mono", default=pcs.EXP_MONO_SCATTER)
    args = parser.parse_args()

    os.makedirs(FIGS, exist_ok=True)
    fam = pbs.load_mono_family(args.sweep)
    missing = [v for v in VARIANTS if v not in fam or set(fam[v]) != set(MONO_E_SEED)]
    if missing:
        raise SystemExit(f"incomplete sweep data for variants {missing}")
    exp = pcs.exp_mono(args.exp_mono)
    budget = load_budget(args.budget)

    m = sweep_metrics(fam, exp)
    fig_spectra(fam, exp)
    fig_summary(m)
    fig_eii_difference(fam)
    fig_electron_map(budget)
    fig_electron_levels(budget)
    fig_eii_budget(budget)
    print_report(m, budget)
    print(f"\nwrote figures to {FIGS}")
