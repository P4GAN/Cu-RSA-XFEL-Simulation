"""Illustrations of the free electrons of the final model (step 5), from its population budget
(scripts/generate_population_budget.sh): where the electrons are, which of them make the 2p holes, and when.

Reads data/population_budget_mono_<job id>/5-electrons/ (default: the newest) at 8048 eV and 1/5/20/30 uJ, and
writes into figs/ (PNG + PDF):
  eii_cascade_20uJ        (a) electrons per atom in each energy group vs time; (b) the electron spectrum at five
                          times; (c) which groups make the 2p3/2 EII holes; (d) 2p3/2 holes made per fs by
                          photoionisation and by EII
  eii_vs_pulse_energy     (a) 2p3/2 holes per atom from each source; (b) EII's share of them; (c) the rate at which
                          an ion is hit in its valence shell, R(t)

Definitions:
- electrons: incident-fluence-weighted beam average, averaged over the foil (planes 1..zgrid-1);
- 2p3/2 hole rates count direct 2p3/2 holes on ground atoms and middlemen: photo = sigma(2p3/2) J, EII =
  sum_g table[2p3/2, g] n_g, per plane in the beam view, then foil-averaged. 2s holes that Coster-Kronig decay
  into 2p3/2 are left out of both;
- R(t) = spatial_factor sum_g n_g n_at v_g [sigma_3s + sigma_3p + sigma_3d](E_g) at the beam centre, foil average.
It only reads saved budgets. Run from the repo root:
  python scripts/plot_eii_diagnostics.py [--budget DIR]
"""

import glob
import os
import argparse

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LogNorm, LinearSegmentedColormap

import plot_population_budget as ppb

DATA, FIGS = ppb.DATA, ppb.FIGS
BUDGETS = sorted(glob.glob(os.path.join(DATA, "population_budget_mono_*", "5-electrons")), key=os.path.getmtime)
BUDGET_DIR = BUDGETS[-1] if BUDGETS else None
INK, INK_2, MUTED, GRIDC = ppb.INK, ppb.INK_2, ppb.MUTED, ppb.GRIDC
UJ = [1.0, 5.0, 20.0, 30.0]
ENERGY = 8048.0
GAMMA_L3_FS = 0.61 / 0.6582119569   # 2p3/2 hole decay rate (fs^-1)
C_PHOTO, C_EII = ppb.SLOTS[1], ppb.SLOTS[0]
UJ_COLOUR = dict(zip(UJ, ppb.RAMP))  # ordered: light = 1 uJ, dark = 30 uJ
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd"] + ppb.RAMP)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11.5, "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "legend.fontsize": 9.5, "axes.linewidth": 1.0, "lines.linewidth": 2.0,
                     "savefig.bbox": "standard"})


class StepRun(ppb.Run):
    """ppb.Run plus the EII tables and cross sections saved alongside (scripts/run_population_budget.py)."""

    def __init__(self, path):
        super().__init__(path)
        with np.load(path, allow_pickle=False) as d:
            self.E_centres = d["E_centres_eV"].astype(float)
            self.table = d["eii_rate_table"].astype(float)
            by_sub = d["eii_rates_by_subshell"].astype(float)
            subs = [str(s) for s in d["eii_subshells"]]
            cfg = yaml.safe_load(str(d["config_yaml"]))
        self.sigma_L3 = float(cfg["sigma1_Ka1_2p3"])
        spatial = float((cfg.get("eii") or {}).get("spatial_factor", 0.5))
        self.valence = spatial * sum(by_sub[subs.index(s)] for s in ("3s", "3p", "3d"))   # (G-1,) fs^-1 per n_g
        self.groups = [n for n in self.names if n.startswith("electrons/")]
        self.dt = float(self.t[1] - self.t[0])
        self.t0, self.t1 = self.pulse_fwhm()

    def levels(self, view="beam"):
        """(G, t) electrons per atom in each level (last = bin below E_bottom), foil average."""
        return np.stack([self.foil(g, view) for g in self.groups])

    def _n(self, view):
        return np.stack([self.channel(g, view)[1:] for g in self.groups[:-1]], axis=1)   # (z, G-1, t)

    def _targets(self, view="beam"):
        return self.channel("ground", view)[1:] + self.channel("middlemen", view)[1:]

    def eii_rate_L3(self):
        """(G-1, t) direct 2p3/2 EII holes per atom per fs made by each level, foil average."""
        n = self._n("beam")
        return np.einsum("zgt,g,zt->gt", n, self.table[0, :-1], self._targets()) / n.shape[0]

    def photo_rate_L3(self):
        """(t,) direct 2p3/2 photoionisation holes per atom per fs, foil average."""
        return (self.sigma_L3 * self.channel("flux")[1:] * self._targets()).mean(axis=0)

    def valence_rate(self, view="centre"):
        """(t,) valence (3s+3p+3d) collision rate per atom, fs^-1, foil average."""
        return np.einsum("zgt,g->zt", self._n(view), self.valence).mean(axis=0)

    def cumulative(self, rate_t):
        return np.cumsum(rate_t) * self.dt

    def in_pulse(self, rate_t):
        m = (self.t >= self.t0) & (self.t <= self.t1)
        return float(rate_t[..., m].sum() * self.dt)


def load(root):
    runs = [StepRun(os.path.join(root, f)) for f in sorted(os.listdir(root)) if f.endswith(".npz")]
    return {r.E_seed: r for r in runs if np.isclose(r.energy, ENERGY)}


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), dpi=200)
    plt.close(fig)


def _log_energy_axis(ax, lo, hi):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ticks = [t for t in (20, 50, 100, 200, 500, 1000, 2000, 5000) if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" if t < 1000 else f"{t / 1000:g}k" for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def fig_cascade(runs, uJ=20.0):
    r = runs[uJ]
    lv = r.levels()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), layout="constrained")

    ax = axes[0, 0]
    hot = lv[:-1]
    norm = LogNorm(vmin=hot.max() * 1e-4, vmax=hot.max())
    t_edges = np.append(r.t - 0.5 * r.dt, r.t[-1] + 0.5 * r.dt)
    mesh = ax.pcolormesh(t_edges, r.E_edges[::-1], np.maximum(hot, norm.vmin * 0.5)[::-1], cmap=SEQ, norm=norm,
                         shading="flat")
    ax.set_yscale("log")
    ax.axhline(945.0, color=INK, lw=0.9, ls="--")
    ax.text(r.t[-1] - 0.5, 1000, "2p threshold", ha="right", va="bottom", fontsize=9.5, color=INK)
    for tt in (r.t0, r.t1):
        ax.axvline(tt, color=INK_2, lw=0.9, ls=":")
    ax.text(0.5 * (r.t0 + r.t1), r.E_edges[-1] * 1.08, "pulse FWHM", ha="center", va="bottom", fontsize=9.5, color=INK,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlim(r.t[0], r.t[-1])
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Electron energy level (eV)")
    ax.set_title("(a) Electrons per atom in each energy level")
    ax.grid(False)
    cb = fig.colorbar(mesh, ax=ax, pad=0.01)
    cb.set_label("electrons per atom")

    ax = axes[0, 1]
    it_peak = int(np.argmin(np.abs(r.t - 0.5 * (r.t0 + r.t1))))
    snaps = [(int(np.argmin(np.abs(r.t - r.t0))), "FWHM start"), (it_peak, "pulse peak"),
             (int(np.argmin(np.abs(r.t - r.t1))), "FWHM end"), (int(np.argmin(np.abs(r.t - (r.t1 + 10)))),
                                                               "FWHM end + 10 fs"), (len(r.t) - 1, "end of window")]
    shades = ["#b7d3f5"] + ppb.RAMP
    width = -np.diff(r.E_edges)
    for (it, tag), c in zip(snaps, shades):
        ax.stairs(hot[:, it][::-1] / width[::-1], r.E_edges[::-1], color=c, lw=2.0,
                  label=f"{tag} ({r.t[it]:.0f} fs): {hot[:, it].sum():.2g} per atom")
    ax.axvline(945.0, color=INK, lw=0.9, ls="--")
    _log_energy_axis(ax, r.E_edges[-1], r.E_edges[0])
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(1e-9, (hot[:, snaps[0][0]] / width).max() * 1e-3))
    ax.set_xlabel("Electron energy (eV)")
    ax.set_ylabel("electrons per atom per eV")
    ax.set_title("(b) The electron spectrum as it builds up and cascades down")
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    ax = axes[1, 0]
    rate = r.eii_rate_L3()
    for in_pulse, ls, tag in ((False, "--", "whole 45 fs window"), (True, "-", "during the pulse FWHM")):
        m = (r.t >= r.t0) & (r.t <= r.t1) if in_pulse else slice(None)
        h = rate[:, m].sum(axis=1) * r.dt
        ax.stairs(h[::-1], r.E_edges[::-1], color=C_EII, lw=2.0, ls=ls, label=f"{tag}: {h.sum():.2g} per atom")
    _log_energy_axis(ax, 800, 9000)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Electron energy (eV)")
    ax.set_ylabel("2p₃/₂ holes made by the level, per atom")
    ax.set_title("(c) Which electrons make the 2p₃/₂ holes")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, 1]
    photo, eii_t = r.photo_rate_L3(), rate.sum(axis=0)
    ax.axvspan(r.t0, r.t1, color=GRIDC, alpha=0.6, lw=0)
    ax.plot(r.t, photo, color=C_PHOTO, label=f"photoionisation: {r.cumulative(photo)[-1]:.3f} per atom "
                                              f"({100 * r.in_pulse(photo) / r.cumulative(photo)[-1]:.0f}% in FWHM)")
    ax.plot(r.t, eii_t, color=C_EII, label=f"electron impact: {r.cumulative(eii_t)[-1]:.3f} per atom "
                                           f"({100 * r.in_pulse(eii_t) / r.cumulative(eii_t)[-1]:.0f}% in FWHM)")
    ax.set_yscale("log")
    ax.set_ylim(max(photo.max(), eii_t.max()) * 1e-3, max(photo.max(), eii_t.max()) * 2)
    ax.set_xlim(r.t[0], r.t[-1])
    ax.text(0.5 * (r.t0 + r.t1), ax.get_ylim()[1] * 0.7, "pulse FWHM", ha="center", va="top", fontsize=9,
            color=INK_2)
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("2p₃/₂ holes made per atom per fs")
    ax.set_title("(d) When the 2p₃/₂ holes are made")
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    for ax in axes.flat[1:]:
        ax.grid(True, color=GRIDC, lw=0.7)
    fig.suptitle(f"Free electrons in the full model (step 5), {uJ:g} µJ on Kα1 ({ENERGY:.0f} eV); "
                 "beam-weighted, foil average", fontsize=13)
    _save(fig, f"eii_cascade_{uJ:g}uJ")


def fig_vs_pulse_energy(runs):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), layout="constrained")
    uJ = [u for u in UJ if u in runs]
    photo = {u: runs[u].photo_rate_L3() for u in uJ}
    eii_t = {u: runs[u].eii_rate_L3().sum(axis=0) for u in uJ}

    ax = axes[0]
    for src, rates, c, mk in (("photoionisation", photo, C_PHOTO, "o"), ("electron impact", eii_t, C_EII, "s")):
        ax.plot(uJ, [runs[u].in_pulse(rates[u]) for u in uJ], color=c, marker=mk, ms=6,
                label=f"{src}, during FWHM")
        ax.plot(uJ, [runs[u].cumulative(rates[u])[-1] for u in uJ], color=c, marker=mk, ms=6, ls="--", lw=1.5,
                mfc="white", label=f"{src}, whole window")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("2p₃/₂ holes per atom")
    ax.set_title("(a) Direct 2p₃/₂ holes by source")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    for in_pulse, ls, mfc, tag in ((True, "-", None, "during the pulse FWHM"), (False, "--", "white", "whole window")):
        share = []
        for u in uJ:
            p = runs[u].in_pulse(photo[u]) if in_pulse else runs[u].cumulative(photo[u])[-1]
            e = runs[u].in_pulse(eii_t[u]) if in_pulse else runs[u].cumulative(eii_t[u])[-1]
            share.append(100 * e / (e + p))
        ax.plot(uJ, share, color=C_EII, marker="s", ms=6, ls=ls, mfc=mfc, label=tag)
    ax.set_xscale("log")
    ax.set_ylim(0, 40)
    ax.set_ylabel("share made by electrons (%)")
    ax.set_title("(b) Electron-impact share of the 2p₃/₂ holes")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[2]
    for u in uJ:
        ax.plot(runs[u].t, runs[u].valence_rate(), color=UJ_COLOUR[u], label=f"{u:g} µJ")
    r = runs[uJ[-1]]
    ax.axvspan(r.t0, r.t1, color=GRIDC, alpha=0.6, lw=0)
    ax.axhline(GAMMA_L3_FS, color=INK, ls="--", lw=1.1)
    ax.text(r.t[-1] - 0.5, GAMMA_L3_FS * 1.03, "2p₃/₂ hole decay rate", ha="right", va="bottom", fontsize=9.5)
    ax.set_xlim(r.t[0], r.t[-1])
    ax.set_ylim(0, None)
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("valence hits per atom per fs")
    ax.set_title("(c) Valence hits per ion (beam centre)")
    ax.legend(frameon=False, loc="upper left", title="pulse energy")

    for ax in axes[:2]:
        ax.set_xlabel("Pulse energy (µJ)")
        ax.set_xticks(uJ)
        ax.set_xticklabels([f"{u:g}" for u in uJ])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for ax in axes:
        ax.grid(True, color=GRIDC, lw=0.7)
    fig.suptitle(f"Electron-impact ionisation against photoionisation on Kα1 ({ENERGY:.0f} eV), full model (step 5)",
                 fontsize=13)
    _save(fig, "eii_vs_pulse_energy")


def print_report(runs):
    print(f"direct 2p3/2 holes per atom at {ENERGY:.0f} eV (beam, foil average): photo / EII, in FWHM and whole window")
    for u in sorted(runs):
        r = runs[u]
        p, e = r.photo_rate_L3(), r.eii_rate_L3().sum(axis=0)
        R = r.valence_rate()
        it = int(np.argmin(np.abs(r.t - 0.5 * (r.t0 + r.t1))))
        print(f"  {u:4g} uJ  FWHM {r.in_pulse(p):.4f} / {r.in_pulse(e):.4f}   total {r.cumulative(p)[-1]:.4f} / "
              f"{r.cumulative(e)[-1]:.4f}   EII share {100 * r.in_pulse(e) / (r.in_pulse(e) + r.in_pulse(p)):.1f}% / "
              f"{100 * r.cumulative(e)[-1] / (r.cumulative(e)[-1] + r.cumulative(p)[-1]):.1f}%   "
              f"hot e- at peak {r.levels()[:-1, it].sum():.3g}   R centre peak/FWHM end {R[it]:.2f} / "
              f"{R[int(np.argmin(np.abs(r.t - r.t1)))]:.2f} fs^-1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget", default=BUDGET_DIR, help="population-budget folder of one step-5 variant")
    args = parser.parse_args()
    if not args.budget:
        raise SystemExit("no data/population_budget_mono_*/5-electrons folder; pass --budget")
    runs = load(args.budget)
    os.makedirs(FIGS, exist_ok=True)
    fig_cascade(runs)
    fig_vs_pulse_energy(runs)
    print_report(runs)
    print(f"\nread {args.budget}; wrote eii_cascade_20uJ and eii_vs_pulse_energy to {FIGS}")
