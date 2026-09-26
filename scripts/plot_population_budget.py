"""Population budget plots: where the atoms and electrons are during the pulse.

Reads the families written by scripts/submit_population_budget.sh
(data/population_budget_{mono,sase}_<jobid>/<variant>/*.npz, recorder layout in
XLO_sim/population_budget.py) and writes into figs/:

  budget_timeline_mono, budget_timeline_sase
        R, beam- and foil-averaged populations vs time for each pulse energy: everything that left the
        ground state, the middleman pool, 2p3/2 / 1s / 2p1/2 / 2s holes (base + every satellite block),
        and the free electrons still hot vs already below the ladder (too slow to ionise). Grey band: incident pulse FWHM.
  budget_vs_fluence
        end-of-pulse ionised fraction, middlemen, free electrons, and the peak 2p3/2-hole and 1s-hole
        populations vs pulse energy; R solid, raman10 dashed; mono on Kalpha1 / on the wing, SASE
  budget_depth
        end-of-pulse middlemen and free electrons vs depth in the foil (R, beam average)
  budget_electron_spectrum
        electrons per atom in each energy group at the end of the pulse (R, beam and foil average)
  budget_blocks
        time-integrated hole population (fs) carried by the base block and by each satellite block (R)

"Beam" = incident-fluence-weighted (x, y) average; "foil average" = mean over planes 1..zgrid-1 (plane 0
repeats plane 1's depth, see the recorder docstring).

Run from the repo root:  python scripts/plot_population_budget.py [--mono DIR] [--sase DIR]
(default: the newest data/population_budget_{mono,sase}_* directories).
"""

import os
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
FIGS = os.path.join(REPO, "figs")

INK, INK_2, MUTED, GRIDC = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal blue, light surface
ATOM_SERIES = [  # (key, label, colour) in fixed order
    ("ionised", "left the ground state", INK),
    ("middlemen", "middleman pool (3d$^{-n}$ ions)", SLOTS[0]),
    ("L3", "2p$_{3/2}$ holes", SLOTS[1]),
    ("K", "1s holes", SLOTS[2]),
    ("L2", "2p$_{1/2}$ holes", SLOTS[3]),
    ("2s", "2s holes", SLOTS[4]),
    ("other", "other", SLOTS[5]),
]
VARIANT_LS = {"R": "-", "raman10": "--"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8, "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.grid": True, "grid.color": GRIDC,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 130, "savefig.bbox": "tight",
})


# --------------------------------------------------------------------------- loading

class Run:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as d:
            self.names = [str(n) for n in d["names"]]
            self.t = d["t"]
            self.beam = d["beam_mean"].astype(float)
            self.centre = d["centre_mean"].astype(float)
            self.depth_nm = d["depth_nm"]
            self.E_seed = float(d["E_seed_uJ"])
            self.energy = float(d["target_energy_eV"])
            self.T = float(d["T_integrated"])
            self.E_edges = d["E_edges_eV"]
            self.sat_names = [str(n) for n in d["satellite_channel_names"]]
            self.n_shots = int(d["n_shots"])
            self.trace_dev = float(d["trace_centre_max_dev"])
        self.index = {n: i for i, n in enumerate(self.names)}

    def channel(self, key, view="beam"):
        """(z, t) array for a derived quantity."""
        a = getattr(self, view)
        ix = self.index
        if key == "ionised":
            return 1.0 - a[:, ix["ground"]]
        if key in ("L3", "K", "L2"):
            return sum(a[:, i] for n, i in ix.items() if n.endswith("/" + key) and not n.startswith("electrons"))
        if key == "electrons_hot":
            return sum(a[:, i] for n, i in ix.items() if n.startswith("electrons/") and n != self._thermal())
        if key == "electrons_thermal":
            return a[:, ix[self._thermal()]] if self._thermal() else np.zeros(a.shape[::2])
        if key == "electrons":
            return self.channel("electrons_hot", view) + self.channel("electrons_thermal", view)
        if key.startswith("block/"):
            name = key[len("block/"):]
            prefix = "base/" if name == "base" else f"sat/{name}/"
            return sum(a[:, i] for n, i in ix.items() if n.startswith(prefix))
        return a[:, ix[key]]

    def _thermal(self):
        groups = [n for n in self.names if n.startswith("electrons/")]
        return groups[-1] if groups else None

    def foil(self, key, view="beam"):
        return self.channel(key, view)[1:].mean(axis=0)

    def pulse_fwhm(self):
        f = self.beam[0, self.index["flux"]]
        above = self.t[f >= f.max() / 2]
        return above.min(), above.max()


def load_family(root):
    """{variant: [Run, ...]} for one family directory (None if missing)."""
    if not root or not os.path.isdir(root):
        return None
    out = {}
    for vdir in sorted(glob.glob(os.path.join(root, "*"))):
        runs = [Run(p) for p in sorted(glob.glob(os.path.join(vdir, "*.npz")))]
        if runs:
            out[os.path.basename(vdir)] = sorted(runs, key=lambda r: (np.nan_to_num(r.energy), r.E_seed))  # SASE: energy NaN
    return out or None


def newest(pattern):
    dirs = sorted(glob.glob(os.path.join(DATA, pattern)), key=os.path.getmtime)
    return dirs[-1] if dirs else None


def _save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".png"))
    fig.savefig(os.path.join(FIGS, name + ".pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- figures

def fig_timeline(runs, beam, energy=None):
    runs = [r for r in runs if energy is None or np.isclose(r.energy, energy)]
    if not runs:
        return
    fig, axes = plt.subplots(len(runs), 2, figsize=(10.0, 2.5 * len(runs)), sharex=True, squeeze=False,
                             constrained_layout=True)
    for row, r in enumerate(runs):
        t0, t1 = r.pulse_fwhm()
        ax = axes[row][0]
        for key, label, colour in ATOM_SERIES:
            y = r.foil(key)
            if np.max(y) > 1e-7:
                ax.plot(r.t, np.maximum(y, 1e-7), color=colour, lw=1.5, label=label)
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1.2)
        ax.set_ylabel(f"{r.E_seed:g} µJ\nper atom")
        ax = axes[row][1]
        ax.plot(r.t, r.foil("electrons_hot"), color=SLOTS[6], lw=1.5, label="hot (still slowing down)")
        ax.plot(r.t, r.foil("electrons_thermal"), color=SLOTS[7], lw=1.5, label=f"below {r.E_edges[-1]:g} eV (no longer ionising)" if len(r.E_edges) else "below the ladder")
        ax.plot(r.t, r.foil("electrons"), color=INK, lw=1.0, ls=":", label="all produced")
        ax.set_ylabel("free electrons per atom")
        for a in axes[row]:
            a.axvspan(t0, t1, color=GRIDC, alpha=0.6, lw=0, zorder=0)
    for a in axes[-1]:
        a.set_xlabel("Time (fs)")
    handles = axes[0][0].get_legend_handles_labels()
    e_handles = axes[0][1].get_legend_handles_labels()
    fig.legend(handles[0] + e_handles[0], handles[1] + e_handles[1], loc="outside lower center", ncol=5,
               frameon=False)
    axes[0][0].set_title("Atoms (beam- and foil-averaged)")
    axes[0][1].set_title("Free electrons")
    where = "SASE" if beam == "sase" else f"self-seeded, {energy:.0f} eV"
    fig.suptitle(f"Population budget, R, {where}. Grey band: incident pulse FWHM", fontsize=10.5)
    _save(fig, f"budget_timeline_{beam}")


def _end_and_peak(runs, key, how):
    E = np.array([r.E_seed for r in runs])
    y = np.array([r.foil(key)[-1] if how == "end" else r.foil(key).max() for r in runs])
    order = np.argsort(E)
    return E[order], y[order]


def fig_vs_fluence(mono, sase):
    cases = []
    if mono:
        energies = sorted({r.energy for runs in mono.values() for r in runs})
        for e in energies:
            cases.append((f"self-seeded, {e:.0f} eV", {v: [r for r in runs if np.isclose(r.energy, e)]
                                                        for v, runs in mono.items()}))
    if sase:
        cases.append(("SASE", sase))
    if not cases:
        return
    rows = [("end of pulse: atoms", [("ionised", "left the ground state", INK),
                                     ("middlemen", "middleman pool", SLOTS[0])], "end"),
            ("end of pulse: free electrons per atom", [("electrons", "all produced", INK),
                                                        ("electrons_hot", "still hot", SLOTS[6])], "end"),
            ("peak core-hole population", [("L3", "2p$_{3/2}$ holes", SLOTS[1]), ("K", "1s holes", SLOTS[2])], "peak")]
    fig, axes = plt.subplots(len(rows), len(cases), figsize=(3.4 * len(cases), 2.7 * len(rows)),
                             squeeze=False, constrained_layout=True)
    for col, (title, fam) in enumerate(cases):
        for row, (ylabel, series, how) in enumerate(rows):
            ax = axes[row][col]
            for key, label, colour in series:
                for variant, runs in fam.items():
                    if not runs:
                        continue
                    E, y = _end_and_peak(runs, key, how)
                    ax.plot(E, y, color=colour, ls=VARIANT_LS.get(variant, "-."), marker="o", ms=3.2, lw=1.4,
                            label=label if variant == "R" else None)
            ax.set_xscale("log")
            E_ticks = sorted({r.E_seed for runs in fam.values() for r in runs})
            ax.xaxis.set_major_locator(FixedLocator(E_ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == len(rows) - 1:
                ax.set_xlabel("Pulse energy (µJ)")
            if col == 0:
                ax.legend(frameon=False, fontsize=6.8)
    fig.suptitle("Population budget vs pulse energy (beam- and foil-averaged). Solid R, dashed raman10",
                 fontsize=10.5)
    _save(fig, "budget_vs_fluence")


def fig_depth(mono, sase, mono_energy=8048.0):
    cases = []
    if mono and "R" in mono:
        cases.append((f"self-seeded, {mono_energy:.0f} eV", [r for r in mono["R"] if np.isclose(r.energy, mono_energy)]))
    if sase and "R" in sase:
        cases.append(("SASE", sase["R"]))
    cases = [c for c in cases if c[1]]
    if not cases:
        return
    fig, axes = plt.subplots(2, len(cases), figsize=(4.2 * len(cases), 5.6), squeeze=False, constrained_layout=True)
    for col, (title, runs) in enumerate(cases):
        for row, (key, ylabel) in enumerate([("middlemen", "middleman pool per atom"),
                                             ("electrons", "free electrons per atom")]):
            ax = axes[row][col]
            for i, r in enumerate(sorted(runs, key=lambda r: r.E_seed)):
                ax.plot(r.depth_nm[1:] / 1e3, r.channel(key)[1:, -1], color=RAMP[min(i, len(RAMP) - 1)],
                        marker="o", ms=2.8, lw=1.5, label=f"{r.E_seed:g} µJ")
            ax.set_xlabel("Depth (µm)")
            ax.set_ylabel(f"{ylabel}, end of pulse")
            if row == 0:
                ax.set_title(title)
            ax.legend(frameon=False)
    fig.suptitle("Ionisation vs depth in the foil (R, beam average). The sweep diagnostics sample only the "
                 "exit face", fontsize=10.5)
    _save(fig, "budget_depth")


def fig_electron_spectrum(mono, sase, mono_energy=8048.0):
    cases = []
    if mono and "R" in mono:
        cases.append((f"self-seeded, {mono_energy:.0f} eV", [r for r in mono["R"] if np.isclose(r.energy, mono_energy)]))
    if sase and "R" in sase:
        cases.append(("SASE", sase["R"]))
    cases = [c for c in cases if c[1] and len(c[1][0].E_edges)]
    if not cases:
        return
    fig, axes = plt.subplots(1, len(cases), figsize=(4.4 * len(cases), 3.4), squeeze=False, constrained_layout=True)
    for ax, (title, runs) in zip(axes[0], cases):
        for i, r in enumerate(sorted(runs, key=lambda r: r.E_seed)):
            groups = [n for n in r.names if n.startswith("electrons/")]
            hot = np.array([r.foil(n)[-1] for n in groups[:-1]])
            # groups run from the highest energy down (XLO_sim/eii.py build_ladder); stairs wants rising edges
            ax.stairs(hot[::-1], r.E_edges[::-1], baseline=None, color=RAMP[min(i, len(RAMP) - 1)], lw=1.6,
                      label=f"{r.E_seed:g} µJ (+{r.foil(groups[-1])[-1]:.3f} below {r.E_edges[-1]:g} eV)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Electron energy group (eV)")
        ax.set_ylabel("electrons per atom, end of pulse")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Free-electron energy distribution at the end of the pulse (R, beam and foil average)",
                 fontsize=10.5)
    _save(fig, "budget_electron_spectrum")


def fig_blocks(mono, sase, mono_energy=8048.0):
    columns = []
    if mono and "R" in mono:
        columns += [(f"mono {r.E_seed:g} µJ", r) for r in mono["R"] if np.isclose(r.energy, mono_energy)]
    if sase and "R" in sase:
        columns += [(f"SASE {r.E_seed:g} µJ", r) for r in sase["R"]]
    if not columns:
        return
    blocks = ["base"] + columns[0][1].sat_names
    M = np.array([[max(np.trapz(r.foil(f"block/{b}"), r.t), 1e-9) for _, r in columns] for b in blocks])
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * len(columns), 0.9 + 0.36 * len(blocks)), constrained_layout=True)
    ax.imshow(M, cmap="Blues", norm=LogNorm(vmin=max(M.min(), M.max() * 1e-4), vmax=M.max()), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.1e}", ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if M[i, j] > M.max() * 0.08 else INK)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([c for c, _ in columns], rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels(blocks, fontsize=8)
    ax.grid(False)
    ax.set_title("Time-integrated core-hole population per block (fs per atom; R, beam and foil average)")
    _save(fig, "budget_blocks")


def print_summary(fams):
    for beam, fam in fams.items():
        if not fam:
            continue
        print(f"\n=== {beam}: T | end of pulse (beam, foil average): left ground, middlemen, electrons (hot) | "
              f"peak 2p3/2, 1s holes | worst trace deviation")
        for variant, runs in fam.items():
            for r in runs:
                print(f"  {variant:8s} {r.energy:7.0f} eV {r.E_seed:5g} uJ  T {r.T:.4f} | "
                      f"{r.foil('ionised')[-1]:.4f} {r.foil('middlemen')[-1]:.4f} {r.foil('electrons')[-1]:.4f} "
                      f"({r.foil('electrons_hot')[-1]:.4f}) | {r.foil('L3').max():.2e} {r.foil('K').max():.2e} | "
                      f"{r.trace_dev:.1e}  [{r.n_shots} shots]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mono", default=newest("population_budget_mono_*"))
    parser.add_argument("--sase", default=newest("population_budget_sase_*"))
    args = parser.parse_args()
    os.makedirs(FIGS, exist_ok=True)
    mono, sase = load_family(args.mono), load_family(args.sase)
    if not mono and not sase:
        raise SystemExit(f"no population budget data under {args.mono} / {args.sase}")
    if mono and "R" in mono:
        fig_timeline(mono["R"], "mono", 8048.0)
    if sase and "R" in sase:
        fig_timeline(sase["R"], "sase")
    fig_vs_fluence(mono, sase)
    fig_depth(mono, sase)
    fig_electron_spectrum(mono, sase)
    fig_blocks(mono, sase)
    print_summary({"mono": mono, "sase": sase})
    print(f"\nwrote figures to {FIGS}")
