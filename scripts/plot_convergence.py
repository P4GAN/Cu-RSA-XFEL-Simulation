"""Grid convergence of the final model (scripts/generate_convergence_sweep.sh): the absorbance ln(T(8000 eV) / T(E))
at each photon energy and pulse energy against the time step, the transverse grid and the number of z planes,
each varied around the reference grid of the final sweep. Writes figs/convergence.png (+ .pdf) and prints the
values.

Run from the repo root:  python scripts/plot_convergence.py [--data DIR]
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_final import DATA, FIGS, INK_2, GRIDC, load_family

# variant -> (axis, grid value); ref is the final sweep's grid and belongs to every axis
AXES = {"tgrid (time steps in 45 fs)": {"t6000": 6000, "ref": 12000, "t24000": 24000},
        "xgrid = ygrid": {"xy3": 3, "ref": 5, "xy7": 7, "xy9": 9},
        "zgrid (planes through 20 µm)": {"z8": 8, "ref": 15, "z30": 30}}
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#eda100", "#4a3aa7"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None, help="convergence_sweep_mono_<job id> folder (default: the newest)")
    args = parser.parse_args()
    root = args.data
    if root is None:
        found = sorted(glob.glob(os.path.join(DATA, "convergence_sweep_mono_*")), key=os.path.getmtime)
        if not found:
            raise SystemExit("no data/convergence_sweep_mono_* folder; pass --data")
        root = found[-1]
    fam = load_family(root)

    points = sorted({(uJ, E) for v in fam.values() for uJ, by_E in v.items() for E in by_E if E != 8000.0})
    fig, axes = plt.subplots(1, len(AXES), figsize=(14, 4.6), sharey=True, layout="constrained")
    for ax, (label, variants) in zip(axes, AXES.items()):
        present = [(g, v) for v, g in sorted(variants.items(), key=lambda kv: kv[1]) if v in fam]
        print(f"\n{label}:  " + "  ".join(f"{uJ:g} uJ {E:g} eV" for uJ, E in points))
        for k, (uJ, E) in enumerate(points):
            grid, A = [], []
            for g, v in present:
                by_E = fam[v].get(uJ, {})
                if E in by_E and 8000.0 in by_E:
                    grid.append(g)
                    A.append(np.log(by_E[8000.0] / by_E[E]))
            ax.plot(grid, A, marker="o", color=COLOURS[k % len(COLOURS)], label=f"{uJ:g} µJ, {E:g} eV")
        for g, v in present:
            vals = [np.log(fam[v][uJ][8000.0] / fam[v][uJ][E]) if E in fam[v].get(uJ, {}) else np.nan
                    for uJ, E in points]
            print(f"  {v:8s} {g:6d}  " + "  ".join(f"{a:13.4f}" for a in vals))
        ax.set_xlabel(label)
        ax.grid(True, color=GRIDC, lw=0.7)
    axes[0].set_ylabel(r"Absorbance  $\ln(T_{8000\,eV}/T)$")
    axes[-1].legend(frameon=False, fontsize=9)
    fig.suptitle("Grid convergence of the final model (other axes at the final sweep's grid)", color=INK_2)
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, "convergence.png"), dpi=200)
    fig.savefig(os.path.join(FIGS, "convergence.pdf"))
    print(f"\nread {root}; wrote {FIGS}/convergence.png")


if __name__ == "__main__":
    main()
