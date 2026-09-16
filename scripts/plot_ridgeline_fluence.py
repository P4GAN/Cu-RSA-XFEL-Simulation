#!/usr/bin/env python3
"""Ridgeline ("Unknown Pleasures") plot of the SASE absorbance spectrum vs pulse energy.

Reads the pathway sweep's full-model variant (data/pathway_sweep_sase_24689982/mid-eii: middlemen +
EII, post 2026-09-15 population fix, 200 SASE shots per point, 3x3 xy grid). Each ridge is the
shot-averaged absorbance -ln(I_out(E) / I_in(E)) at one pulse energy, stacked bottom (2 uJ) to top
(80 uJ) with a fixed vertical offset, so height differences between ridges mix the offset with the
real absorbance change. Reverse saturable absorption shows up as the resonant Kalpha1 feature growing
and broadening with pulse energy.

The x axis is the detuning from Kalpha1 (womega_ar; the simulation's rotating frame is resonant with
Kalpha1), which sidesteps the two absolute-energy calibrations used elsewhere in the repo (8045 + w
for the slide/experiment figures, 8047.91 + w physically).

    python scripts/plot_ridgeline_fluence.py            # figs/ridgeline_absorbance_vs_fluence.png
"""

import argparse
import glob
import os
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

BG = "#07080c"
FG = "#c9d2e0"


def load_spectra(folder):
    """{E_seed_uJ: (energy, absorbance)}, combining every chunk in each runs_seed_* folder."""
    out = {}
    for run_dir in glob.glob(os.path.join(folder, "runs_seed_*_uJ")):
        E_seed = float(re.search(r"runs_seed_([\d.]+)_uJ", run_dir).group(1))
        acc = None
        for path in sorted(glob.glob(os.path.join(run_dir, "*.npz"))):
            z = np.load(path)
            part = {k: z[k].astype(float) for k in ("I_int_thy_w_0_sum", "I_int_thy_w_0_count",
                                                    "I_int_thy_w_last_sum", "I_int_thy_w_last_count")}
            acc = part if acc is None else {k: acc[k] + part[k] for k in acc}
            w = z["womega_ar"].astype(float)
        I_in = acc["I_int_thy_w_0_sum"] / acc["I_int_thy_w_0_count"]
        I_out = acc["I_int_thy_w_last_sum"] / acc["I_int_thy_w_last_count"]
        out[E_seed] = (w, -np.log(I_out / I_in))
    return dict(sorted(out.items()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.path.join(REPO_ROOT, "data/pathway_sweep_sase_24689982/mid-eii"))
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "figs/ridgeline_absorbance_vs_fluence.png"))
    p.add_argument("--emin", type=float, default=-11.0, help="detuning range (eV)")
    p.add_argument("--emax", type=float, default=11.0)
    p.add_argument("--overlap", type=float, default=3.0,
                   help="tallest ridge height in units of the ridge spacing (bigger = more overlap)")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    spectra = load_spectra(args.data)
    fig, ax = plt.subplots(figsize=(7.5, 8.2), facecolor=BG)
    ax.set_facecolor(BG)

    cmap = plt.get_cmap("magma")
    n = len(spectra)
    in_range = [A[(E >= args.emin) & (E <= args.emax)] for E, A in spectra.values()]
    base = min(a.min() for a in in_range)
    spacing = (max(a.max() for a in in_range) - base) / args.overlap
    # Draw top ridge first so each lower ridge's black fill hides the lines behind it.
    for i, (E_seed, (E, A)) in reversed(list(enumerate(spectra.items()))):
        m = (E >= args.emin) & (E <= args.emax)
        y = (A[m] - base) * 1.0 + i * spacing
        colour = cmap(0.35 + 0.6 * i / max(n - 1, 1))
        ax.fill_between(E[m], i * spacing - 0.3 * spacing, y, color=BG, zorder=2 * (n - i))
        ax.plot(E[m], y, color=colour, lw=1.8, zorder=2 * (n - i) + 1)
        ax.text(args.emin - 0.4, i * spacing + (A[m][0] - base), f"{E_seed:g} µJ",
                color=colour, ha="right", va="center", fontsize=10.5)

    ax.set_xlim(args.emin - 4.5, args.emax + 0.5)
    ax.set_ylim(-0.5 * spacing, (n - 1) * spacing + max(a.max() for a in in_range) - base + 0.4 * spacing)
    ax.set_yticks([])
    ax.set_xticks(np.arange(np.ceil(args.emin / 5) * 5, args.emax + 0.1, 5))
    ax.tick_params(colors=FG, labelsize=10)
    ax.set_xlabel("photon energy relative to the Kα₁ line  (eV)", color=FG, fontsize=11)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#2a3142")

    fig.text(0.07, 0.955, "Copper gets darker the harder you hit it", color="#ffffff", fontsize=16)
    fig.text(0.07, 0.925, "X-ray absorbance of a 20 µm Cu foil vs SASE pulse energy, 200 simulated shots each",
             color="#8fa0bb", fontsize=10)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.08)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor=BG)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
