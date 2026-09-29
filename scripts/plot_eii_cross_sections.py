"""Electron-impact ionisation cross sections of Cu: Bote-Salvat (the `eii.cross_section: bote_salvat` reference)
against Burgess-Chidichimo (the old default), and what the difference does to one photoelectron.
docs/bote-salvat-and-final-experiments.md explains both formulas.

Writes figs/eii_cross_sections.png (+ .pdf):
  (a) sigma(E) of the L subshells, Bote-Salvat solid, Burgess-Chidichimo dashed; the dotted line marks U = 16,
      where Bote-Salvat switches from its distorted-wave fit to the relativistic Bethe (plane-wave) form
  (b) Burgess-Chidichimo / Bote-Salvat for every subshell both formulas cover
  (c) 2p holes made by one 7.09 keV photoelectron while it slows down from its birth energy to E
      (continuous slowing down: dN/dE = n sigma(E) / S(E), Joy-Luo stopping power, no spatial_factor)

Pure formula evaluation (XLO_sim/eii.py), no simulation. Run from the repo root:
  python scripts/plot_eii_cross_sections.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from XLO_sim import eii
import plot_population_budget as ppb

FIGS = ppb.FIGS
INK, INK_2, MUTED, GRIDC = ppb.INK, ppb.INK_2, ppb.MUTED, ppb.GRIDC
SUBSHELLS = ["2p3/2", "2p1/2", "2s", "3s", "3p"]
COLOUR = dict(zip(SUBSHELLS, ppb.SLOTS[:5]))
LABEL = {"2p3/2": "2p₃/₂", "2p1/2": "2p₁/₂", "2s": "2s", "3s": "3s", "3p": "3p"}
E_PHOTO = eii.DEFAULT_BIRTH_ENERGIES_EV["photo"]
N_ATOMS_NM3 = 8.96 / 63.546 * 6.02214076e23 * 1e-21

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12.5, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 10.5, "axes.linewidth": 1.0, "lines.linewidth": 2.2})


def sigma(E, name, model):
    if model == "bote_salvat":
        return eii.bote_salvat_cross_section_nm2(E, name)
    I, zeta, n, l = eii.DEFAULT_SUBSHELLS[name]
    return eii.bcf_cross_section_nm2(E, I, zeta, n, l)


def holes_along_path(E, name, model):
    """2p holes made between the birth energy E[-1] and each E (E ascending), per primary electron."""
    dNdE = N_ATOMS_NM3 * sigma(E, name, model) / eii.stopping_power_eV_nm(E)
    seg = 0.5 * (dNdE[1:] + dNdE[:-1]) * np.diff(E)
    return np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])


def main():
    fig, (ax_s, ax_r, ax_n) = plt.subplots(1, 3, figsize=(16, 5.2), layout="constrained")

    E = np.geomspace(900.0, 3.0e4, 1500)
    for name in ["2p3/2", "2p1/2", "2s"]:
        edge = eii.BOTE_SALVAT_CU[name][0]
        keep = E > edge
        ax_s.plot(E[keep] / 1e3, 1e6 * sigma(E[keep], name, "bote_salvat"), color=COLOUR[name],
                  label=f"{LABEL[name]}  Bote–Salvat")
        ax_s.plot(E[keep] / 1e3, 1e6 * sigma(E[keep], name, "bcf"), color=COLOUR[name], ls="--", lw=1.6,
                  label=f"{LABEL[name]}  Burgess–Chidichimo")
    ax_s.axvline(16 * eii.BOTE_SALVAT_CU["2p3/2"][0] / 1e3, color=MUTED, ls=":", lw=1.2)
    ax_s.text(16 * eii.BOTE_SALVAT_CU["2p3/2"][0] / 1e3 * 1.04, 8.6, "U = 16 (2p₃/₂)\nDWBA fit → Bethe",
              color=INK_2, fontsize=9.5, va="top")
    ax_s.axvline(E_PHOTO / 1e3, color=MUTED, lw=1.0)
    ax_s.text(E_PHOTO / 1e3 * 0.96, 0.3, "photoelectron\n7.09 keV", color=INK_2, fontsize=9.5, ha="right")
    ax_s.set_xscale("log")
    ax_s.set_xlim(0.9, 30)
    ax_s.set_xticks([1, 2, 5, 10, 20])
    ax_s.set_xticklabels(["1", "2", "5", "10", "20"])
    ax_s.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_s.set_ylim(0, 9.2)
    ax_s.set_xlabel("Electron energy (keV)")
    ax_s.set_ylabel("Cross section (10⁻⁶ nm²)")
    ax_s.set_title("(a) L-shell ionisation of Cu")
    ax_s.legend(frameon=False, fontsize=9.5, loc="upper left")

    E = np.geomspace(1000.0, 8000.0, 800)
    for name in SUBSHELLS:
        keep = E > 1.02 * max(eii.DEFAULT_SUBSHELLS[name][0], eii.BOTE_SALVAT_CU.get(name, (0.0,))[0])
        ratio = sigma(E[keep], name, "bcf") / sigma(E[keep], name, "bote_salvat")
        ax_r.plot(E[keep] / 1e3, ratio, color=COLOUR[name], label=LABEL[name])
    ax_r.axhline(1.0, color=MUTED, lw=0.9)
    ax_r.axvline(E_PHOTO / 1e3, color=MUTED, lw=1.0)
    ax_r.set_xscale("log")
    ax_r.set_xlim(1.0, 8.0)
    ax_r.set_xticks([1, 2, 3, 5, 8])
    ax_r.set_xticklabels(["1", "2", "3", "5", "8"])
    ax_r.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_r.set_ylim(0.4, 1.3)
    ax_r.set_xlabel("Electron energy (keV)")
    ax_r.set_ylabel("Burgess–Chidichimo / Bote–Salvat")
    ax_r.set_title("(b) Old formula relative to Bote–Salvat")
    ax_r.legend(frameon=False, ncol=2, loc="lower left")

    E = np.linspace(eii.BOTE_SALVAT_CU["2p3/2"][0] * 1.0005, E_PHOTO, 3000)
    for name in ["2p3/2", "2p1/2"]:
        for model, ls, lw in (("bote_salvat", "-", 2.2), ("bcf", "--", 1.6)):
            N = holes_along_path(E, name, model)
            tag = "Bote–Salvat" if model == "bote_salvat" else "Burgess–Chidichimo"
            ax_n.plot(E / 1e3, N, color=COLOUR[name], ls=ls, lw=lw, label=f"{LABEL[name]}  {tag}  (total {N[0]:.3f})")
    ax_n.invert_xaxis()
    ax_n.set_xlim(E_PHOTO / 1e3, 0.9)
    ax_n.set_ylim(0, None)
    ax_n.set_xlabel("Energy left (keV)   ← slowing down")
    ax_n.set_ylabel("2p holes made so far, per electron")
    ax_n.set_title("(c) One 7.09 keV photoelectron slowing down")
    ax_n.legend(frameon=False, fontsize=9.5, loc="upper left")

    for ax in (ax_s, ax_r, ax_n):
        ax.grid(True, color=GRIDC, lw=0.7)
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"eii_cross_sections.{ext}"), dpi=200)
    plt.close(fig)

    print(f"holes per {E_PHOTO:g} eV photoelectron (continuous slowing down, no spatial_factor):")
    for name in ["2p3/2", "2p1/2", "2s"]:
        E = np.linspace(eii.BOTE_SALVAT_CU[name][0] * 1.0005, E_PHOTO, 3000)
        print(f"  {name:6s} Bote-Salvat {holes_along_path(E, name, 'bote_salvat')[0]:.3f}"
              f"   Burgess-Chidichimo {holes_along_path(E, name, 'bcf')[0]:.3f}")
    print(f"wrote {os.path.join(FIGS, 'eii_cross_sections.png')}")


if __name__ == "__main__":
    main()
