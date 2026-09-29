"""xgrid/ygrid numerical-convergence check for the mono full-spectrum transmittance scan.

data/mono_fullspectrum_xygrid_<job>/ holds one Cu-seed-mono_xygrid{N}_{Eseed}uJ_{E}eV/
folder per (xgrid=ygrid value, target energy), each with a single runs_.../ subfolder --
one level deeper than tools.data_from_folder expects (which wants runs_.../ folders
directly under folder_path), so this script walks the extra level itself instead of
calling data_from_folder.

T(E) = sum(I_int_thy_w_last) / sum(I_int_thy_w_0), matching plot-mono-vs-experiment.ipynb.

Run from the repo root:  python scripts/plot_mono_fullspectrum_xygrid_convergence.py
"""

import os
import re
import glob

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data", "mono_fullspectrum_xygrid_24677579")
FIGS = os.path.join(REPO, "figs")
os.makedirs(FIGS, exist_ok=True)


def load_run(run_dir):
    yaml_file = next(f for f in os.listdir(run_dir) if f.endswith(".yaml"))
    with open(os.path.join(run_dir, yaml_file)) as f:
        cfg = yaml.safe_load(f)

    npz_files = glob.glob(os.path.join(run_dir, "*.npz"))
    I_last_sum = I_0_sum = None
    for npz_path in npz_files:
        d = np.load(npz_path)
        last = d["I_int_thy_w_last_sum"] / np.maximum(d["I_int_thy_w_last_count"], 1)
        zero = d["I_int_thy_w_0_sum"] / np.maximum(d["I_int_thy_w_0_count"], 1)
        I_last_sum = last if I_last_sum is None else I_last_sum + last
        I_0_sum = zero if I_0_sum is None else I_0_sum + zero

    T = np.sum(I_last_sum) / np.sum(I_0_sum)
    return cfg, T


def collect(data_dir, grid_key="xgrid"):
    """{grid_value: (sorted energy_values array, T_values array)}.

    Skips run folders with no .npz yet (in-progress sweep)."""
    per_grid = {}
    cfg = None
    for entry in sorted(os.listdir(data_dir)):
        entry_path = os.path.join(data_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        run_dirs = [os.path.join(entry_path, d) for d in os.listdir(entry_path)
                    if os.path.isdir(os.path.join(entry_path, d)) and d.startswith("runs_")]
        if not run_dirs or not glob.glob(os.path.join(run_dirs[0], "*.npz")):
            continue
        this_cfg, T = load_run(run_dirs[0])
        cfg = this_cfg
        grid_value = cfg[grid_key]
        energy = cfg["monochromator_target_energy_eV"]
        per_grid.setdefault(grid_value, {})[energy] = T

    hwKalpha1N = cfg["hwKalpha1N"]
    E_seed_uJ = cfg["E_seed_uJ"]

    sorted_per_grid = {}
    for grid_value, energy_to_T in sorted(per_grid.items()):
        energies = np.array(sorted(energy_to_T.keys()))
        T_values = np.array([energy_to_T[e] for e in energies])
        sorted_per_grid[grid_value] = (energies, T_values)
    return sorted_per_grid, hwKalpha1N, E_seed_uJ


def plot_overlay(per_grid, hwKalpha1N, E_seed_uJ, grid_label, out_name):
    grid_values = list(per_grid.keys())
    norm = plt.matplotlib.colors.LogNorm(vmin=min(grid_values), vmax=max(grid_values))
    cmap = plt.matplotlib.colormaps["viridis"]

    plt.figure(figsize=(7, 5))
    for grid_value, (energies, T_values) in per_grid.items():
        plt.plot(energies, T_values, "o-", ms=4, color=cmap(norm(grid_value)), label=f"{grid_label}={grid_value}")
    plt.axvline(hwKalpha1N, color="grey", ls=":", lw=1, label=r"$K\alpha_1$")
    plt.xlabel("Photon energy (eV)")
    plt.ylabel("Transmittance")
    plt.title(f"Mono full-spectrum transmittance vs {grid_label}\nE_seed = {E_seed_uJ:g} uJ")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, out_name))
    plt.close()


def plot_convergence(per_grid, hwKalpha1N, E_seed_uJ):
    grid_values = list(per_grid.keys())
    norm = plt.matplotlib.colors.LogNorm(vmin=min(grid_values), vmax=max(grid_values))
    cmap = plt.matplotlib.colormaps["viridis"]

    plot_overlay(per_grid, hwKalpha1N, E_seed_uJ, "xgrid=ygrid", "mono_fullspectrum_xygrid_convergence_overlay.pdf")

    # --- per-energy convergence: T vs xgrid, one line per energy point ---
    all_energies = sorted(set(e for energies, _ in per_grid.values() for e in energies))
    cmap2 = plt.matplotlib.colormaps["plasma"]
    norm2 = plt.matplotlib.colors.Normalize(vmin=min(all_energies), vmax=max(all_energies))

    plt.figure(figsize=(7, 5))
    for energy in all_energies:
        xs, ys = [], []
        for xgrid, (energies, T_values) in per_grid.items():
            idx = np.where(np.isclose(energies, energy))[0]
            if len(idx):
                xs.append(xgrid)
                ys.append(T_values[idx[0]])
        plt.plot(xs, ys, "o-", ms=4, color=cmap2(norm2(energy)))
    plt.xlabel("xgrid = ygrid")
    plt.ylabel("Transmittance")
    plt.title(f"Per-energy convergence vs xgrid/ygrid\nE_seed = {E_seed_uJ:g} uJ (color = photon energy)")
    sm = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
    plt.colorbar(sm, ax=plt.gca(), label="Photon energy (eV)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "mono_fullspectrum_xygrid_convergence_per_energy.pdf"))
    plt.close()

    # --- relative change vs finest grid, per energy ---
    finest = max(grid_values)
    finest_energies, finest_T = per_grid[finest]
    plt.figure(figsize=(7, 5))
    for xgrid, (energies, T_values) in per_grid.items():
        if xgrid == finest:
            continue
        rel_diff = []
        for e, T in zip(energies, T_values):
            idx = np.where(np.isclose(finest_energies, e))[0]
            if len(idx):
                rel_diff.append(T - finest_T[idx[0]])
            else:
                rel_diff.append(np.nan)
        plt.plot(energies, rel_diff, "o-", ms=4, color=cmap(norm(xgrid)), label=f"xgrid={xgrid} - xgrid={finest}")
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("Photon energy (eV)")
    plt.ylabel(f"T(xgrid) - T(xgrid={finest})")
    plt.title(f"Transmittance deviation from finest grid (xgrid={finest})\nE_seed = {E_seed_uJ:g} uJ")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "mono_fullspectrum_xygrid_convergence_deviation.pdf"))
    plt.close()


if __name__ == "__main__":
    per_grid, hwKalpha1N, E_seed_uJ = collect(DATA)
    plot_convergence(per_grid, hwKalpha1N, E_seed_uJ)

    # tgrid/zgrid sweeps are still running -- overlay-only, from whatever's landed so far.
    tgrid_dir = os.path.join(REPO, "data", "mono_fullspectrum_tgrid_24677578")
    if os.path.isdir(tgrid_dir):
        per_grid_t, hwKalpha1N_t, E_seed_uJ_t = collect(tgrid_dir, grid_key="tgrid")
        plot_overlay(per_grid_t, hwKalpha1N_t, E_seed_uJ_t, "tgrid",
                     "mono_fullspectrum_tgrid_convergence_overlay.pdf")

    zgrid_dir = os.path.join(REPO, "data", "mono_fullspectrum_zgrid_24677580")
    if os.path.isdir(zgrid_dir):
        per_grid_z, hwKalpha1N_z, E_seed_uJ_z = collect(zgrid_dir, grid_key="zgrid")
        plot_overlay(per_grid_z, hwKalpha1N_z, E_seed_uJ_z, "zgrid",
                     "mono_fullspectrum_zgrid_convergence_overlay.pdf")
