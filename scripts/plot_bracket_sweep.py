"""Batch A (one-change brackets) and Batch B1 (double-L-hole absorbers) against experiment.

Reads, from data/:
  bracket_sweep_mono_<id>/<variant>/   scripts/generate_bracket_sweeps.sh, 8 variants x 1/5/20/30 uJ x
                                       15 photon energies, 10 reps (the SASE family of this batch is
                                       missing, see the note printed at the end)
  b1_sweep_mono_<id>/{R,dLL}/          scripts/generate_b1_sweeps.sh, 5/20/50/80 uJ x 8040-8090 eV
  b1_sweep_sase_<id>/dLL/              SASE, 2/9/20/40 uJ, 200 reps. dLL moves the mono T by < 6e-4,
                                       so it stands in for the post-J(z=0)-fix SASE reference
  pathway_sweep_sase_24689982/mid-eii/ the same model before the J(z=0) fix
and writes into figs/:
  bracket_mono_absorbance   resonant absorbance A(E) per 20 um, every variant vs experiment
  bracket_mono_levers       share of the model-experiment gap each variant closes
  bracket_mono_fluence      Kalpha1 depth and band area vs pulse energy: the model's plateau
  b1_dll                    what the 2s^-1 2p^-1 / 2p^-2 absorbers change (nothing measurable)
  sase_line_shape           SASE line depth, width and area before/after the J(z=0) fix

Definitions (as scripts/plot_pathway_sweep.py): mono T = sum(I_int_thy_w_last)/sum(I_int_thy_w_0)
per monochromator setting; A(E) = ln(T_wing/T(E)) with T_wing = T(8000 eV) for the model and the
mean of 8000/8005 eV for the hand-digitised seeded data; foil10 is multiplied by 2 so every curve is
an absorbance per 20 um (equal values = no front-loading). "Band area" is the trapezoid integral of
A over 8012-8072 eV on the sweep's own 15-point grid, applied identically to model and experiment.

Run from the repo root:  python scripts/plot_bracket_sweep.py [--bracket-mono DIR] [--b1-mono DIR] ...
"""

import os
import re
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

import plot_pathway_sweep as pps  # loaders, experiment readers and rcParams shared with the pathway plots

REPO, DATA, FIGS = pps.REPO, pps.DATA, pps.FIGS
KA1_EV, KA2_EV = pps.KA1_EV, pps.KA2_EV

BRACKET_MONO = os.path.join(DATA, "bracket_sweep_mono_24722858")
B1_MONO = os.path.join(DATA, "b1_sweep_mono_24729292")
B1_SASE = os.path.join(DATA, "b1_sweep_sase_24729291")
PRE_FIX_SASE = os.path.join(DATA, "pathway_sweep_sase_24689982")

# Categorical slots in fixed order (the dataviz reference palette); a colour follows its variant
# through every figure. Experiment is primary ink.
VARIANTS = ["R", "mixsat-pop", "mixall-pop", "eii1", "det0", "grasp", "spot71", "foil10"]
COLOUR = dict(zip(VARIANTS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                             "#4a3aa7", "#e34948"]))
LABEL = {
    "R": "R: middlemen + CK + EII",
    "mixsat-pop": "mixing 2/fs, satellites, populations only",
    "mixall-pop": "mixing 2/fs, all blocks, populations only",
    "eii1": "EII spatial factor 1",
    "det0": "satellites at the bare lines",
    "grasp": "GRASP atomic data",
    "spot71": "spot 0.71x (2x peak flux)",
    "foil10": "10 um foil (A x 2)",
}
INK, INK_2, MUTED, GRIDC = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
PER_20UM = {"foil10": 2.0}
MONO_E_SEED = [1.0, 5.0, 20.0, 30.0]
AREA_BAND = (8012.0, 8072.0)

plt.rcParams.update({"grid.color": GRIDC, "axes.edgecolor": MUTED, "xtick.color": INK_2,
                     "ytick.color": INK_2})


# --------------------------------------------------------------------------- loading

def load_mono_family(root):
    """{variant: {E_seed: {energy: T}}} for every variant folder under root."""
    out = {}
    for vdir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(vdir):
            continue
        for run in glob.glob(os.path.join(vdir, "runs_seed_*")):
            m = re.search(r"runs_seed_([\d.]+)_uJ__energy_([\d.]+)_eV", os.path.basename(run))
            if not m or not glob.glob(os.path.join(run, "*.npz")):
                continue
            I_last, I_0 = pps._npz_mean(run, "I_int_thy_w_last"), pps._npz_mean(run, "I_int_thy_w_0")
            out.setdefault(os.path.basename(vdir), {}).setdefault(float(m.group(1)), {})[float(m.group(2))] = \
                float(I_last.sum() / I_0.sum())
    return out


def load_sase_variant(vdir):
    """{E_seed: (energies, T)} on the hwKalpha1N + womega_ar axis, restricted to pps.SASE_VALID."""
    out = {}
    for run in glob.glob(os.path.join(vdir, "runs_seed_*")):
        npz = glob.glob(os.path.join(run, "*.npz"))
        if not npz:
            continue
        e_seed = float(re.search(r"runs_seed_([\d.]+)_uJ", os.path.basename(run)).group(1))
        I_last, I_0 = pps._npz_mean(run, "I_int_thy_w_last"), pps._npz_mean(run, "I_int_thy_w_0")
        with np.load(npz[0]) as d:
            E = KA1_EV + d["womega_ar"].astype(float)
        keep = (E >= pps.SASE_VALID[0]) & (E <= pps.SASE_VALID[1])
        out[e_seed] = (E[keep], (I_last / np.maximum(I_0, 1e-300))[keep])
    return dict(sorted(out.items()))


def exp_mono_curves():
    """{uJ: (E, T)} from the hand-digitised seeded data (plot_pathway_sweep.load_experiment)."""
    mono, _, _ = pps.load_experiment()
    return mono


# --------------------------------------------------------------------------- metrics

def model_curve(fam, variant, uJ):
    by_E = fam[variant][uJ]
    E = np.array(sorted(by_E))
    return E, np.array([by_E[e] for e in E])


def absorbance_model(fam, variant, uJ):
    E, T = model_curve(fam, variant, uJ)
    return E, np.log(fam[variant][uJ][8000.0] / T) * PER_20UM.get(variant, 1.0)


def absorbance_exp(exp, uJ):
    E, T = exp[uJ]
    T_wing = np.mean(np.interp([8000.0, 8005.0], E, T))
    return E, np.log(T_wing / T)


def at(E, A, energies):
    return float(np.mean(np.interp(energies, E, A)))


def area(E, A, band=AREA_BAND, grid=None):
    """Trapezoid area of A over the band. grid: resample onto these energies first, so model and
    experiment are integrated with the same (coarse) quadrature."""
    if grid is not None:
        grid = np.asarray([g for g in grid if band[0] <= g <= band[1]])
        return float(np.trapz(np.interp(grid, E, A), grid))
    m = (E >= band[0]) & (E <= band[1])
    return float(np.trapz(A[m], E[m]))


OBSERVABLES = [  # (name, energies averaged) -- the sweep's own photon energies
    ("Kα1 line (8048)", [8048.0]),
    ("Kα2 line (8028)", [8028.0]),
    ("between lines (8032-8040)", [8032.0, 8036.0, 8040.0]),
    ("red side (8015-8020)", [8015.0, 8020.0]),
    ("blue side (8055-8060)", [8055.0, 8060.0]),
]


def gap_table(fam, exp):
    """{observable: {variant: {uJ: fraction of (exp - R) closed}}}, plus the raw values."""
    grid = sorted(fam["R"][20.0])
    frac, raw = {}, {}
    for name, energies in OBSERVABLES + [("band area", None)]:
        frac[name], raw[name] = {}, {}
        for uJ in MONO_E_SEED:
            Ee, Ae = absorbance_exp(exp, uJ)
            exp_val = area(Ee, Ae, grid=grid) if energies is None else at(Ee, Ae, energies)
            vals = {}
            for v in VARIANTS:
                E, A = absorbance_model(fam, v, uJ)
                vals[v] = area(E, A) if energies is None else at(E, A, energies)
            raw[name][uJ] = dict(vals, experiment=exp_val)
            for v in VARIANTS[1:]:
                frac[name].setdefault(v, {})[uJ] = (vals[v] - vals["R"]) / (exp_val - vals["R"])
    return frac, raw


# --------------------------------------------------------------------------- figures

def _save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".png"))
    fig.savefig(os.path.join(FIGS, name + ".pdf"))
    plt.close(fig)


def _line_markers(ax):
    for E_line in (KA1_EV, KA2_EV):
        ax.axvline(E_line, color=MUTED, ls=":", lw=0.7, zorder=0)


def fig_absorbance(fam, exp):
    """A(E) per 20 um, every variant, split into two rows so no panel carries more than 5 series."""
    groups = [("Coherence and atomic data", ["R", "mixsat-pop", "mixall-pop", "det0", "grasp"]),
              ("Flux, geometry and electrons", ["R", "eii1", "spot71", "foil10"])]
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.2), sharex=True, sharey=True, constrained_layout=True)
    for row, (title, variants) in enumerate(groups):
        for col, uJ in enumerate(MONO_E_SEED):
            ax = axes[row][col]
            Ee, Ae = absorbance_exp(exp, uJ)
            ax.plot(Ee, Ae, color=INK, ls="--", marker="s", ms=2.8, lw=1.0, label="experiment", zorder=5)
            for v in variants:
                E, A = absorbance_model(fam, v, uJ)
                ax.plot(E, A, color=COLOUR[v], lw=1.8 if v == "R" else 1.3, marker="o", ms=2.2,
                        label=LABEL[v], zorder=4 if v == "R" else 3)
            _line_markers(ax)
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.set_xlim(8000, 8072)
            ax.set_ylim(-0.02, 0.43)
            if row == 0:
                ax.set_title(f"{uJ:g} µJ")
            if row == 1:
                ax.set_xlabel("Photon energy (eV)")
        axes[row][0].set_ylabel(f"{title}\n" + r"$\ln(T_{wing}/T)$ per 20 µm")
        axes[row][0].legend(frameon=False, fontsize=7, loc="upper left")  # the 1 uJ panel is empty above 0.15
    fig.suptitle("Self-seeded beam, 20 µm Cu: resonant absorbance. Every model variant sits far below "
                 "the measured band", fontsize=10.5)
    _save(fig, "bracket_mono_absorbance")


def fig_levers(frac, raw):
    """Heatmap: % of the model-experiment gap each variant closes, per observable and pulse energy."""
    names = [n for n, _ in OBSERVABLES] + ["band area"]
    variants = VARIANTS[1:]
    fig, axes = plt.subplots(1, len(names), figsize=(15.5, 3.6), sharey=True, constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    norm = TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=0.6)
    for ax, name in zip(axes, names):
        M = np.array([[frac[name][v][u] for u in MONO_E_SEED] for v in variants])
        ax.imshow(np.clip(M, -0.6, 0.6), cmap=cmap, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{100 * M[i, j]:+.0f}", ha="center", va="center", fontsize=7.5,
                        color=INK if abs(M[i, j]) < 0.45 else "#ffffff")
        gaps = [raw[name][u]["experiment"] - raw[name][u]["R"] for u in MONO_E_SEED]
        ax.set_xticks(range(len(MONO_E_SEED)))
        ax.set_xticklabels([f"{u:g} µJ\ngap {g:+.{1 if name == 'band area' else 2}f}" for u, g in zip(MONO_E_SEED, gaps)],
                           fontsize=7)
        ax.set_title(name + (" (gap in eV)" if name == "band area" else ""), fontsize=9)
        ax.grid(False)
        ax.tick_params(length=0)
    axes[0].set_yticks(range(len(variants)))
    axes[0].set_yticklabels([LABEL[v] for v in variants], fontsize=8)
    fig.suptitle("Share of the model-experiment gap each one-change variant closes (%, relative to R; "
                 "colour clipped at ±60%)", fontsize=10.5)
    _save(fig, "bracket_mono_levers")


def fig_fluence(fam, exp, raw):
    """Kalpha1 depth and band area vs pulse energy: the model plateaus by 5 uJ, the data do not."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    grid = sorted(fam["R"][20.0])
    exp_uJ = sorted(exp)
    for ax, (name, ylabel) in zip(axes, [("Kα1 line (8048)", r"Kα$_1$ depth  $\ln(T_{wing}/T)$ per 20 µm"),
                                         ("band area", "Integrated absorbance 8012-8072 eV (eV)")]):
        for v in VARIANTS:
            y = [raw[name][u][v] for u in MONO_E_SEED]
            ax.plot(MONO_E_SEED, y, color=COLOUR[v], marker="o", ms=3.4, lw=1.8 if v == "R" else 1.2,
                    label=LABEL[v], zorder=4 if v == "R" else 3)
        ye = []
        for u in exp_uJ:
            Ee, Ae = absorbance_exp(exp, u)
            ye.append(at(Ee, Ae, [8048.0]) if name != "band area" else area(Ee, Ae, grid=grid))
        ax.plot(exp_uJ, ye, color=INK, ls="--", marker="s", ms=4, lw=1.1, label="experiment", zorder=5)
        ax.set_xscale("log")
        ax.set_xlabel("Pulse energy (µJ)")
        ax.set_ylabel(ylabel)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    axes[0].set_title("Line depth: model saturates at 0.17-0.19 by ~5 µJ; measured rises to 0.37-0.41")
    axes[1].set_title("Band area: 2.5x short at 1 µJ, 3.3-3.6x at 20-30 µJ")
    _save(fig, "bracket_mono_fluence")


def fig_b1(b1, exp):
    """dLL against R on the blue side."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), constrained_layout=True)
    ramp = {5.0: "#86b6ef", 20.0: "#3987e5", 50.0: "#1c5cab", 80.0: "#0d366b"}
    ax = axes[0]
    for uJ, c in ramp.items():
        E, T = model_curve(b1, "R", uJ)
        T_far = b1["R"][uJ][8090.0]
        ax.plot(E, np.log(T_far / T), color=c, lw=1.6, marker="o", ms=2.6, label=f"model R, {uJ:g} µJ")
        E, T = model_curve(b1, "dLL", uJ)
        ax.plot(E, np.log(T_far / T), color=c, lw=0, marker="x", ms=4.5)
        if uJ in exp:
            Ee, Te = exp[uJ]
            T_far_e = np.interp(8090.0, Ee, Te)
            ax.plot(Ee, np.log(T_far_e / Te), color=c, ls="--", lw=1.0, marker="s", ms=2.6,
                    label=f"experiment, {uJ:g} µJ")
    for E_line, name in [(8054.4, "2s2p"), (8062.7, "2p2"), (8074.8, "2s2p"), (8083.4, "2p2")]:
        ax.axvline(E_line, color=MUTED, ls=":", lw=0.7)
        ax.annotate(name, (E_line, 0.96), xycoords=("data", "axes fraction"), fontsize=6.5, color=INK_2,
                    ha="center")
    ax.set_xlim(8038, 8092)
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel(r"$\ln(T(8090)/T)$")
    ax.set_title("Blue side: dLL (×) lands on R (o); the measured shoulder is untouched")
    ax.legend(frameon=False, fontsize=6.5, ncol=1, loc="upper right", bbox_to_anchor=(1.0, 0.88))

    ax = axes[1]
    for uJ, c in ramp.items():
        E, T = model_curve(b1, "R", uJ)
        _, Td = model_curve(b1, "dLL", uJ)
        ax.plot(E, 1e4 * (Td - T), color=c, lw=1.6, marker="o", ms=2.6, label=f"{uJ:g} µJ")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel(r"$T_{dLL} - T_R$  (×10$^{-4}$)")
    ax.set_title("Effect of the double-L-hole absorbers: < 6e-4 in T")
    ax.legend(frameon=False, fontsize=7.5)
    _save(fig, "b1_dll")


def sase_metrics(E, T, smooth_eV=0.0):
    T_wing = pps.wing_T(E, T, pps.SASE_WING)
    A = pps.absorbance(E, T, T_wing, smooth_eV)
    m = (E >= 8030) & (E <= 8060)
    Em, Am = E[m], A[m]
    i = int(np.argmax(Am))
    above = Em[Am >= Am[i] / 2]
    return dict(T_wing=T_wing, depth=float(Am[i]), E_peak=float(Em[i]),
                fwhm=float(above.max() - above.min()), area=pps.band_area(E, T, T_wing, pps.SASE_BAND, smooth_eV))


def fig_sase(post, pre, exp_sase):
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.3), sharey=True, constrained_layout=True)
    for ax, uJ in zip(axes, [2.0, 9.0, 20.0, 40.0]):
        for curves, colour, label, lw in [(pre, "#86b6ef", "model before J(z=0) fix", 1.2),
                                         (post, "#2a78d6", "model after fix (B1 run)", 1.6)]:
            E, T = curves[uJ]
            A = pps.absorbance(E, T, pps.wing_T(E, T, pps.SASE_WING))
            ax.plot(E, A, color=colour, lw=lw, label=label)
        if uJ in exp_sase:
            Ee, Te = exp_sase[uJ][0], exp_sase[uJ][1]
            Ae = pps.absorbance(Ee, Te, pps.wing_T(Ee, Te, pps.SASE_WING), 2.0)
            ax.plot(Ee, Ae, color=INK, ls="--", lw=1.1, label="experiment (2 eV boxcar)")
        _line_markers(ax)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xlim(8024, 8070)
        ax.set_title(f"{uJ:g} µJ")
        ax.set_xlabel("Photon energy (eV)")
    axes[0].set_ylabel(r"$\ln(T_{wing}/T)$, $T_{wing}$ = 8060-8070 eV")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("SASE beam, 20 µm Cu (xgrid 3): the fix adds ~4% to the line; the measured line is 2-3x "
                 "wider with 2.2-4x the area", fontsize=10.5)
    _save(fig, "sase_line_shape")


# --------------------------------------------------------------------------- report

def print_report(frac, raw, b1, post, pre, exp_sase):
    print("\n=== mono: resonant absorbance per 20 um (model variants | experiment) ===")
    for name in raw:
        print(f"\n{name}")
        print(f"{'uJ':>5} " + "".join(f"{v:>11}" for v in VARIANTS) + f"{'exp':>9}")
        for u in MONO_E_SEED:
            cells = "".join(f"{raw[name][u][v]:11.3f}" for v in VARIANTS)
            print(f"{u:5g} {cells}{raw[name][u]['experiment']:9.3f}")
    print("\n=== share of the gap closed (%) ===")
    for name in frac:
        print(f"{name:28s}" + "  ".join(
            f"{v}: " + "/".join(f"{100 * frac[name][v][u]:+.0f}" for u in MONO_E_SEED) for v in VARIANTS[1:]))
    print("\n=== B1: max |T_dLL - T_R| over 8040-8090 eV ===")
    for u in sorted(b1["R"]):
        d = [abs(b1["dLL"][u][e] - b1["R"][u][e]) for e in b1["R"][u]]
        print(f"  {u:g} uJ: {max(d):.1e}")
    print("\n=== SASE: depth / FWHM (eV) / area (eV) ===")
    for u in [2.0, 9.0, 20.0, 40.0]:
        row = []
        for name, curves in [("pre-fix", pre), ("post-fix", post)]:
            m = sase_metrics(*curves[u])
            row.append(f"{name} {m['depth']:.3f}/{m['fwhm']:.1f}/{m['area']:.2f}")
        if u in exp_sase:
            m = sase_metrics(exp_sase[u][0], exp_sase[u][1], 2.0)
            row.append(f"exp {m['depth']:.3f}/{m['fwhm']:.1f}/{m['area']:.2f}")
        print(f"  {u:4g} uJ: " + "   ".join(row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bracket-mono", default=BRACKET_MONO)
    parser.add_argument("--b1-mono", default=B1_MONO)
    parser.add_argument("--b1-sase", default=B1_SASE)
    parser.add_argument("--pre-fix-sase", default=PRE_FIX_SASE)
    args = parser.parse_args()

    os.makedirs(FIGS, exist_ok=True)
    fam = load_mono_family(args.bracket_mono)
    missing = [v for v in VARIANTS if v not in fam]
    if missing:
        raise SystemExit(f"{args.bracket_mono} lacks variants {missing}")
    b1 = load_mono_family(args.b1_mono)
    post = load_sase_variant(os.path.join(args.b1_sase, "dLL"))
    pre = load_sase_variant(os.path.join(args.pre_fix_sase, "mid-eii"))
    exp = exp_mono_curves()
    _, exp_sase, _ = pps.load_experiment()

    frac, raw = gap_table(fam, exp)
    fig_absorbance(fam, exp)
    fig_levers(frac, raw)
    fig_fluence(fam, exp, raw)
    fig_b1(b1, exp)
    fig_sase(post, pre, exp_sase)
    print_report(frac, raw, b1, post, pre, exp_sase)
    print(f"\nwrote figures to {FIGS}")
