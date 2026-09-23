#!/usr/bin/env python3
"""Two-level Maxwell-Bloch runs (scripts/run_twolevel_sweep.py) vs the analytic RSA model.

Reads every <beam>_<mode>_<E>uJ.npz in --data and compares it with the attempt-1 analytic model
(rsa_transmittance.py in --analytic-dir, evaluated with the same atomic and beam parameters,
taken from the run's own config) and with its closed-form regime limits:

  fig1_spectra_<mode>   resonant absorption coefficient kappa_res(E) for several pulse energies
                        (log scale, and normalised to the peak to show the line shape)
  fig2_peak             peak kappa_res vs pulse energy (MB, RE, attempt 1, weak-field limit
                        Eq. 33, RSA ceiling Eq. 35/36 with and without ground-state depletion),
                        ratio to attempt 1, and the non-resonant kappa (bleaching)
  fig3_bloch            mono: per-hole Bloch vector (u, v, w)/(l + u) at the entrance plane, on
                        axis, against the quasi-steady-state locus
  fig4_fwhm             FWHM of kappa_res(E) vs pulse energy with the power-broadening estimates
  fig5_thickness        apparent peak kappa_res = -ln(T/T_nonres)/L for several foil thicknesses
  table_<view>.csv      every plotted number (peak, FWHM, kappa_nr, photons per hole) per run

kappa_res is the resonant (Kalpha1) part: at L -> 0 it is read directly from the entrance-plane
polarisation, at finite L as -ln(T / T_nonres) / L, with T_nonres from the same pulses with the
Kalpha1 coupling switched off (numerical) or sigma0 -> 0 (analytic).

--view axis   on-axis (v = 1) local absorption, the cleanest regime test (default)
--view beam   energy-weighted average over the Gaussian focus, what a detector sees
--view both   both sets, in figs/twolevel/<data folder>/<view>/

Regime markers (on-axis, pulse peak): s = W/W_sat = 1 (RSA saturation), Omega = gamma (mono) or
in-band Omega_in = gamma, i.e. 2W = gamma (SASE), and sigma_g F = 1 (ground-state depletion).

Example:
    python scripts/plot_twolevel_vs_analytic.py --data data/twolevel_<jobid> --view both
"""

import argparse
import csv
import functools
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from scipy.special import erfcx, exp1, voigt_profile  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from XLO_sim import twolevel as tl  # noqa: E402

HBAR = tl.HBAR
EULER_GAMMA = 0.5772156649015329

# Reference palette of the dataviz method (light surface), validated with its script:
# categorical slots 1-2 for the two numerical models, a 5-step blue ordinal ramp for ordered
# series (pulse energy, foil thickness), ink for the analytic model, secondary ink for limits.
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
C_MODE = {"mb": "#2a78d6", "re": "#eb6834"}
MODE_LABEL = {"mb": "numerical, Maxwell-Bloch", "re": "numerical, rate equations"}
RAMP5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
BEAM_TITLE = {"mono": "Mono (transform-limited 6 fs, photon-energy scan)",
              "sase": "SASE (spectrally resolved)"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "font.family": "sans-serif", "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.8, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def ramp(n):
    """n colours from the validated 5-step ramp (light = low); continuous interpolation past 5."""
    if n <= 5:
        idx = np.round(np.linspace(0, 4, n)).astype(int) if n > 1 else [2]
        return [RAMP5[i] for i in idx]
    rgb = np.array([matplotlib.colors.to_rgb(c) for c in RAMP5])
    x = np.linspace(0, 1, 5)
    return [matplotlib.colors.to_hex([np.interp(s, x, rgb[:, k]) for k in range(3)]) for s in np.linspace(0, 1, n)]


def fmt_E(E):
    return f"{E:g} µJ"


# ----------------------------------------------------------------------------------------------
# Loading and numerical reductions
# ----------------------------------------------------------------------------------------------
def load_runs(data_path):
    runs = {}
    for path in sorted(glob.glob(os.path.join(data_path, "*uJ.npz"))):
        m = re.match(r"(mono|sase)_(mb|re)_([0-9.eE+-]+)uJ\.npz$", os.path.basename(path))
        if not m:
            continue
        with np.load(path, allow_pickle=False) as f:
            if m[2] == "re" and "re_closure" not in f.files:
                print(f"skipping {os.path.basename(path)}: rate-equation run from before the photon-"
                      "conserving closure (XLO_sim/twolevel.py docstring)")
                continue
            runs.setdefault((m[1], m[2]), {})[float(m[3])] = {k: f[k] for k in f.files}
    if not runs:
        sys.exit(f"no <beam>_<mode>_<E>uJ.npz files in {data_path}")
    return runs


def run_config(runs):
    """Config of the runs; refuses mixed physics (different atom/beam/grid blocks)."""
    cfgs = {}
    for key, per_E in runs.items():
        for E, d in per_E.items():
            cfg = yaml.safe_load(str(d["config_yaml"]))
            sig = yaml.safe_dump({k: cfg[k] for k in ("atom", "mono", "sase", "grid")}, sort_keys=True)
            cfgs.setdefault(sig, []).append(f"{key[0]}_{key[1]}_{E:g}")
    if len(cfgs) > 1:
        sys.exit("runs in this folder use different configs: " + "; ".join(v[0] + "..." for v in cfgs.values()))
    return yaml.safe_load(str(next(iter(next(iter(runs.values())).values()))["config_yaml"]))


def node_weights(d, view):
    v = d["v"]
    return np.eye(v.size)[v.size - 1] if view == "axis" else d["wv"]


def peak_fwhm(x, y):
    """Peak value, its position and the FWHM (linear interpolation of the half-maximum crossings
    on either side of the maximum; NaN if a crossing is outside the grid)."""
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if not ok.any():
        return np.nan, np.nan, np.nan
    i = int(np.nanargmax(np.where(ok, y, -np.inf)))
    pk, h = y[i], 0.5 * y[i]
    xl = xr = np.nan
    for k in range(i - 1, -1, -1):
        if y[k] < h:
            xl = np.interp(h, [y[k], y[k + 1]], [x[k], x[k + 1]])
            break
    for k in range(i + 1, y.size):
        if y[k] < h:
            xr = np.interp(h, [y[k], y[k - 1]], [x[k], x[k - 1]])
            break
    return pk, x[i], xr - xl


class Numeric:
    """kappa spectra of one run in one view. x = E - E_ul [eV]. 'local' = L -> 0."""

    def __init__(self, d, view, n_a, E_ul):
        self.beam = str(d["beam"])
        self.d, self.view, self.n_a = d, view, n_a
        self.w = node_weights(d, view)
        self.z_rec_um = d["z_rec_um"]
        if self.beam == "mono":
            self.x = d["dE"]
            Phi = d["int0"][..., 0]
            self.kres0 = n_a * (d["int0"][..., 1] / Phi) @ self.w
            self.knr0 = n_a * (d["int0"][..., 2] / Phi) @ self.w
            self.T = np.einsum("jmr,m->rj", d["T"], self.w)
            self.T_ref = (d["T_ref"].T @ self.w)[:, None]
            self.smooth = 0.0
        else:
            self.x = d["E_axis"] - E_ul
            den = d["den"].sum(axis=0)
            self.valid = den > 1e-3 * den.max()
            self.kres0 = self._mask(np.einsum("bmw,m->w", d["kres_num"], self.w) / den)
            self.knr0 = self._mask(np.einsum("bmw,m->w", d["knr_num"], self.w) / den)
            iv = 0 if view == "axis" else 1          # views stored by run_sase: [axis, beam]
            self.T = d["num_out_view"][iv].sum(axis=0) / den
            self.T_ref = d["num_ref_view"][iv] / d["den_ref"]
            self.smooth = 0.06 / (self.x[1] - self.x[0])     # 0.06 eV Gaussian: < 0.2 % peak bias

    def _mask(self, y):
        return np.where(self.valid, y, np.nan)

    def kres(self, L_index=None):
        """Resonant kappa [cm^-1] at L -> 0 (None) or at recorded thickness L_index."""
        if L_index is None:
            return self.kres0
        L = self.z_rec_um[L_index] * 1e-4
        k = -np.log(self.T[L_index] / self.T_ref[L_index]) / L
        return self._mask(k) if self.beam == "sase" else k

    def peak_fwhm(self, L_index=None, y=None):
        y = self.kres(L_index) if y is None else y
        if self.smooth:
            good = np.isfinite(y)
            y = np.where(good, gaussian_filter1d(np.where(good, y, 0.0), self.smooth), np.nan)
        return peak_fwhm(self.x, y)

    def _leave_one_out(self):
        d = self.d
        num = np.einsum("bmw,m->bw", d["kres_num"], self.w)
        tot_n, tot_d = num.sum(axis=0), d["den"].sum(axis=0)
        return [self._mask((tot_n - num[b]) / (tot_d - d["den"][b])) for b in range(num.shape[0])]

    @staticmethod
    def _jk_error(est):
        est = np.asarray(est)
        nb = est.shape[0]
        return np.sqrt((nb - 1) / nb * np.sum((est - est.mean(axis=0)) ** 2, axis=0))

    def jackknife(self):
        """Leave-one-block-out errors of the L -> 0 peak and FWHM (SASE); zeros for mono."""
        if self.beam != "sase":
            return 0.0, 0.0
        return tuple(self._jk_error([self.peak_fwhm(y=y)[::2] for y in self._leave_one_out()]))

    def kres_display(self, smooth_eV=0.1, max_rel_err=None):
        """L -> 0 kappa_res for plotting. SASE: Gaussian-smoothed by smooth_eV and, if max_rel_err is
        set, masked where the block-jackknife error exceeds that fraction of the value."""
        y = self.kres0
        if self.beam != "sase":
            return y
        sig = smooth_eV / (self.x[1] - self.x[0])

        def sm(v):
            good = np.isfinite(v)
            return np.where(good, gaussian_filter1d(np.where(good, v, 0.0), sig), np.nan)

        ys = sm(y)
        if max_rel_err is not None:
            err = self._jk_error([sm(v) for v in self._leave_one_out()])
            ys = np.where(err <= max_rel_err * np.abs(ys), ys, np.nan)
        return ys

    def knr_center(self):
        """Non-resonant kappa at L -> 0 at the line (mono: energy-weighted; SASE: at E_ul)."""
        return float(np.interp(0.0, self.x, np.nan_to_num(self.knr0)))

    def photons_per_hole(self):
        """Resonant photons absorbed per 2p3/2 hole at the entrance plane, at the line centre."""
        d = self.d
        if self.beam == "mono":
            j = int(np.argmin(np.abs(d["dE"])))
            wa = self.w / d["v"] if self.view == "beam" else self.w     # per atom, not per photon
            wa = np.where(np.isfinite(wa), wa, 0.0)
            return float((d["int0"][j, :, 5] @ wa) / (d["int0"][j, :, 3] @ wa))
        wa = self.w / d["v"] if self.view == "beam" else self.w
        s = d["int0"].sum(axis=0)
        return float((s[:, 5] @ wa) / (s[:, 3] @ wa))


# ----------------------------------------------------------------------------------------------
# Analytic model (attempt 1) and closed-form limits
# ----------------------------------------------------------------------------------------------
class Analytic:
    """rsa_transmittance.py evaluated with the runs' parameters. kappa_res = -ln(T/T_nonres)/L
    with T_nonres from sigma0 -> 0; L -> 0 is evaluated at L = 1 nm."""

    def __init__(self, module, cfg):
        self.m = module
        atom = dict(cfg["atom"])
        self.r = module.rates(atom)
        self.r_nr = module.rates(dict(atom, sigma0_prefactor=1e-30))
        mono, sase = cfg["mono"], cfg["sase"]
        self.exp = {
            "mono": dict(bw_eV=tl.tl_bandwidth_eV(mono["tau_fs"]), tau_fs=mono["tau_fs"],
                         wx_nm=mono["wx_nm"], wy_nm=mono["wy_nm"], T_beamline=mono["T_beamline"]),
            "sase": dict(E_c=sase["E_c"], bw_eV=sase["bw_eV"], tau_fs=sase["tau_fs"],
                         wx_nm=sase["wx_nm"], wy_nm=sase["wy_nm"], T_beamline=sase["T_beamline"]),
        }
        t, wt = np.polynomial.hermite.hermgauss(48)
        self.quad = {"beam": module.pulse_quadrature(),
                     "axis": (np.ones((1, t.size)), t[None, :], (wt / np.sqrt(np.pi))[None, :])}

    @functools.lru_cache(maxsize=None)
    def _T(self, beam, x_key, E_uJ, view, L_um, nonres):
        r = dict(self.r_nr if nonres else self.r, L=max(L_um, 1e-3) * 1e-4)
        E = np.array(x_key) + self.r["E_ul"]
        if beam == "mono":
            return self.m.transmittance_exp2(E, E_uJ, r, self.exp["mono"], self.quad[view])
        return self.m.transmittance_exp1(E, E_uJ, r, self.exp["sase"], self.quad[view])

    def kres(self, beam, x, E_uJ, view, L_um=0.0):
        key = tuple(np.round(np.atleast_1d(x), 9))
        L_cm = max(L_um, 1e-3) * 1e-4
        T = self._T(beam, key, float(E_uJ), view, float(L_um), False)
        T_nr = self._T(beam, key, float(E_uJ), view, float(L_um), True)
        return -np.log(T / T_nr) / L_cm, -np.log(T_nr) / L_cm

    def mean_saturation(self, beam, E_uJ, view):
        """Mean saturation parameter s = W/W_sat over the attempt-1 slices of a thin foil, weighted
        by each slice's contribution to the measured line (energy weight x local line-centre
        kappa_res), so time and space where the ground state is already depleted count little."""
        m, ex = self.m, self.exp[beam]
        r = dict(self.r, L=1e-7)
        V, Tt, w = self.quad[view]
        E_ph = ex["E_c"] if beam == "sase" else self.r["E_ul"]
        F_pk, J_pk = m.beam_peak(ex, E_uJ, E_ph)
        J, Phi = m.slices(F_pk, J_pk, V, Tt)
        a, b, gam, Wsat, rho, k = m.slice_state(J, Phi, r)
        if beam == "sase":
            S_ul = voigt_profile((r["E_ul"] - ex["E_c"]) / HBAR, ex["bw_eV"] / HBAR * tl.FWHM_TO_STD, gam)
            W = 0.5 * np.pi * r["sigma0"] * r["Gsp"] * S_ul * J
        else:
            W = m.sigma_voigt(r["E_ul"], gam, r, ex) * J
        s = W / Wsat
        kap = r["sig_L3"] * J * rho / (b * gam * (1.0 + s))
        return float(np.sum(w * kap * s) / np.sum(w * kap))


def ein_over_x(X):
    """(1/X) int_0^X (1 - e^-y)/y dy: the uniform-v average of (1 - e^{-X v})/(X v)."""
    X = np.asarray(X, float)
    small = X < 1e-3
    Xs = np.where(small, 1.0, X)
    return np.where(small, 1 - X / 4 + X**2 / 18, (exp1(Xs) + np.log(Xs) + EULER_GAMMA) / Xs)


class Limits:
    """Closed-form limits of the analytic model (derivation Eqs. 7, 11, 33-36)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.r = tl.rates(cfg["atom"])
        self.wf = tl.weak_field_constants(self.r)
        r, wf = self.r, self.wf
        g0 = wf["gamma"]
        mono, sase = cfg["mono"], cfg["sase"]
        sig_G = tl.tl_bandwidth_eV(mono["tau_fs"]) / HBAR * tl.FWHM_TO_STD
        self.sig_line = {"mono": 0.5 * np.pi * r["s0G"] * voigt_profile(0.0, sig_G, g0),
                         "sase": wf["sig_pk"]}
        sig_w = sase["bw_eV"] / HBAR * tl.FWHM_TO_STD
        self.S_ul = voigt_profile((r["E_ul"] - sase["E_c"]) / HBAR, sig_w, g0)
        self.eps = np.pi * g0 * self.S_ul
        self.ceiling = {"mono": wf["c0"] * wf["k0"], "sase": wf["c0"] * wf["k0"] / self.eps}
        # W per unit peak flux at the line (mono: Voigt cross section; SASE: Eq. 26)
        self.W_per_J = {"mono": self.sig_line["mono"], "sase": 0.5 * np.pi * r["s0G"] * self.S_ul}

    def beam_cfg(self, beam):
        return self.cfg[beam], (self.r["E_ul"] if beam == "mono" else self.cfg["sase"]["E_c"])

    def peak(self, beam, E_uJ):
        b, E_ph = self.beam_cfg(beam)
        return tl.beam_peak(b, E_uJ, E_ph)

    def memory_R(self, beam):
        """Eq. (34): QSS overstates the weak-field hole population by 1/R for a Gaussian pulse."""
        y = self.r["Gl"] * self.cfg[beam]["tau_fs"] / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return float(np.sqrt(np.pi) * y * erfcx(y))

    def linear(self, beam, E_uJ, view):
        """Weak-field limit, Eq. (33) (QSS, no memory factor): kappa_res = n_a sig_line sig_L3 <J>/Gamma_l."""
        _, J_pk = self.peak(beam, E_uJ)
        Jbar = J_pk / np.sqrt(2.0) * (1.0 if view == "axis" else 0.5)
        return self.r["n_a"] * self.sig_line[beam] * self.r["sig_L3"] * Jbar / self.r["Gl"]

    def ceiling_depleted(self, beam, E_uJ, view):
        """RSA ceiling times the energy-weighted ground-state population <J g>/<J>."""
        F_pk, _ = self.peak(beam, E_uJ)
        X = self.r["sig_g"] * F_pk
        depl = -np.expm1(-X) / X if view == "axis" else ein_over_x(X)
        return self.ceiling[beam] * depl

    def regime_energies(self, beam):
        """Pulse energies [uJ] at which, on axis at the pulse peak: s = W/W_sat = 1; Omega = gamma
        (mono) or 2 W = gamma (SASE, in-band Rabi frequency = gamma); sigma_g F = 1."""
        F1, J1 = self.peak(beam, 1.0)
        W1 = self.W_per_J[beam] * J1
        g0 = self.wf["gamma"]
        E_coh = g0**2 / (self.r["s0G"] * J1) if beam == "mono" else g0 / (2.0 * W1)
        return {"sat": self.wf["Wsat"] / W1, "coh": E_coh, "depl": 1.0 / (self.r["sig_g"] * F1)}

    def fwhm_weak(self, beam):
        g_eV = 2.0 * self.wf["gamma"] * HBAR
        if beam == "sase":
            return g_eV
        fG = tl.tl_bandwidth_eV(self.cfg["mono"]["tau_fs"])
        return 0.5346 * g_eV + np.sqrt(0.2166 * g_eV**2 + fG**2)      # Olivero-Longbothum Voigt

    def fwhm_power_peak(self, E_uJ):
        """mono: local power-broadened width 2 gamma sqrt(1 + s) with s at the pulse peak on axis,
        an upper bound for the pulse-averaged line."""
        _, J_pk = self.peak("mono", E_uJ)
        return 2.0 * self.wf["gamma"] * HBAR * np.sqrt(1.0 + self.W_per_J["mono"] * J_pk / self.wf["Wsat"])

    def fwhm_power_mean(self, s_mean):
        return 2.0 * self.wf["gamma"] * HBAR * np.sqrt(1.0 + np.asarray(s_mean))


def qss_bloch(r, J, delta):
    """Per-hole Bloch vector (u, v, w)/(l + u) in the quasi-steady state (derivation Eq. 12 and the
    Bloch steady state c = (i/2) Om w / (gamma - i delta)) for flux J and detuning delta [fs^-1]."""
    J = np.asarray(J, float)
    a = r["Gu"] + r["sig_u"] * J
    b = r["Gl"] + r["sig_l"] * J
    gam = 0.5 * (a + b) + r["gphi"]
    W = r["s0G"] * J * gam / (2.0 * (gam**2 + delta**2))
    wn = a / (a + 2.0 * W)
    cn = 0.5j * np.sqrt(r["s0G"] * J) * wn / (gam - 1j * delta)
    return 2.0 * cn.real, 2.0 * cn.imag, wn


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
def mark_regimes(ax, lim, beam, labels=True):
    """Vertical hairlines at the regime boundaries; labels in a strip above the plotting area
    (callers leave room with the title pad). Boundaries closer than a factor 1.5 share a label."""
    reg = lim.regime_energies(beam)
    marks = sorted([(reg["sat"], "s = 1"), (reg["coh"], "Ω = γ" if beam == "mono" else "Ω_in = γ"),
                    (reg["depl"], "σ_g F = 1")])
    groups = [[marks[0]]]
    for E, lab in marks[1:]:
        if E / groups[-1][-1][0] < 1.5:
            groups[-1].append((E, lab))
        else:
            groups.append([(E, lab)])
    for grp in groups:
        for E, _ in grp:
            ax.axvline(E, color=AXIS, lw=0.9, zorder=0)
        if labels:
            E_mid = np.exp(np.mean([np.log(E) for E, _ in grp]))
            ax.text(E_mid, 1.01, ", ".join(lab for _, lab in grp), transform=ax.get_xaxis_transform(),
                    color=INK2, fontsize=7.5, va="bottom", ha="center")


def energies_in(runs, key, wanted):
    have = sorted(runs.get(key, {}))
    if wanted is None:
        return have
    return [E for E in wanted if any(np.isclose(E, h) for h in have)]


def get(runs, key, E):
    for h, d in runs.get(key, {}).items():
        if np.isclose(h, E):
            return d
    return None


def fig_spectra(runs, an, lim, cfg, view, mode, energies, out):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), constrained_layout=True)
    r = lim.r
    for row, beam in enumerate(("mono", "sase")):
        Es = energies_in(runs, (beam, mode), energies)
        cols = ramp(len(Es))
        ax_log, ax_n = axes[row]
        for E, col in zip(Es, cols):
            num = Numeric(get(runs, (beam, mode), E), view, r["n_a"], r["E_ul"])
            k_an, _ = an.kres(beam, num.x, E, view)
            ax_log.semilogy(num.x, num.kres_display(max_rel_err=0.25), color=col, lw=1.8)
            ax_log.semilogy(num.x, k_an, color=col, lw=1.1, ls=(0, (4, 2)))
            k = num.kres_display()
            ax_n.plot(num.x, k / np.nanmax(k), color=col, lw=1.8)
            ax_n.plot(num.x, k_an / np.nanmax(k_an), color=col, lw=1.1, ls=(0, (4, 2)))
        ax_log.set(xlim=(-20, 20), ylabel="κ_res  (cm⁻¹)", title=f"{BEAM_TITLE[beam]}: resonant κ, L → 0")
        if beam == "sase":
            ax_log.text(0.01, 0.02, "0.1 eV smoothing; bins with > 25 % statistical error hidden",
                        transform=ax_log.transAxes, color=INK2, fontsize=7.5)
        ax_n.set(xlim=(-12, 12), ylim=(-0.05, 1.1), ylabel="κ_res / peak",
                 title="line shape (each curve normalised to its peak)")
        for ax in (ax_log, ax_n):
            ax.set_xlabel("E − E_Kα1  (eV), photon energy of the scan" if beam == "mono"
                          else "E − E_Kα1  (eV), spectrometer")
        handles = [Line2D([], [], color=c, lw=2.2, label=fmt_E(E)) for E, c in zip(Es, cols)]
        if row == 0:
            handles += [Line2D([], [], color=INK2, lw=1.8, label=MODE_LABEL[mode].replace(", ", ",\n")),
                        Line2D([], [], color=INK2, lw=1.1, ls=(0, (4, 2)), label="analytic, attempt 1")]
        ax_n.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="pulse energy",
                    title_fontsize=8)
    fig.suptitle(f"Resonant absorption coefficient spectra ({view} view, {MODE_LABEL[mode]})", color=INK)
    save(fig, out)


def peak_series(runs, beam, mode, view, r, L_index=None):
    Es, pk, dpk, fw, dfw, knr, eta = [], [], [], [], [], [], []
    for E in energies_in(runs, (beam, mode), None):
        num = Numeric(get(runs, (beam, mode), E), view, r["n_a"], r["E_ul"])
        p, _, f = num.peak_fwhm(L_index)
        e_p, e_f = num.jackknife() if L_index is None else (0.0, 0.0)
        Es.append(E); pk.append(p); fw.append(f); dpk.append(e_p); dfw.append(e_f)
        knr.append(num.knr_center()); eta.append(num.photons_per_hole())
    return dict(E=np.array(Es), pk=np.array(pk), dpk=np.array(dpk), fwhm=np.array(fw),
                dfwhm=np.array(dfw), knr=np.array(knr), eta=np.array(eta))


def analytic_peak(an, beam, Es, view, L_um=0.0):
    k, knr = zip(*[an.kres(beam, [0.0], E, view, L_um) for E in Es])
    return np.array(k)[:, 0], np.array(knr)[:, 0]


def analytic_fwhm(an, beam, E, view, x):
    k, _ = an.kres(beam, x, E, view)
    return peak_fwhm(x, k)[2]


def energy_grid(runs):
    Es = sorted({E for per in runs.values() for E in per})
    return np.logspace(np.log10(Es[0]), np.log10(Es[-1]), 49)


def fig_peak(runs, an, lim, view, out):
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10.5), sharex="col", constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[3.0, 1.3, 1.3]))
    r = lim.r
    E_dense = energy_grid(runs)
    for col, beam in enumerate(("mono", "sase")):
        ax, ax_r, ax_n = axes[:, col]
        k_an, knr_an = analytic_peak(an, beam, E_dense, view)
        ax.loglog(E_dense, k_an, color=INK, lw=1.3, label="analytic, attempt 1", zorder=3)
        k_lin = lim.linear(beam, E_dense, view)
        ax.loglog(E_dense, np.where(k_lin < 5.0 * lim.ceiling[beam], k_lin, np.nan), color=INK2, lw=1.0,
                  ls=(0, (5, 3)), label="weak-field limit, Eq. (33)")
        ax.axhline(lim.ceiling[beam], color=INK2, lw=1.0, ls=(0, (6, 2, 1, 2)),
                   label="RSA ceiling c₀k₀" + ("" if beam == "mono" else "/ε") + ", Eq. " + ("(35)" if beam == "mono" else "(36)"))
        ax.loglog(E_dense, lim.ceiling_depleted(beam, E_dense, view), color=INK2, lw=1.0, ls=(0, (1, 2)),
                  label="ceiling × ⟨ground-state population⟩")
        ax_n.plot(E_dense, knr_an, color=INK, lw=1.3)
        ratios = []
        for mode, marker in (("mb", "o"), ("re", "s")):
            if (beam, mode) not in runs:
                continue
            s = peak_series(runs, beam, mode, view, r)
            face = C_MODE[mode] if mode == "mb" else SURFACE
            ax.errorbar(s["E"], s["pk"], yerr=s["dpk"], color=C_MODE[mode], marker=marker, ms=5,
                        mfc=face, lw=1.6, capsize=0, label=MODE_LABEL[mode], zorder=4)
            k_at, _ = analytic_peak(an, beam, s["E"], view)
            ratios.append(s["pk"] / k_at)
            ax_r.errorbar(s["E"], ratios[-1], yerr=s["dpk"] / k_at, color=C_MODE[mode], marker=marker, ms=5,
                          mfc=face, lw=1.6, capsize=0)
            ax_n.plot(s["E"], s["knr"], color=C_MODE[mode], marker=marker, ms=5, mfc=face, lw=1.6)
        R = lim.memory_R(beam)
        ax_r.axhline(1.0, color=INK, lw=1.0)
        ax_r.axhline(R, color=INK2, lw=1.0, ls=(0, (5, 3)))
        ax_r.text(E_dense[0], R, f" hole-memory factor R = {R:.3f}, Eq. (34)", color=INK2, fontsize=7.5, va="top")
        ax.set(ylabel="peak κ_res  (cm⁻¹), L → 0")
        ax.set_title(BEAM_TITLE[beam], pad=16)
        ax.set_ylim(bottom=max(1e-4, 0.3 * np.nanmin(k_an)))
        allr = np.concatenate(ratios) if ratios else np.array([1.0])
        ax_r.set(ylabel="numerical / attempt 1",
                 ylim=(min(0.8, np.nanmin(allr) - 0.05), max(1.2, np.nanmax(allr) + 0.05)))
        ax_n.set(ylabel="κ_nr at E_Kα1  (cm⁻¹)", xlabel="pulse energy on target (µJ)", xscale="log")
        ax_n.text(0.01, 0.05, "non-resonant absorption (ground-state bleaching)", transform=ax_n.transAxes,
                  color=INK2, fontsize=7.5)
        for a in (ax, ax_r, ax_n):
            mark_regimes(a, lim, beam, labels=a is ax)
        ax.legend(loc="lower right")
    fig.suptitle(f"Peak resonant absorption coefficient vs pulse energy ({view} view)", color=INK)
    save(fig, out)
    return


def fig_fwhm(runs, an, lim, view, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    r = lim.r
    E_dense = energy_grid(runs)
    for ax, beam in zip(axes, ("mono", "sase")):
        some = next((get(runs, (beam, m), E) for m in ("mb", "re") for E in energies_in(runs, (beam, m), None)), None)
        if some is None:
            continue
        x = Numeric(some, view, r["n_a"], r["E_ul"]).x
        E_an = E_dense[::2]
        ax.plot(E_an, [analytic_fwhm(an, beam, E, view, x) for E in E_an], color=INK, lw=1.3,
                label="analytic, attempt 1")
        ax.axhline(lim.fwhm_weak(beam), color=INK2, lw=1.0, ls=(0, (5, 3)),
                   label="weak-field width " + ("(2γ ⊗ pulse spectrum)" if beam == "mono" else "2γ"))
        if beam == "mono":
            ax.plot(E_dense, lim.fwhm_power_peak(E_dense), color=INK2, lw=1.0, ls=(0, (6, 2, 1, 2)),
                    label="2γ√(1+s), s at the pulse peak on axis (upper bound)")
        s_mean = [an.mean_saturation(beam, E, view) for E in E_an]
        ax.plot(E_an, lim.fwhm_power_mean(s_mean), color=INK2, lw=1.0, ls=(0, (1, 2)),
                label="2γ√(1+⟨s⟩), ⟨s⟩ absorption-weighted" + (" (heuristic, not attempt 1)" if beam == "sase" else ""))
        tops = [2.5 * lim.fwhm_weak(beam)]
        for mode, marker in (("mb", "o"), ("re", "s")):
            if (beam, mode) not in runs:
                continue
            s = peak_series(runs, beam, mode, view, r)
            face = C_MODE[mode] if mode == "mb" else SURFACE
            ax.errorbar(s["E"], s["fwhm"], yerr=s["dfwhm"], color=C_MODE[mode], marker=marker, ms=5,
                        mfc=face, lw=1.6, label=MODE_LABEL[mode], zorder=4)
            tops.append(1.15 * np.nanmax(s["fwhm"] + s["dfwhm"]))
        ax.set(xscale="log", xlabel="pulse energy on target (µJ)", ylabel="FWHM of κ_res(E), L → 0  (eV)")
        ax.set_title(BEAM_TITLE[beam], pad=16)
        ax.set_ylim(0, max(tops))
        mark_regimes(ax, lim, beam)
        ax.legend(loc="upper left")
    fig.suptitle(f"Line width vs pulse energy ({view} view)", color=INK)
    save(fig, out)


def fig_bloch(runs, lim, energies, out):
    key = ("mono", "mb")
    Es = energies_in(runs, key, energies)
    if not Es:
        return
    r = lim.r
    cols = ramp(len(Es))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), constrained_layout=True)
    (ax_vw, ax_uv), (ax_w, ax_v) = axes
    d0 = get(runs, key, Es[0])
    det = d0["ts_dE"]
    j0 = int(np.argmin(np.abs(det)))
    j1 = int(np.argmin(np.abs(det - 2.0))) if det.size > 1 else j0
    Jl = np.logspace(12, 21.5, 400)
    for idx, ax, (ia, ib) in ((j0, ax_vw, (1, 2)), (j1, ax_uv, (0, 1))):
        loc = qss_bloch(r, Jl, det[idx] / HBAR)
        ax.plot(loc[ia], loc[ib], color=INK, lw=1.1, ls=(0, (4, 2)), zorder=5)
    th = np.linspace(0, 2 * np.pi, 300)
    for ax in (ax_vw, ax_uv):
        ax.plot(np.cos(th), np.sin(th), color=AXIS, lw=0.8, zorder=0)
        ax.set_aspect("equal")
    for E, col in zip(Es, cols):
        d = get(runs, key, E)
        t = d["ts_t"]
        for idx, ax, (ia, ib) in ((j0, ax_vw, (1, 2)), (j1, ax_uv, (0, 1))):
            g, l, u, cr, ci, J = d["ts"][idx].astype(float)
            n = l + u
            ok = (J > 1e-3 * J.max()) & (n > 1e-3 * n.max())
            b = (2 * cr / np.where(ok, n, 1), 2 * ci / np.where(ok, n, 1), (l - u) / np.where(ok, n, 1))
            ax.plot(np.where(ok, b[ia], np.nan), np.where(ok, b[ib], np.nan), color=col, lw=1.8)
            i_pk = int(np.argmax(J))
            ax.plot(b[ia][i_pk], b[ib][i_pk], "o", color=col, ms=5, mec=SURFACE, mew=1.0, zorder=6)
            if ax is ax_vw:
                qu, qv, qw = qss_bloch(r, J, det[idx] / HBAR)
                ax_w.plot(t, np.where(ok, b[2], np.nan), color=col, lw=1.8)
                ax_w.plot(t, np.where(ok, qw, np.nan), color=col, lw=1.0, ls=(0, (4, 2)))
                ax_v.plot(t, np.where(ok, b[1], np.nan), color=col, lw=1.8)
                ax_v.plot(t, np.where(ok, qv, np.nan), color=col, lw=1.0, ls=(0, (4, 2)))
    ax_vw.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), xlabel="v/n = 2 Im ρ_ul / (ρ_ll + ρ_uu)  (absorptive)",
              ylabel="w/n = (ρ_ll − ρ_uu) / (ρ_ll + ρ_uu)", title=f"resonant drive (δ = {det[j0]:+.1f} eV)")
    ax_uv.set(xlim=(-1.05, 1.05), ylim=(-0.05, 1.05), xlabel="u/n = 2 Re ρ_ul / n  (dispersive)",
              ylabel="v/n", title=f"detuned drive (δ = {det[j1]:+.1f} eV)")
    for ax, lab in ((ax_w, "w/n  (inversion per hole)"), (ax_v, "v/n  (absorptive coherence per hole)")):
        ax.set(xlim=(-12, 12), xlabel="time (fs), pulse peak at 0", ylabel=lab)
    ax_w.set_ylim(-0.05, 1.05)
    handles = [Line2D([], [], color=c, lw=2.2, label=fmt_E(E)) for E, c in zip(Es, cols)]
    handles += [Line2D([], [], color=INK2, lw=1.8, label="Maxwell-Bloch,\nz = 0, on axis"),
                Line2D([], [], color=INK, lw=1.1, ls=(0, (4, 2)), label="quasi-steady state\n(Eq. 12, same J(t))"),
                Line2D([], [], color=INK2, marker="o", ls="", ms=5, label="pulse peak"),
                Line2D([], [], color=AXIS, lw=0.8, label="|b| = 1 (pure state)")]
    ax_uv.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="pulse energy",
                 title_fontsize=8)
    fig.suptitle("Per-hole Bloch vector during the pulse (mono, transform-limited 6 fs); "
                 "n = ρ_ll + ρ_uu, shown where J > 10⁻³ J_max", color=INK)
    save(fig, out)


def fig_thickness(runs, an, lim, view, out, L_show=(0.0, 1.0, 5.0, 20.0)):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    r = lim.r
    E_dense = energy_grid(runs)
    for ax, beam in zip(axes, ("mono", "sase")):
        key = (beam, "mb")
        if key not in runs:
            continue
        z_rec = next(iter(runs[key].values()))["z_rec_um"]
        Ls = [L for L in L_show if L == 0.0 or np.any(np.isclose(z_rec, L))]
        for L, col in zip(Ls, ramp(len(Ls))):
            iL = None if L == 0.0 else int(np.argmin(np.abs(z_rec - L)))
            s = peak_series(runs, beam, "mb", view, r, L_index=iL)
            ax.loglog(s["E"], s["pk"], color=col, marker="o", ms=4.5, lw=1.8,
                      label=("L → 0" if L == 0 else f"L = {L:g} µm"))
            ax.loglog(E_dense, analytic_peak(an, beam, E_dense, view, L)[0], color=col, lw=1.0, ls=(0, (4, 2)))
        ax.set(xlabel="pulse energy on target (µJ)", ylabel="apparent peak κ_res = −ln(T/T_nr)/L  (cm⁻¹)")
        ax.set_title(BEAM_TITLE[beam], pad=16)
        handles = ax.get_legend_handles_labels()[0]
        handles += [Line2D([], [], color=INK2, lw=1.8, marker="o", ms=4.5, label="Maxwell-Bloch"),
                    Line2D([], [], color=INK2, lw=1.0, ls=(0, (4, 2)), label="analytic, attempt 1")]
        ax.legend(handles=handles, loc="lower right")
        mark_regimes(ax, lim, beam)
    fig.suptitle(f"Foil thickness: apparent peak absorption coefficient ({view} view)", color=INK)
    save(fig, out)


def write_table(runs, an, lim, view, path):
    rows = []
    r = lim.r
    for beam in ("mono", "sase"):
        for mode in ("mb", "re"):
            if (beam, mode) not in runs:
                continue
            s = peak_series(runs, beam, mode, view, r)
            x = Numeric(get(runs, (beam, mode), s["E"][0]), view, r["n_a"], r["E_ul"]).x
            k_an, knr_an = analytic_peak(an, beam, s["E"], view)
            for i, E in enumerate(s["E"]):
                rows.append(dict(beam=beam, mode=mode, view=view, E_uJ=E,
                                 kres_peak_cm=s["pk"][i], kres_peak_err=s["dpk"][i],
                                 kres_peak_attempt1=k_an[i], kres_linear_eq33=lim.linear(beam, E, view),
                                 fwhm_eV=s["fwhm"][i], fwhm_err=s["dfwhm"][i],
                                 fwhm_attempt1=analytic_fwhm(an, beam, E, view, x),
                                 knr_cm=s["knr"][i], knr_attempt1=knr_an[i],
                                 photons_per_hole=s["eta"][i]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in row.items()})
    return rows


def save(fig, path):
    fig.savefig(path + ".png", dpi=170)
    fig.savefig(path + ".pdf")
    plt.close(fig)
    print("wrote", path + ".png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="folder written by submit_twolevel_sweep.sh")
    parser.add_argument("--analytic-dir", default=os.path.join(REPO_ROOT, "..", "RSA-derivation"),
                        help="folder with rsa_transmittance.py")
    parser.add_argument("--out", default=None, help="default: figs/twolevel/<data folder name>")
    parser.add_argument("--view", choices=("axis", "beam", "both"), default="both")
    parser.add_argument("--spectra-energies", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--bloch-energies", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--sase-blocks", type=int, default=None,
                        help="use only the first N SASE blocks (same shots as a shorter run, for paired checks)")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.analytic_dir))
    try:
        import rsa_transmittance
    except ImportError:
        sys.exit(f"rsa_transmittance.py not found in {args.analytic_dir} (--analytic-dir)")

    runs = load_runs(args.data)
    if args.sase_blocks:
        for (beam, _), per_E in runs.items():
            for d in per_E.values() if beam == "sase" else ():
                for k in ("den", "kres_num", "knr_num", "int0"):
                    d[k] = d[k][: args.sase_blocks]
                d["num_out_view"] = d["num_out_view"][:, : args.sase_blocks]
    cfg = run_config(runs)
    an = Analytic(rsa_transmittance, cfg)
    lim = Limits(cfg)
    out_root = args.out or os.path.join(REPO_ROOT, "figs", "twolevel", os.path.basename(os.path.normpath(args.data)))
    print("runs:", {f"{b}_{m}": len(v) for (b, m), v in sorted(runs.items())})
    for beam in ("mono", "sase"):
        reg = lim.regime_energies(beam)
        print(f"{beam}: on-axis s = 1 at {reg['sat']:.3g} uJ, coherent (Omega{'_in' if beam == 'sase' else ''} = gamma) "
              f"at {reg['coh']:.3g} uJ, sigma_g F = 1 at {reg['depl']:.3g} uJ; RSA ceiling "
              f"{lim.ceiling[beam]:.1f} /cm; memory factor R = {lim.memory_R(beam):.3f}")

    views = ("axis", "beam") if args.view == "both" else (args.view,)
    for view in views:
        out = os.path.join(out_root, view)
        os.makedirs(out, exist_ok=True)
        for mode in ("mb", "re"):
            if any(k[1] == mode for k in runs):
                fig_spectra(runs, an, lim, cfg, view, mode, args.spectra_energies, os.path.join(out, f"fig1_spectra_{mode}"))
        fig_peak(runs, an, lim, view, os.path.join(out, "fig2_peak"))
        if view == "axis":
            fig_bloch(runs, lim, args.bloch_energies, os.path.join(out, "fig3_bloch"))
        fig_fwhm(runs, an, lim, view, os.path.join(out, "fig4_fwhm"))
        fig_thickness(runs, an, lim, view, os.path.join(out, "fig5_thickness"))
        write_table(runs, an, lim, view, os.path.join(out, f"table_{view}.csv"))
        print("wrote", os.path.join(out, f"table_{view}.csv"))


if __name__ == "__main__":
    main()
