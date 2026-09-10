"""Diagnostic plots for one production sweep of the full (double-satellite + L2) model.

Reads the accumulated .npz chunk files written by run_intensity_sweep.py for
data/production_sweep_sase_24437070/Cu-seed-SASE-double-satellite and produces four
figures into figs/:

  1. transmittance spectra at every pulse energy, plus dip depth vs pulse energy
  2. the satellite cascade in time at 40 uJ (which block is populated when)
  3. fluence scaling of every tracked manifold, on log-log axes
  4. temporal reshaping of the pulse by the foil

Run from the repo root:  python scripts/plot_double_satellite_diagnostics.py

NOTE on what is and is not trustworthy in these files. The per-block traces
(rho_l3/rho_l2/rho_ee, and their _sat counterparts) are sampled at the transverse
centre pixel, as intended. rho_ground_t_last / rho_2s_t_last / rho_other_t_last are
NOT: tools.compute_run_outputs clamps cx,cy to 0 based on rho_ijtxyz's length-1
footprint, but Sample.py's lean path stores rho_ground/2s/other_txyz on the full
(x, y) grid, so those three are read at the transverse corner instead of the centre.
They are therefore excluded from every plot here.
"""

import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(REPO, "data", "production_sweep_sase_24437070",
                     "Cu-seed-SASE-double-satellite")
FIGS = os.path.join(REPO, "figs")
E_SEED_VALUES = [2.0, 9.0, 40.0, 60.0]

# Seed centre energy used by the sweep configs (seed_center_E), so that
# womega_ar (a detuning axis) maps onto absolute photon energy.
SEED_CENTRE_EV = 8045.0
E_KA1, E_KA2 = 8047.91, 8027.98

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
E_COLOURS = {2.0: PALETTE[0], 9.0: PALETTE[1], 40.0: PALETTE[2], 60.0: PALETTE[3]}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "lines.linewidth": 1.9, "figure.dpi": 120, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})


def log_axis(ax, values, axis="x"):
    """Label a log axis with just the sampled values, with no minor-tick clutter."""
    from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(values))
    target.set_major_formatter(FixedFormatter([f"{v:g}" for v in values]))
    target.set_minor_formatter(NullFormatter())
    target.set_minor_locator(FixedLocator([]))


def load(e_seed):
    """Combine every chunk file for one pulse energy into element-wise means."""
    files = sorted(glob.glob(os.path.join(SWEEP, f"runs_seed_{e_seed}_uJ", "*.npz")))
    if not files:
        raise FileNotFoundError(f"no chunk files for {e_seed} uJ under {SWEEP}")

    totals, out, n_reps = {}, {}, 0
    for path in files:
        d = np.load(path, allow_pickle=True)
        n_reps += int(d["n_reps"])
        for key in d.files:
            if key.endswith("_sum"):
                stem = key[:-4]
                s, c = d[key], d.get(f"{stem}_count")
                c = np.broadcast_to(np.asarray(c), np.shape(s)) if c is not None \
                    else np.full(np.shape(s), int(d["n_reps"]))
                if stem in totals:
                    totals[stem] = (totals[stem][0] + s, totals[stem][1] + c)
                else:
                    totals[stem] = (s.copy(), c.copy())
            elif not (key.endswith("_sumsq") or key.endswith("_count")):
                out[key] = d[key]

    for stem, (s, c) in totals.items():
        out[stem] = np.real(s / np.maximum(c, 1))
    out["n_reps"] = n_reps
    return out


DATA = {e: load(e) for e in E_SEED_VALUES}
NAMES = [str(n) for n in DATA[40.0]["satellite_channel_names"]]
T_AXIS = DATA[40.0]["t_axis"]
DT = T_AXIS[1] - T_AXIS[0]
os.makedirs(FIGS, exist_ok=True)


def transmittance(arrays):
    energy = SEED_CENTRE_EV + arrays["womega_ar"]
    return energy, arrays["I_int_thy_w_last"] / arrays["I_int_thy_w_0"]


def dip_metrics(energy, trans, centre, window=(8025, 8070), base_hw=15.0, dip_hw=8.0):
    keep = (energy >= window[0]) & (energy <= window[1])
    energy, trans = energy[keep], trans[keep]
    baseline = np.mean(trans[np.abs(energy - centre) > base_hw])
    near = np.abs(energy - centre) <= dip_hw
    return baseline, np.min(trans[near]), baseline - np.min(trans[near])


# =============================================================================
# 1. transmittance spectra + dip depth vs pulse energy
# =============================================================================
fig, (ax_spec, ax_dip) = plt.subplots(1, 2, figsize=(12.4, 4.6),
                                      gridspec_kw={"width_ratios": [1.75, 1]})

for e_seed in E_SEED_VALUES:
    energy, trans = transmittance(DATA[e_seed])
    ax_spec.plot(energy, trans, color=E_COLOURS[e_seed],
                 label=f"{e_seed:g} " + r"$\mu$J")

for line_e, label in ((E_KA1, r"$K\alpha_1$"), (E_KA2, r"$K\alpha_2$")):
    ax_spec.axvline(line_e, color="0.45", lw=0.9, ls="--", zorder=0)
    ax_spec.text(line_e, 0.468, label, ha="center", va="bottom", fontsize=9.5, color="0.35")

ax_spec.set_xlim(8010, 8070)
ax_spec.set_ylim(0.16, 0.47)
ax_spec.set_xlabel("Photon energy (eV)")
ax_spec.set_ylabel(r"Transmittance $T(\omega)$")
ax_spec.set_title("Simulated transmittance, full model")
ax_spec.legend(title="Pulse energy", loc="lower left")

depths = {}
for centre, label, marker in ((E_KA1, r"$K\alpha_1$ dip", "o"), (E_KA2, r"$K\alpha_2$ dip", "s")):
    d = [dip_metrics(*transmittance(DATA[e]), centre=centre)[2] for e in E_SEED_VALUES]
    depths[label] = d
    ax_dip.plot(E_SEED_VALUES, d, marker=marker, color="#2a78d6" if marker == "o" else "#eb6834",
                label=label)

ax_dip.set_xscale("log")
ax_dip.set_xlabel(r"Pulse energy ($\mu$J)")
ax_dip.set_ylabel("Dip depth below baseline")
ax_dip.set_title("Growth of the dip with fluence")
log_axis(ax_dip, E_SEED_VALUES)
ax_dip.set_ylim(0, 0.185)
ax_dip.legend(loc="upper left")
ax_dip.annotate(r"$K\alpha_1$ saturating", xy=(57, 0.1465), xytext=(9.5, 0.093),
                fontsize=9, color="0.35",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=1.0,
                                connectionstyle="arc3,rad=-0.25"))

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "double_satellite_transmittance_and_dip.pdf"))
fig.savefig(os.path.join(FIGS, "double_satellite_transmittance_and_dip.png"))
plt.close(fig)

# =============================================================================
# 2. the cascade in time at 40 uJ
# =============================================================================
arrays = DATA[40.0]
sat_l3 = arrays["rho_l3_t_last_sat"]

groups = [
    ("base, $2p_{3/2}^{-1}$ (L3)", arrays["rho_l3_t_last"], "#111111", "-", 2.6),
    ("base, $2p_{1/2}^{-1}$ (L2)", arrays["rho_l2_t_last"], "#111111", "--", 1.8),
    ("base, $1s^{-1}$ (K)", arrays["rho_ee_t_last"], "#111111", ":", 2.0),
]
single = [n for n in NAMES if len(n) <= 3]
double = [n for n in NAMES if len(n) > 3]
for i, name in enumerate(single):
    groups.append((f"1 spectator, ${name}$", sat_l3[NAMES.index(name)], PALETTE[i % 4], "-", 1.5))
for i, name in enumerate(double):
    groups.append((f"2 spectators, ${name}$", sat_l3[NAMES.index(name)], PALETTE[4 + i % 4], "-", 1.5))

fig, (ax_abs, ax_norm) = plt.subplots(1, 2, figsize=(12.4, 4.6), sharex=True)

seed = np.abs(arrays["I_t_0"])
peak = max(arrays["rho_l3_t_last"].max(), sat_l3.max())
ax_abs.fill_between(T_AXIS, 0, seed / seed.max() * peak, color="0.88", zorder=0,
                    label="incident pulse (arb. scale)")
ax_norm.fill_between(T_AXIS, 0, seed / seed.max(), color="0.88", zorder=0)

for label, trace, colour, ls, lw in groups:
    ax_abs.plot(T_AXIS, trace, color=colour, ls=ls, lw=lw, label=label)
    ax_norm.plot(T_AXIS, trace / trace.max(), color=colour, ls=ls, lw=lw)

ax_abs.set_ylim(0, peak * 1.06)
ax_abs.set_xlabel("Time (fs)")
ax_abs.set_ylabel("Dipole-weighted population (arb.)")
ax_abs.set_title(r"Populations at the exit face, 40 $\mu$J")

ax_norm.set_xlim(6, 18)
ax_norm.set_ylim(0, 1.12)
ax_norm.set_xlabel("Time (fs)")
ax_norm.set_ylabel("Normalised to own peak")
ax_norm.set_title("Same traces, peak-normalised")

# annotate the cascade delay using intensity-weighted centroids
c_base = float((T_AXIS * arrays["rho_l3_t_last"]).sum() / arrays["rho_l3_t_last"].sum())
c_double = float(np.mean([(T_AXIS * sat_l3[NAMES.index(n)]).sum() / sat_l3[NAMES.index(n)].sum()
                          for n in double]))
ax_norm.annotate("", xy=(c_double, 1.05), xytext=(c_base, 1.05),
                 arrowprops=dict(arrowstyle="<->", color="#4a3aa7", lw=1.5))
ax_norm.text(0.5 * (c_base + c_double), 1.065,
             f"{c_double - c_base:.2f} fs later", ha="center", va="bottom",
             fontsize=9.5, color="#4a3aa7")

handles, labels = ax_abs.get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, 0.005), frameon=False)
fig.savefig(os.path.join(FIGS, "double_satellite_cascade_in_time.pdf"))
fig.savefig(os.path.join(FIGS, "double_satellite_cascade_in_time.png"))
plt.close(fig)

# =============================================================================
# 3. fluence scaling of every manifold
# =============================================================================
fig, ax = plt.subplots(figsize=(8.2, 5.2))

series = [("base $1s^{-1}$ (K)", [DATA[e]["rho_ee_t_last"].sum() * DT for e in E_SEED_VALUES],
           "#e34948", "-", "D", 2.6),
          ("base $2p_{3/2}^{-1}$ (L3)", [DATA[e]["rho_l3_t_last"].sum() * DT for e in E_SEED_VALUES],
           "#111111", "-", "o", 2.6),
          ("base $2p_{1/2}^{-1}$ (L2)", [DATA[e]["rho_l2_t_last"].sum() * DT for e in E_SEED_VALUES],
           "#111111", "--", "o", 1.8)]
for i, name in enumerate(NAMES):
    style = "-" if len(name) <= 3 else "--"
    series.append((f"${name}$",
                   [DATA[e]["rho_l3_t_last_sat"][i].sum() * DT for e in E_SEED_VALUES],
                   PALETTE[i % len(PALETTE)], style, "s" if len(name) <= 3 else "^", 1.4))

for label, values, colour, ls, marker, lw in series:
    slope = np.polyfit(np.log(E_SEED_VALUES), np.log(values), 1)[0]
    ax.plot(E_SEED_VALUES, values, color=colour, ls=ls, marker=marker, lw=lw, ms=5,
            label=f"{label}  ($p={slope:.2f}$)")

guide = np.array([2.0, 60.0])
ax.plot(guide, 2.2e-4 * (guide / 2.0) ** 1.0, color="0.55", lw=1.1, ls=":", zorder=0)
ax.text(26, 2.2e-4 * (26 / 2.0) ** 1.0 * 0.62, r"$\propto E$", color="0.4", fontsize=10)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Pulse energy ($\mu$J)")
ax.set_ylabel("Time-integrated dipole-weighted population (arb.)")
ax.set_title(r"Fluence scaling of each manifold, fitted as $\propto E^{\,p}$")
log_axis(ax, E_SEED_VALUES)
ax.legend(fontsize=8.6, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "double_satellite_fluence_scaling.pdf"))
fig.savefig(os.path.join(FIGS, "double_satellite_fluence_scaling.png"))
plt.close(fig)

# =============================================================================
# 4. temporal reshaping of the pulse
# =============================================================================
fig, (ax_shape, ax_ratio) = plt.subplots(1, 2, figsize=(12.4, 4.6), sharex=True)

i_in40, i_out40 = np.abs(DATA[40.0]["I_t_0"]), np.abs(DATA[40.0]["I_t_last"])
ratio40 = i_out40 / np.maximum(i_in40, i_in40.max() * 1e-3)
c_in40 = float((T_AXIS * i_in40).sum() / i_in40.sum())
c_out40 = float((T_AXIS * i_out40).sum() / i_out40.sum())
w_in = float(np.sqrt(((T_AXIS - c_in40) ** 2 * i_in40).sum() / i_in40.sum()))
w_out = float(np.sqrt(((T_AXIS - c_out40) ** 2 * i_out40).sum() / i_out40.sum()))

for e_seed in E_SEED_VALUES:
    i_in = np.abs(DATA[e_seed]["I_t_0"])
    i_out = np.abs(DATA[e_seed]["I_t_last"])
    ax_shape.plot(T_AXIS, i_out / i_out.max(), color=E_COLOURS[e_seed],
                  label=f"out, {e_seed:g} " + r"$\mu$J")
    ax_ratio.plot(T_AXIS, i_out / np.maximum(i_in, i_in.max() * 1e-3),
                  color=E_COLOURS[e_seed], label=f"{e_seed:g} " + r"$\mu$J")

i_in = np.abs(DATA[40.0]["I_t_0"])
ax_shape.plot(T_AXIS, i_in / i_in.max(), color="0.35", ls="--", lw=1.6, label="incident")
ax_shape.set_xlim(4, 17)
ax_shape.set_xlabel("Time (fs)")
ax_shape.set_ylabel("Power, normalised to own peak")
ax_shape.set_title("Pulse shape at entrance and exit")
ax_shape.legend(fontsize=9, loc="upper right")
ax_shape.text(0.035, 0.97, f"40 " + r"$\mu$J:" + "\n"
              + f"centroid {c_out40 - c_in40:+.3f} fs\n"
              + f"rms width {100 * (w_out / w_in - 1):+.1f} %",
              transform=ax_shape.transAxes, fontsize=9.5, color="0.3", va="top", ha="left")

window = (T_AXIS > 6) & (T_AXIS < 15)
t_min = float(T_AXIS[window][np.argmin(ratio40[window])])
ratio_min = float(ratio40[window].min())
ax_ratio.axvline(c_in40, color="0.5", lw=1.1, ls="--", zorder=0)
ax_ratio.text(c_in40 - 0.15, 0.437, f"pulse centroid, {c_in40:.1f} fs", rotation=90,
              ha="right", va="top", fontsize=9, color="0.4")
ax_ratio.plot([t_min], [ratio_min], marker="v", ms=8, color="#0f7d57", zorder=5)
ax_ratio.text(t_min + 0.25, ratio_min - 0.001,
              f"least transmitting, {t_min:.1f} fs", fontsize=9, color="#0f7d57", va="top")
ax_ratio.set_xlim(4, 17)
ax_ratio.set_ylim(0.325, 0.445)
ax_ratio.set_xlabel("Time (fs)")
ax_ratio.set_ylabel(r"Instantaneous transmission $I_{\rm out}/I_{\rm in}$")
ax_ratio.set_title("Transmission through the pulse")
ax_ratio.legend(fontsize=9, loc="upper left", ncol=2)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "double_satellite_pulse_reshaping.pdf"))
fig.savefig(os.path.join(FIGS, "double_satellite_pulse_reshaping.png"))
plt.close(fig)

# =============================================================================
# console summary
# =============================================================================
print(f"repetitions per pulse energy: "
      f"{ {e: DATA[e]['n_reps'] for e in E_SEED_VALUES} }")
print("\ndip depth below baseline")
for label, values in depths.items():
    print(f"  {label:14s} " + "  ".join(f"{e:g} uJ: {v:.4f}" for e, v in zip(E_SEED_VALUES, values)))

print("\nfluence-scaling exponent p  (population ~ E^p, fitted over 2-60 uJ)")
for label, values, *_ in series:
    print(f"  {label:26s} p = {np.polyfit(np.log(E_SEED_VALUES), np.log(values), 1)[0]:.2f}")

print("\ntemporal reshaping")
for e_seed in E_SEED_VALUES:
    i_in = np.abs(DATA[e_seed]["I_t_0"])
    i_out = np.abs(DATA[e_seed]["I_t_last"])
    c_in = (T_AXIS * i_in).sum() / i_in.sum()
    c_out = (T_AXIS * i_out).sum() / i_out.sum()
    w_in = np.sqrt(((T_AXIS - c_in) ** 2 * i_in).sum() / i_in.sum())
    w_out = np.sqrt(((T_AXIS - c_out) ** 2 * i_out).sum() / i_out.sum())
    print(f"  {e_seed:5g} uJ  centroid {c_in:.3f} -> {c_out:.3f} fs "
          f"({c_out - c_in:+.3f}),  rms width {w_in:.3f} -> {w_out:.3f} fs "
          f"({100 * (w_out / w_in - 1):+.1f} %),  energy transmission {i_out.sum() / i_in.sum():.4f}")

print(f"\nfigures written to {FIGS}")
