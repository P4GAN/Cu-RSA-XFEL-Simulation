#!/usr/bin/env python3
"""Animated level diagram: what the pulse does to the copper atoms at one point in the foil.

Companion to scripts/render_movie.py, from the same scripts/run_movie.py HDF5 file. Left: a
schematic (not to scale) level scheme whose bars glow with each population on a log scale,
photoionisation arrows that glow with the instantaneous flux, and the Kalpha resonances that glow
with the base-block 1s-2p coherence |rho_eg|. Right: the pulse and the populations vs time, revealed
up to the current frame. All at the beam centre of one depth plane (default plane 1, 0.33 um in;
plane 0 is never photoionised, see render_movie.py).

Satellite blocks are summed into single-spectator (3d+, 3d-, 3p+, 3p-) and double-spectator
(3dx3dx) groups, each split into its 1s-hole and 2p-hole (L3 + L2) manifolds.

    python scripts/render_level_diagram.py --still 9      # one PNG
    python scripts/render_level_diagram.py --gif          # figs/xlo_levels.mp4 + .gif
"""

import argparse
import os
import sys

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from render_movie import BG, FG, REPO_ROOT, to_gif  # noqa: E402

PANEL_BG = "#04050a"
MUTED = "#8fa0bb"
PULSE = "#ffd27a"
COLOURS = {
    "ground": "#9fb3c8", "L3": "#63e0d0", "L2": "#3fb8c9", "L1": "#7aa2ff", "K": "#ff5fa2",
    "satL1": "#b48cff", "satL2": "#8f6bff", "satK1": "#e07bff", "satK2": "#c45cff",
    "mid": "#ff8a5b", "electrons": "#8be38b",
}
LOG_FLOOR = 1e-4


def load_pixel(h5_path, plane, refresh=False):
    """Every time trace the animation needs at the beam centre of `plane`, cached as .npz."""
    cache = os.path.splitext(h5_path)[0] + f"_pixel_z{plane}.npz"
    if os.path.exists(cache) and not refresh:
        return dict(np.load(cache))
    with h5py.File(h5_path, "r") as f:
        cx, cy = (int(v) for v in f.attrs["center_pixel"])
        names = [c.decode() if isinstance(c, bytes) else str(c) for c in f["pop/satellite"].attrs["channels"]]
        base = f["pop/base"][plane, :, :, cx, cy]
        sat = f["pop/satellite"][plane, :, :, :, cx, cy]  # (channel, manifold L3/K/L2, t)
        single = np.array([n.count("3") == 1 for n in names])
        rho = f["center/rho_base"][plane]
        pairs = f["coherence/base"].attrs["pairs_upper_lower"]
        ka1 = sum(np.abs(rho[i, j]) for i, j in pairs if j < 4)
        ka2 = sum(np.abs(rho[i, j]) for i, j in pairs if j >= 6)
        el = f["electrons"][plane, :, :, cx, cy]
        out = {
            "t": f["t"][:], "z_nm": np.asarray(f["z"][plane]),
            "flux": f["flux"][plane, :, cx, cy],
            "ground": f["pop/ground"][plane, :, cx, cy],
            "L1": f["pop/2s"][plane, :, cx, cy],
            "mid": f["pop/middlemen"][plane, :, cx, cy],
            "L3": base[0:4].sum(0), "K": base[4:6].sum(0), "L2": base[6:8].sum(0),
            "satL1": (sat[single, 0] + sat[single, 2]).sum(0), "satK1": sat[single, 1].sum(0),
            "satL2": (sat[~single, 0] + sat[~single, 2]).sum(0), "satK2": sat[~single, 1].sum(0),
            "coh_ka1": ka1, "coh_ka2": ka2,
            "electrons": el, "E_edges": f["electrons"].attrs["E_edges_eV"],
        }
    np.savez_compressed(cache, **out)
    return out


def log_level(p):
    return np.clip(np.log10(np.maximum(p, 1e-30) / LOG_FLOOR) / -np.log10(LOG_FLOOR), 0, 1)


def fmt_pop(p):
    if p >= 0.01:
        return f"{100 * p:.1f} %"
    if p >= 1e-4:
        return f"{100 * p:.2f} %"
    return "<0.01 %"


class Glow:
    """A line drawn as a few stacked strokes (wide + faint under narrow + bright), dimmable per frame."""

    def __init__(self, ax, xs, ys, colour, widths=(16, 9, 3.2), alphas=(0.07, 0.16, 1.0), zorder=3, **kw):
        ax.plot(xs, ys, color=colour, lw=widths[-1], alpha=0.18, zorder=zorder - 1, solid_capstyle="round", **kw)
        self.lines = [ax.plot(xs, ys, color=colour, lw=w, alpha=0, zorder=zorder, solid_capstyle="round", **kw)[0]
                      for w in widths]
        self.alphas = alphas

    def set(self, v):
        for line, a in zip(self.lines, self.alphas):
            line.set_alpha(float(np.clip(a * v, 0, 1)))


def build(data, args):
    t = data["t"]
    fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
    ax = fig.add_axes([0.03, 0.07, 0.57, 0.80], facecolor=BG)
    ax.set_xlim(-0.6, 10.4)
    ax.set_ylim(-1.2, 10.2)
    ax.axis("off")

    # (key, label, x0, x1, y). Schematic energies: ground 0, 2p ~0.93 keV, 2s ~1.1 keV, 1s ~9 keV.
    levels = [
        ("ground", "Cu atom", 0.3, 2.0, 0.8),
        ("L3", "2p₃/₂ hole", 2.8, 4.6, 3.0),
        ("L2", "2p₁/₂ hole", 2.8, 4.2, 4.0),
        ("L1", "2s hole", 2.8, 3.6, 5.1),
        ("K", "1s hole", 2.8, 4.6, 8.6),
        ("satL1", "2p hole + 3d/3p hole", 5.3, 7.5, 3.3),
        ("satL2", "2p hole + two 3d holes", 5.3, 7.5, 4.3),
        ("satK1", "1s hole + 3d/3p hole", 5.3, 7.5, 8.2),
        ("satK2", "1s hole + two 3d holes", 5.3, 7.5, 9.2),
        ("mid", "3d-ionised Cu", 8.2, 9.9, 2.4),
    ]
    glows, values = {}, {}
    for key, label, x0, x1, y in levels:
        c = COLOURS[key]
        glows[key] = Glow(ax, [x0, x1], [y, y], c)
        ax.text(x0, y + 0.18, label, color=c, fontsize=9, va="bottom", alpha=0.9)
        values[key] = ax.text(x0, y - 0.2, "", color="#ffffff", fontsize=9, va="top", family="monospace")

    # photoionisation fan from the ground state (glows with the flux)
    pi_arrows = []
    for key in ("K", "L1", "L2", "L3"):
        y = dict((k, yy) for k, _, _, _, yy in levels)[key]
        pi_arrows.append(Glow(ax, [2.0, 2.75], [0.8, y], PULSE, widths=(8, 3.5, 1.2), alphas=(0.12, 0.3, 1.0)))
    ax.text(1.15, 5.9, "X-ray\nphotoionisation", color=PULSE, fontsize=9.5, ha="center", alpha=0.9)

    # Kalpha resonances (glow with the 1s-2p coherence)
    ka1 = Glow(ax, [4.45, 4.45], [3.15, 8.45], "#fff1c9", widths=(12, 5, 1.6), alphas=(0.12, 0.3, 1.0))
    ka2 = Glow(ax, [3.95, 3.95], [4.15, 8.45], "#fff1c9", widths=(12, 5, 1.6), alphas=(0.12, 0.3, 1.0))
    ax.text(4.58, 6.9, "Kα₁", color="#fff1c9", fontsize=11)
    ax.text(3.82, 6.9, "Kα₂", color="#fff1c9", fontsize=11, ha="right")

    # decay routes (static, dim)
    for (x0, y0), (x1, y1), (lx, ly), label in (((3.65, 5.1), (5.25, 3.45), (5.15, 5.25), "Coster–Kronig"),
                                                ((4.65, 3.0), (8.15, 2.45), (7.75, 2.72), "Auger")):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#56627a", lw=1.0, ls=(0, (3, 3))))
        ax.text(lx, ly, label, color="#6b7890", fontsize=8.5, ha="center")

    # energy axis + scale break
    ax.annotate("", xy=(-0.35, 9.9), xytext=(-0.35, 0.2), arrowprops=dict(arrowstyle="-|>", color="#56627a", lw=1))
    ax.text(-0.5, 3.0, "energy (not to scale)", color="#56627a", fontsize=9, rotation=90, ha="center", va="center")
    for yy in (6.6, 6.85):
        ax.plot([-0.5, -0.2], [yy - 0.1, yy + 0.1], color=BG, lw=5, zorder=5)
        ax.plot([-0.5, -0.2], [yy - 0.1, yy + 0.1], color="#56627a", lw=1, zorder=6)
    for x, text in ((1.15, "neutral"), (3.7, "one core hole"), (6.4, "+ spectator holes"), (9.05, "after Auger decay")):
        ax.text(x, -0.95, text, color=MUTED, fontsize=10, ha="center")

    # free electrons, by energy group (last = thermalised)
    axe = fig.add_axes([0.331, 0.165, 0.124, 0.115], facecolor=PANEL_BG)  # under the satellite column
    E = data["E_edges"]
    names = [f"{E[g]/1000:.0f}k" if E[g] >= 1000 else f"{E[g]:.0f}" for g in range(len(E) - 1)] + ["cold"]
    bars = axe.bar(range(len(names)), np.full(len(names), 1e-6), color=COLOURS["electrons"], width=0.7)
    axe.set_yscale("log")
    axe.set_ylim(1e-3, 2.0)
    axe.set_xticks(range(len(names)), names, fontsize=6.5, color=MUTED)
    axe.tick_params(axis="y", labelsize=7, colors=MUTED)
    axe.set_title("free electrons per atom (eV)", color=COLOURS["electrons"], fontsize=9)
    for s in axe.spines.values():
        s.set_color("#2a3142")

    # right: pulse and populations vs time
    axp = fig.add_axes([0.67, 0.62, 0.31, 0.24], facecolor=PANEL_BG)
    axn = fig.add_axes([0.67, 0.10, 0.31, 0.43], facecolor=PANEL_BG, sharex=axp)
    flux = data["flux"] / data["flux"].max()
    axp.plot(t, flux, color=PULSE, lw=0.8, alpha=0.18)
    (pulse_line,) = axp.plot([], [], color=PULSE, lw=1.0)
    axp.set_ylim(0, 1.05)
    axp.set_ylabel("pulse intensity", color=FG, fontsize=9.5)
    axp.tick_params(colors=FG, labelsize=8.5, labelbottom=False)
    traces = [("1 - ground", 1 - data["ground"], COLOURS["ground"]), ("3d-ionised", data["mid"], COLOURS["mid"]),
              ("2p holes", data["L3"] + data["L2"], COLOURS["L3"]), ("1s holes", data["K"], COLOURS["K"]),
              ("with spectators", data["satL1"] + data["satL2"] + data["satK1"] + data["satK2"], COLOURS["satL1"]),
              ("2s holes", data["L1"], COLOURS["L1"])]
    trace_lines = []
    for label, y, c in traces:
        axn.plot(t, np.maximum(y, 1e-12), color=c, lw=0.8, alpha=0.15)
        trace_lines.append((axn.plot([], [], color=c, lw=1.5, label=label)[0], y))
    axn.set_yscale("log")
    axn.set_ylim(LOG_FLOOR, 1.2)
    axn.set_xlim(t[0], t[-1])
    axn.set_xlabel("time (fs)", color=FG, fontsize=10)
    axn.set_ylabel("fraction of atoms", color=FG, fontsize=9.5)
    axn.tick_params(colors=FG, labelsize=8.5)
    leg = axn.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=8, frameon=False, ncol=3)
    for text in leg.get_texts():
        text.set_color(FG)
    cursors = [a.axvline(t[0], color="#ffffff", lw=0.8, alpha=0.5) for a in (axp, axn)]
    for a in (axp, axn):
        a.grid(alpha=0.1, color=FG)
        for s in a.spines.values():
            s.set_color("#2a3142")

    fig.text(0.03, 0.945, "What the X-ray pulse does to the copper atoms", color="#ffffff", fontsize=17)
    fig.text(0.03, 0.905, f"beam centre, {float(data['z_nm']) / 1000:.2f} µm into the foil  ·  glow ∝ log population",
             color=MUTED, fontsize=10)
    time_text = fig.text(0.98, 0.935, "", color="#ffffff", fontsize=15, family="monospace", ha="right")

    coh_max = max(data["coh_ka1"].max(), data["coh_ka2"].max())
    el = data["electrons"]

    def draw(T):
        i = int(np.clip(np.searchsorted(t, T), 0, t.size - 1))
        for key, *_ in levels:
            p = float(data[key][i])
            glows[key].set(log_level(p))
            values[key].set_text(fmt_pop(p))
        for g in pi_arrows:
            g.set(flux[i] ** 0.6)
        ka1.set(data["coh_ka1"][i] / coh_max)
        ka2.set(data["coh_ka2"][i] / coh_max)
        for bar, h in zip(bars, el[:, i]):
            bar.set_height(max(float(h), 1e-6))
        pulse_line.set_data(t[: i + 1], flux[: i + 1])
        for line, y in trace_lines:
            line.set_data(t[: i + 1], np.maximum(y[: i + 1], 1e-12))
        for c in cursors:
            c.set_xdata([t[i], t[i]])
        time_text.set_text(f"t = {t[i]:5.1f} fs")

    return fig, draw


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", default=os.path.join(REPO_ROOT, "data/movie_24699095/movie.h5"))
    p.add_argument("--plane", type=int, default=1)
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "figs/xlo_levels.mp4"))
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--t-start", type=float, default=2.0)
    p.add_argument("--t-end", type=float, default=25.0)
    p.add_argument("--hold", type=float, default=1.2)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--still", type=float, default=None)
    p.add_argument("--gif", action="store_true")
    p.add_argument("--gif-width", type=int, default=960)
    p.add_argument("--gif-fps", type=int, default=20)
    p.add_argument("--refresh-cache", action="store_true")
    args = p.parse_args()

    data = load_pixel(args.h5, args.plane, refresh=args.refresh_cache)
    fig, draw = build(data, args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.still is not None:
        draw(args.still)
        png = os.path.splitext(args.out)[0] + f"_t{args.still:g}fs.png"
        fig.savefig(png, dpi=args.dpi, facecolor=BG)
        print(f"Saved {png}")
        return

    times = np.linspace(args.t_start, args.t_end, args.frames)
    times = np.concatenate([times, np.full(int(round(args.hold * args.fps)), args.t_end)])
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=5000, codec="libx264",
                                    extra_args=["-pix_fmt", "yuv420p", "-preset", "slow"])
    with writer.saving(fig, args.out, dpi=args.dpi):
        for T in times:
            draw(T)
            writer.grab_frame(facecolor=BG)
    print(f"Saved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    if args.gif:
        gif = os.path.splitext(args.out)[0] + ".gif"
        to_gif(args.out, gif, args.gif_fps, args.gif_width)
        print(f"Saved {gif} ({os.path.getsize(gif) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
