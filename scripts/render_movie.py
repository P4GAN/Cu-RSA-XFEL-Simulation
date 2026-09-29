#!/usr/bin/env python3
"""Render the hero animation (and stills) from a scripts/run_movie.py HDF5 file.

The simulation marches in retarded time t at each depth plane z; this script maps that back to the
lab frame, where at lab time T the plane at depth z shows the state at t = T - z/c. The pulse then
appears as a ~1.8 um sheet of light crossing the 20 um foil in ~67 fs, leaving ionised copper
behind it. Sampling along a fine z axis at fixed T walks along the t axis, so the SASE spike train
shows up as structure in depth even though there are only 61 stored planes.

Two layers are composited additively on black: the photon flux (warm) and the ionised fraction
1 - rho_ground (cool), both from the y = 0 slice. Lab times before the pulse arrives or after the
stored 25 fs window are held at the first/last stored step, so the late frames freeze the
long-lived populations rather than decaying them.

Examples:
    python scripts/render_movie.py --still 40                 # one PNG, for checking the look
    python scripts/render_movie.py --gif                      # figs/xlo_hero.mp4 + .gif
    python scripts/render_movie.py --frames 400 --fps 30 --out figs/xlo_hero_long.mp4
"""

import argparse
import os
import subprocess
import sys

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
C_NM_FS = 299.792458  # speed of light

BG = "#07080c"
FG = "#c9d2e0"
PULSE_CMAP = LinearSegmentedColormap.from_list("pulse", ["#2a0a3f", "#b02a5b", "#f97d2a", "#ffe7a8", "#ffffff"])
TRAIL_CMAP = LinearSegmentedColormap.from_list("trail", ["#050814", "#123a6b", "#1f8ab0", "#63e0d0"])


def load_slices(h5_path, refresh=False):
    """y = 0 slices of everything the animation needs, cached next to the HDF5 file.

    Reading them out of the 6 GB file means decompressing whole (t, x, y) chunks, so it is worth
    doing once (~1 min) rather than on every re-render.
    """
    cache_path = os.path.splitext(h5_path)[0] + "_slices.npz"
    if os.path.exists(cache_path) and not refresh:
        return dict(np.load(cache_path))

    print(f"extracting slices from {h5_path} (once; cached to {os.path.basename(cache_path)})", flush=True)
    with h5py.File(h5_path, "r") as f:
        cy = int(f.attrs["center_pixel"][1])
        flux_factor = float(f.attrs["flux_factor"])
        out = {
            "t": f["t"][:], "x": f["x"][:], "z": f["z"][:],
            "flux": f["flux"][:, :, :, cy],                      # (z, t, x) photons nm^-2 fs^-1
            "ground": f["pop/ground"][:, :, :, cy],              # (z, t, x)
            "khole": f["pop/base"][:, 4:6, :, :, cy].sum(axis=1),  # 1s holes
            "electrons": f["electrons"][:, :, :, :, cy].sum(axis=1),
            "E_seed_uJ": np.asarray(f.attrs["E_seed_uJ"]),
            "energy_transmission": np.asarray(f.attrs["energy_transmission"]),
        }
        fe = f["field_exit"][:, :, :, :, cy]
        out["flux_exit"] = np.real(fe[0, 0] * fe[1, 0] + fe[0, 1] * fe[1, 1]) / flux_factor  # (t, x)
    np.savez_compressed(cache_path, **out)
    return out


class LabFrame:
    """Resamples the stored (z, t, x) arrays onto a fine depth axis at a given lab time."""

    def __init__(self, data, n_zfine=800, skip_first_plane=True):
        # The stored z = 0 plane is never photoionised (the flux arrays that drive photoionisation
        # start at zero, so plane 0 sees no flux), which would show up as a hard edge at the front
        # face. Start the render one plane in; the 0.33 um shift is 1.7% of the foil.
        self.i0 = 1 if skip_first_plane else 0
        self.d = {k: (v[self.i0:] if getattr(v, "ndim", 0) == 3 else v) for k, v in data.items()}
        self.dt = data["t"][1] - data["t"][0]
        self.dz = data["z"][1] - data["z"][0]
        self.z0 = data["z"][self.i0]
        self.z_fine = np.linspace(self.z0, data["z"][-1], n_zfine)
        self.nx = data["x"].size
        zi = (self.z_fine - self.z0) / self.dz
        xi = np.arange(self.nx)
        self.z_c = np.repeat(zi[:, None], self.nx, axis=1)
        self.x_c = np.repeat(xi[None, :], n_zfine, axis=0)
        self.delay = self.z_fine / C_NM_FS  # retardation of each depth

    def t_index(self, T):
        # Held at the ends: before the pulse arrives the sample is unperturbed, and after the
        # stored window the long-lived populations simply stay put.
        return np.clip((T - self.delay) / self.dt, 0, self.d["t"].size - 1)

    def slice_at(self, key, T):
        ti = np.repeat(self.t_index(T)[:, None], self.nx, axis=1)
        return map_coordinates(self.d[key], [self.z_c, ti, self.x_c], order=1, mode="nearest")


def composite(flux_img, ion_img, flux_max, ion_max, gamma=0.5):
    """Additive blend of the two layers, as an (nx, nz, 3) RGB image ready for imshow."""
    p = np.clip(flux_img / flux_max, 0, 1) ** gamma
    q = np.clip(ion_img / ion_max, 0, 1) ** 0.8
    rgb = (TRAIL_CMAP(q)[..., :3] * (0.20 + 0.60 * q[..., None])
           + PULSE_CMAP(p)[..., :3] * np.clip(1.7 * p, 0, 1)[..., None])
    return np.clip(rgb, 0, 1).transpose(1, 0, 2)


def build_figure(data, lab, args):
    z_um = lab.z_fine / 1000.0
    x_um = data["x"] / 1000.0
    flux_max = float(lab.d["flux"].max())
    ion_max = float((1.0 - lab.d["ground"]).max())

    fig = plt.figure(figsize=(12.0, 6.4), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.28,
                          left=0.07, right=0.98, top=0.86, bottom=0.11)
    ax = fig.add_subplot(gs[0], facecolor="#04050a")
    axl = fig.add_subplot(gs[1], facecolor="#04050a", sharex=ax)

    extent = [z_um[0], z_um[-1], x_um[0], x_um[-1]]
    blank = np.zeros((x_um.size, lab.z_fine.size, 3))
    im = ax.imshow(blank, origin="lower", extent=extent, aspect="auto", interpolation="bicubic",
                   zorder=1)
    ax.set_ylabel("x  (µm)", color=FG)
    ax.tick_params(colors=FG, labelbottom=False)
    for spine in list(ax.spines.values()) + list(axl.spines.values()):
        spine.set_color("#2a3142")

    time_text = ax.text(0.012, 0.93, "", transform=ax.transAxes, color="#ffffff", fontsize=15,
                        family="monospace", zorder=5)
    ax.text(0.012, 0.055,
            f"Cu foil, 20 µm   ·   {float(data['E_seed_uJ']):.0f} µJ SASE pulse, 8.05 keV, 6 fs FWHM"
            f"   ·   transmission {float(data['energy_transmission']):.2f}",
            transform=ax.transAxes, color="#8fa0bb", fontsize=10.5, zorder=5)
    ax.text(0.988, 0.055, "transverse scale stretched ×40", transform=ax.transAxes,
            color="#5d6a78", fontsize=9, ha="right", zorder=5)

    (line_flux,) = axl.plot([], [], color="#ffb457", lw=1.6, label="pulse intensity (on axis)")
    (line_ion,) = axl.plot([], [], color="#63e0d0", lw=1.6, label="atoms ionised (on axis)")
    axl.set_xlim(z_um[0], z_um[-1])
    axl.set_ylim(0, 1.05)
    axl.set_xlabel("depth into the foil,  z  (µm)", color=FG)
    axl.set_ylabel("normalised", color=FG, fontsize=10)
    axl.tick_params(colors=FG)
    axl.grid(alpha=0.12, color=FG)
    leg = axl.legend(loc="upper right", frameon=False, fontsize=9.5, ncol=2)
    for text in leg.get_texts():
        text.set_color(FG)

    fig.suptitle("An X-ray laser pulse burning through copper", color="#ffffff", fontsize=17, y=0.955)
    fig.text(0.98, 0.955, "Maxwell–Bloch + Fresnel simulation", color="#7b8ba6", fontsize=10,
             ha="right", va="center")

    cx = data["x"].size // 2
    ion_axis_max = float((1.0 - lab.d["ground"][:, :, cx]).max())

    def draw(T):
        flux_img = lab.slice_at("flux", T)
        ion_img = 1.0 - lab.slice_at("ground", T)
        im.set_data(composite(flux_img, ion_img, flux_max, ion_max, gamma=args.gamma))
        time_text.set_text(f"t = {T:5.1f} fs")
        line_flux.set_data(lab.z_fine / 1000.0, flux_img[:, cx] / flux_max)
        line_ion.set_data(lab.z_fine / 1000.0, ion_img[:, cx] / ion_axis_max)
        return im, time_text, line_flux, line_ion

    return fig, draw


def to_gif(mp4_path, gif_path, fps, width):
    """Two-pass palette GIF (ffmpeg's default 256-colour quantisation looks bad on these gradients)."""
    palette = os.path.join(os.path.dirname(gif_path) or ".", ".palette.png")
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4_path, "-vf", f"{vf},palettegen=stats_mode=diff",
                    palette], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4_path, "-i", palette, "-lavfi",
                    f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3", gif_path], check=True)
    os.remove(palette)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", default=os.path.join(REPO_ROOT, "data/movie_24699095/movie.h5"))
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "figs/xlo_hero.mp4"))
    p.add_argument("--frames", type=int, default=260)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--t-start", type=float, default=2.0, help="first lab time (fs)")
    p.add_argument("--t-end", type=float, default=88.0, help="last lab time (fs)")
    p.add_argument("--hold", type=float, default=1.2, help="seconds to hold the last frame (pause before a GIF loops)")
    p.add_argument("--gamma", type=float, default=0.5, help="brightness curve of the pulse layer")
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--still", type=float, default=None, help="render one PNG at this lab time (fs) instead")
    p.add_argument("--gif", action="store_true", help="also write a GIF next to the MP4")
    p.add_argument("--gif-width", type=int, default=900)
    p.add_argument("--gif-fps", type=int, default=20)
    p.add_argument("--refresh-cache", action="store_true")
    args = p.parse_args()

    data = load_slices(args.h5, refresh=args.refresh_cache)
    lab = LabFrame(data)
    fig, draw = build_figure(data, lab, args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.still is not None:
        draw(args.still)
        png = os.path.splitext(args.out)[0] + f"_t{args.still:g}fs.png"
        fig.savefig(png, dpi=args.dpi, facecolor=BG)
        print(f"Saved {png}", flush=True)
        return

    times = np.linspace(args.t_start, args.t_end, args.frames)
    times = np.concatenate([times, np.full(int(round(args.hold * args.fps)), args.t_end)])
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=6000, codec="libx264",
                                    extra_args=["-pix_fmt", "yuv420p", "-preset", "slow"])
    with writer.saving(fig, args.out, dpi=args.dpi):
        for i, T in enumerate(times):
            draw(T)
            writer.grab_frame(facecolor=BG)
            if (i + 1) % 25 == 0 or i == len(times) - 1:
                print(f"  frame {i + 1}/{len(times)}", flush=True)
    print(f"Saved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)", flush=True)

    if args.gif:
        gif = os.path.splitext(args.out)[0] + ".gif"
        to_gif(args.out, gif, args.gif_fps, args.gif_width)
        print(f"Saved {gif} ({os.path.getsize(gif) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
