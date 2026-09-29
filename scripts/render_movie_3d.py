#!/usr/bin/env python3
"""3D version of the hero animation: the pulse crossing the foil, seen by an orbiting camera.

Same data and lab-frame mapping as scripts/render_movie.py (plane z at lab time T shows retarded
time T - z/c), but over the whole (x, y, z) volume instead of the y = 0 slice. Rendered with a small
emission-only volume renderer: every voxel of a fine lab-frame grid is projected through a
perspective camera and splatted (bilinearly) into the image, so brightness adds up along each line
of sight like a glowing gas. No occlusion, which suits light and a translucent ionised trail. Then a
tone curve and a bloom pass.

Transverse axes are stretched (--stretch, default x10) or the 0.5 um wide beam would be a line in a
20 um foil. The wireframe box is the simulated volume.

    python scripts/render_movie_3d.py --still 40      # one PNG
    python scripts/render_movie_3d.py --gif           # figs/xlo_hero_3d.mp4 + .gif (~15 min)
"""

import argparse
import os
import sys
import time

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from scipy.ndimage import gaussian_filter, map_coordinates  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from render_movie import BG, C_NM_FS, PULSE_CMAP, REPO_ROOT, TRAIL_CMAP, to_gif  # noqa: E402


def load_volume(h5_path):
    """flux and ionised fraction over (z, t, x, y), skipping the never-ionised z = 0 plane."""
    t0 = time.perf_counter()
    with h5py.File(h5_path, "r") as f:
        vol = {
            "t": f["t"][:], "x": f["x"][:], "y": f["y"][:], "z": f["z"][1:],
            "flux": f["flux"][1:],
            "ion": 1.0 - f["pop/ground"][1:],
            "E_seed_uJ": float(f.attrs["E_seed_uJ"]),
        }
    print(f"loaded volume in {time.perf_counter() - t0:.0f} s", flush=True)
    return vol


class VolumeRenderer:
    def __init__(self, vol, args):
        self.v = vol
        self.args = args
        self.W, self.H = args.width, args.height
        t, x, y, z = vol["t"], vol["x"], vol["y"], vol["z"]
        self.dt, self.dx, self.dz = t[1] - t[0], x[1] - x[0], z[1] - z[0]
        self.nt = t.size
        self.flux_max = float(vol["flux"].max())
        self.ion_max = float(vol["ion"].max())

        # fine lab-frame sample grid (nm)
        zf = np.linspace(z[0], z[-1], args.nz)
        xf = np.linspace(-args.half_width, args.half_width, args.nxy)
        yf = np.linspace(-args.half_width, args.half_width, args.nxy)
        Z, Xg, Yg = np.meshgrid(zf, xf, yf, indexing="ij")
        # Sub-voxel jitter (fixed seed, so it doesn't shimmer between frames): a regular lattice
        # projected at an angle beats against the pixel grid into moire.
        rng = np.random.default_rng(0)
        Z = Z + rng.uniform(-0.5, 0.5, Z.shape) * (zf[1] - zf[0])
        Xg = Xg + rng.uniform(-0.5, 0.5, Xg.shape) * (xf[1] - xf[0])
        Yg = Yg + rng.uniform(-0.5, 0.5, Yg.shape) * (yf[1] - yf[0])
        self.shape = Z.shape
        self.delay = (Z / C_NM_FS).ravel()
        self.zi = ((Z - z[0]) / self.dz).ravel()
        self.xi = ((Xg - x[0]) / self.dx).ravel()
        self.yi = ((Yg - y[0]) / self.dx).ravel()
        # world coordinates for the camera: um, foil centred on the origin, transverse stretched
        self.world = np.stack([Xg.ravel() / 1000 * args.stretch, Yg.ravel() / 1000 * args.stretch,
                               Z.ravel() / 1000 - 10.0])
        self.z_um = (z[0] / 1000, z[-1] / 1000)
        self.bg = np.array(matplotlib.colors.to_rgb(BG))

    def camera(self, azimuth_deg, elevation_deg):
        a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
        Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
        return Rx @ Ry

    def project(self, pts, R):
        """world (3, N) -> screen pixel coordinates (px, py)."""
        c = R @ pts
        depth = self.args.distance - c[2]
        scale = self.args.focal * self.H / depth
        return self.W / 2 + c[0] * scale, self.H / 2 - c[1] * scale + self.args.shift_y * self.H

    def sample(self, key, T):
        ti = np.clip((T - self.delay) / self.dt, 0, self.nt - 1)
        return map_coordinates(self.v[key], [self.zi, ti, self.xi, self.yi], order=1, mode="nearest")

    def render(self, T, azimuth, elevation):
        a = self.args
        p = np.clip(self.sample("flux", T) / self.flux_max, 0, 1)
        q = np.clip(self.sample("ion", T) / self.ion_max, 0, 1)
        emit = np.zeros((p.size, 3), dtype=np.float32)
        on_p = p > 1e-3
        on_q = q > 1e-3
        pg = p[on_p] ** a.gamma
        emit[on_p] += PULSE_CMAP(0.3 + 0.7 * pg)[:, :3] * (pg * a.pulse_gain)[:, None]
        emit[on_q] += TRAIL_CMAP(q[on_q] ** 0.8)[:, :3] * (q[on_q] * a.trail_gain)[:, None]
        keep = on_p | on_q
        px, py = self.project(self.world[:, keep], self.camera(azimuth, elevation))
        img = self.splat(px, py, emit[keep])
        # tone curve + bloom
        img = 1.0 - np.exp(-img)
        bloom = gaussian_filter(img, sigma=(a.bloom, a.bloom, 0))
        img = np.clip(gaussian_filter(img, sigma=(a.blur, a.blur, 0)) + a.bloom_weight * bloom, 0, 1)
        return self.bg + img * (1.0 - self.bg)

    def splat(self, px, py, values):
        W, H = self.W, self.H
        ix, iy = np.floor(px).astype(int), np.floor(py).astype(int)
        fx, fy = px - ix, py - iy
        img = np.zeros((H * W, 3))
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            jx, jy = ix + dx, iy + dy
            ok = (jx >= 0) & (jx < W) & (jy >= 0) & (jy < H)
            idx = jy[ok] * W + jx[ok]
            for c in range(3):
                img[:, c] += np.bincount(idx, weights=values[ok, c] * w[ok], minlength=H * W)
        return img.reshape(H, W, 3)

    def box_edges(self, R):
        """Projected wireframe of the simulated volume, as a list of (xs, ys) polylines."""
        h = self.args.half_width / 1000 * self.args.stretch
        z0, z1 = self.z_um[0] - 10.0, self.z_um[1] - 10.0
        corners = [(sx * h, sy * h) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))]
        lines = []
        for zz in (z0, z1):
            pts = np.array([[cx, cy, zz] for cx, cy in corners]).T
            lines.append(self.project(pts, R))
        for cx, cy in corners[:4]:
            pts = np.array([[cx, cy, z0], [cx, cy, z1]]).T
            lines.append(self.project(pts, R))
        return lines


def build_figure(renderer, args):
    fig = plt.figure(figsize=(args.width / 100, args.height / 100), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, args.width)
    ax.set_ylim(args.height, 0)
    ax.axis("off")
    im = ax.imshow(np.zeros((args.height, args.width, 3)), extent=(0, args.width, args.height, 0), zorder=1)
    edges = [ax.plot([], [], color="#3a4560", lw=0.9, alpha=0.7, zorder=2)[0] for _ in range(6)]
    labels = [ax.text(0, 0, s, color="#6b7890", fontsize=10, ha="center", zorder=3) for s in ("front", "back")]
    ax.text(0.03 * args.width, 0.075 * args.height, "An X-ray laser pulse burning through copper",
            color="#ffffff", fontsize=18, zorder=4)
    ax.text(0.03 * args.width, 0.115 * args.height,
            f"{renderer.v['E_seed_uJ']:.0f} µJ SASE pulse, 8.05 keV  ·  20 µm Cu foil  ·  "
            f"transverse scale stretched ×{args.stretch:g}",
            color="#8fa0bb", fontsize=10.5, zorder=4)
    clock = ax.text(0.97 * args.width, 0.075 * args.height, "", color="#ffffff", fontsize=16,
                    family="monospace", ha="right", zorder=4)
    ax.text(0.97 * args.width, 0.955 * args.height, "Maxwell–Bloch + Fresnel simulation",
            color="#5d6a78", fontsize=9.5, ha="right", zorder=4)

    def draw(T, azimuth, elevation):
        im.set_data(renderer.render(T, azimuth, elevation))
        R = renderer.camera(azimuth, elevation)
        for line, (xs, ys) in zip(edges, renderer.box_edges(R)):
            line.set_data(xs, ys)
        h = args.half_width / 1000 * args.stretch
        for text, zz in zip(labels, (renderer.z_um[0] - 10.0, renderer.z_um[1] - 10.0)):
            xs, ys = renderer.project(np.array([[0.0], [-h * 1.35], [zz]]), R)
            text.set_position((xs[0], ys[0] + 14))
        clock.set_text(f"t = {T:5.1f} fs")

    return fig, draw


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", default=os.path.join(REPO_ROOT, "data/movie_24699095/movie.h5"))
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "figs/xlo_hero_3d.mp4"))
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--nz", type=int, default=720, help="lab-frame depth samples")
    p.add_argument("--nxy", type=int, default=48, help="transverse samples per axis")
    p.add_argument("--half-width", type=float, default=220.0, help="transverse half-width rendered (nm)")
    p.add_argument("--stretch", type=float, default=10.0)
    p.add_argument("--distance", type=float, default=30.0)
    p.add_argument("--focal", type=float, default=1.75)
    p.add_argument("--shift-y", type=float, default=0.03)
    p.add_argument("--gamma", type=float, default=0.75)
    p.add_argument("--pulse-gain", type=float, default=1.4)
    p.add_argument("--trail-gain", type=float, default=0.22)
    p.add_argument("--bloom", type=float, default=6.0)
    p.add_argument("--bloom-weight", type=float, default=0.35)
    p.add_argument("--blur", type=float, default=1.3, help="splat smoothing (px)")
    p.add_argument("--azimuth", type=float, nargs=2, default=(52.0, 70.0), help="camera orbit start/end (deg)")
    p.add_argument("--elevation", type=float, default=22.0)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--t-start", type=float, default=2.0)
    p.add_argument("--t-end", type=float, default=88.0)
    p.add_argument("--hold", type=float, default=1.5, help="seconds of camera-only orbit at the end")
    p.add_argument("--still", type=float, default=None)
    p.add_argument("--gif", action="store_true")
    p.add_argument("--gif-width", type=int, default=900)
    p.add_argument("--gif-fps", type=int, default=20)
    args = p.parse_args()

    renderer = VolumeRenderer(load_volume(args.h5), args)
    fig, draw = build_figure(renderer, args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.still is not None:
        t0 = time.perf_counter()
        draw(args.still, np.mean(args.azimuth), args.elevation)
        png = os.path.splitext(args.out)[0] + f"_t{args.still:g}fs.png"
        fig.savefig(png, dpi=100, facecolor=BG)
        print(f"Saved {png} ({time.perf_counter() - t0:.1f} s)")
        return

    n_hold = int(round(args.hold * args.fps))
    times = np.concatenate([np.linspace(args.t_start, args.t_end, args.frames), np.full(n_hold, args.t_end)])
    azimuths = np.linspace(args.azimuth[0], args.azimuth[1] + (args.azimuth[1] - args.azimuth[0]) * n_hold / args.frames,
                           times.size)
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=8000, codec="libx264",
                                    extra_args=["-pix_fmt", "yuv420p", "-preset", "slow"])
    t0 = time.perf_counter()
    with writer.saving(fig, args.out, dpi=100):
        for i, (T, az) in enumerate(zip(times, azimuths)):
            draw(T, az, args.elevation)
            writer.grab_frame(facecolor=BG)
            if (i + 1) % 30 == 0 or i == times.size - 1:
                print(f"  frame {i + 1}/{times.size} ({time.perf_counter() - t0:.0f} s)", flush=True)
    print(f"Saved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)", flush=True)
    if args.gif:
        gif = os.path.splitext(args.out)[0] + ".gif"
        to_gif(args.out, gif, args.gif_fps, args.gif_width)
        print(f"Saved {gif} ({os.path.getsize(gif) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
