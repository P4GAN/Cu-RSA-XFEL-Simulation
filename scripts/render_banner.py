#!/usr/bin/env python3
"""Text-free wide stills of the 3D volume render, sized for a LinkedIn banner (1584x396, 4:1).

Reuses VolumeRenderer from render_movie_3d.py but skips every text/wireframe overlay -- just the
glowing pulse and its ionised trail on a plain background, cropped to banner proportions.

    python scripts/render_banner.py                # a handful of candidate crops to figs/banner/
"""
import os
import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from render_movie import BG, REPO_ROOT
from render_movie_3d import VolumeRenderer, load_volume


class Args:
    def __init__(self, **kw):
        self.width = 1584
        self.height = 396
        self.nz = 1400
        self.nxy = 84
        self.half_width = 220.0
        self.stretch = 10.0
        self.distance = 30.0
        self.focal = 1.75
        self.shift_y = 0.0
        self.gamma = 0.75
        self.pulse_gain = 1.4
        self.trail_gain = 0.22
        self.bloom = 7.0
        self.bloom_weight = 0.35
        self.blur = 1.6
        self.__dict__.update(kw)


def save(img, path):
    fig = plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(img, extent=(0, img.shape[1], img.shape[0], 0))
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    h5 = os.path.join(REPO_ROOT, "data/movie_24699095/movie.h5")
    vol = load_volume(h5)
    out_dir = os.path.join(REPO_ROOT, "figs/banner")
    os.makedirs(out_dir, exist_ok=True)

    candidates = [
        dict(name="final_beam", T=68.0, azimuth=95.0, elevation=4.0, focal=3.3, distance=18.0, shift_y=0.0),
        dict(name="final_striated", T=44.0, azimuth=90.0, elevation=2.0, focal=3.3, distance=18.0, shift_y=0.0),
        dict(name="final_beam_alt", T=64.0, azimuth=92.0, elevation=6.0, focal=3.1, distance=19.0, shift_y=-0.03),
        dict(name="final_beam_bright", T=58.0, azimuth=93.0, elevation=3.0, focal=3.3, distance=18.0, shift_y=0.0),
    ]

    for c in candidates:
        t0 = time.perf_counter()
        args = Args(focal=c["focal"], distance=c["distance"], shift_y=c["shift_y"])
        renderer = VolumeRenderer(vol, args)
        img = renderer.render(c["T"], c["azimuth"], c["elevation"])
        save(img, os.path.join(out_dir, f"{c['name']}.png"))
        print(f"  ({time.perf_counter() - t0:.1f} s)")


if __name__ == "__main__":
    main()
