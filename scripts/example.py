"""
Quick start: one shot of the final model on a coarse grid (config/example.yaml, about two minutes on a laptop,
most of it compiling the numba kernel on the first run), then the incident and transmitted spectra.

    python scripts/example.py
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO_ROOT)

from xraymb_sim import Simulation  # noqa: E402
from xraymb_sim.analysis import spectrum_w  # noqa: E402

t0 = time.perf_counter()
sim = Simulation(os.path.join(REPO_ROOT, "config", "example.yaml"))
sim.configure()          # the incident pulse named by the config's `pulse` key
sim.run()                # march the pulse and the atoms through the foil
print(f"simulated {sim.zmax / 1e3:g} um of Cu in {time.perf_counter() - t0:.0f} s")

E, I_in, _ = spectrum_w(sim, 0, ypad=64, tpad=1000)
_, I_out, _ = spectrum_w(sim, -1, ypad=64, tpad=1000)
print(f"transmitted pulse energy: {I_out.sum() / I_in.sum():.3f} of the incident")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(E + sim.hwKalpha1N, I_in / I_in.max(), label="incident")
ax.plot(E + sim.hwKalpha1N, I_out / I_in.max(), label="transmitted")
ax.set_xlim(sim.hwKalpha1N - 5, sim.hwKalpha1N + 5)
ax.set_xlabel("Photon energy (eV)")
ax.set_ylabel("Spectral intensity (normalised)")
ax.legend()
fig.tight_layout()
plt.show()
