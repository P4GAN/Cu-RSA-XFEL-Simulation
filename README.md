# Cu-RSA-XFEL-Simulation

Numerical simulation of **Reverse Saturable Absorption (RSA)** in copper targets irradiated by ultra-intense **X-ray Free-Electron Laser (XFEL)** pulses — developed during a research internship at **DESY**.

## What this does

At XFEL intensities, X-ray pulses can ionize a copper target fast enough to change their *own* absorption as they propagate through it — a nonlinear, self-modifying interaction rather than simple linear attenuation. This project solves that problem numerically by coupling:

- A **multilevel density-matrix (Maxwell–Bloch) model** of the Cu atomic populations, tracking photoionization, spontaneous decay, and K-shell fluorescence across `nlevel` atomic states.
- A **3D (transverse + longitudinal + time) optical propagation solver**, handling Gaussian/SASE pulse profiles, beam geometry, and auto-gridding based on diffraction angle.
- Support for both **seeded** and **SASE** (Self-Amplified Spontaneous Emission) XFEL pulses, generated via a bundled mini X-ray FEL simulator (`mini_ocelot`).

The result is a first-principles prediction of how a copper target's transmittance evolves under intense, ultrafast X-ray exposure — directly relevant to XFEL beam diagnostics and sample damage studies at facilities like European XFEL.

## Highlights

- **Physics**: multilevel atomic Bloch equations × nonlinear pulse propagation, coupled self-consistently.
- **Performance**: core density-matrix kernels are JIT-compiled with `numba` for large 3D (t, x, y) grids.
- **Configurable**: simulation parameters (cross-sections, pulse shape, geometry, grid resolution) are defined declaratively in YAML (see [`config/`](config/)), enabling sweeps like the pulse-energy scans in [`config/Cu-seed-SASE_*.yaml`](config/).
- **Reproducible**: results and figures for varying cross-sections and pulse durations in [`figs/`](figs/) and [`notebooks/`](notebooks/).

## Project structure

```
XLO_sim/        Core simulation package (Maxwell-Bloch solver, optics, sample physics)
mini_ocelot/    Lightweight SASE X-ray FEL pulse generator
config/         YAML configs for seeded / SASE runs, pulse-energy sweeps
notebooks/      Analysis notebooks (transmittance, SASE runs)
data/           Simulation outputs (HDF5)
figs/           Result plots
```

## Quick start

```bash
pip install -e .
python example_script.py
```

```python
from XLO_sim.XLO_sim import XLO_sim
from XLO_sim import tools

sim = XLO_sim("config/Cu-seed.yaml")
seed_field = tools.Gaussian_pulse_aniso_seed(sim)
sim.configure(seed_field)
sim.run_3D()
```

## Requirements

Python ≥ 3.9, `numpy`, `scipy`, `matplotlib`, `pyyaml`, `h5py`, `numba`.
