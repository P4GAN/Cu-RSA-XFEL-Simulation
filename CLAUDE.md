# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Numerical simulation of Reverse Saturable Absorption (RSA) in copper targets irradiated by X-ray
Free-Electron Laser (XFEL) pulses (DESY internship project). It couples a multilevel density-matrix
(Maxwell–Bloch) model of Cu atomic populations to a 3D (t, x, y, z) optical propagation solver, so
that the target's absorption evolves self-consistently as the pulse propagates through it. Supports
both seeded and SASE XFEL pulses.

## Setup

```bash
pip install -e .   # editable install so `import XLO_sim` works from anywhere, incl. notebooks/
```

Dependencies (see `pyproject.toml`): `numpy<2`, `scipy`, `matplotlib`, `pyyaml`, `h5py`, `numba`,
`ocelot-collab`. No lint/test/CI tooling is configured in this repo (no pytest, no linter config) —
correctness is checked by running simulations and inspecting outputs/plots, not by an automated suite.

## Running simulations

Minimal interactive run:

```python
from XLO_sim.XLO_sim import XLO_sim
from XLO_sim import tools

X = XLO_sim("config/base/Cu-seed.yaml")   # loads YAML, builds level structure, grids, precomputes coupling tensors
seed_field = tools.Gaussian_pulse_aniso_seed(X)   # or tools.Ocelot_SASE_seed_pstxy(X) for SASE
X.configure(seed_field)                   # builds XLO_sample (Green's function, initial conditions)
X.run_3D()                                # runs the z-marching Maxwell-Bloch + Fresnel propagation loop
```

`scripts/example_script.py` is the canonical smoke-test entry point (`python scripts/example_script.py`).

For batch/statistics jobs (many repetitions, e.g. SASE noise realizations), use the
`generate_*_sweep_configs.py` → `submit_*_sweep.sh` (SLURM array) → `run_*_sweep.py` pipeline (see
Architecture below) rather than driving `XLO_sim` directly.

## Architecture

### Core package: `XLO_sim/`

The simulation is one z-marching loop coupling two subsystems at each longitudinal step:

- **`XLO_sim.py`** — `XLO_sim` class: the top-level entry point/config object. Loads a YAML config,
  hangs every key onto `self` (so `X.tgrid`, `X.nlevel`, etc. are read directly off the config
  throughout the rest of the code), and precomputes all the physics tensors needed by `Model.py`:
  `Tijs`/`Tijs_plus`/`Tijs_minus` (dipole coupling matrix elements, direction-aware so the right
  Rabi field couples to the right transition), `Gij`/`Gamma_sp_Gij` (spontaneous-decay feed
  fractions), `S_ground_Fi`/`S_ion_Fi` (photoionization cross sections in/out of each level), `Mij`
  (coherence dephasing rates), `Delta_ij` (per-pair detuning). `configure()` builds the `XLO_sample`;
  `run_3D()` runs it and copies results back onto `X`.
- **`Model.py`** — the Maxwell–Bloch RHS functions (`MB_nlevel_regular`, `MB_ground_regular`,
  `MB_other_regular`, `MB_2s_regular`, `MB_satellite_block_regular`, `Omega_source_regular`,
  `absorption`). The hot inner kernel (`_MB_nlevel_regular_core`) is `@njit`-compiled with numba over
  the full (level, level, x, y) tensor per RK4 substep — this is the performance-critical code path.
- **`Sample.py`** — `XLO_sample`: owns the actual t/z marching loop
  (`evaluate_n_level_3D` → `_evaluate_n_level_3D_full` or `_evaluate_n_level_3D_lean`). At each z
  step it RK4-integrates the density matrices forward in t (`tools.RK45_step` calling into
  `Model.py`), computes the absorption coefficient, Fresnel-propagates the field to the next z plane
  (`Optics.py`), then adds the field sourced by the updated density matrix
  (`Model.Omega_source_regular`). Two implementations of the same physics:
  - `_evaluate_n_level_3D_full`: keeps every z-plane of every array (needed by `Plot.py`/notebooks
    to show propagation through depth). Tens of GB at production grid sizes.
  - `_evaluate_n_level_3D_lean`: identical numerics, only keeps a rolling 2-slot z buffer plus the
    z=0/last snapshot — this is what `X.keep_z_history = False` selects, and is what all
    `run_*_sweep.py` batch scripts use (they only ever read z=0/z=-1 via `tools.compute_run_outputs`).
- **`Optics.py`** — `XLO_optics`: FFT-based Fresnel propagation (`Fresnel_propagator_with_absorption`
  / `_no_absorption`), the numerical Green's function for the sample, k-space grids/filters, thin
  lens / drift kernels. `enable_self_diffraction=False` in a config short-circuits propagation to a
  no-op.
- **`tools.py`** — the largest module; grab-bag of: seed pulse generators (`gaussian_pulse`,
  `Gaussian_pulse_aniso_seed`, `Ocelot_SASE_seed_pstxy` — the latter wraps `ocelot` to generate
  realistic SASE spectra/statistics), polarization conversion (`linear_to_circular`/
  `circular_to_linear`), noise generators, the shared RK4 stepper (`RK45_step`), post-processing
  (`fft_field_t_y_to_w_thy`, `SF_spectrum_w`, `compute_run_outputs`, `accumulate_run_outputs`,
  `data_from_folder`), and the sweep-runner harness (`run_sweep_chunk`).
- **`Plot.py`** — `XLO_plot`: all visualization, driven off a finished/loaded `XLO_sim` run.

### Level structure (config-driven, not a fixed enum)

`nlevel` in the YAML (6 or 2) sets the base atomic block: 6-level is the full sublevel-resolved model
(2p₃/₂ hole "L3": 4 sublevels, 1s hole "K": 2 sublevels, coherent density matrix between them —
this is where stimulated Kα1 emission lives); 2-level collapses each manifold to one level. Ground,
2s-hole ("L1"), and "other" are always separate incoherent population scalars alongside the coherent
block. Two optional extensions, mutually incompatible with each other and gated by config flags in
`XLO_sim.py.__init__`, add to this base:

- **`use_L2_pathway: true`** (requires `nlevel: 6`) — appends the 2p₁/₂ hole ("L2", Kα2) manifold as
  2 more local levels *inside the same density matrix* (it shares the base block's 1s population, so
  it can't be a separate block). See `docs/2p1_2-implementation-plan.md`.
- **`satellite_channels: [...]`** (requires `nlevel: 6`, incompatible with `use_L2_pathway`) — each
  entry describes an independent detuned 6-level block (2s-hole spectator satellite pathway, e.g.
  `2p+3d+`), evolved by `Model.MB_satellite_block_regular` alongside the base block and fed by it.
  See `docs/theory-and-2s-satellite-pathways.md` Part II.

Cross sections/widths referenced by these config keys (`sigma1_Ka1_2p1`, `GammaL2eVN`,
`sigma_Ka1_from_2p`, etc.) are computed from the XATOM atomic-structure code via `xatom/xatom_tools.py`
(`run_xatom`/`run_xatom_cached` shells out to the `xatom` binary and parses its output;
`satellite_channel_parameters`/`l2_pathway_parameters` assemble the final config-ready values).
`xatom/print_satellite_parameters.py` is the CLI for regenerating these numbers.

### Config files (`config/base/*.yaml`)

Each YAML fully describes one simulation: photoionization cross sections, radiative rates, sample
composition/density, level structure flags (above), the (t, x, y, z) grid, seed pulse parameters, and
run-mode flags (`enable_self_diffraction`, `enable_self_absorption`, `is_use_stochastic`,
`run_mode: simultaneous|consecutive`, ...). The `Cu-seed*.yaml` variants in `config/base/` are
starting points for different level structures/pulse types (SASE, mono, satellite, L2, 2-level);
`config/generated/` (gitignored) holds YAML manifests produced by `scripts/generate_*_sweep_configs.py`
for parameter sweeps.

### Batch sweep pipeline (`scripts/`)

Three-stage pipeline for cluster (SLURM) parameter sweeps, e.g. transmittance vs. seed intensity:

1. `generate_<X>_sweep_configs.py` — writes one YAML per sweep point into `config/generated/.../` plus
   a `manifest.txt`, and prints the `sbatch --array=...` command to submit (the array size depends on
   how many configs it just wrote, so the printed command is authoritative over any hardcoded
   `#SBATCH --array` pragma in the `.sh` file).
2. `submit_<X>_sweep.sh` — SLURM array job; each array task picks one config/repetition-chunk out of
   the manifest and calls `run_<X>_sweep.py`.
3. `run_<X>_sweep.py` — runs a chunk of repetitions (`X.keep_z_history = False`, `X.random_seed = rep`
   for SASE noise realizations), computes outputs via `tools.compute_run_outputs`, and writes one
   accumulated (sum/sumsq/n_reps) `.npz`/output file per chunk into `data/`.
   `tools.data_from_folder` losslessly combines all chunks for a sweep point afterward, so chunk
   boundaries don't affect the final aggregated statistics. Every `run_*_sweep.py` script pins
   `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/`NUMEXPR_NUM_THREADS=1` before importing
   numpy, since each multiprocessing worker gets its own BLAS thread pool otherwise.

### Notebooks (`notebooks/`)

Interactive counterparts to the batch scripts (e.g. `run-transmittance-vs-intensity.ipynb` mirrors
`run_intensity_sweep.py`'s multiprocessing loop) plus plotting notebooks (`plot-*.ipynb`) that read
from `data/` and render into `figs/`. `Cu-RSA-v0.ipynb` and `monochromator.ipynb` are exploratory.

### Physics reference

`docs/theory-and-2s-satellite-pathways.md` is the authoritative theory writeup — it derives every
matrix (`Tijs`, `Gij`, `Mij`, `Delta_ij`) from first principles, cross-references equation numbers
against `docs/XLO-sim_equations.pdf`, and explicitly flags where the code has drifted ahead of the
PDF. `docs/2p1_2-implementation-plan.md` covers the L2/Kα2 pathway specifically. When touching
`Model.py` or `XLO_sim.py`'s tensor construction, check these docs first — the sign conventions
(e.g. `Tijs_plus` vs `Tijs_minus`, the `Delta_ij` detuning sign) are physically derived and
non-obvious from the code alone; get one wrong and a spectral feature flips sign or lands at the
wrong detuning without erroring.
