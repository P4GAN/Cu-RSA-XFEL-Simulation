# Repository guide for AI coding agents

This file provides guidance to AI coding agents (Claude Code, Codex CLI, and similar tools) working
in this repository. It is published under two filenames: `CLAUDE.md` (this file, the real one) and
`AGENTS.md`, which is a symlink to it — so they are always byte-identical and there is no separate
copy to keep in sync. `CLAUDE.md` is the real file rather than `AGENTS.md` specifically because
Claude Code's own file-editing tools refuse to write *through* a symlink ("refusing to write: it is
a symbolic link") — so if the link ran the other way, editing `CLAUDE.md` from inside Claude Code
would break. Always edit `CLAUDE.md`; don't turn `AGENTS.md` into a real file (check `ls -la
AGENTS.md` if unsure which way it points). If the link ever gets broken (e.g. some tool copies
instead of following it), recreate it from the repo root with:
`rm AGENTS.md && ln -s CLAUDE.md AGENTS.md`.

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
  - An optional `X.movie_recorder` (`movie.py`, lean path only) streams a float32 reduction of the
    full (t, x, y, z) history to HDF5 one z-plane at a time. It exists for visualisation runs
    (`scripts/run_movie.py` / `submit_movie.sh`, config `config/base/Cu-seed-SASE-movie.yaml`) and
    only reads state, so the numerics are bit-identical with or without it.
  - `population_budget.py` is a lighter recorder on the same hook, sized for sweeps. It keeps every
    population (middleman pool, base and each satellite block per manifold, each free-electron
    group) at every z plane. It stores the centre pixel and an incident-fluence-weighted beam average,
    averaged over shots, and drives `scripts/run_population_budget.py`. The sweep outputs keep only
    the exit plane's centre pixel.
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

`nlevel` must be 6 (the full sublevel-resolved model: 2p₃/₂ hole "L3" — 4 sublevels, 1s hole "K" — 2
sublevels, coherent density matrix between them — this is where stimulated Kα1 emission lives);
`XLO_sim.__init__` raises `ValueError` if it isn't. A collapsed-manifold `nlevel: 2` mode existed
historically (`config/base/Cu-seed-2-level.yaml` is a leftover config from it) but is no longer
supported by the code — don't use that file as a starting point. Ground, 2s-hole ("L1"), and "other"
are always separate incoherent population scalars alongside the coherent block. Two optional
extensions, gated by config flags in `XLO_sim.py.__init__` and freely combinable with each other,
add to this base:

- **`use_L2_pathway: true`** — appends the 2p₁/₂ hole ("L2", Kα2) manifold as 2 more local levels
  *inside the same density matrix* (it shares the base block's 1s population, so it can't be a
  separate block). See `docs/2p1_2-implementation-plan.md`.
- **`satellite_channels: [...]`** — each entry describes an independent detuned 6-level block
  (2s-hole spectator satellite pathway, e.g. `2p+3d+`), evolved by `Model.MB_satellite_block_regular`
  alongside the base block and fed by it. See `docs/theory-and-2s-satellite-pathways.md` Part II.
  `double_satellite_channels: [...]` (same entry schema) appends a further generation of
  double-M-shell-spectator channels (e.g. `2p+3d+3d+`) onto the same list — see
  `docs/double-spectator-satellite-implementation-plan.md`.

When both `use_L2_pathway` and `satellite_channels` are set, each satellite block *also* auto-gains
its own 2p₁/₂ extension (`use_L2_satellite_pathway`, `satellite_nlevel = 8`). This combination — L2 +
double satellite — is the "full model" that most current production configs
(`Cu-seed-SASE-double-satellite*.yaml`) actually run; `*-no-L2.yaml`/`*-satellite-no-L2.yaml`
variants exist specifically to isolate the L2 contribution by turning it back off.

Cross sections/widths referenced by these config keys (`sigma1_Ka1_2p1`, `GammaL2eVN`,
`sigma_Ka1_from_2p`, etc.) are computed from the XATOM atomic-structure code via `xatom/xatom_tools.py`
(`run_xatom`/`run_xatom_cached` shells out to the `xatom` binary and parses its output;
`satellite_channel_parameters`/`l2_pathway_parameters` assemble the final config-ready values).
`xatom/print_satellite_parameters.py` and `xatom/assemble_satellite_blocks.py` (bulk-regenerates the
`xatom/assembled/*.yaml` snippets spliced into `config/base/*.yaml`) are the CLIs for regenerating
these numbers. `grasp/` (GRASP2018+RATIP Dirac-Fock) and `jac/` (JAC.jl) are independent cross-checks
of a subset of these same cross sections/detunings — not part of the simulation pipeline itself, but
the right place to look if an XATOM-derived number looks physically off; start at `grasp/README.md`
(it has already caught real sign errors in XATOM's satellite detunings once).

**Population bookkeeping and the Part VI/VII extensions** (`XLO_sim._build_pathway_extensions`; all
off by default, and with every flag off results match the pre-extension code to round-off).
`__init__` precomputes, per level, the *untracked* outflow: decay and further photoionisation that
has no other destination. A feed from one population to another must be a **branch** of the
source's own width, never a reduction of it (`feed_diag_*` only add to the destination). A config
whose `double_satellite_channels` feeds exceed the parent's width is rejected. That was a real
population-creating bug until 2026-09-15 (`docs/theory-middlemen-and-pathway-audit.md` §2.3).

- **`use_middlemen: true`** + `middlemen: {targets: ...}` — one scalar pool of 3d⁻ⁿ ions that
  collects every untracked outflow, and the old ground→"other" pump. It absorbs like ground and is
  photoionised into the satellite blocks named in `targets` (or `'none'`). With it,
  `total_population_t_last` is 1 to timestep accuracy. `docs/middlemen-implementation-plan.md`.
- **`L2_CK_feed`**, **`GammaA_L1_to_L2eVN`** — extra feeds into existing blocks (metal-only
  2p₁/₂→2p₃/₂+3d Coster–Kronig; 2s→bare 2p₁/₂).
- **`L3_sublevel_mixing_fs_inv`** / **`..._satellite_fs_inv`** — depolarising mixing of the 2p₃/₂
  sublevels inside `_MB_nlevel_regular_core` (dark-state test, Part VI §1.5).
  `L3_sublevel_mixing_coherence_factor` (default 1) scales the coherence-damping half of that map:
  1 is the Lindblad form (also dephases the optical coherence by ħγ/2), 0 keeps only the population
  exchange. Population exchange alone leaves the dark state intact, because the dark state is a
  Raman *coherence* between 2p₃/₂ sublevels, not a population imbalance. `Model.MODEL_FEATURES`
  names such kernel features so `tools.verify_code` can refuse a config that needs one the imported
  code lacks.
- **`sublevel_raman_dephasing_fs_inv`** — extra pure dephasing of the 2p–2p and 1s–1s coherences
  only, in every block. It removes the dark state without broadening the line. It is added to the
  off-diagonal `Mij` in `XLO_sim.__init__` (`raman_coherence_mask`); only the kernel reads those
  entries. The broadening counterpart is the older scalar `additional_dephasing`, which is added to
  every optical coherence and to the 2p₃/₂–2p₁/₂ coherence.
- **`use_eii: true`** + `eii: {...}` — free-electron slowing-down ladder (`XLO_sim/eii.py`:
  Burgess–Chidichimo cross sections, Joy–Luo stopping) whose EII rates feed base/2s/middleman
  populations. `docs/eii-free-electrons-implementation-plan.md`. `docs/theory-eii-electron-ladder-explained.md`
  explains the model as built. The `eii:` keys `anchor_birth_energies`, `secondary_spectrum` and
  `slowing_down` (all off by default, so old configs are unchanged) give the finer anchored ladder,
  the δ-electron cascade and the fixed-energy bound. The low-energy electron counts with
  `secondary_spectrum` measure deposited energy, not physical free electrons (that doc, §5).

`config/base/Cu-seed-{SASE,mono-SASE}-middlemen.yaml` and `...-middlemen-eii.yaml` are the base
configs: the double-satellite configs of the same name plus the extension blocks. The `eii:`,
`middlemen:` and `L2_CK_feed:` sub-blocks reject unknown keys. `...-middlemen-eii-dLL.yaml` adds
the 2s⁻¹2p⁻¹ and 2p⁻² absorbers as two more satellite channels (`xatom/double_L_hole_parameters.py`;
new channel key `sigma_Ka1_from_2p_to_L2`). Photoionisation feeds out of a base manifold carry that
manifold's own further-ionisation sublevel pattern (`XLO_sim.pi_feed_pattern_Fi`), so they can be
drained sublevel by sublevel in the untracked bookkeeping.
New flags go into `tools.PATHWAY_EXTENSION_KEYS`, so `tools.verify_code` (called by
`run_intensity_sweep.py` and `run_mono_sweep.py`) refuses to run them from a stale XLO_sim import.

### Config files (`config/base/*.yaml`)

Each YAML fully describes one simulation: photoionization cross sections, radiative rates, sample
composition/density, level structure flags (above), the (t, x, y, z) grid, seed pulse parameters, and
run-mode flags (`enable_self_diffraction`, `enable_self_absorption`, `is_use_stochastic`,
`use_rate_equations` — adiabatically eliminates the coherences, see `Model.physical_rho` — `run_mode:
simultaneous|consecutive`, ...). The `Cu-seed*.yaml` variants in `config/base/` are starting points
for different level structures/pulse types (SASE, mono, satellite, double-satellite, L2,
GRASP-recomputed cross sections — see Level structure above for which of these are still live);
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

The `production_*` sweep family generates from a `.sh` script (`generate_production_sweeps.sh`)
instead of a `.py` one — it just points the existing `generate_intensity_sweep_configs.py`/
`generate_mono_sweep_configs.py` generators at several base configs in turn — but still follows the
same generate → submit → run shape, ending in `run_production_config.py` rather than a per-sweep
runner. The `pathway_sweep` family (`generate_pathway_sweeps.sh` → `submit_pathway_sweep_sase.sh` /
`submit_pathway_sweep_mono.sh` → `run_intensity_sweep.py` / `run_mono_sweep.py`) has the same
shape. It runs the Part VI/VII variants of the double-satellite model and writes one flat
`<variant> <yaml>` manifest per family. Variants that differ only in a scalar are expressed as
`--set KEY=VALUE` overrides, which both generators accept; they refuse keys missing from the base
config. Structured edits (every satellite detuning, the foil thickness, another config's extension
blocks) go through `scripts/derive_config.py`, which writes a variant base config with a `derived:`
record of its transforms; `generate_bracket_sweeps.sh` uses it to build one-change-at-a-time
brackets around a reference model. The two `submit_pathway_sweep_*.sh` scripts take the family
directory as `$1`. `generate_b1_sweeps.sh` (double-L-hole absorbers) and
`generate_coherence_sweeps.sh` (Raman dephasing and homogeneous broadening) reuse the same pair.
Several things are easy to get wrong with these families:
- A family's data lands in `data/<family dir basename>_<array job id>/<variant>/`.
- Omitting `$1` silently runs the pathway manifest instead.
- The SASE base configs are 3×3. Use `xgrid=ygrid=5` for anything compared with experiment:
  `generate_coherence_sweeps.sh` does, and `SASE_GRID=... generate_bracket_sweeps.sh` can.
- 5×5 SASE needs `sbatch --mem=64G`, and the generators print it.
`generate_progression_sweeps.sh` (mono only, same submit script) runs the model one addition at a
time: L2 only (`Cu-seed-mono-SASE-L2-original.yaml`), + single satellites and middlemen
(`Cu-seed-mono-SASE-satellite-middlemen.yaml`), + double satellites, + EII in three forms. It also
writes a population-budget family for the EII variants.
`scripts/plot_bracket_sweep.py` plots the bracket (mono) and B1 families, and
`scripts/plot_coherence_sweep.py` the coherence family. The latter reads its mono experiment from
`../RSA-derivation-bloch/data/exp_mono_scatter_slide9bins.csv`. The population-budget
family (`generate_population_budget.sh` → `submit_population_budget.sh <family dir>`, with `NREP`
and `CONFIGS_PER_TASK` passed via `--export` → `run_population_budget.py` →
`plot_population_budget.py`) records where atoms and electrons go during the pulse.
`scripts/plot_*.py` are standalone (non-notebook) counterparts to the `plot-*.ipynb`
notebooks below; each is tied to one specific sweep's output directory (check the file's own
docstring for which `data/...` folder it expects).

### Notebooks (`notebooks/`)

Interactive counterparts to the batch scripts (e.g. `run-transmittance-vs-intensity.ipynb` mirrors
`run_intensity_sweep.py`'s multiprocessing loop) plus plotting notebooks (`plot-*.ipynb`) that read
from `data/` and render into `figs/`. `Cu-RSA-v0.ipynb` and `monochromator.ipynb` are exploratory.

### Physics reference

`docs/theory-and-2s-satellite-pathways.md` is the authoritative theory writeup — it derives every
matrix (`Tijs`, `Gij`, `Mij`, `Delta_ij`) from first principles, cross-references equation numbers
against `docs/XLO-sim_equations.pdf`, and explicitly flags where the code has drifted ahead of the
PDF. `docs/2p1_2-implementation-plan.md` (L2/Kα2) and
`docs/double-spectator-satellite-implementation-plan.md` (double-satellite) cover those two
*implemented* extensions specifically; `docs/2s-satellite-implementation-plan.md` and
`docs/auger-branching-and-ci-mixing.md` record earlier/supporting derivations for the same pathways.
When touching `Model.py` or `XLO_sim.py`'s tensor construction, check these docs first — the sign
conventions (e.g. `Tijs_plus` vs `Tijs_minus`, the `Delta_ij` detuning sign) are physically derived
and non-obvious from the code alone; get one wrong and a spectral feature flips sign or lands at the
wrong detuning without erroring.

`docs/model-changes-2026-09-physics.md` and `docs/model-changes-2026-09-code.md` summarise every
change made 14–17 September 2026, physically step by step and file by file (commits, config keys,
outputs, run families, validation). Start there before the individual plans.
`docs/theory-middlemen-and-pathway-audit.md`/`docs/middlemen-implementation-plan.md` (Part VI) and
`docs/theory-eii-and-free-electrons.md`/`docs/eii-free-electrons-implementation-plan.md` (Part VII)
are the theory and the implementation record of the pathway extensions described under Level
structure. Each plan opens with a status block saying which steps were built, and ends with the
validation results. `config/base/Cu-seed-satellite-eii.yaml` belongs to an earlier, superseded EII
design. Its `eii:` block (`tau_th_fs`, `birth_energy_eV`) is now rejected at load time, so don't
start from it.
