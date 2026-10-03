# Guide for AI coding agents

Maxwell–Bloch simulation of reverse saturable absorption in Cu under XFEL pulses. Read README.md first for the
layout and the quick start.

## Rules
- **No local simulation runs** beyond tiny smoke tests (a few minutes at most). Production runs go to the
  Maxwell cluster: write or reuse a generate script and hand the user the sbatch command it prints.
- There is no test suite. After touching `xraymb_sim/`, compare a tiny-grid run (e.g. config/example.yaml with a
  fixed random_seed) before and after the change; the numerics should be bit-identical unless the physics
  was meant to change.

## How the code fits together
- `Simulation` (simulation.py) loads a YAML config (unknown keys are rejected), sets every key as an attribute,
  and builds the coupling tensors: `Tijs_plus`/`Tijs_minus` (dipole couplings, split by which level is upper),
  `Gij` (spontaneous-decay branching), `Mij` (decay matrix), `Delta_ij` (detunings), photoionisation cross
  sections `S_ground_Fi`/`S_ion_Fi`, one namespace per satellite channel, and the middleman and EII tables.
- `Sample` (sample.py) marches z plane by plane; at each plane and time step it RK4-integrates every
  population (model.py), then propagates the field to the next plane (optics.py) and adds the field radiated
  by the coherences. `keep_z_history = False` selects the memory-lean path the sweeps use; an optional
  `X.recorder` (scripts/movie.py, scripts/population_budget.py) sees every step of that path.
- The sign conventions in `Tijs` (the relative sign of the 2p1/2 couplings) and `detuning_matrix` (a level
  sits at minus its energy offset, so a satellite with detuning d absorbs at Kα1 + d) are physical: getting
  one wrong mirrors or flips a spectral feature without any error.
- Population bookkeeping: a feed from one population to another must be a branch of the source's own width;
  `_build_satellite_blocks` and `_build_untracked_outflow` refuse configs that would create population.

## Sweeps
`scripts/generate_sweep.py` writes one config per sweep point into a family folder with a
`<variant> <config>` manifest; `scripts/submit_sweep.sh` runs it as a SLURM array (knobs NREP, CHUNKS,
PER_TASK); `scripts/run_sweep.py` writes `data/<family>_<job id>/<variant>/runs_.../`, read back by
`xraymb_sim.analysis.data_from_folder` or `scripts/plot_final.py`.
