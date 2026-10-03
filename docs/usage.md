# Using xraymb_sim

This guide covers installing the code, running a single simulation, the config files, running sweeps on
DESY's Maxwell cluster, and reading the results. The physics and the numerical scheme are described in the
technical write-up.

## Install

```bash
git clone https://github.com/P4GAN/Cu-RSA-XFEL-Simulation.git
cd Cu-RSA-XFEL-Simulation
pip install -e .
```

`numpy<2` is pinned for numba and OCELOT. The first run of a session spends about a minute compiling the
numba kernel; later runs reuse the cache.

## One simulation

```python
from xraymb_sim import Simulation

sim = Simulation("config/example.yaml")
sim.configure()     # incident pulse from the config's `pulse` key; or sim.configure(field) with your own
sim.run()
```

`python scripts/example.py` does this on a coarse grid and plots the incident and transmitted spectra.

After `run()` the results are attributes of `sim`:

| attribute | shape | meaning |
|---|---|---|
| `Omega_pstxyz` | (2, 2, t, x, y, z) | field as a Rabi frequency (fs⁻¹); p = 0 field, 1 conjugate; s = circular polarisation |
| `rho_ijtxyz` | (8, 8, t, x, y, z) | base-block density matrix (levels 0–3 2p₃/₂, 4–5 1s, 6–7 2p₁/₂ hole) |
| `rho_sat_ijtxyz` | list of the same | one per satellite channel, in config order |
| `rho_ground_txyz`, `rho_2s_txyz`, `rho_other_txyz` | (t, x, y, z) | incoherent populations |
| `rho_mid_txyz` | (t, x, y, z) or None | middleman pool (with `use_middlemen`) |
| `rho_e_gtxyz` | (group, t, x, y, z) or None | free electrons per atom per energy group (with `use_eii`) |

By default (`keep_z_history: true`) every z plane is kept, which `xraymb_sim/plot.py` needs. On a production
grid that takes tens of GB, so for anything large set `sim.keep_z_history = False` before `run()`. The
arrays then hold only the entrance and exit fields (z axis of length 2) and the centre pixel of the last
plane.

`xraymb_sim.analysis.spectrum_w(sim, z_index, ypad, tpad)` gives the spectrum of the field at a plane. The
energy axis is relative to Kα1, `hwKalpha1N`. `analysis.compute_run_outputs(sim)` gives everything a sweep
saves for one shot.

## How the simulation works

`Sample.run` marches through the foil one z plane at a time. At each plane, for each time step:

1. RK4-integrate every population one step (`model.py`), driven by the local field and the photon flux.
2. Propagate the field to the next plane: half the photoabsorption, a Fresnel diffraction step by FFT,
   the other half (`optics.py`).
3. Add the field radiated by the atomic coherences, which is where resonant absorption and stimulated
   emission enter.

The field that leaves plane iz drives plane iz + 1. All times are retarded times (t − z/c).

## Config files

Each YAML file fully describes one simulation. Unknown keys are rejected, so a typo fails loudly.

| file | what |
|---|---|
| `config/mono/1-bare.yaml` … `5-electrons.yaml` | the five steps of the final model, self-seeded pulse, as run in the final sweep |
| `config/sase/5-electrons.yaml` | the final model with SASE pulses |
| `config/movie.yaml` | the final model on a fine grid for the movie |
| `config/example.yaml` | the final model on a coarse grid with a Gaussian pulse |

The five steps add physics one block at a time, so the files differ only in the blocks they add:

1. `1-bare`: 2p₃/₂, 2p₁/₂ and 1s holes in one density matrix (Kα1 and Kα2).
2. `2-single`: adds the 2s hole and the single-spectator satellites (3d±, 3p±).
3. `3-double`: adds the double-3d satellites, fed by the 3p satellites.
4. `4-middlemen`: adds the middleman pool and the metal-only Coster–Kronig feeds.
5. `5-electrons`: adds the free electrons and electron-impact ionisation.

### Keys

Units are nm, fs and eV; cross sections are in nm² (1 kbarn = 1e-7 nm²).

**Sample and lines**: `n` (atoms per nm³), `zmax` (foil thickness, nm), `hwKalpha1N`, `hwKalpha2N` (line
energies, eV).

**Widths and decay**: `GammaKeVN`, `GammaL3eVN`, `GammaL2eVN`, `GammaL1eVN` are the total widths of the 1s,
2p₃/₂, 2p₁/₂ and 2s holes. `GammarKalpha1eVN` and `GammarKalpha2eVN` are the radiative Kα widths.
`GammaA_L1_to_L3M45eVN` and `GammaA_L1_to_L2eVN` are the 2s-hole decay into the bare 2p₃/₂ and 2p₁/₂ holes.

**Cross sections**: `sigma1_Ka1_<hole>` is photoionisation of the neutral atom leaving that hole (`2p3`,
`2p1`, `2s`, `other` = every other subshell). `sigma2_Ka1_<hole>` is further photoionisation of an ion that
carries that hole, which removes it from the tracked states.

**Pathways**: `use_2s_pathway` (track the 2s hole).

**Satellite channels**: `satellite_channels` and `double_satellite_channels`. Each entry is one block of
ions with spectator holes:

- `name`: the channel name.
- `detuning_eV`: shift of its Kα1 line from Kα1.
- `detuning_eV_L2_split`: its Kα1–Kα2 splitting.
- `Gamma_L_eV`, `Gamma_K_eV`, `Gamma_L2_eV`: widths of its 2p₃/₂, 1s and 2p₁/₂ holes.
- `Gamma_A_2s_eV`, `Gamma_A_2s_to_L2_eV`: feed from 2s-hole Coster–Kronig decay.
- `Gamma_A_K_eV`, `Gamma_A_K_to_L2_eV`: feed from 1s-hole KLM Auger decay.
- `sigma_Ka1_from_2p`, `sigma_Ka1_from_2p1`, `sigma_Ka1_from_1s`: feed from photoionisation of the spectator
  shell of base-block ions.
- `sigma_ion_from_2p`, `sigma_ion_from_1s`: further-photoionisation loss.

A double-spectator channel is fed only through `feed_from`, a list of `{channel, manifold, Gamma_feed_eV}`,
where manifold is `lower` (2p₃/₂), `upper` (1s) or `L2` (2p₁/₂). A parent's widths must include its feeds:
the code refuses configs that would create population.

**Middlemen** (step 4 on): `use_middlemen`, `middlemen: {targets: {channel: weight}}` (which satellite
blocks a middleman's 2p hole lands in), and `L2_CK_feed: {rate_eV, targets}` (2p₁/₂ → 2p₃/₂ + 3d
Coster–Kronig, a branch of `GammaL2eVN`).

**Free electrons** (step 5): `use_eii` and `eii`, with these keys:

- `n_groups`, `E_top_eV`, `E_bottom_eV`: the energy ladder.
- `spatial_factor`: the share of each electron's ionisations that happen inside the focus.
- `M_shell_scale`: scale of the 3s/3p/3d electron-impact ionisation; 0 means L shell only.

**Pulse**: `pulse` is `gaussian`, `sase` or `sase_dcm` (SASE through the Si(111) monochromator). The other
pulse keys:

- `E_seed_uJ`: pulse energy.
- `seed_duration_FWHM_t` (fs) and `seed_width_FWHM_x`, `seed_width_FWHM_y` (nm): duration and focal spot.
- `t_peak`: pulse centre in the time window, fs.
- `seed_center_E`, `seed_FEL_bandwidth`: SASE centre (eV) and relative bandwidth.
- `monochromator_target_energy_eV`, `seed_delay`: for `sase_dcm`.
- `random_seed`: −1 draws new SASE noise every run; sweeps set it to the shot number.

**Grid**: `tgrid`, `tmax` (time window, fs), `xgrid`/`ygrid` and `xmax`/`ymax` (transverse half-width, nm),
`zgrid` (planes through the foil), and `xpad`, `ypad`, `tpad` (zero padding for the FFTs).

Grid advice from the convergence work:

- Use `xgrid = ygrid ≥ 5`. A 3×3 grid puts too little fluence on the centre pixel and overstates the
  low-fluence dip.
- The final sweep used `tgrid 12000`, `zgrid 15`. `scripts/generate_convergence_sweep.sh` checks these.

## Sweeps on the Maxwell cluster

Every sweep follows the same four steps.

1. `bash scripts/generate_<family>_sweep.sh` writes one config per sweep point into
   `config/generated/<family>/<variant>/`, plus a `manifest.txt` listing them. It prints the `sbatch` command.
2. `sbatch --array=... scripts/submit_sweep.sh config/generated/<family>`. Each array task runs `PER_TASK`
   configs side by side; each config's `NREP` shots can be split into `CHUNKS` tasks. Defaults: a whole
   40-core node of the `allcpu` partition for up to 16 h.
3. `scripts/run_sweep.py` runs the shots. Shot `r` uses SASE seed `r`, so any split gives the same shots.
   The output is `data/<family>_<job id>/<variant>/runs_seed_<E>_uJ[__energy_<E>_eV]/`, holding a copy of the
   config, a `.provenance.txt` (package path and git commit), and a `.npz` of summed outputs.
4. Copy the data folder back (e.g. with `rsync`) and plot.

| family | generate | plot |
|---|---|---|
| final model, five steps, 25 measured energies × 4 pulse energies | `generate_final_sweep.sh` | `plot_final.py` |
| final model, SASE, 7 pulse energies × 200 shots | `generate_sase_sweep.sh` | `xraymb_sim.analysis.data_from_folder` |
| grid convergence of step 5 | `generate_convergence_sweep.sh` | `plot_convergence.py` |
| where atoms and electrons go (population budget) | `generate_population_budget.sh` (uses `submit_population_budget.sh`) | `plot_population_budget.py`, `plot_eii_diagnostics.py` |
| one shot for the movies | `sbatch scripts/submit_movie.sh` | `render_movie.py`, `render_movie_3d.py`, `render_level_diagram.py` |

Practical notes:

- Run the generate scripts and `sbatch` from the repo root, and `mkdir -p logs` first. Logs go to
  `logs/<job name>_<job id>_<task>.out`.
- `squeue -u $USER` shows the queue; `sacct -j <job id>` shows finished tasks and their memory.
- Each task writes its results only when its shots finish. A task killed partway leaves a `.partial.npz`
  with the shots done so far, which the loaders pick up.
- SASE runs on a 5×5 grid need about 64 GB per task; the mono defaults use `--mem=0` (a whole node).
- The movie run writes up to ~15 GB; point `OUT_DIR` at large storage.
- After changing the code, check the `.provenance.txt` commit before trusting new results.

## Reading sweep results

```python
from xraymb_sim.analysis import data_from_folder

means, aux, n_shots = data_from_folder("data/final_sweep_mono_<id>/5-electrons",
                                       group_keys=("E_seed_uJ", "monochromator_target_energy_eV"))
d = means[20.0][8048.0]
T = d["I_int_thy_w_last"].sum() / d["I_int_thy_w_0"].sum()    # transmitted / incident pulse energy
```

Each output is averaged over shots, with `<name>_std` alongside:

- `I_int_thy_w_0` and `I_int_thy_w_last`: incident and transmitted spectra on `womega_ar`, in eV from Kα1.
- `I_t_0` and `I_t_last`: the same in time.
- Centre-pixel exit-plane populations: `rho_ground_t_last`, `rho_2s_t_last`, `rho_mid_t_last`,
  `base_pop_t_last`, `sat_pop_t_last`.
- `total_population_t_last`: their sum, which stays 1 with middlemen.
- `rho_K_t_last`, `rho_l3_t_last`, `rho_l2_t_last`: dipole-weighted hole populations, also per satellite as
  `_sat`.
- `n_e_hot_t_last` and `n_e_total_t_last`: free electrons per atom.

The measured self-seeded transmittance is in `data/experiment/cu_20um_self_seeded_transmittance.csv`.

## Atomic parameters

The satellite and 2p₁/₂ parameters come from XATOM (`xatom/`). `xatom/print_satellite_parameters.py` prints
them as config-ready YAML. `xatom/recompute_all_parameters.py` recomputes everything into
`xatom/recomputed_parameters.json`. Both need XATOM installed and `XATOM_PATH` set. The base cross sections
of the bare ion are the original GRASP/RATIP values.
