# Model changes, 14–17 September 2026: the code

What changed in the code, config and scripts to implement the physics in
`docs/model-changes-2026-09-physics.md` (referred to below as "physics §n"). Branch
`electron-impact-ionization`. Every new model option defaults to off. With all of them off, results
match the pre-extension code to round-off (not bit-for-bit, because the numba kernel uses `fastmath`
and the new terms reorder its sums).

---

## 0. Commits

| Commit | Date | Content |
|---|---|---|
| ff828c4 (`main`) | 15 Sep | `compute_run_outputs` splits the population trace into `base_pop_t_last` / `sat_pop_t_last` |
| 3d30b18 | 15 Sep | Population bookkeeping and double-satellite feed check (physics §1). Middleman pool (§2), CK feeds (§3), sublevel mixing (§6), EII ladder (§4), `verify_code` for both runners, new base configs, pathway sweep family |
| 2297546 | 16 Sep | `movie.py` recorder hook in the lean path; movie runner and config |
| 472cd62 | 16 Sep | J(z=0) fix (§5), mixing coherence factor (§6), `MODEL_FEATURES`, `derive_config.py`, bracket sweep family, pathway plots |
| a78237e | 16 Sep | Double-L-hole absorbers (§7), `sigma_Ka1_from_2p_to_L2`, `pi_feed_pattern_Fi` (§8), B1 sweep family, population-record runner (analytic comparison) |
| 6521581 | 17 Sep | Raman dephasing (§6), coherence sweep family, `SASE_GRID`, bracket plots |
| uncommitted | 17 Sep | Population-budget recorder, runner, generator, submit and plot scripts; these two docs |

---

## 1. Invariants the code now enforces

1. **Feeds are branches, not reductions** (physics §1). `XLO_sim._build_pathway_extensions`:
   - Sums every `feed_from` out of each parent manifold and raises if the sum exceeds that manifold's
     `Mij` diagonal (its total width).
   - Computes, per level of every block, the **untracked outflow**: decay width minus radiative return
     inside the block minus every tracked feed, and further-ionisation cross section minus every
     tracked photoionisation feed. It raises if any entry is negative. The middleman pool and the
     electron production both read these vectors.
   - The existing check that the K-hole Auger feeds into satellites fit inside the non-radiative K
     width is unchanged.
2. **Population is conserved.** With `use_middlemen`, ground + other + 2s + middlemen + every block
   diagonal = 1 to RK4 accuracy. `compute_run_outputs` saves it as `total_population_t_last`.
3. **Structured config blocks reject unknown keys.** `middlemen:`, `eii:` and `L2_CK_feed:` go through
   `check_keys`, so a typo or the superseded EII schema (`tau_th_fs`, `birth_energy_eV` in
   `config/base/Cu-seed-satellite-eii.yaml`) fails at load instead of running on defaults.
4. **Stale code refuses new flags.** `tools.verify_code(X, repo_root)` runs before any repetition in
   `run_intensity_sweep.py`, `run_mono_sweep.py`, `run_population_record.py` and
   `run_population_budget.py`, and writes a provenance string (checkout path, commit, dirty flag,
   active extensions) next to every output. It refuses to run if:
   - the imported `XLO_sim` is not this checkout;
   - a rate-equation config would run the Maxwell–Bloch kernel;
   - a config key in `tools.PATHWAY_EXTENSION_KEYS` is active and needs a feature missing from
     `Model.MODEL_FEATURES`.
5. **Full and lean paths cannot drift apart.** `Sample._step_increments` holds every per-step RK4 call
   and is shared by `_evaluate_n_level_3D_full` and `_lean`. `Sample._photon_flux` is the single
   field → flux formula.

---

## 2. `XLO_sim/` by file

### `XLO_sim.py`

- **`raman_coherence_mask(ei_L, ei_K)`** (module level). 1 on 2p–2p and 1s–1s off-diagonal entries,
  0 elsewhere. `__init__` adds `sublevel_raman_dephasing_fs_inv` times this mask to the base `Mij`,
  and to every satellite channel's `Mij` after its L2 extension. Only the kernel reads off-diagonal
  `Mij`; everything else reads the diagonal. (Physics §6.)
- **Satellite channel parameters.** New channel key `sigma_Ka1_from_2p_to_L2` → `chan.S_feed_2p_to_L2`,
  a photoionisation feed from the base 2p₃/₂ manifold into the channel's 2p₁/₂ manifold. Allowed only
  with the L2 satellite extension. (§7.)
- **`_build_pathway_extensions()`**, called at the end of `__init__`. Sections and the attributes they
  set:

  | Section | Attributes |
  |---|---|
  | feed-from check | (raises) |
  | sublevel mixing | `L3_mixing_base_fs`, `L3_mixing_sat_fs`, `L3_mixing_coh`, `mix_mask_base`, `mix_mask_sat` |
  | L2 CK feed | `chan.Gamma_CK_L2_fs`, `Gamma_CK_L2_total_fs` |
  | untracked bookkeeping | `decay_untracked_base`, `S_untracked_base`, `pi_feed_pattern_Fi`, `e_emit_base` {KLL, LMM, CK}; per channel `chan.decay_untracked`, `chan.S_untracked`, `chan.e_emit`; `twos_decay_tracked_fs`, `twos_decay_untracked_fs` |
  | middleman pool | `sigma_mid_F`, `chan.mid_share`, `mid_ck_L3`, `mid_ck_L2`, `mid_S_out`, `mid_L2k_available` |
  | EII ladder | `eii_ladder` (from `eii.build_ladder`), `eii_G` (= `n_groups` + 1 thermal bin), `eii_k_down`, `eii_rate_table` (rows 2p₃/₂, 2p₁/₂, 2s, M shell; × `spatial_factor`, M row × `M_shell_scale`), `eii_birth` |

  `pi_feed_pattern_Fi` (§8) is the base further-ionisation cross section per sublevel, divided by its
  manifold mean. Photoionisation feeds out of the base block are multiplied by it, and so is the
  untracked PI subtraction.

### `Model.py`

- **`MODEL_FEATURES`** = {`pathway_extensions`, `mixing_coherence_factor`, `raman_dephasing`}.
- **`_MB_nlevel_regular_core(..., mix_mask, gamma_mix, mix_coh)`**, the numba kernel.
  - Depolarising mixing inside the flagged (2p₃/₂) levels: populations relax to the manifold mean at
    rate γ; coherences with one or both indices flagged are damped at `mix_coh`·γ/2·(mask_i + mask_j).
  - In rate-equation mode the same damping enters the filtered-field rate. `gamma_mix = 0` skips all
    of it.
- **`feed_diag_base_block`**. Adds L-shell EII of ground atoms into the base 2p₃/₂ and 2p₁/₂ holes,
  spread evenly over sublevels.
- **`feed_diag_satellite_block`**. Adds:
  - photoionisation feeds weighted by `pi_feed_pattern_Fi` (§8);
  - `S_feed_2p_to_L2` (§7);
  - the L2 CK feed from the base 2p₁/₂ population (§3);
  - middleman photoionisation (2p₃/₂ with the ground sublevel pattern, 2p₁/₂, 2s followed by instant
    CK) scaled by `chan.mid_share`, plus L-shell EII of middlemen along the same routing (§2, §4).
- **`MB_other_regular`**. With `use_middlemen`, "other" is no longer pumped: its pump goes to the pool.
  Without middlemen it gains M-shell EII of ground atoms.
- **`MB_2s_regular`**. Adds 2s EII of ground atoms.
- **`MB_ground_regular`**. Adds the EII loss, frozen at its start-of-step value (exactly what the
  destinations receive; draining it by RK4 overshot the trace by 0.9% at dt = 0.03 fs).
- **New functions:**
  - `eii_rates_xy(X, rho_e_gxy)`: EII rate per target atom (4 rows).
  - `middleman_gain_loss(...)`: pool gain = every untracked outflow + the old "other" pump + M-shell
    EII; loss rate = the routed photoionisation and EII.
  - `MB_middleman_regular`.
  - `electron_production_gxy(...)`: electrons per atom per fs by birth group. Photoionisation of any
    population → photo group; non-radiative K decay → KLL group; untracked L Auger and 2s L₁–MM → LMM
    group; tracked CK, super-CK and every EII event → CK group.
  - `MB_electron_regular`: production minus downward flow at `eii_k_down`; the last bin only
    accumulates.
- **`absorption(..., rho_mid_xyz=None)`** adds n·σ_mid·ρ_mid.

### `Sample.py`

- **`_step_increments(...)`** (static). Every population's RK4 increment over one step from
  start-of-step values: base, other, 2s, ground, each satellite block, middlemen, electron ladder.
  Used by both z-marching loops.
- **`_photon_flux(X, Omega_ps)`** (static). J = Re(Ω₀ₛΩ₁ₛ)/`flux_factor`. **J(z=0) fix (§5):** both
  loops now initialise `J_Omega_*_txy` from the incident field instead of zeros.
- **New state:** `rho_mid` (x, y) and `rho_e` (G, x, y) in both loops. The full path keeps
  `rho_mid_txyz`, `rho_e_gtxyz`. The lean path keeps the pool's two-slot z buffer (float32) for the
  absorption window, and the centre pixel of the last plane for outputs.
- **Recorder hook (lean path only).** If `X.movie_recorder` is set, it gets
  `record_step(it, Omega_it, rho_phys_ijxy, rho_sat_phys_ijxy, ground, other, 2s, mid, rho_e_gxy)`
  after every step and `end_plane(iz, Omega_pstxy)` after every plane. It only reads, so the numerics
  are identical with or without it. The full path raises if a recorder is attached.
- **Not changed:** the readout off-by-one (`Omega_pstxyz_zlast` is taken after N−2 of N−1 steps).

### `eii.py` (new)

Pure functions, evaluated at `__init__` from the `eii:` block.

- `bcf_cross_section_nm2`: Burgess–Chidichimo cross section.
- `speed_nm_fs`.
- `stopping_power_eV_nm`: Joy–Luo, Cu, J = 322 eV, k = 0.83.
- `build_ladder`: log-spaced groups, `k_down_fs`, per-subshell n σ v, birth groups (defaults: photo
  7090, KLL 7100, LMM 870, CK 60 eV). Subshell binding energies default to XATOM neutral Cu
  (`DEFAULT_SUBSHELLS`).
- `ladder_yields`: the single-electron yield test.

### `tools.py`

- **`compute_run_outputs`** adds `base_pop_t_last`, `sat_pop_t_last` (ff828c4), `rho_mid_t_last`,
  `n_e_hot_t_last` and `n_e_total_t_last`. All are the last plane's centre pixel.
  `total_population_t_last` includes the pool.
- **`PATHWAY_EXTENSION_KEYS`**, `_EXTENSION_INACTIVE_VALUE`, `_EXTENSION_REQUIRED_FEATURE`,
  **`active_pathway_extensions(config)`** and **`verify_code(X, repo_root)`**, moved here from
  `run_intensity_sweep.py` and extended (§1 item 4).

### `movie.py` (new, 2297546)

`MovieRecorder` writes every (t, x, y, z) point of one shot to HDF5, one plane at a time: field, flux,
populations per level and per satellite manifold, middlemen, electron groups, and base coherences.
Driven by `scripts/run_movie.py`.

### `population_budget.py` (new, uncommitted)

`PopulationBudgetRecorder(X, weights_xy, stride)`: the sweep-sized counterpart of the movie recorder.

- **Channels:** flux; ground; other; 2s; middlemen; base 2p₃/₂ / 1s / 2p₁/₂; each satellite block's
  three manifolds; each electron group. Every channel except flux is a population per atom.
- **Views:** the centre pixel and the incident-fluence-weighted (x, y) average, per z plane, every
  `stride` steps.
- **Helpers:** `incident_fluence_weights(X, seed)`, `channel_names(X)`, and `trace(view)`, which should
  be 1 with middlemen.
- **Check:** on a test shot its last-plane centre channels reproduce `compute_run_outputs` to float32
  round-off (ground, pool, 2s, base and satellite totals, electrons hot and total, trace).

---

## 3. Config

### New keys

| Key | Default | Meaning |
|---|---|---|
| `use_middlemen` | false | middleman pool (physics §2) |
| `middlemen.targets` | required with the pool | `'none'` or {channel: weight}; R: {3d+3d+: 0.3333, 3d−3d+: 0.5333, 3d−3d−: 0.1334} |
| `middlemen.twos_ck` | `auto` | 2s-on-middleman CK fractions {L3, L2}; `auto` = Σ Γ_A,2s / Γ_L₁ |
| `middlemen.sigma_Ka1_total` | ground total | pool photoabsorption cross section |
| `L2_CK_feed: {rate_eV, targets}` | off | base 2p₁/₂ → satellite 2p₃/₂ CK; R: 0.374 eV, {3d+: 0.6, 3d−: 0.4} |
| `GammaA_L1_to_L2eVN` | 0 | 2s → bare base 2p₁/₂ (R: 0.0508 eV). Its 2p₃/₂ counterpart is the existing `GammaA_L1_to_L3M45eVN`, which R sets to 0.0903 eV (0 in the double-satellite configs) |
| `L3_sublevel_mixing_fs_inv`, `L3_sublevel_mixing_satellite_fs_inv` | 0 | mixing rate in the base / satellite blocks |
| `L3_sublevel_mixing_coherence_factor` | 1 | 1 = Lindblad, 0 = population exchange only |
| `sublevel_raman_dephasing_fs_inv` | 0 | extra damping of the 2p–2p and 1s–1s coherences |
| `use_eii` | false | EII ladder (physics §4) |
| `eii: {n_groups, E_top_eV, E_bottom_eV, subshells, birth_energies_eV, stopping, spatial_factor, M_shell_scale}` | 6, 7100, 30, XATOM, XATOM, Joy–Luo, 0.5, 0 | ladder definition |
| channel `sigma_Ka1_from_2p_to_L2` | 0 | base 2p₃/₂ → channel 2p₁/₂ photoionisation feed |

Used but not new: `additional_dephasing` (extra optical dephasing in every block; the broadening
lever of the coherence sweep).

Every key above that changes the model is in `tools.PATHWAY_EXTENSION_KEYS`. `...coherence_factor`
needs `mixing_coherence_factor`, and `sublevel_raman_dephasing_fs_inv` needs `raman_dephasing`.

### Config files

| File | Change |
|---|---|
| every `config/base/*double-satellite*.yaml` and `Cu-seed-SASE-duration.yaml` | 3p± parent widths restored to totals, commented (physics §1) |
| `Cu-seed-{SASE,mono-SASE}-middlemen.yaml` | double satellite + middlemen + L2 CK + 2s → bare-2p CK |
| `Cu-seed-{SASE,mono-SASE}-middlemen-eii.yaml` | the same + EII: **the reference model R** |
| `Cu-seed-{SASE,mono-SASE}-middlemen-eii-dLL.yaml` | R + the `2s2p` and `2p2` channels (from `xatom/double_L_hole_parameters.py`) |
| `Cu-seed-SASE-movie.yaml` | fine-grid movie config |

`xatom/double_L_hole_parameters.py` (new) generates the dLL channel entries. XATOM ΔSCF gives the
detunings and widths. Feeds are XATOM Auger (KLL) rates and XATOM subshell branching fractions times
the config's own further-ionisation totals.

---

## 4. Outputs

| Output | Where | Contents |
|---|---|---|
| sweep `.npz` | `run_{intensity,mono}_sweep.py` | as before, plus `rho_mid_t_last`, `n_e_hot_t_last`, `n_e_total_t_last`, `base_pop_t_last`, `sat_pop_t_last` (last plane, centre pixel, sum/sumsq over reps). `rho_*_t_last_sat` are Tijs-weighted dipole contractions, **not** populations |
| `*.provenance.txt` | next to every sweep, record and budget output | `verify_code` string |
| movie `.h5` | `run_movie.py` | everything at every (t, x, y, z) point of one shot (`movie.py` docstring) |
| population record `.npz` | `run_population_record.py` | per shot, centre pixel, every plane: flux, ground, other, 2s, base diagonal. For the analytic two-level comparison, so extensions are refused unless `--allow-extensions` |
| population budget `.npz` | `run_population_budget.py` | shot mean and std of every population channel (centre and beam view, every plane, ~0.04 fs); transmitted spectra and integrated T; electron group edges; EII rate table; weights; worst trace deviation; config and provenance |

---

## 5. Scripts: run families

All families follow generate → submit → run. Generators write `config/generated/<family>/manifest.txt`
(`<variant> <yaml>` lines) and print the exact `sbatch` lines. Data lands in
`data/<family>_<array job id>/<variant>/`.

| Family | Generator | Submit (`$1` = family dir) | Runner | Plot |
|---|---|---|---|---|
| pathway (dsat → mid → mid-eii → mixing) | `generate_pathway_sweeps.sh` | `submit_pathway_sweep_{sase,mono}.sh` | `run_{intensity,mono}_sweep.py` | `plot_pathway_sweep.py` |
| bracket (one change at a time around R) | `generate_bracket_sweeps.sh` (`SASE_GRID` env override) | same | same | `plot_bracket_sweep.py` |
| B1 (double-L-hole absorbers) | `generate_b1_sweeps.sh` | same | same | `plot_bracket_sweep.py` |
| coherence (Raman dephasing, broadening) | `generate_coherence_sweeps.sh` | same | same | (to write when the data arrive) |
| population budget | `generate_population_budget.sh` | `submit_population_budget.sh` (env `NREP`, `CONFIGS_PER_TASK`) | `run_population_budget.py` | `plot_population_budget.py` |
| movie | — | `submit_movie.sh` | `run_movie.py` | `render_movie.py`, `render_movie_3d.py` |
| population record | — | `submit_population_record.sh` | `run_population_record.py` | (analytic comparison, outside this repo) |

Supporting changes:

- `generate_intensity_sweep_configs.py` / `generate_mono_sweep_configs.py` accept `--set KEY=VALUE`,
  refusing keys missing from the base config.
- `derive_config.py` (new) writes variant base configs with a `derived:` record. Transforms:
  `spot_scale`, `foil_um`, `detunings_zero`, `eii_spatial`, `mixing=rate,scope,coh`, `raman`,
  `extensions_from`, `set`.
- `submit_pathway_sweep_{sase,mono}.sh` take the family directory as `$1`. Without it they silently
  run the pathway manifest.
- Other new plots and renders: `plot_ridgeline_fluence.py`, `plot_fwhm_comparison.py`,
  `render_level_diagram.py`.

---

## 6. Validation done

| Check | Result | Where |
|---|---|---|
| Reduction: every new flag off vs the pre-extension code | T to 6 digits; worst relative output difference 4.8×10⁻¹⁶ | middlemen plan, Results |
| Trace with middlemen | [0.99970, 1.00003] at dt = 0.015 fs; converges with dt | middlemen plan |
| Full vs lean, all extensions | 4×10⁻¹⁰ (spectrum), 6×10⁻⁸ (pool) | middlemen plan |
| Maxwell–Bloch shot vs rate-equation estimator, each extension | every row within 0.004 in T | middlemen plan, Results table |
| EII single-electron yields | 0.108 / 0.052 / 0.035 / 4.80 / 63.1 (2p₃/₂ / 2p₁/₂ / 2s / 3p / 3d) vs continuous 0.109 / 0.053 / 0.035 / 4.7 / 63 | EII plan |
| Mixing kernel | coh = 1 matches the Lindblad formula to 4×10⁻¹⁵; coh = 0 leaves off-diagonals untouched; trace preserved | session test |
| J(z=0) fix A/B | T −0.0016 at 60 µJ with 30 planes, −0.0034 with 15 (scales with dz); full = lean | memory note, commit 472cd62 |
| dLL feeds | construction-time conservation check passes; R reproduced to 10⁻⁷ after the `pi_feed_pattern` change | B1 sweep |
| Raman dephasing, 0D kernel | photons per 2p₃/₂ hole 0.39 → 0.60 at 10 fs⁻¹ (RE 0.64); weak-field line unchanged to 2×10⁻⁴; mask only on 2p–2p and 1s–1s in 6- and 8-level blocks | session test |
| Stale-code refusal | a Model without `raman_dephasing` refuses a Raman config | session test |
| Population budget recorder | last-plane centre channels = `compute_run_outputs` to float32 round-off; trace 1±10⁻³ on the test grid | session test |

---

## 7. Pitfalls

- **Pre-fix data.** Double-satellite data from before 15 September overstate the dip (feed bug), and
  data from before 16 September have an un-ionised first plane.
- **Grid.** The SASE base configs are 3×3 (0.67× on-axis fluence). The coherence and population-budget
  families use 5×5, which needs `--mem=64G`.
- **"other".** With `use_middlemen`, "other" is only drained, so compare old "other" with the pool, not
  with "other".
- **Satellite outputs.** `rho_l3/K/l2_t_last_sat` in sweep outputs are not populations. Use
  `sat_pop_t_last`, the movie, or the population budget.
- **Exit plane.** Sweep population outputs are the exit plane only, where the pulse is attenuated. Use
  the population budget for depth-resolved or beam-averaged values.
- **Recorders.** They work only on the lean path (`keep_z_history = False`).
