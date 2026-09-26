# Part VIII — How the free-electron (EII) model works

An explanation of the free-electron model as it is implemented (26 September 2026). It covers what
the energy "ladder" is, the electron-flux formula, which processes make electrons, and what the three
EII variants of the progression runs (`scripts/generate_progression_sweeps.sh`) do. Part VII
(`docs/theory-eii-and-free-electrons.md`) is the original derivation and estimate. This document
explains the model and records the 26 September changes:

- 24 levels instead of 6, with each source born at the top of its own level;
- M-shell photoelectrons born at 7.95 keV;
- no cut-off above the lowest threshold;
- δ electrons followed;
- a fixed-energy option.

Every number below comes from `XLO_sim/eii.py`, nothing else.

---

## 0. Short answers

1. **The "ladder"** is a set of discrete free-electron energy levels, like atomic levels but for
   electrons in the continuum. Each level holds a population, n_g = free electrons per atom with
   energy near E_g, at every (x, y, z, t). An electron moves down one level at a time, at the rate
   its collisions take its energy away. That is **slowing down, not thermalisation**. Nothing in the
   model ever becomes a Maxwellian. (§1–2)
2. **Electron flux of level g:** Φ_g = n · n_g · v(E_g) electrons nm⁻² fs⁻¹. It enters exactly
   like the photon flux J: an atom is ionised in subshell i at rate σ_i(E_g) Φ_g, summed over
   levels. (§3)
3. **Non-radiative decays:** every non-radiative decay the model *tracks* emits exactly one electron
   into the ladder: Auger, Coster–Kronig, KLL/KLM, and every photoabsorption. Missing are the later
   electrons of cascades the model shortcuts straight into the middleman pool. Examples are the two
   L vacancies a KLL decay leaves, and the Auger decay after a further photoionisation of an ion.
   All of the missing electrons are below 1 keV, so none can make a 2p hole. (§4)
4. **Secondaries:** yes, electrons that ionise make more electrons. In the original ladder, each
   EII event that changes an atom's state made one secondary, always born at 60 eV. In the
   progression's 4a/4b, *every* ionisation by a ladder electron (valence included) makes one
   secondary, at an energy drawn from the binary-encounter spectrum. That secondary then slows down
   and ionises in turn. The cascade conserves energy: about one electron per 30 eV deposited. (§5)

---

## 1. The picture: free electrons as energy levels

A photoabsorption at 8048 eV ejects a 2p electron with 7.09 keV. Nothing else in the problem has
that energy. The electron is not bound, so it is not an atomic level, and it moves, so it is not a
static population either. The model keeps only the property that matters for ionisation, its
kinetic energy E, and bins it:

- levels g = 0 … G−1, from high to low energy, each covering an energy interval
  [E_{g+1}, E_g] and represented by its centre E_g^c = √(E_g E_{g+1});
- one extra bin, g = G, for electrons below E_bottom. With E_bottom at the lowest tracked
  threshold (3d, 16.5 eV), those electrons cannot ionise anything, so the bin only counts them.

n_g(x, y, z, t) is a scalar per pixel, in the same units as every other population (per atom). The
sum over all levels plus the bin is the total number of free electrons produced per atom so far.
Atoms have their density matrices and scalar pools; the electrons have this vector of G+1 numbers.

## 2. Why electrons move down, and why that is not thermalisation

A fast electron in solid Cu keeps colliding with bound and valence electrons. Each collision costs
it some energy: a 3d ionisation costs 16.5 eV plus whatever the knocked-out electron carries away,
a collective valence excitation a few tens of eV, a 2p ionisation ~1 keV. The average loss per path length is the stopping
power S(E) (Joy–Luo modified Bethe formula for Cu, `eii.stopping_power_eV_nm`). Along the path,

$$
\frac{dE}{dt} = -S(E)\,v(E).
\tag{VIII.1}
$$

For a 7.09 keV photoelectron that gives 7.4 fs to fall below 1 keV and 8.7 fs to reach 30 eV, over
~290 nm of path. A 0.87 keV Auger electron reaches 30 eV in ~1 fs.

**Thermalisation** is something else. It is the electron *distribution* relaxing to a Maxwellian
through electron–electron collisions among the free electrons, and it is slow for keV electrons. The
model never does that: no level exchanges energy with another except by slowing down, and nothing
ever acquires a temperature. The one element that looked like thermalisation was the bottom bin
("thermalised bin" in the old comments). It is only a counter for electrons that have slowed below
E_bottom. With E_bottom = 16.5 eV (the 3d threshold, used by 4a–4c), anything in it genuinely
cannot ionise a tracked shell any more.

**From Eq. VIII.1 to the ladder.** Many electrons slowing down obey a continuity equation in energy,
∂f/∂t = ∂(S v f)/∂E + (sources), where f(E) is the number per unit energy. Integrate f over level
g: the flux S v f leaves through the level's bottom edge and enters the next level down. Approximating
f as flat across the level gives

$$
\frac{dn_g}{dt} = P_g + k_{g-1}\,n_{g-1} - k_g\,n_g,
\qquad
k_g = \frac{S(E_g^c)\,v(E_g^c)}{E_g - E_{g+1}}
\tag{VIII.2}
$$

(Part VII Eq. VII.3; `Model.MB_electron_regular`). P_g is the production into level g (§4, §5).
1/k_g is the average time an electron spends in level g: for the 24-level ladder, 2.2 fs just below
the 2p photoelectron birth energy, 0.3 fs near 1 keV, 0.03 fs near 20 eV.

**What the discretisation costs.** Each level empties exponentially, so the time an electron needs to
cross n levels is spread (relative spread ≈ 1/√n) about the correct mean. The *total* number of
ionisations along the way comes out right almost regardless of the level count (it is a midpoint
sum of ∫ nσ/S dE). The *timing* does not. With few levels, some electrons stay in the top level far
longer than a real electron stays at those energies, and their 2p holes come after the pulse. That is
why the level count matters (§7).

## 3. Ionisation by the electrons: the electron flux

The ionisation rate of one atom by a beam of particles is always σ × flux. For photons the code uses
Γ = σ J with J in photons nm⁻² fs⁻¹. For the electrons of level g, the flux through an atom is their
number density times their speed:

$$
\Phi_g = n_{e,g}\,v(E_g^c) = n\,n_g\,v(E_g^c)
\qquad[\text{electrons nm}^{-2}\,\text{fs}^{-1}],
\tag{VIII.3}
$$

with n = 82.5 atoms nm⁻³ and n_g in electrons per atom. The ionisation rate of subshell i of one
target atom is then

$$
\Gamma_i^{EII} = s \sum_g \sigma_i(E_g^c)\,\Phi_g
= \sum_g \underbrace{s\,n\,\sigma_i(E_g^c)\,v(E_g^c)}_{\texttt{eii\_rate\_table}[i,\,g]}\;n_g
\tag{VIII.4}
$$

(`Model.eii_rates_xy`). σ_i is the Burgess–Chidichimo cross section with XATOM binding energies
(`eii.bcf_cross_section_nm2`, Part VII Eq. VII.1). s is the `spatial_factor`, explained below. The
table's rows are 2p₃/₂, 2p₁/₂, 2s and the M shell (3s + 3p + 3d, times `M_shell_scale`). Like J, the
rates are frozen over one time step.

**Scale at 20 µJ (beam centre, pulse peak).** The photon flux is ~1.3 × 10⁵ nm⁻² fs⁻¹, so the
2p₃/₂ photoionisation rate is 1.52 × 10⁻⁷ × 1.3 × 10⁵ ≈ 0.019 fs⁻¹. About 0.3 fast electrons per atom
at ~4 keV give Φ ≈ 82.5 × 0.3 × 37 ≈ 900 nm⁻² fs⁻¹. That is a flux 140× smaller than the photons',
but σ_2p₃/₂(4 keV) = 5.2 × 10⁻⁶ nm² is 34× larger, so the EII rate is 0.0048 fs⁻¹ (0.0024 with
s = 0.5). EII therefore makes ~10–25% as many 2p₃/₂ holes as the photons do. That is Part VII's
+25% on f before losses.

**Spatial factor s.** A 7 keV electron travels ~150 nm (path) before it has made half its 2p holes,
while the focus is 100 × 170 nm. The model has no electron transport. Instead, s = 0.5 keeps half of
every EII event, and of every secondary electron, inside the pixel (Part VII §6.1).

**Where the new holes go** (`feed_diag_base_block`, `feed_diag_satellite_block`,
`middleman_gain_loss`, `MB_2s_regular`):

| Target | Subshell | Destination |
|---|---|---|
| neutral (ground) atom | 2p₃/₂, 2p₁/₂ | bare base L3 / L2 hole, spread evenly over sublevels (a fast electron picks no m) |
| neutral atom | 2s | the 2s hole, which then Coster–Kronig decays like a photo-made one |
| neutral atom | M shell (× `M_shell_scale`) | middleman pool |
| middleman | 2p, 2s | the satellite blocks the middlemen route to, as for their photoabsorption |
| middleman | M shell | stays a middleman (the pool is not resolved in 3d-hole number) |
| atom that already has a core hole | any | **not modelled** (for the L shell negligible: a 2p hole has a ~0.5% chance of it within its 1 fs life; for the M shell see §8) |

The ground population loses exactly what these destinations gain in each step (its EII loss is
frozen at the start-of-step value), so population stays conserved.

## 4. Where the electrons come from

`Model.electron_production_gxy` adds one electron per event, into the level of its birth energy:

| Event (populations it applies to) | Electron | Birth level |
|---|---|---|
| photoabsorption by *any* population: ground, base and satellite ions, 2s, "other", middlemen | photoelectron, 7.09 keV (2s: 6.95) | `photo` |
| … of the M shell of a ground atom or middleman (when 2s and 2p₁/₂ are tracked separately) | photoelectron, ~7.95 keV | `photo_M` |
| every non-radiative 1s-hole decay, base and satellites (KLL, and the KLM feeds into satellites) | ~7.1 keV | `KLL` |
| every L3/L2 hole decay that isn't a tracked feed (the L–MM Auger), base and satellites | ~0.87 keV | `LMM` |
| the untracked part of the 2s-hole decay (L1–MM) | ~0.87 keV | `LMM` |
| every tracked Coster–Kronig decay: 2s → 2p + spectator, L2 → L3 + 3d, 3p-satellite super-CK into the double satellites | ~60 eV | `CK` |
| every electron-impact ionisation | secondary | §5 |

Radiative decays (Kα₁, Kα₂) emit a photon, not an electron.

**Not counted.** When a decay or ionisation leaves the atom somewhere the model does not track, the
atom goes straight into the middleman pool. Only the first electron of what follows is counted:

- **KLL.** A KLL decay leaves two L vacancies, each of which Auger-decays: two more ~0.9 keV
  electrons per KLL decay (one per KLM). K holes exist only on resonance (~0.05–0.1 per absorbed
  photon at 8048 eV, Part VII §1), so this is a few percent of the primary electrons there, and 0
  off resonance.
- **Further photoionisation of an ion** into an untracked state: the photoelectron is counted, the
  new core hole's Auger electron is not.
- **Middlemen photoabsorption** that isn't routed to a satellite block (the M shell, and the non-CK
  part of a 2s hole): the photoelectron is counted, the Auger/CK electron is not.
- **The M-shell cascade** after an L–M₂₃M₄₅ Auger decay, or after M-shell photoionisation: the 3p
  hole's super-Coster–Kronig electron, 10–40 eV.

All of these electrons are below the 952 eV 2p threshold, so they cannot change the L-shell EII
(4a, 4c). They would only add M-shell ionisation in 4b and raise the low-energy electron counts. The
secondary cascade (§5) produces far more of those anyway.

## 5. Secondary (δ) electrons

Every ionisation releases a second electron. By convention it is the slower of the two outgoing
electrons, with energy W between 0 and (E − I)/2. The model gives W the binary-encounter
(Rutherford-like) spectrum

$$
\frac{d\sigma}{dW} \propto \frac{1}{(W + I)^2},\qquad 0 \le W \le W_{max} = \frac{E - I}{2}
\tag{VIII.5}
$$

(`eii.secondary_fractions`). Its median is about I and its mean is about I (ln(W_max/I) − 1). For a
7.09 keV primary:

| subshell ionised | mean W | median W | W > 16.5 eV (can ionise 3d) | W > 952 eV (can make 2p holes) |
|---|---|---|---|---|
| 2p₃/₂ | 845 eV | 590 eV | 98% | 34% |
| 3p | 243 eV | 82 eV | 84% | 6% |
| 3d | 73 eV | 16 eV | 50% | 1.2% |

**4a/4b (`secondary_spectrum: true`).** Every ionisation of every subshell by every ladder electron
puts one secondary into the level of its energy (`eii.secondary_matrix`, times s). This includes the
63 valence (3d) ionisations per 7 keV primary, whether or not the model changes the atom's state
for them. In a metal the 3d vacancy delocalises, but the δ electron is real either way. The matrix
only feeds levels below the source (W < E/2), so the cascade cannot run away.

- *Energy.* The primary's loss already includes the secondary's energy (it is part of S). The
  secondary spends it again on its own collisions: the energy is handed on, not created. Summed over
  its own events (binding + mean W), a 7.09 keV primary spends 6.0 keV of the 7.07 keV that S takes
  from it between 7.09 keV and 16.5 eV. The Burgess–Chidichimo cross sections and the Joy–Luo
  stopping power agree to ~15%.
- *Effect on the dip.* Fast δ electrons add 3% to the 2p₃/₂ EII (0.113 instead of 0.109 holes per
  primary).
- *Effect on the electron count.* The full cascade makes 232 electrons per 7.09 keV primary (70 with
  s = 0.5), i.e. one per ~30 eV deposited, the textbook mean energy per ion pair.

  **Caution when reading 4a/4b electron populations:** at 20 µJ the front of the foil absorbs ~0.5
  photons per atom. The cascade then puts ~30 electrons per atom into the bottom bin, more than the
  11 valence electrons Cu has. The model never depletes the M shell and has no hot electron gas, so
  below ~100 eV the counts measure deposited energy (per 30 eV), not free electrons that could
  exist. The hot levels (> 1 keV), the ones that make 2p holes, are unaffected.

**The original ladder (`secondary_spectrum: false`, the old reference configs).** One secondary per
state-changing EII event, born at the CK energy (60 eV). With M-shell EII off this changes nothing:
60 eV cannot ionise the L shell. With M-shell EII on, the cascade would be unbounded. A 60 eV
secondary makes ~2–3 3d ionisations, each making another 60 eV secondary. That is why 4b uses the
spectrum.

## 6. Fixed-energy electrons (4c, the bound)

`slowing_down: false` (`eii.build_fixed_levels`) keeps one level per birth energy: 7.95, 7.1
(photo and KLL merged), 0.87 and 0.06 keV. k_g = 0, so an electron keeps its birth energy for the
rest of the window. This is the literal "discrete electron levels that never thermalise or slow
down" model. It is the upper bound, because it drops energy conservation. A 7.1 keV electron
ionises 2p₃/₂ at a constant 1/(nσv) = 1/66 fs forever:

| 2p₃/₂ holes per 7.09 keV electron (s = 1) | total within 35 fs | made while a 6 fs pulse is still there (vs continuous slowing-down) |
|---|---|---|
| continuous slowing-down (exact) | 0.109 | 1.00 |
| old 6-level ladder | 0.108 | 0.82 |
| 24-level ladder (4a) | 0.109 | 0.93 |
| 24-level ladder + δ electrons (4a) | 0.113 | 0.96 |
| fixed 7.1 keV (4c) | 0.533 | 1.02 |

The fixed-energy electron makes 5× more holes in total, but almost all the extra come after the pulse
has gone, when they can no longer absorb. A slowing-down electron's 2p₃/₂ rate stays at ~0.015 fs⁻¹
for its first ~7 fs, i.e. through the pulse. **Expect 4c to transmit almost exactly like 4a.** It will
differ in the populations after the pulse (more L holes, more middlemen), not in the spectrum. In 4c
secondaries go straight to the bin, because at a fixed energy they would ionise for the rest of the
window without ever losing energy.

## 7. Choosing the levels

The original ladder had 6 levels log-spaced from 7.1 keV to 30 eV. Only 2 of them are above the 2p
threshold, the top one spans 7.1–2.9 keV, and every source was placed in the level that *contains*
its birth energy. Three problems followed:

1. **Timing.** With only 2 levels doing all the 2p-hole making, the exponential waits spread the
   holes late: 82% of the in-pulse 2p holes of the exact slowing-down (table above).
2. **M-shell photoelectrons** (20% of photoabsorption, ~7.95 keV) were born in the same level as the
   7.09 keV ones. Their 2p₃/₂ yield is 0.127, not 0.109.
3. **The bottom** was at 30 eV. Electrons of 16.5–30 eV can still ionise the 3d shell, which matters
   once M-shell EII is on.

`anchor_birth_energies: true` (`eii.ladder_edges`) makes every birth energy a level edge (7.95,
7.1, 0.87, 0.06 keV). Each source then starts at the *top* of its own level. The requested number of
levels is shared among the segments between those edges so that the widest level is as narrow as
possible. The tested ladders (2p₃/₂ holes made during the pulse, relative to the exact
slowing-down):

| levels | levels above the 2p threshold | 7.09 keV electron | 7.95 keV electron | fastest k (fs⁻¹) | k·dt, mono (dt 0.00375 fs) |
|---|---|---|---|---|---|
| 6 (original) | 2 | 0.82 | 0.81 (and yield 15% low) | 8 | 0.03 |
| 16 | 6 | 0.90 | 0.93 | 20 | 0.07 |
| **24 (4a/4b)** | **9** | **0.93** | **0.95** | **33** | **0.12** |
| 32 | 12 | 0.95 | 0.96 | 47 | 0.18 |
| 48 | 16 | 0.96 | 0.97 | 67 | 0.25 |

24 levels leave a 7% timing error on a ~10% effect, and cost nothing measurable: 25 scalars per
pixel against 7 satellite density matrices. RK4 stays far inside its stability limit on the mono grid.
(The SASE grid, dt = 0.075 fs, would need k·dt < 2.8, i.e. at most ~24 levels; not used here.)

The 24 levels (centres, eV): g0 7513 (photo_M born here), g1 6227 (photo and KLL born here),
g2–g7 4790 … 1290, g8 992 (straddles the 952–1099 eV L thresholds), g9 761 (LMM born here),
g10–g18 583 … 69, g19 53 (CK born here), g20–g23 41 … 19, then the bin below 16.5 eV.

Levels 1–8 make 2p holes (level 8 straddles the 952–1099 eV thresholds). Everything below only
ionises the M shell.

## 8. The three EII variants of the progression runs

All three are `config/base/Cu-seed-mono-SASE-middlemen-eii.yaml` (stage 3 + EII) with the `eii:` block
changed by `scripts/derive_config.py`:

| variant | ladder | L-shell EII | M-shell EII (neutral → middleman) | secondaries |
|---|---|---|---|---|
| 4a-eii-L | 24 anchored levels, 7.95 keV → 16.5 eV | yes | no (`M_shell_scale` 0) | spectrum, every subshell |
| 4b-eii-LM | same | yes | yes, atomic BCF rate (`M_shell_scale` 1) | same |
| 4c-eii-fixed | 4 fixed levels, no slowing down | yes | no | into the bin |

`spatial_factor` = 0.5 in all three.

What to expect:

- **4a vs the old reference.** The better timing (+13%), the M-shell photoelectrons (+3%) and the δ
  electrons (+3%) make the L-shell EII ~20% stronger. The old EII moved T(8048, 20 µJ) by −0.0055, so
  4a should sit ~0.001 below the old reference. The L-shell EII remains a ~10% effect on the dip.
- **4c vs 4a:** nearly identical spectra (§6). They differ in the post-pulse populations.
- **4b** is the only variant where the low-energy levels matter. Each primary makes ~70 M-shell
  ionisations in the focus (with the cascade and s = 0.5): ~1.7 per atom at 1 µJ at the front of the
  foil, ~8 at 5 µJ. So it should turn a large part of the neutral atoms in the focus into middlemen
  at 1 µJ, and nearly all of them from 5 µJ on, during the pulse. The middlemen's photo-made 2p holes
  then land in the double-3d satellites at −1.7 eV instead of the bare line. So expect 4b to move
  resonant absorption to the red and broaden the dip as the pulse goes on, rather than deepen it.
  Two caveats. It is an upper bound: in the metal a single 3d vacancy delocalises. And M-shell EII of
  atoms that *already* have a 2p or 1s hole (which would turn bare holes into satellites within
  their 1 fs life) is not modelled, so 4b under-represents that half of the effect.

The population-budget runs of the same family record every level at every depth, as centre pixel
and beam average. Which levels matter can be read from them directly. The saved `eii_rate_table`
(and `eii_rates_by_subshell`) times n_g gives each level's ionisation rate per target atom. For the
L shell, levels 1–8 are all of it.

## 9. What the model leaves out

- **Transport.** Electrons stay in their pixel, and `spatial_factor` stands in for leaving the focus.
  Transverse diffusion is sketched in Part VII §9 and the implementation plan's step 6.
- **M-shell occupancy.** Ionising the M shell never empties it (ζ is fixed), and middlemen are not
  resolved in 3d-hole number. That is harmless for the L shell but is why the low-energy electron
  counts exceed what the atoms could supply at high fluence (§5).
- **Thermalisation of the slow electrons and heating.** In reality, electrons below ~50 eV share
  their energy quickly and form a hot electron gas. That gas can ionise and recombine (three-body
  recombination) and lowers the continuum. None of this is modelled: the non-thermal ladder is the
  requested picture.
- **EII of atoms with a core hole** (§3 table).
- **The later electrons of shortcut cascades** (§4).
- **Continuum lowering** (R₀ = 1 in the cross section). It would lower the M-shell thresholds in the
  heated focus.

## 10. Where it is in the code

| What | Where |
|---|---|
| cross section, stopping power, speed | `eii.bcf_cross_section_nm2`, `eii.stopping_power_eV_nm`, `eii.speed_nm_fs` |
| ladder (edges, k_g, rates, birth levels) | `eii.build_ladder` (`eii.ladder_edges` when anchored); fixed levels `eii.build_fixed_levels` |
| secondary spectrum and matrix | `eii.secondary_fractions`, `eii.secondary_matrix` |
| config → tables | `XLO_sim._build_pathway_extensions`, EII section: `eii_rate_table`, `eii_k_down`, `eii_birth`, `eii_secondary_matrix` |
| rates from the ladder (Eq. VIII.4) | `Model.eii_rates_xy` |
| production (§4, §5) | `Model.electron_production_gxy` |
| ladder kinetics (Eq. VIII.2) | `Model.MB_electron_regular` |
| where the holes go | `Model.feed_diag_base_block`, `feed_diag_satellite_block`, `middleman_gain_loss`, `MB_2s_regular`, `MB_ground_regular` |
| one time step, all populations | `Sample._step_increments` |
| outputs | `n_e_hot_t_last` / `n_e_total_t_last` (sweeps, exit plane, centre); every level at every plane: `population_budget.py`, `scripts/run_population_budget.py` |

`eii:` config keys (unknown keys are rejected):

| key | default | meaning |
|---|---|---|
| `n_groups` | 6 | number of ladder levels (slowing-down mode) |
| `E_top_eV`, `E_bottom_eV` | 7100, 30 | ladder range; below E_bottom is the bin |
| `anchor_birth_energies` | false | birth energies as level edges (§7) |
| `slowing_down` | true | false = fixed-energy levels (§6) |
| `secondary_spectrum` | false | true = δ electrons from every ionisation, binary-encounter spectrum (§5) |
| `spatial_factor` | 0.5 | fraction of EII events (and their secondaries) kept in the focus |
| `M_shell_scale` | 0 | M-shell EII of neutral atoms → middlemen, in units of the atomic BCF rate |
| `birth_energies_eV` | photo 7090, photo_M 7950, KLL 7100, LMM 870, CK 60 | per source |
| `subshells`, `stopping` | XATOM neutral Cu; Joy–Luo Cu | overrides |

With the defaults, the configs written before 26 September (`Cu-seed-*-middlemen-eii*.yaml`) give
bit-identical electron production to the code they were run with. That was checked by calling
`electron_production_gxy` from both versions on the same state.
