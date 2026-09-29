# The model, step by step (final presentation)

This document explains the five steps in which the Cu reverse-saturable-absorption (RSA) model is built up,
the physics each one adds, and three final experiments. It goes with the run family
`scripts/generate_final_sweeps.sh` and the figures of `scripts/plot_final.py`.

The effect sizes quoted below come from the final family itself (`data/final_sweep_mono_24980068`, 28
September: every step with every fix, 25 measured photon energies, 1/5/20/30 µJ, 10 shots). "Kα1 peak" is the
largest absorbance in 8040–8054 eV, "area" the integral over 8012–8072 eV. Where a number comes from an
earlier family, it says so.

| model / experiment | Kα1 peak, 1 / 5 / 20 / 30 µJ | Kα2 peak | area |
|---|---|---|---|
| 1 bare lines | 0.43 / 0.31 / 0.21 / 0.19 | 0.27 / 0.29 / 0.23 / 0.20 | 0.25 / 0.21 / 0.15 / 0.12 |
| 2 + single satellites | 0.63 / 0.50 / 0.37 / 0.33 | 0.39 / 0.48 / 0.42 / 0.38 | 0.43 / 0.37 / 0.28 / 0.24 |
| 3 + double satellites | 0.69 / 0.56 / 0.41 / 0.36 | 0.44 / 0.54 / 0.47 / 0.42 | 0.50 / 0.43 / 0.30 / 0.26 |
| 4 + middlemen, metal CK | 0.76 / 0.62 / 0.47 / 0.43 | 0.44 / 0.55 / 0.50 / 0.46 | 0.53 / 0.45 / 0.33 / 0.29 |
| 5 + free electrons | 0.85 / 0.71 / 0.56 / 0.52 | 0.49 / 0.62 / 0.58 / 0.54 | 0.59 / 0.51 / 0.37 / 0.33 |

(Model on a true 20 µm foil, which absorbs 7–9% more than the measured foil; see §7.)

---

## 0. What is measured, and what "RSA" means here

A 20 µm Cu foil is illuminated by self-seeded (monochromatised) XFEL pulses of 1–30 µJ, tuned across the Kα
lines (8000–8070 eV). The measurement is the transmission T(E). We plot the **absorbance**
A(E) = ln(T_wing/T(E)), the extra absorption relative to the off-resonant wing at 8000 eV.

A cold Cu foil has no resonance at 8 keV: every photon is simply photoabsorbed (mostly by the L and M
shells), and T is flat. RSA appears because the pulse makes its own absorbers:

1. A photon photoionises a 2p electron, leaving an ion with a **2p hole**.
2. This ion can absorb a second photon *resonantly* on the Kα transition, 2p ← 1s. That is Kα emission run
   backwards: a 1s electron is lifted into the 2p vacancy, leaving a **1s hole**.
3. The 1s hole either decays radiatively (Kα emission, back to the 2p-hole state, ready to absorb again) or
   by Auger decay (the ion leaves the cycle).

The number of resonant absorbers grows with the intensity. Near Kα1 (8048 eV) and Kα2 (8028 eV) the foil
therefore absorbs more at high fluence: a dip in T, a peak in A, that deepens with pulse energy.

## 1. The machinery shared by every step

- **Atoms.** A density matrix per ion "block" with every magnetic sublevel. There are four for the 2p₃/₂
  hole, two for 1s and two for 2p₁/₂, with the coherences between them (Maxwell–Bloch equations). The field
  couples 1s ↔ 2p sublevels with Clebsch–Gordan weights. Photoionisation feeds the blocks at σ·J (J the
  photon flux). Radiative and Auger decays empty them.
- **Light.** The field is propagated through the foil in 15 planes. At each plane it is absorbed by
  everything that absorbs (photoionisation of all populations plus the resonant Kα coupling) and diffracts
  (Fresnel).
- **Pulse.** A SASE pulse filtered by a Si(111) double-crystal monochromator: a ~1 eV band at the chosen
  photon energy, 6 fs, 100 × 170 nm focus. Ten shots per point.
- **Output.** The transmitted spectrum after the last plane gives T(E) at each monochromator setting.

## 2. Step 1 — the bare lines (2p₃/₂, 1s and 2p₁/₂ holes)

*Config `Cu-seed-mono-SASE-L2-original.yaml`: the original model plus the 2p₁/₂ hole; original GRASP/RATIP
cross sections.*

**Physics.**

- *Where the holes come from.* A ground-state atom is photoionised into a 2p₃/₂ hole (σ = 1.52 × 10⁻⁷ nm²)
  or a 2p₁/₂ hole (6.9 × 10⁻⁸ nm²). All other photoabsorption (2s, M shell; 3.65 × 10⁻⁷ nm², together
  5.86 × 10⁻⁷ nm²) makes ions that absorb only non-resonantly.
- *Line widths.* The 2p₃/₂ hole lives 1.1 fs (Γ = 0.61 eV) and the 1s hole 0.44 fs (1.49 eV). The Kα1
  resonance is therefore 2.1 eV wide. Kα2 (from 2p₁/₂, Γ = 1.04 eV) sits 19.9 eV lower, in the same density
  matrix, because both share the 1s hole.

**The dark state.** A linearly polarised field couples each 1s sublevel to two 2p₃/₂ sublevels (Δm = ±1)
at once. The field then drives the 2p₃/₂ hole into the one superposition of the two that cannot absorb.
A 2p₃/₂ hole therefore absorbs only ~0.4 photons before it decays, instead of ~0.6 if the sublevels were
independent. 2p₁/₂ has no such dark state.

**Result.** Resonances at 8048 and 8028 eV, but the line saturates by 5 µJ at ~1/5 of the measured
saturated depth (0.43 of the measured Kα1 peak at 1 µJ, 0.19 at 30 µJ). Off resonance the foil gets *more* transparent with pulse energy, which the measurement
does not do. When a 2p hole decays by Auger emission, the ion (now with two holes in the M shell) is simply
dropped from the model, so it stops absorbing.

## 3. Step 2 — 2s holes and single-spectator satellites

*Config `Cu-seed-mono-SASE.yaml`.*

The 2s shell has the largest L-shell photoabsorption (2.45 × 10⁻⁷ nm²). A 2s hole decays in 0.08 fs by
Coster–Kronig decay into a 2p hole *plus* a hole in the M shell (3d or 3p), which is called a spectator.
The spectator shifts the Kα line: by −0.8 eV for a 3d hole, and by +2.8 and +2.9 eV for 3p. Four satellite
blocks (3d±, 3p±, split by the spectator's j) carry these holes, each with its own Kα1- and Kα2-type
resonances. Photoionising the M shell of an ion that already has a 2p hole also feeds them.

**Result.** The biggest single step: Kα1 +48 / 61 / 73 / 78% at 1 / 5 / 20 / 30 µJ, and the area nearly
doubles. Satellite holes become about half of all resonant absorbers.

## 4. Step 3 — double satellites

*Config `Cu-seed-mono-SASE-double-satellite.yaml`.*

A 3p spectator hole decays in ~0.25 fs by super-Coster–Kronig decay (3p → 3d 3d), leaving a 2p hole with
*two* 3d spectators, which is resonant at ~−1.7 eV. Three double-3d blocks keep these atoms resonant
instead of dropping them. The feed is a branch of the parent's own width, which fixed a
population-creating bug in September.

**Result.** Kα1 +9–11%, area +7–16%.

## 5. Step 4 — middlemen and the metal Coster–Kronig decays

*Config `Cu-seed-mono-SASE-middlemen.yaml` (= step 3 + the `use_middlemen`, `middlemen:`, `L2_CK_feed:` and
2s→2p4s blocks). Theory: `docs/theory-middlemen-and-pathway-audit.md` (Part VI); code:
`docs/middlemen-implementation-plan.md`.*

### 5.1 The problem: atoms that vanish

Steps 1–3 track a fixed list of states: the neutral atom, the 2s hole, "other" (an M-shell hole), the bare 2p
and 1s holes, and the satellite blocks. A process whose product is not on the list simply removes the atom
from the calculation, and a removed atom absorbs nothing. That is most processes:

| tracked state | process | what it really makes | share | steps 1–3 |
|---|---|---|---|---|
| 2p₃/₂ hole | Auger decay (the way ~99% of 2p holes decay) | 3d⁻², 3p⁻¹3d⁻¹, 3p⁻², … | ~99% | vanishes |
| 1s hole | KLL Auger | 2p⁻², 2s⁻¹2p⁻¹ → after two L-Auger decays, 3d⁻⁵… | 40% | vanishes |
| 2p₁/₂ hole | Auger; L₂-L₃M₄₅ Coster–Kronig (metal) | M-shell holes; 2p₃/₂⁻¹3d⁻¹ | 64% / 36% | vanishes |
| any core-holed ion | further photoionisation | ion with two core holes | ~4% at 20 µJ | vanishes |
| every satellite | core Auger | 3d⁻³…3d⁻⁶ | ~100% | vanishes |

(XATOM branchings; Part VI §2.2.) At 20–30 µJ a sizeable fraction of the foil is removed this way during the
pulse, so the foil **bleaches**: off resonance it gets *more* transparent with pulse energy. In steps 1–3,
T(8000 eV) rises from 0.381 at 1 µJ to 0.397–0.402 at 30 µJ. The measurement *falls*, from 0.413 to 0.402.

### 5.2 What the atoms really become: middlemen

Whatever the first event, about a femtosecond later the ion is in a **3d⁻ⁿ** configuration (plus 4s holes,
which the metal refills), and it stays there for the rest of the pulse:

- **3d holes are Auger-stable.** Filling a 3d hole from 4s releases too little energy to eject an electron.
- **3p and 3s holes are not.** A 3p hole decays by super-Coster–Kronig (3p → 3d 3d) in ~0.2 fs, and a 3s
  hole in a similar time, so they collapse into extra 3d holes.

The typical number of 3d holes left behind:

| after | mean n |
|---|---|
| a 2p hole's Auger decay | 2.9 |
| a 1s hole's KLL Auger + two L-Auger | ~6 |
| a single satellite's core Auger | 3.9 |
| a double satellite's core Auger | 4.9 |

These are the **middlemen**: ions between the core-hole states and the end of the cascade.

- **They absorb like neutral Cu.** At 8 keV, photoabsorption is dominated by the L shell, which barely feels
  the 3d holes. XATOM gives σ_tot = 5.15 × 10⁻⁷ nm² for the neutral atom and 5.32 × 10⁻⁷ nm² for 3d⁻⁶, and
  the same branching into 2p₃/₂, 2p₁/₂ and 2s to 0.2%.
- **They make resonant holes.** A photon that makes a 2p hole in a middleman leaves 2p⁻¹3d⁻ⁿ: a Kα
  satellite, red-shifted by ~0.9 eV per 3d hole (−0.8, −1.7, −2.65, −3.6 eV for n = 1–4).

### 5.3 How the model does it

One extra population per pixel, the **middleman pool** ρ_mid, sits alongside the blocks:

$$
\frac{d\rho_{mid}}{dt} = \underbrace{\sum_{\text{blocks}}\sum_i \big(\gamma_i^{untracked} + \sigma_i^{untracked}J\big)\rho_{ii}
+ \sigma_M J\,\rho_{ground}}_{\text{everything that used to vanish}}
\;-\; \sigma_{routed}\,J\,\rho_{mid}
$$

- **In.** For each level i of each block, `XLO_sim` precomputes the part of its decay width and of its
  further photoionisation that has *no* tracked destination (γ_i^untracked and σ_i^untracked above). That flow now goes
  into the pool instead of vanishing. The ground state's M-shell photoionisation (the old "other") also goes
  there. A feed is always a branch of the source's own width, never an extra loss, so the total population is
  exactly 1 at every pixel and time (to the time-step error).
- **Absorption.** The pool absorbs with the neutral atom's total cross section, 5.86 × 10⁻⁷ nm².
- **Out, into the satellites.** When a middleman is photoionised:
  - a 2p₃/₂ or 2p₁/₂ hole goes into the **double-3d satellite blocks**, split by the statistical weights of two
    3d holes, 3d₅/₂3d₅/₂ : 3d₃/₂3d₅/₂ : 3d₃/₂3d₃/₂ = 15 : 24 : 6 (`middlemen.targets`);
  - a 2s hole Coster–Kronig decays within 0.08 fs into 2p + spectator, and goes to the same blocks with the
    same fractions as a 2s hole on a neutral atom (`twos_ck: auto`);
  - an M-shell hole just adds 3d holes, so the ion stays in the pool.

One pool lumps every n together, and all of its resonant holes sit in the 3d⁻² blocks at −1.7 eV. In reality
n = 3, 4, 5 resonate at −2.65, −3.6 and −4.6 eV. This is the approximation §9 comes back to.

### 5.4 Two Coster–Kronig decays that the metal opens

These land in blocks that already exist, so they need no new state:

- **2p₁/₂ → 2p₃/₂ + 3d hole (L₂-L₃M₄₅).** In the free Cu atom this decay is energetically closed (XATOM
  Γ(2p₁/₂) = 0.67 eV). In the metal it is open, which is why the measured width, used by the model since
  step 1, is 1.04 eV. The 0.374 eV difference is this channel. Until step 4 that part of the width led nowhere.
  Its product, 2p₃/₂⁻¹3d⁻¹, is exactly the 3d single satellite (split 3d₅/₂ : 3d₃/₂ = 6 : 4). So 36% of 2p₁/₂
  holes turn from Kα2-type into Kα1-type absorbers: about 4% of all photoabsorptions, +9% on the Kα1-type
  holes per photon (`L2_CK_feed`).
- **2s → 2p + 4s hole (L₁-L₂,₃N₁).** XATOM gives 0.090 eV (to 2p₃/₂) and 0.051 eV (to 2p₁/₂) of the 8.13 eV
  2s width. The 4s hole is refilled from the conduction band, so these land on the *bare* 2p holes. That is
  1.7% of 2s holes (`GammaA_L1_to_L3M45eVN`, `GammaA_L1_to_L2eVN`).

### 5.5 Result

- **Kα1 +10 / 11 / 15 / 19%** at 1 / 5 / 20 / 30 µJ, and area +6–13%. The gain grows with pulse energy
  because middlemen are *sequential*: an atom must absorb once to become one, then again to make a satellite
  hole. The Kα2 gain is 0–9%, since the CK feed moves holes from Kα2 to Kα1.
- **The wing changes sign.** T(8000 eV) now *falls* with pulse energy, from 0.379 to 0.362, as measured. The
  size is too large, though: the model's wing absorbance grows by 0.047 from 1 to 30 µJ, the measured one by
  0.028.

## 6. Step 5 — free electrons and electron-impact ionisation

*Config `Cu-seed-mono-SASE-middlemen-eii.yaml` + the `eii:` settings of the generator. Detail:
`docs/theory-eii-electron-ladder-explained.md`, `docs/eii-model-evaluation.md`; the Bote–Salvat formula:
`docs/bote-salvat-and-final-experiments.md` Part A.*

**Physics.** Every photoabsorption releases a 7–8 keV photoelectron, every Auger decay a ~0.9 keV electron
(KLL: 7 keV), and every Coster–Kronig decay a slow one. These electrons ionise other atoms before they stop.
Only electrons above ~1 keV can make 2p holes. A 7 keV electron makes ~0.17 2p₃/₂ holes (Bote–Salvat cross
sections), which adds ~39% to the Kα1-type holes per absorbed photon.

The model follows the electrons on **24 energy levels** from 7.95 keV down to 16.5 eV:

- **Slowing down.** Every collision costs the electron energy, so it steps down the levels at the rate the
  stopping power gives. A 7 keV electron takes ~8 fs to stop. This is energy loss, not thermalisation:
  nothing ever becomes a Maxwellian.
- **Ionisation rate.** Level g ionises an atom at σ(E_g)·Φ_g, where Φ_g = n·n_g·v_g is its electron flux,
  exactly like σ·J for photons.
- **Secondary electrons.** Each ionisation releases a secondary electron, which is followed down the levels
  in turn.
- **Transport.** Half the ionisations are counted inside the focus (`spatial_factor` 0.5), because
  electrons travel ~100 nm before they ionise.

**Cross section.** Step 5 uses the Bote–Salvat cross sections: an analytic fit to quantum (distorted- and
plane-wave Born) calculations, where the old Burgess–Chidichimo formula is a generic semi-classical shape.
For 2p at 3–8 keV Bote–Salvat is 1.5–1.8× larger. The formula and a worked example are in
`docs/bote-salvat-and-final-experiments.md` Part A.

**Result.** Kα1 +11 / 15 / 18 / 20%, area +10–14%. Two things limit it (`figs/eii_cascade_20uJ.png`,
`figs/eii_vs_pulse_energy.png`, from the step-5 population budget):

- **The holes come late.** The electrons that can make 2p holes are the 2.5–7.5 keV ones, and they are still
  building up when the pulse ends: the EII 2p hole rate peaks at the end of the pulse's FWHM. 81% of the
  photoionisation holes are made inside the FWHM, but only 46% of the EII holes.
- **The share is fixed.** EII makes 18% of the direct 2p₃/₂ holes during the pulse (28% counting the 45 fs
  after it), the same at 1 and at 30 µJ. It is linear in the number of absorbed photons, so it scales the
  whole curve by a similar factor. It does not add the extra absorption the measurement shows at high
  fluence.

## 7. Fixes in every step

| Fix | What was wrong | Effect |
|---|---|---|
| true 20 µm foil (`read_field_after_last_plane`) | the transmitted field was read after 13 of 14 absorption steps: an 18.6 µm foil | absorbance +7% |
| satellite line positions (`satellite_detuning_sign_fix`) | every satellite absorbed at the mirror image of its shift, so the 3d spectators sat on the blue side of Kα1 | lines move ~0.7 eV red, toward the measured lines |
| first plane photoionised | the first plane saw no photoionising flux | +4% on the SASE line (September) |
| double-satellite feed | the feed was carved out of the parent's width, which created population | the dip was 0.014 too deep before |
| Bote–Salvat (step 5) | Burgess–Chidichimo is 35–45% low for 2p ionisation at 3–8 keV | EII +58% |

**The foil.** With every step counted, the model's off-resonant transmission of 20 µm Cu is 0.379. The
tabulated Cu attenuation gives 0.388, and the measurement 0.413. The measured foil absorbs about 7–9% less
than 20 µm of Cu would, most likely because it is a little thin (foil tolerances are typically ±10%). The
figures use 20 µm as asked. On a thinner foil, the model's resonant absorbance would be lower by about the
same 7–9%.

## 8. Final experiments (each on top of step 5)

The physics and equations of each are in `docs/bote-salvat-and-final-experiments.md` Part B.

**Results** (`figs/final_experiments.png`), change against step 5 at 1 / 5 / 20 / 30 µJ:

| | Kα1 peak | area | Kα2 peak | valley 8032–8041 eV, model / exp. |
|---|---|---|---|---|
| step 5 | — | — | — | 0.31 / 0.29 / 0.28 / 0.28 |
| E1 dephasing | +0 / 3 / 11 / 14% | +1 / 7 / 21 / 26% | 0 | 0.32 / 0.33 / 0.36 / 0.37 |
| E2 core-hole EII | +1 / 5 / 13 / 16% | +1 / 8 / 22 / 27% | 0 to +3% | 0.32 / 0.34 / 0.37 / 0.38 |
| E3 exchange | +6 / 10 / 13 / 15% | +2 / 4 / 6 / 7% | 0 | 0.31 / 0.30 / 0.30 / 0.30 |

- **E1 and E2 are almost the same curve.** Both broaden the lines at the valence collision rate, so they add
  area in the wings and a little in the valley at 20–30 µJ, and nothing at 1 µJ. E2's population transfer to
  the satellites adds only 1–2% on top. The broadening, not where the atoms end up, is what matters.
- **E3 deepens the Kα1 peak and nothing else**: 6% even at 1 µJ, 15% at 30 µJ, with little area and no Kα2
  change, as expected for pure dark-state removal. The 1 µJ gain shows that the 2p₃/₂ dark state already forms
  at 1 µJ.
- The best case (E2) reaches **0.60 of the measured saturated Kα1 peak, 0.55 of Kα2 and 0.42 of the area**.
  None of them reaches more than 0.38 of the measured valley.

**E1 — collisional dephasing.** A free electron that hits an ion carrying a core hole disturbs its outer
shell, which shifts the Kα transition for a moment and scrambles the phase of the Kα coherence. The rate is
the valence collision rate: 0.06 fs⁻¹ at 1 µJ, ~1 fs⁻¹ at 20 µJ, ~1.6 fs⁻¹ at 30 µJ. That makes the line
broader the more electrons there are. The same collisions rotate the 2p₃/₂ sublevels and dissolve the dark
state. E1 applies the upper bound (every collision randomises the phase).

In the gap family, E1 also gave about 20% more 1s holes per 2p hole. For the SASE line width, though, it
gave only +0.5 eV at 40 µJ, against the +8 eV measured.

**E2 — electron-impact ionisation of ions that already carry a core hole.** In the atomic picture, the
collision leaves a spectator hole:

- a bare 2p hole becomes a single satellite, and a single becomes a double;
- a third spectator is kept in the double block (last time these atoms were thrown into the non-resonant
  pool, which lost 30–37% of all resonant holes at 20–30 µJ);
- collisions also dephase the 2p₃/₂ sublevels.

This is the "electrons feed the satellite ladder" picture done without losing absorbers.

**E3 — 2p–3d exchange in the satellites.** In an ion with an open M shell, the 2p hole couples to the
spectator by exchange; the multiplet splitting is ~1–2 eV. Its sublevels are therefore not stationary: they
precess into each other at ~splitting/ħ, whatever the fluence. That dissolves the dark state for satellite
holes, which are about half of all resonant absorbers. E3 applies 2 fs⁻¹ (a 1.3 eV splitting) of pure
sublevel dephasing to the satellite blocks only; the bare 2p hole has no open-shell partner. It is the
physical, fluence-independent counterpart of E1's Raman half. The phenomenological version on every block at
10 fs⁻¹ raised the saturated depth by 45%.

## 9. Where the model stands

The full model (step 5) reaches:

- **Kα1 peak:** 0.85 of the measured at 1 µJ, 0.52 at 30 µJ;
- **Kα2 peak:** 0.49–0.62;
- **area:** 0.59 at 1 µJ, 0.33 at 30 µJ.

The best experiment (E2) lifts the saturated values to 0.60, 0.55 and 0.42. Three features of the data are
still missing:

- **The broad absorption.** At 20–30 µJ the measurement absorbs from 8010 to 8070 eV and fills the valley
  between the lines. At 8032–8041 eV the measured absorbance is 0.25–0.29; the model has 0.07–0.08, and E1/E2
  raise it to 0.09–0.11. On the blue side (8055–8065 eV) the model has 0.14–0.21 of the measured absorbance at
  20–30 µJ, and on the red side (8015–8022 eV) 0.39–0.41. The measured spectrum gets much broader with pulse
  energy, while the model's shape barely changes.
- **Kα2 is weak even in the linear regime.** At 1 µJ the measured Kα2/Kα1 peak ratio is 0.63; the model's is
  0.36. Nothing nonlinear acts at 1 µJ, so this points at the number or lifetime of the 2p₁/₂-type absorbers
  (the L₂ CK feed of step 4 moves 36% of them to Kα1), or at something that absorbs near Kα2 which the model
  lacks.
- **Line position.** The measured Kα1 centroid (8040–8054 eV) moves 0.26 eV to the blue from 1 to 30 µJ
  (8046.84 → 8047.10 eV). The model's sits 0.2–0.7 eV higher and moves 0.3 eV to the red.

A spread of charge states, which the model lumps into one pool, would explain the broad absorption. Each extra
3d hole shifts the line ~0.9 eV to the red, and each 3p hole or double core hole shifts it to the blue. A pool
resolved in n, with a satellite block for each n, is the natural next step.

## 10. Figures

| Figure | Use |
|---|---|
| `figs/final_steps.png` | 2×2 overview: every step at 1/5/20/30 µJ with the measurement |
| `figs/final_buildup_20uJ_1..5.png` | one slide per step at 20 µJ; earlier steps in grey |
| `figs/final_experiments.png` | step 5 with E1–E3 |
| `figs/eii_cascade_20uJ.png` | step 5's electrons at 20 µJ: populations by level and time, spectrum, which levels make 2p holes, and when |
| `figs/eii_vs_pulse_energy.png` | EII against photoionisation vs pulse energy; the valence collision rate of E1–E3 |
| `figs/eii_scale_*.png` | the L-shell EII scale test (`docs/bote-salvat-and-final-experiments.md` Part C) |

Regenerate with `python scripts/plot_final.py` (it reads the newest `data/final_sweep_mono_*`),
`python scripts/plot_eii_diagnostics.py` and `python scripts/plot_eii_scale.py`.
