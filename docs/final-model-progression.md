# The model, step by step (final presentation)

This document explains the five steps in which the Cu reverse-saturable-absorption (RSA) model is built up,
the physics each one adds, and three final experiments. It goes with the run family
`scripts/generate_final_sweeps.sh` and the figures of `scripts/plot_final.py`.

Unless stated otherwise, the effect sizes quoted below come from the progression and gap families of 26–27
September (`data/progression_sweep_mono_24911420`, `data/gap_sweep_mono_24933261`); the final family
reproduces them with every fix in place.

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
saturated depth. Off resonance the foil gets *more* transparent with pulse energy, which the measurement
does not do. When a 2p hole decays by Auger emission, the ion (now with two holes in the M shell) is simply
dropped from the model, so it stops absorbing.

## 3. Step 2 — 2s holes and single-spectator satellites

*Config `Cu-seed-mono-SASE.yaml`.*

The 2s shell has the largest L-shell photoabsorption (2.45 × 10⁻⁷ nm²). A 2s hole decays in 0.08 fs by
Coster–Kronig decay into a 2p hole *plus* a hole in the M shell (3d or 3p), which is called a spectator.
The spectator shifts the Kα line: by −0.8 eV for a 3d hole, and by +2.8 and +2.9 eV for 3p. Four satellite
blocks (3d±, 3p±, split by the spectator's j) carry these holes, each with its own Kα1- and Kα2-type
resonances. Photoionising the M shell of an ion that already has a 2p hole also feeds them.

**Result.** The dip roughly doubles: satellite holes become about half of all resonant absorbers.

## 4. Step 3 — double satellites

*Config `Cu-seed-mono-SASE-double-satellite.yaml`.*

A 3p spectator hole decays in ~0.25 fs by super-Coster–Kronig decay (3p → 3d 3d), leaving a 2p hole with
*two* 3d spectators, which is resonant at ~−1.7 eV. Three double-3d blocks keep these atoms resonant
instead of dropping them. The feed is a branch of the parent's own width, which fixed a
population-creating bug in September.

**Result.** About +7% on the line depth.

## 5. Step 4 — middlemen and the metal Coster–Kronig decays

*Config `Cu-seed-mono-SASE-middlemen.yaml`.*

**Nothing leaves the model any more.** Until this step, every decay without a tracked destination removed
the atom from the calculation. Examples are the Auger decay of a 2p hole, the Auger decay of a 1s hole, and
further photoionisation of an ion. The atom simply stopped absorbing. Now all of these land in one pool of
"middlemen": ions with a few 3d holes (3d⁻ⁿ).

- They photoabsorb like neutral Cu; the XATOM cross sections differ by less than 3%.
- Their own 2p photoholes resonate in the double-3d satellite blocks.

With this, the population adds up to 1 at every point and time, and the off-resonant wing *darkens* with
pulse energy, as measured.

Two decays that are closed in the free atom but open in the metal feed existing blocks:

- 2p₁/₂ → 2p₃/₂ + 3d hole (0.37 eV of the 1.04 eV 2p₁/₂ width): a Kα2-type hole becomes a Kα1-type absorber.
- 2s → 2p + 4s hole: the 4s is refilled from the conduction band, so these land on the bare 2p holes.

**Result.** About +15% on the saturated depth, and the wing now behaves as measured.

## 6. Step 5 — free electrons and electron-impact ionisation

*Config `Cu-seed-mono-SASE-middlemen-eii.yaml` + the `eii:` settings of the generator. Detail:
`docs/theory-eii-electron-ladder-explained.md`, `docs/eii-model-evaluation.md`.*

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

**Result.** +12% on the saturated depth with the old Burgess–Chidichimo cross sections, ~+17–19% with
Bote–Salvat. The holes arrive a few fs after the photoabsorption that made the electron, so part of them
miss the 6 fs pulse. EII scales the whole curve rather than adding
absorption only at high fluence.

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

**E1 — collisional dephasing.** A free electron that hits an ion carrying a core hole disturbs its outer
shell, which shifts the Kα transition for a moment and scrambles the phase of the Kα coherence. The rate is
the valence collision rate: 0.06 fs⁻¹ at 1 µJ, ~1 fs⁻¹ at 20 µJ, ~1.6 fs⁻¹ at 30 µJ. That makes the line
broader the more electrons there are. The same collisions rotate the 2p₃/₂ sublevels and dissolve the dark
state. E1 applies the upper bound (every collision randomises the phase).

In the gap family this gave +11–15% on the saturated depth and +20–26% on the area, and about 20% more 1s
holes per 2p hole. For the SASE line width, though, it gave only +0.5 eV at 40 µJ, against the +8 eV
measured.

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

The full model reaches about half the measured saturated Kα1 depth and a third to two fifths of the
absorbance area. The deficit is not mainly the peak heights: the measured spectra at 20–30 µJ absorb
broadly from 8015 to 8065 eV and **fill the valley between the two lines** (A ≈ 0.22–0.30 at 8032–8040 eV,
against 0.05–0.10 in the model). The measured lines also move ~1 eV to the blue with pulse energy.

Both point to a spread of ion charge states that the model lumps together. Each additional 3d hole shifts
the line ~0.85 eV to the red and each 3p hole ~2.8 eV to the blue. A pool resolved in the number of M-shell
holes, with a satellite block for each, is the natural next step.

## 10. Figures

| Figure | Use |
|---|---|
| `figs/final_steps.png` | 2×2 overview: every step at 1/5/20/30 µJ with the measurement |
| `figs/final_buildup_20uJ_1..5.png` | one slide per step at 20 µJ; earlier steps in grey |
| `figs/final_experiments.png` | step 5 with E1–E3 |

Regenerate with `python scripts/plot_final.py` (it reads the newest `data/final_sweep_mono_*`).
