# Part IX — Evaluating the EII model, electron-impact dephasing, and two bugs

Written 27 September 2026, after the progression runs (`data/progression_*_2491144x`,
`scripts/plot_progression.py`). The model as built is explained in
`docs/theory-eii-electron-ladder-explained.md` (Part VIII). This part checks each ingredient against
references and adds three options:

- the Bote–Salvat cross section;
- electron-impact dephasing;
- core-hole EII.

It also records two bugs that are older than EII. The run family `scripts/generate_gap_sweeps.sh` tests
all of it (§9).

Every number comes from `XLO_sim/eii.py`, the saved progression data, or a single-atom kernel
integration. No XLO_sim run was made.

---

## 0. Summary

| Ingredient | Verdict | Size | Change |
|---|---|---|---|
| L-shell cross section (Burgess–Chidichimo) | **35–45% low** against Bote–Salvat at 3–8 keV | 2p₃/₂ EII per primary 0.109 → 0.172 (+58%) | `eii.cross_section: bote_salvat` |
| M-shell cross section (3s, 3p) | BCF ~15% high | little on the dip | same key |
| 3d (valence) | BCF consistent with the inelastic mean free path | — | none (Bote–Salvat has no 3d for Cu) |
| stopping power (Joy–Luo) | = Bethe to 1–6% above 2 keV, where the 2p holes are made | — | none |
| ladder (24 levels) | 93% of the exact in-pulse 2p EII | — | none |
| `spatial_factor` 0.5 | geometry gives 0.44–0.69 | ±30% on the EII effect | none; 0.5 is central |
| EII of core-holed atoms (M shell) | **missing**. At 20 µJ, about half of all base 2p holes take a valence hit before they decay | moves holes into satellites | `eii.core_hole_EII` |
| electron-impact dephasing | **missing**. Rate ≈ valence collision rate, 0.06 → 1.1 → 1.6 fs⁻¹ at 1/20/30 µJ (pulse peak) | +0.07 → 1.4 → 2.1 eV Lorentzian FWHM; partial dark-state removal | `eii.dephasing` |
| triple-hole states | not worth a block yet | ≲1% of atoms, ≲3% on the dip | none (§7) |
| **satellite line positions** | **mirrored about Kα1** since the satellites were built | 3d satellites sit 1.6–3.4 eV on the wrong (blue) side | `satellite_detuning_sign_fix: true` |
| transmitted-field readout | read after 13 of 14 absorption steps | absorbance −7% everywhere | `read_field_after_last_plane: true` |

All of these default to off, so every earlier config gives the same result to round-off. The kernel
changes reorder floating-point sums by ~1e-16.

---

## 1. The ladder and the stopping power

The discretisation was checked in Part VIII §7. With 24 anchored levels the model makes 93% of the 2p
EII holes that exact slowing-down makes during a 6 fs pulse, against 82% with 6 levels.

The stopping power is the Joy–Luo modified Bethe formula. Above 2 keV, where every 2p EII hole is made,
it equals the Bethe formula to 1–6%. At 10 keV it gives 12.9 MeV cm² g⁻¹, in line with the tabulated
collision stopping power of Cu. Below 1 keV it is 18% above Bethe by construction (Bethe fails there),
which only affects the M shell. Energy closes to 15% (Part VIII §5).

**Verdict:** the ladder is not the weak point.

## 2. Cross sections

Part VII took the Burgess–Chidichimo (BCF) formula from van den Berg et al. (PRL 120, 055002, 2018). It
was built and validated for ions in dense plasmas, which is what the M shell of a hot, ionised focus
becomes. For the L shell of a nearly neutral atom hit by a keV electron, the standard is Bote, Salvat,
Jablonski & Powell (At. Data Nucl. Data Tables 95, 871, 2009). That is an analytic fit to
distorted-wave-Born (overvoltage U ≤ 16) and plane-wave-Born (U > 16) calculations, and is the reference
the measured K/L/M data are compared with (Llovet et al., J. Phys. Chem. Ref. Data 43, 013102, 2014).
The Cu coefficients are copied from NIST's `BoteSalvatICX.jl` into `eii.BOTE_SALVAT_CU`; the Python
port reproduces that implementation exactly.

σ in nm², BCF / Bote–Salvat:

| E (eV) | 2p₃/₂ | 2p₁/₂ | 2s | 3s | 3p |
|---|---|---|---|---|---|
| 1500 | 5.6e-6 / 6.5e-6 (0.86) | 2.6e-6 / 3.0e-6 (0.88) | 1.3e-6 / 1.3e-6 (0.98) | 6.7e-5 / 5.6e-5 (1.18) | 3.6e-4 / 3.1e-4 (1.16) |
| 3000 | 5.9e-6 / 8.6e-6 (0.68) | 2.8e-6 / 4.1e-6 (0.69) | 1.8e-6 / 2.4e-6 (0.78) | 4.1e-5 / 3.6e-5 (1.15) | 2.1e-4 / 1.8e-4 (1.14) |
| 5000 | 4.6e-6 / 7.8e-6 (0.60) | 2.2e-6 / 3.7e-6 (0.60) | 1.6e-6 / 2.3e-6 (0.69) | 2.8e-5 / 2.4e-5 (1.15) | 1.4e-4 / 1.2e-4 (1.14) |
| 7000 | 3.8e-6 / 6.7e-6 (0.56) | 1.8e-6 / 3.2e-6 (0.57) | 1.3e-6 / 2.0e-6 (0.67) | 2.2e-5 / 1.9e-5 (1.15) | 1.1e-4 / 9.3e-5 (1.14) |

Per 7.09 keV primary on the 24-level ladder:

| | 2p₃/₂ | 2p₁/₂ | 2s | 3s | 3p | 3d |
|---|---|---|---|---|---|---|
| BCF | 0.109 | 0.053 | 0.035 | 0.89 | 4.75 | 62.9 |
| Bote–Salvat | 0.172 | 0.082 | 0.049 | 0.75 | 4.00 | 62.9 |

Kα1-type holes per absorbed photon from EII go from 0.131 to 0.202, i.e. **+39% on f instead of +25%**
(Part VI Eq. VI.2). In the progression runs, BCF L-shell EII added ~12% to the saturated Kα1 depth
(3-dsat → 4a), so expect ~19% with Bote–Salvat.

There is no Bote–Salvat 3d for Cu: it is a valence band. BCF's 3d cross section at 7 keV
(1.17 × 10⁻³ nm²) is 65–75% of the total inelastic cross section 1/(n λ) ≈ (1.6–1.8) × 10⁻³ nm² implied
by the 6–7 nm inelastic mean free path (Part VII §2). The remainder is plasmons and 4s. That is a
sensible share, so the 3d stays BCF.

## 3. Transport: the spatial factor

Holes made by EII land where the electron is, not where it was born. With a Gaussian beam (FWHM 100 ×
170 nm) and a Gaussian lateral displacement σ_k per axis, the probe weights EII holes by
√(2σ²/(2σ² + σ_k²)) per axis, relative to no transport:

| σ_k (nm) | 30 | 50 | 70 | 85 |
|---|---|---|---|---|
| overlap | 0.86 | 0.69 | 0.54 | 0.44 |

Part VII §6.1 puts the median 2p EII event 100–150 nm of path from its origin, with transport mean free
paths of 40–100 nm. That gives σ_k ≈ 50–85 nm and a factor of 0.44–0.69. The 0.5 in use sits at the
cautious end. This is not worth a run of its own: the bracket's "eii1" (factor 1) closed 9% of the gap.

## 4. EII feeds that are missing

What the ladder electrons do at present:

| Target | L shell | M shell |
|---|---|---|
| neutral atom | → bare base L3/L2/2s holes | → middleman pool (`M_shell_scale`; 0 in the reference) |
| middleman | → its routed satellite (double-3d) | stays a middleman |
| **atom with a core hole** (base, single, double satellite) | not modelled (~0.5% of 2p holes) | **not modelled** |

So EII never feeds the satellite ladder from an existing hole. Physically it should: a valence collision
of a 2p-holed atom leaves 2p⁻¹3d⁻¹, which is exactly the 3d single satellite, and a 3d single hit again
becomes a double.

How fast is it? The valence collision rate per atom is R = Σ_g n_g · s n (σ₃ₛ + σ₃ₚ + σ₃d) v_g. It can be
read off the 4a budget run (beam centre, foil average):

| pulse energy | R at the pulse FWHM start / peak / end (fs⁻¹) |
|---|---|
| 1 µJ | 0.008 / 0.056 / 0.10 |
| 5 µJ | 0.040 / 0.28 / 0.49 |
| 20 µJ | 0.16 / 1.08 / 1.94 |
| 30 µJ | 0.23 / 1.60 / 2.87 |

A base 2p₃/₂ hole decays at 0.93 fs⁻¹. At 20 µJ, therefore, about half the base holes made after the
pulse peak take a valence hit first.

Whether that hit leaves a lasting spectator is the metal question of Part VII §6.3. On a *core-holed*
site the answer leans atomic. The 2p hole pulls the 3d level down (Z+1: the site looks like Zn), so a
3d vacancy there is more localised than on a neutral site. `eii.core_hole_EII` implements the atomic
picture:

- **Loss.** A block loses scale · R_tot of every level. The loss is frozen at the start of each step,
  as the ground's EII loss is, and the coherences decay with it.
- **Routing.**
  - A 3d hit sends base holes to 3d+ : 3d− = 6 : 4 (3d₅/₂ : 3d₃/₂ statistical weight), sublevel by
    sublevel.
  - A 3p hit sends them to 3p+ : 3p− = 4 : 2.
  - A 3d single hit by 3d goes to the doubles by the electrons left in each j-shell (3d+: 5 : 4;
    3d−: 6 : 3).
- **Everything else goes to the middleman pool:** 3s hits, 3p hits on satellites, and a third spectator
  on a double.

The bookkeeping identity (lost = fed + to middlemen) holds to 2 × 10⁻¹⁶. The δ electrons of these
collisions are not added to the ladder, because core-holed atoms are a few percent of all targets.

## 5. Electron-impact dephasing

**The physics.** The Kα transition energy of an ion depends on its outer-shell configuration: each 3d
hole shifts it by ~0.85 eV and each 3p hole by ~2.8 eV (XATOM). A free electron that ionises or excites
the M shell of a core-holed ion therefore shifts the 1s–2p frequency by Δ for as long as the
disturbance lasts, τ. In the impact limit, where collisions are short and independent (the same
approximation as Griem's electron-impact line broadening), each collision multiplies the coherence by
e^{iφ} with φ = Δτ/ħ. The coherence then decays at

$$
\gamma_{coll} = R\,\langle 1-\cos\varphi\rangle \equiv \kappa R ,
\tag{IX.1}
$$

where R is the valence collision rate of §4. It grows with the number of electrons present, i.e. with
fluence and through the pulse. That is exactly the **fluence-dependent width** the coherence sweep
found missing: a fixed dephasing only adds a constant width, while the measured SASE width grows from
6.2 eV at 2 µJ to 14.1 eV at 40 µJ.

The pure-dephasing rate adds a Lorentzian of FWHM 2ħγ. With κ = 1 that is 0.07 / 0.37 / 1.4 / 2.1 eV
at the pulse peak for 1/5/20/30 µJ, and twice that by the end of the pulse.

**How big is κ?**

- **Atomic picture** (the spectator hole stays): φ is effectively random, so κ = 1. This is also what
  core-hole EII does, through population transfer instead of pure dephasing.
- **Metal picture** (a single 3d vacancy delocalises in τ ≈ 0.3 fs): φ(3d) = 0.85 eV × 0.3 fs / ħ =
  0.39 rad, so 1 − cos φ = 0.075. A 3p hit leaves a lasting 3d⁻² after super-Coster–Kronig decay, so
  κ ≈ 1 there. Weighted by the rates (3d ≈ 90% of R, 3p ≈ 8%, 3s ≈ 2%), κ ≈ 0.17.

κ therefore lies between ~0.2 and 1, and the runs use the upper end. If κ = 1 cannot produce the
measured width, this mechanism cannot either.

**The Raman half.** A linearly polarised field parks the 2p₃/₂ hole in a dark superposition of
sublevels (Part VI §1.5); that is the ceiling on photons per hole. A valence collision re-couples the
2p hole to a changed open shell. Through 2p–3d exchange (~1–2 eV) that rotates the sublevels, so
collisions also damp the 2p–2p and 1s–1s (Raman) coherences.

`eii.dephasing: {optical: κ_o, raman: κ_r}` adds κ_o R to every 1s–2p coherence and κ_r R to every
2p–2p and 1s–1s coherence, in every block. It is a per-pixel, time-dependent version of
`additional_dephasing` and `sublevel_raman_dephasing_fs_inv`. The phenomenological raman10 (10 fs⁻¹,
constant) raised the saturated mono Kα1 depth by 45%. Here the rate is 1–3 fs⁻¹ and only present at
high fluence, so expect a smaller, fluence-dependent share of that.

**Not included:** the slow electrons (the bin below 16.5 eV). In the metal they are a hot electron gas,
and the core-hole line shape (the Doniach–Šunjić edge singularity) smears with the electron temperature.
Their collision rate with the ion would be large, but Pauli blocking and screening make a per-electron
cross section meaningless. Modelling it needs T_e(t) from the deposited energy (§10).

## 6. Bug: satellites absorb on the wrong side of Kα1

A feature sits at −(f_upper − f_lower) in the code's frame. The base block sets f[2p₁/₂] = −19.93 eV, and
Kα2 lands at 8028 eV (verified empirically in 2026-09 and by the stage-1 data). The satellite blocks set
f[1s + X] = +detuning_eV (`XLO_sim.py`, `f_local = Delta_fs * ei_K_sat`). A satellite whose config says
d = E_sat − E_Kα1 (theory doc §13, `xatom_tools.satellite_detuning_eV`) therefore absorbs at Kα1 − d.
Its Kα2 partner absorbs at Kα1 − split, instead of Kα1 + d − split.

**Single-atom check** (`_MB_nlevel_regular_core`, weak CW field, anchored on the base Kα2 at −20.00 eV):

| block | config: Kα1-type / Kα2-type (eV) | model, old | model, `satellite_detuning_sign_fix` |
|---|---|---|---|
| 3d+ | −0.80 / −22.07 | +0.75 / −21.25 | −0.75 / −22.00 |
| 3p+ | +2.76 / −18.55 | −2.75 / −21.25 | +2.75 / −18.50 |
| 3d+3d+ | −1.70 / −22.96 | +1.75 / −21.25 | −1.75 / −23.00 |

(0.25 eV scan step.)

**In the data:**
- 4b − 4a turns atoms into middlemen, whose holes go to the double-3d satellites. It removes absorption
  at 8032–8046 eV and adds it at 8050 eV.
- 3-dsat − 2-sat does the same.

Every satellite result so far has had its 3d-type absorption on the blue side. The 3d-type satellites
carry most satellite holes (2s Coster–Kronig into 3d±, L₂ CK, middlemen → doubles). The measured lines
sit ~2 eV *below* the model's, and ~80% of the missing SASE area lies red of the model Kα1. The fix
could therefore matter. It is `satellite_detuning_sign_fix: true` (f[1s + X] = −d, f[2p₁/₂ + X] = −split).

## 7. Triple-hole states

Middlemen (3d⁻ⁿ ions) are already routed into the double-3d satellites. A triple block (2p⁻¹3d⁻³) would
only change where the most-stripped atoms' holes resonate:

- **Population.** In the model without M-shell EII of neutral atoms, triple holes need a second
  photoabsorption by a middleman or a core-hole EII of a double. With core-hole EII at 20 µJ, a double
  (L width 0.57 eV, i.e. 0.87 fs⁻¹) takes another valence hit with probability R/(R + Γ) ≈ 0.5. So the
  triple population is about half the double population: ≲1% of atoms.
- **Leverage.** The doubles added ~7% to the Kα1 depth (2-sat → 3-dsat). Part VI §6.2 found that
  routing the middlemen to the doubles or making them non-resonant changes T(8048) by < 0.006.
  Triples are worth ≲3% on the dip.

Not worth a block now. f-core's population budget shows the flux out of the doubles (their untracked
core-hole EII goes to the middleman pool); build a triple block only if that flux is large.

## 8. Bug: the readout

Both z-marching paths return the field after the second-to-last plane: 13 of 14 absorption steps at
zgrid 15, an effective foil of 18.6 µm (known since 2026-09-16). `read_field_after_last_plane: true`
returns the field after the last plane in both the lean and the full path. Every absorbance grows by
~14/13 = +7.7%.

## 9. The run family (`scripts/generate_gap_sweeps.sh`)

One change at a time from the progression's full model (4a):

| variant | change | question |
|---|---|---|
| a-readout | readout fix | size of the readout bug (~+7.7%) |
| b-sign | + satellite sign fix | do the red-side satellites move the line toward the measured 8046 eV and fill the red wing? |
| c-bs | + Bote–Salvat | L-shell EII +58%: 12% → ~19% on the dip. **The corrected reference** |
| d-deph | + dephasing, optical + Raman, κ = 1 | can collisions produce the fluence-dependent width? |
| e-deph-raman | + Raman half only | how much of d is dark-state removal (d − e = broadening) |
| f-core | + core-hole EII (atomic) | the missing feed: base holes → satellites at 1–3 fs⁻¹ |
| g-core-raman | f + Raman dephasing | everything collisions can do, without counting the optical dephasing twice |

- **Mono:** every variant on the bracket grid (15 energies × 1/5/20/30 µJ, 5×5, 10 reps).
- **SASE:** a, c, d, f, g at 2/9/20/40 µJ, 5×5, 200 reps. SASE shows the width growth best.
- **Population budget:** c, d, f at 8000/8048 eV.

The 24-level ladder is stable on the SASE step too (dt = 0.042 fs, k·dt ≤ 1.4).

## 10. Further ideas, not in this family

- **Pulse duration / flux calibration.** The model needs 2–4× the nominal pulse energy to reach the
  measured onset (Part VI results). Peak flux sets the 2p-hole density in the linear regime. A 3 fs
  bracket of the mono pulse would show how much a shorter pulse buys, but the self-seeded duration
  should be measured, not fitted.
- **2p–3d exchange in the satellites.** Open-shell spectators split the 2p₃/₂ sublevels by 1–2 eV, which
  mixes them at ~1.5–3 fs⁻¹ regardless of the fluence. That is a Raman dephasing on the satellite blocks
  only, with a physical rate (Part VI §1.5). It needs a satellite-only version of the existing key.
- **Hot electron gas.** T_e(t) from the energy in the bin (~30 eV per bin electron), a line-shape
  temperature broadening, and three-body recombination. This is the warm-dense-matter end of §5.
- **Real transport** instead of `spatial_factor` (Part VII implementation plan step 6), only if a
  factor 0.44–0.69 matters.
