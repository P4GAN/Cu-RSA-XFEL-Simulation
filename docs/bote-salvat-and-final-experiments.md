# Part X — The Bote–Salvat cross section and the three final experiments

Written 27–28 September 2026 for the final presentation. It explains three things in more depth than
`docs/final-model-progression.md`:

- the electron-impact ionisation (EII) cross section that step 5 now uses (Bote–Salvat), with its formula;
- the three final experiments E1–E3 of `scripts/generate_final_sweeps.sh`, and their results (B.6);
- a test of whether more EII could match the experiment (Part C, `scripts/generate_eii_scale_sweeps.sh`).

Figure: `figs/eii_cross_sections.png` (`python scripts/plot_eii_cross_sections.py`, formula evaluation only).
Background: `docs/theory-eii-electron-ladder-explained.md` (Part VIII: how the electron ladder works) and
`docs/eii-model-evaluation.md` (Part IX: the evaluation that introduced these options).

---

## Part A — The Bote–Salvat cross section

### A.1 What the model needs a cross section for

The free electrons sit on energy levels E_g (24 of them, 7.95 keV down to 16.5 eV). Level g carries n_g
electrons per atom, so its electron flux is Φ_g = n_at · n_g · v_g (n_at = 84.9 atoms nm⁻³, v_g the speed).
One atom is then ionised in subshell s at the rate

$$
R_s = \sum_g \sigma_s(E_g)\,\Phi_g ,
\tag{X.1}
$$

exactly as a photon flux J photoionises at σ·J. Everything about EII therefore rests on σ_s(E): the
probability, expressed as an area, that an electron of energy E knocks out one electron of subshell s
(2p₃/₂, 2p₁/₂, 2s, 3s, 3p, 3d) of a Cu atom.

The subshells matter at different energies:

- **2p (the resonant holes).** Only electrons above ~1 keV can ionise the 2p shell (binding 945–972 eV).
  The cross section is largest at a few keV (panel a of the figure). Its size there decides how many
  Kα-active holes the photoelectrons make.
- **3s, 3p, 3d.** Every electron above a few tens of eV ionises these. They are the valence collisions of
  experiments E1–E3.

### A.2 The old formula: Burgess–Chidichimo (BCF)

Part VII took the Burgess–Chidichimo formula in the form of van den Berg et al. (PRL 120, 055002, 2018).
Written with the overvoltage U = E/I (I the binding energy) and without continuum lowering, it is

$$
\sigma^{BCF} = \pi a_0^2\,\zeta\left(\frac{\mathrm{Ry}}{I}\right)^2 \frac{1}{U}
\Big[A_{nl}\ln U + B_{nl}\,(1-U^{-1}) + C_{nl}\,(1-U^{-1})^2\Big],
\tag{X.2}
$$

with ζ the number of electrons in the subshell and, for example, A_nl = c₁ + (c₂ + c₃ l)/n. The nine constants
c₁…c₉ are the same for every element and every shell.

This is a semi-classical shape: the ln U / U behaviour of a free-electron (binary-encounter) collision, scaled
by 1/I². The only things it knows about a particular subshell are its binding energy, its occupation and its
n and l. It was built for ions in dense plasmas, where it does well for outer shells.

### A.3 What Bote–Salvat is

Bote, Salvat, Jablonski and Powell (At. Data Nucl. Data Tables 95, 871, 2009) is not fitted to measurements.
It is an analytic fit to a large database of **quantum-mechanical** cross sections, computed by Bote and
Salvat (Phys. Rev. A 77, 042701, 2008):

- for every element from H to Es (Z = 1–99);
- for the K shell and every L and M subshell;
- for electrons and positrons, from threshold to 1 GeV.

The calculation combines two approximations:

- **Plane-wave Born approximation (PWBA).** The projectile interacts with the bound electron once, and
  enters and leaves as a plane wave, unaffected by the atom. This is exact when the projectile is fast. At
  high energy it gives the relativistic Bethe formula. There, most ionisations are *distant* collisions: the
  passing electron's field acts on the atom like a flash of virtual photons. Their strength is therefore set
  by the subshell's **dipole oscillator strength**, the same quantity that sets its photoabsorption.
- **Distorted-wave Born approximation (DWBA).** The projectile's incoming and outgoing waves are computed in
  the atom's own self-consistent potential (Dirac–Hartree–Fock–Slater). That potential attracts and speeds up
  the electron near the nucleus, which raises the cross section near threshold. Exchange is also included:
  the projectile is indistinguishable from the bound electron, so the final state is antisymmetrised
  (direct and exchange amplitudes, with their interference). The DWBA is only computable up to ~25 times the binding energy.

The two are combined into a "corrected PWBA" (Llovet et al., J. Phys. Chem. Ref. Data 43, 013102, 2014,
Eq. 65):

$$
\sigma = \begin{cases}
\sigma^{PWBA} + \big(\sigma^{DWBA} - \sigma^{PWBA}\big)_{\text{Coulomb only}}, & E \le 16\,E_i\\[4pt]
\dfrac{E}{E + b\,E_i}\,\sigma^{PWBA}, & E > 16\,E_i
\end{cases}
\tag{X.3}
$$

The PWBA term includes the transverse (magnetic) interaction; the correction in brackets is computed with
the Coulomb (longitudinal) interaction only. Below 16 E_i, this distortion-and-exchange correction is added
to the PWBA. Above it, the correction is carried
on by a smooth factor whose constant b is fixed by continuity at U = 16.

### A.4 The formula

Bote et al. fit X.3 with two closed forms, one on each side of U = 16. Here U = E/E_i is the overvoltage and
4πa₀² = 0.03519 nm² (a₀ the Bohr radius). This is what `eii.bote_salvat_cross_section_nm2` evaluates.

**Low and intermediate energy, 1 < U ≤ 16** (every 2p ionisation in the foil is in this range, since
16 × 945 eV = 15 keV):

$$
\sigma = 4\pi a_0^2\,\frac{U-1}{U^2}\left[a_1 + a_2 U + \frac{a_3}{1+U} + \frac{a_4}{(1+U)^3} + \frac{a_5}{(1+U)^5}\right]^2
\tag{X.4}
$$

The prefactor (U − 1)/U² makes σ vanish linearly at threshold and fall as 1/U far above it. The bracket
shapes the peak. a₁…a₅ are fitted separately for every element and subshell.

**High energy, U > 16** (relativistic Bethe form):

$$
\sigma = 4\pi a_0^2\,\frac{U}{U+b}\;\frac{A_i}{\beta^2}
\Big\{\big[\ln X^2 - \beta^2\big]\big(1 + g_1/X\big) + g_2 + g_3\,(1-\beta^2)^{1/4} + g_4/X\Big\}
\tag{X.5}
$$

- β = v/c.
- X = p/(m_e c) = √(E(E + 2m_ec²))/(m_ec²) is the projectile momentum in units of m_ec². The combination
  ln X² − β² = ln(β²γ²) − β² is the relativistic Bethe logarithm.
- A_i is the Bethe strength of the subshell, (α²/2)M_i², where M_i² = α²m_ec² ∫ (1/W)(df_i/dW) dW is an
  integral over the subshell's optical oscillator strength df_i/dW. This is the photoabsorption connection
  above.
- g₁…g₄ are small fitted corrections, and U/(U + b) is the continuation factor of X.3.

**Accuracy.**

- *Of the fit.* It reproduces the computed database to about 1% for U > 1.3, and 5% closer to threshold.
- *Of the database against experiment.* Llovet et al. compared it with every set of measured K, L and M
  cross sections that was consistent with others. The overall RMS deviation is 10.9%, and the mean
  deviation 2.5%.
- *Of the older semi-empirical formulas.* The same review finds that the Lotz and Drawin formulas fall
  below the measurements, increasingly with energy and atomic number. Those formulas were fitted to old,
  low-energy data. BCF belongs to that family.

**Cu coefficients** (`eii.BOTE_SALVAT_CU`, copied from NIST's `BoteSalvatICX.jl`). E_i is the
Dirac–Hartree–Fock–Slater binding energy the fit was made with:

| subshell | E_i (eV) | a₁ | a₂ | a₃ | a₄ | a₅ | A_i | b |
|---|---|---|---|---|---|---|---|---|
| 2s (L₁) | 1093.70 | 0.0237 | 1.23e-4 | −0.0273 | 0.0163 | 0.00688 | 4.59e-7 | 1.02 |
| 2p₁/₂ (L₂) | 966.03 | 0.0307 | 1.97e-4 | −0.0370 | 0.0600 | −0.0988 | 8.76e-7 | 0.959 |
| 2p₃/₂ (L₃) | 944.89 | 0.0446 | 2.83e-4 | −0.0537 | 0.0872 | −0.144 | 1.80e-6 | 0.959 |
| 3s (M₁) | 127.93 | 0.149 | 7.06e-4 | −0.181 | −0.197 | 1.08 | 2.02e-6 | 1.69 |
| 3p₁/₂ (M₂) | 87.04 | 0.231 | 8.92e-4 | −0.352 | 0.387 | −0.240 | 2.66e-6 | 1.34 |
| 3p₃/₂ (M₃) | 84.31 | 0.339 | 1.30e-3 | −0.526 | 0.609 | −0.464 | 5.67e-6 | 1.33 |

The g₁…g₄ are in the code. There is no 3d entry: in Cu the 3d is a valence band, and the database covers only
inner shells. The model's 3p is the sum of 3p₁/₂ and 3p₃/₂.

### A.5 A worked example: one photoelectron and a 2p₃/₂ electron

A 2p photoelectron is born at 7.09 keV. For the 2p₃/₂ subshell, U = 7090/944.89 = 7.504. The terms of the
bracket in X.4 are

| a₁ | a₂U | a₃/(1+U) | a₄/(1+U)³ | a₅/(1+U)⁵ | sum |
|---|---|---|---|---|---|
| 0.04460 | 0.00212 | −0.00632 | 0.00014 | −0.000003 | 0.04055 |

and (U − 1)/U² = 0.1155. So

σ = 0.03519 nm² × 0.1155 × 0.04055² = **6.68 × 10⁻⁶ nm²**, against BCF's 3.74 × 10⁻⁶ nm².

As a rate: at one free electron per atom, n_at σ v = 84.9 × 6.68 × 10⁻⁶ × 49.4 fs⁻¹ = 1/(36 fs), against 1/(64 fs)
with BCF.

**One electron slowing down** (figure, panel c). An electron loses energy continuously at the stopping power
S(E), so it makes dN = n_at σ(E) dE / S(E) holes on the way from E + dE to E. From 7.09 keV down to the 2p
threshold:

| holes per 7.09 keV photoelectron | 2p₃/₂ | 2p₁/₂ | 2s |
|---|---|---|---|
| Bote–Salvat, exact slowing down | 0.178 | 0.084 | 0.050 |
| Burgess–Chidichimo, exact slowing down | 0.112 | 0.054 | 0.036 |
| Bote–Salvat, 24-level ladder (Part IX §2) | 0.172 | 0.082 | 0.049 |

The ladder keeps 97% of the exact number. The holes are made fairly evenly all the way down to ~1.5 keV, so
electrons that have had time to slow down still contribute.

### A.6 How far off BCF was, and what that changes

Panel b of the figure shows the ratio of the two:

- **2p₃/₂ and 2p₁/₂.** BCF is 4–8% low just above threshold and 44% low at 7 keV. The gap grows with energy because the
  Bethe (dipole) part, which dominates at U = 3–8, is where BCF's generic shape and Cu's actual 2p oscillator
  strength differ most.
- **2s.** Within 5% up to 1.5 keV, 34% low at 7 keV.
- **3s and 3p.** BCF is 14–22% *high* at 1–8 keV, so Bote–Salvat slightly lowers the valence collision rate. The 3d,
  about 90% of that rate, stays on BCF (no Bote–Salvat entry). BCF's 3d cross section is 65–75% of the total
  inelastic cross section implied by the measured inelastic mean free path, which is a sensible share.

In the model, the Kα1-type holes that EII adds per absorbed photon go from 0.131 to 0.202. EII's share of
the saturated Kα1 depth goes from ~12% to ~17–19%. In the gap family (b-sign → c-bs), the line was 4–6%
deeper and the area 4–5% larger.

### A.7 Caveats

- **Free, neutral atoms.** The 2p shell of an atom in the metal is essentially atomic, and a 2p cross section
  barely changes when the atom has lost a few outer electrons. The 3d is where the solid matters, which is
  why it stays on BCF.
- **Binding energy.** The fit uses its own binding energy (944.9 eV for 2p₃/₂; XATOM gives 951.6 eV). This
  only matters within a few percent of threshold, where σ is small anyway.
- **Transport.** σ says nothing about *where* the hole is made. The `spatial_factor` 0.5 (Part IX §3) is a
  separate, larger uncertainty (±30% on the EII effect).

---

## Part B — The three final experiments

All three sit on top of step 5 (the full model with electrons, every fix, Bote–Salvat). Each asks one
question about what the free electrons do to an ion that *already* carries a core hole. Steps 1–5 only let
electrons ionise neutral atoms and middlemen.

### B.1 The shared ingredient: the valence collision rate

Eq. X.1 applied to the M shell gives the rate at which one given atom is hit in its valence shell:

$$
R(t,x,y) = s \sum_g \Phi_g(t,x,y)\,\big[\sigma_{3s}(E_g) + \sigma_{3p}(E_g) + \sigma_{3d}(E_g)\big]
\tag{X.6}
$$

- s = 0.5 is the `spatial_factor`.
- R changes with time and position, because the electron populations build up during the pulse and are
  largest at the beam centre.
- About 90% of R is 3d, 8% 3p and 2% 3s.

Beam centre, foil average (Part IX §4):

| pulse energy | R at pulse peak (fs⁻¹) | R at end of FWHM (fs⁻¹) |
|---|---|---|
| 1 µJ | 0.06 | 0.10 |
| 5 µJ | 0.28 | 0.49 |
| 20 µJ | 1.08 | 1.94 |
| 30 µJ | 1.60 | 2.87 |

A 2p₃/₂ hole decays at 0.93 fs⁻¹. At 20–30 µJ, therefore, a core-holed ion is about as likely to be hit by an
electron as to decay. At 1 µJ collisions are negligible. That is why all three experiments matter only where
the model falls short: at and above saturation.

**The two kinds of coherence** that collisions can destroy:

- **Optical coherence (1s–2p).** This *is* the Kα line. Its decay rate is the line's half-width, so extra
  decay here broadens the line.
- **Raman coherence (2p–2p between the 2p₃/₂ sublevels, and 1s–1s).** This is the **dark state**. A
  linearly polarised field couples each 1s sublevel to two 2p₃/₂ sublevels, and pumps the 2p hole into the
  superposition of the two that cannot absorb. A 2p₃/₂ hole then absorbs only 0.39 resonant photons before
  it decays, instead of 0.64 without the dark state. Destroying this coherence releases the trapped holes,
  and deepens the line above saturation without broadening it. 2p₁/₂ (Kα2) has no dark state.

### B.2 E1 — collisional dephasing

**Physics.** The Kα frequency of an ion depends on its outer shell: each 3d hole shifts it by ~0.85 eV,
each 3p hole by ~2.8 eV. An electron that ionises or excites the M shell shifts the frequency by Δ for the
time τ the disturbance lasts. The coherence picks up a phase φ = Δτ/ħ.

This is the impact picture of electron line broadening in plasmas: collisions are short and independent,
and arrive at rate R. Averaging the phase factor over a Poisson sequence of kicks gives

$$
\langle e^{i\Phi(t)}\rangle = \exp\!\big[-R\,t\,(1-\langle e^{i\varphi}\rangle)\big]
\;\Rightarrow\; \gamma_{coll} = \kappa R,\qquad \kappa = \langle 1-\cos\varphi\rangle .
\tag{X.7}
$$

(The imaginary part, R⟨sin φ⟩, would be a line shift; it is not included.) κ is how thoroughly one
collision scrambles the phase:

- **Atomic picture, κ = 1.** The spectator hole stays, so φ is effectively random.
- **Metal picture, κ ≈ 0.17.** A single 3d hole spreads into the band in ~0.3 fs, so φ ≈ 0.4 rad.

**In the equations.** E1 adds, in every block, at every pixel and time step:

$$
\dot\rho_{ij}\big|_{coll} = -\kappa_o R\,\rho_{ij}\ \ (\text{1s–2p}),\qquad
\dot\rho_{ij}\big|_{coll} = -\kappa_r R\,\rho_{ij}\ \ (\text{2p–2p, 1s–1s}),
\tag{X.8}
$$

with κ_o = κ_r = 1 (`eii.dephasing: {optical: 1.0, raman: 1.0}`). No population moves.

**What it does:**

- The optical part broadens the line by a Lorentzian FWHM of 2ħκR: 0.07, 0.37, 1.4 and 2.1 eV at the pulse
  peak for 1/5/20/30 µJ, about twice that by the end of the pulse. This is a width that *grows with
  fluence*, which a constant dephasing cannot give.
- The Raman part releases the dark state.

**Earlier result** (gap family, d-deph):

- Saturated Kα1 depth +11–15%, area +20–26%.
- The Raman half alone gave +15–20% on the depth. That is more than both halves together, because
  broadening spreads the same absorption over a wider band.
- The SASE width grew by only 0.5 eV at 40 µJ, against the +8 eV measured. Collisions are an order of
  magnitude too weak to explain the SASE width growth.

**What to look for.** Deeper, broader lines at 20–30 µJ, and no change at 1 µJ. The model's depth, which
drops from 20 to 30 µJ, becomes flat.

### B.3 E2 — electron-impact ionisation of core-holed ions

**Physics.** E1 treats a collision as a phase kick and leaves the atom where it was. But a valence
*ionisation* of an ion with a 2p hole leaves a lasting extra hole: 2p⁻¹ + e⁻ → 2p⁻¹3d⁻¹ + 2e⁻. That is a
**satellite state**, which the model already has, with its own detuned lines.

Will the hole last? On a neutral site of the metal, a 3d hole would be screened and spread into the band.
On a core-holed site the 2p hole pulls the 3d level down: the site looks like the next element, Zn (the
"Z + 1" picture). The 3d hole there is more localised. E2 takes the atomic picture: every valence ionisation
leaves a spectator hole that lasts.

**In the equations.** Every level of a core-holed block empties at rate R, sublevel by sublevel:

$$
\dot\rho_{ii}^{(src)} \supset -R\,\rho_{ii}^{(src)},\qquad
\dot\rho_{ii}^{(dst)} \supset +\sum_{s\,\in\,3p,3d} w^{s}_{src\to dst}\,R_s\,\rho_{ii}^{(src)} .
\tag{X.9}
$$

- The same sublevel i lands in the target block. The collision doesn't change the core hole.
- The coherences of the source decay at R, because both of their levels empty.
- Anything not routed (3s hits) goes to the middleman pool.
- The loss is taken from the populations at the start of each step, so the bookkeeping is exact to 10⁻¹⁶.

**Routing** (the weights w: statistical, from the number of electrons in each j-shell):

| ion hit | by a 3d ionisation | by a 3p ionisation |
|---|---|---|
| bare 2p hole (base block) | 3d₅/₂ : 3d₃/₂ single = 6 : 4 | 3p₃/₂ : 3p₁/₂ single = 4 : 2 |
| 3d₅/₂ single | double 3d₅/₂3d₅/₂ : 3d₃/₂3d₅/₂ = 5 : 4 (electrons left) | same weights |
| 3d₃/₂ single | double 3d₃/₂3d₅/₂ : 3d₃/₂3d₃/₂ = 6 : 3 | same weights |
| 3p singles | doubles 3d₅/₂3d₅/₂ : 3d₃/₂3d₅/₂ : 3d₃/₂3d₃/₂ = 15 : 24 : 6 (pair statistics) | same weights |
| any double (third spectator) | stays in the same double block | same |

Two rows need explaining:

- **A 3p hit on a single, or any hit on a 3p single.** The 3p hole super-Coster–Kronig decays (3p → 3d 3d)
  in ~0.25 fs. The atom therefore ends up with several 3d holes and is put in the double blocks.
- **A third spectator.** A 2p⁻¹3d⁻³ ion would absorb at about −2.5 eV. The nearest block is the 3d⁻² double
  at −1.7 eV. E2 keeps the atom there instead of dropping it: it stays resonant, and its coherence is reset
  by the collision.

**Why this version.** Its predecessor in the gap family (f-core) had no route for a third spectator or a 3p hit
on a satellite, so those atoms went to the non-resonant pool. That lost 30–37% of all resonant-hole time
at 20–30 µJ, and the line got 4% *shallower*. In E2, only the 3s hits (~2% of R) leave the resonant blocks.

**The dephasing that comes with it.**

- The transfer already damps every coherence of the source block at R. That includes the optical one, so
  E2 contains E1's optical broadening (κ_o = 1) automatically. Adding `optical` as well would count it twice.
- E2 adds only the Raman half (κ_r = 1): the collision also rotates the sublevels of the atoms it hits.

This matches the gap family's g-core-raman, which lost atoms and gave +1–2% on the depth and +9–12% on the
area.

**What to look for.** Absorption moves from the bare lines to the satellites: mostly to the red of Kα1
(3d-type, −0.8 to −1.7 eV), partly to the blue (3p-type, +2.8 eV). The lines broaden as in E1. It is the only
experiment that changes *which* absorbers exist rather than how they absorb.

### B.4 E3 — 2p–3d exchange in the satellites

**Physics.** In a bare 2p hole (2p⁻¹ with a closed 3d¹⁰ shell; the 4s electron is delocalised in the metal),
nothing couples to the hole's angular momentum. Its four m_j sublevels are degenerate and stationary, and a
dark superposition, once made, lasts the hole's whole ~1 fs life.

In a satellite (2p⁻¹3d⁻¹, or with two 3d holes) the M shell is open:

- The 2p hole couples to the 3d hole through the Coulomb **exchange** interaction (the Slater integrals
  G¹, G³ between 2p and 3d) and the 3d spin–orbit coupling.
- The stationary states are now multiplet levels |(j₂ₚ j₃d) J M⟩, spread over ~1–2 eV.
- Only the *total* M is conserved, not the 2p hole's own m_j.

A dark superposition of 2p m_j sublevels, with the spectator in whatever state it happens to be in,
therefore precesses into absorbing combinations at frequencies ΔE/ħ ≈ 1.5–3 fs⁻¹. Averaged over the
unobserved spectator, the 2p–2p coherence decays. This happens **at any fluence**, because it is a property
of the ion, not of the electrons.

The model's satellite blocks reuse the bare-hole sublevel structure (spectator approximation). They put each
satellite line at XATOM's configuration-average position, and cannot show this mixing.

**In the equations.** E3 adds pure Raman dephasing of 2 fs⁻¹ (≈ 1.3 eV / ħ) to every 2p–2p and 1s–1s
coherence of the **satellite blocks only** (`sublevel_raman_dephasing_satellite_fs_inv: 2.0`):

$$
\dot\rho_{ij}^{(sat)}\big|_{ex} = -\gamma_{ex}\,\rho_{ij}^{(sat)},\quad i,j\ \text{both 2p or both 1s},\qquad \gamma_{ex} = 2\ \text{fs}^{-1}.
\tag{X.10}
$$

- The optical coherences are untouched, so the line does not broaden.
- The base block is untouched: the bare 2p hole has no open-shell partner.
- The 1s–1s part comes with the existing mask. The 1s–3d exchange is much smaller, but the dark states live
  in the 2p₃/₂ manifold, so this hardly matters.

**Size.** γ_ex = 2 fs⁻¹ is about twice the 2p₃/₂ decay rate (0.93 fs⁻¹), so most of the satellite holes'
dark-state trapping is lifted. Satellite holes are about half of all resonant absorbers.

For comparison:

- The phenomenological Raman dephasing of 10 fs⁻¹ on *every* block raised the saturated depth by 45%.
- A 2 fs⁻¹ sublevel mixing on every block was estimated at +27%.

Expect roughly +10% on the saturated Kα1 depth, nothing at 1 µJ, and no change to Kα2. This is a rough
estimate; the run decides. (Result, B.6: +13–15% at 20–30 µJ, and +6% already at 1 µJ.)

**What it does not include.** The same multiplet splitting also spreads each satellite *line* over ~1–2 eV.
That would add broad absorption around the satellite positions, the kind of absorption the model is missing
between the lines. E3 only tests the sublevel-mixing half; the line spread needs multiplet-resolved satellite
positions (GRASP/JAC).

### B.5 The three side by side

| | E1 dephasing | E2 core-hole EII | E3 exchange |
|---|---|---|---|
| cause | valence collisions scramble the phase | valence collisions leave a lasting spectator | the spectator's exchange field rotates the 2p hole |
| rate | κR, 0–3 fs⁻¹, grows with fluence | R, same | 2 fs⁻¹, constant |
| acts on | every block | every core-holed block | satellite blocks only |
| optical coherence (width) | +2ħR | +2ħR (via the transfer) | — |
| Raman coherence (dark state) | released | released | released in satellites |
| populations | unchanged | moved: bare → single → double | unchanged |
| config | `eii.dephasing` | `eii.core_hole_EII` + `eii.dephasing.raman` | `sublevel_raman_dephasing_satellite_fs_inv` |

None of the three is expected to fill the valley between the lines (8032–8040 eV), where the measured
absorbance is 3–5 times the model's. That remains the main open deficit (`docs/final-model-progression.md` §9).

### B.6 Results (final family, `data/final_sweep_mono_24980068`)

Change against step 5 at 1 / 5 / 20 / 30 µJ (`figs/final_experiments.png`):

| | Kα1 peak | area 8012–8072 eV | Kα2 peak |
|---|---|---|---|
| E1 dephasing | +0 / 3 / 11 / 14% | +1 / 7 / 21 / 26% | 0 |
| E2 core-hole EII | +1 / 5 / 13 / 16% | +1 / 8 / 22 / 27% | 0 to +3% |
| E3 exchange | +6 / 10 / 13 / 15% | +2 / 4 / 6 / 7% | 0 |

- **E1 ≈ E2.** Both broaden the lines at the collision rate (E2 through the transfer), so they add area in the
  wings and a little in the valley at 20–30 µJ, and nothing at 1 µJ. Moving the atoms into the satellites adds
  only 1–2% on top of the broadening. Unlike the gap family's f-core, E2 no longer loses depth, because the
  triple holes stay resonant.
- **E3 deepens Kα1 only**, already by 6% at 1 µJ, with little area and no Kα2 change, as B.4 predicted for
  pure dark-state removal. The 1 µJ gain means the 2p₃/₂ dark state already forms at 1 µJ.
- **Best case (E2):** 0.60 of the measured saturated Kα1 peak, 0.55 of Kα2, 0.42 of the area. The valley
  (8032–8041 eV) reaches at most 0.38 of the measured absorbance.

---

## Part C — Could more EII match the experiment? (`eii.L_shell_scale`)

**The question.** Step 5's L-shell EII is fixed by the Bote–Salvat cross section and the transport factor
`spatial_factor` 0.5. Suppose the model under-counts EII by some factor, whatever the cause. Could enough of
it reproduce the measurement?

**The test.** `eii.L_shell_scale: k` multiplies the three L-shell rows of the EII rate table (the rates that
make 2p₃/₂, 2p₁/₂ and 2s holes, from neutral atoms and middlemen) by k. Nothing else changes: the ladder, the
stopping power, the secondary electrons and the valence collisions stay as in step 5. `scripts/generate_eii_scale_sweeps.sh`
runs k = 2, 4, 8, 16 on the final family's grid, plus population budgets; `scripts/plot_eii_scale.py` plots
them against step 5 (k = 1) and the experiment.

**How far k can physically go.**

- **k ≈ 2 is the most that is defensible.** The cross section is good to ~10% (A.4). The transport factor 0.5
  could be at most 1 (no electron ever leaves the focus; Part IX §3 estimates 0.44–0.69).
- **k = 16 is the energy ceiling.** At ×16, one 7.09 keV photoelectron would make 16 × (0.178 + 0.084 +
  0.050) ≈ 5 L-shell holes. Each hole costs ≥ 945 eV, so that is ~5 of the electron's 7 keV. The stopping
  power doesn't know about the extra holes, so larger k would make holes the electron cannot pay for.

**What to expect.** From the step-5 budget (`figs/eii_vs_pulse_energy.png`), EII makes 18% of the direct
2p₃/₂ holes during the pulse at every pulse energy, and most of its holes arrive at or after the end of the
FWHM.

- ×k multiplies the in-pulse EII holes by k: ×2 gives ~1.2× the holes, ×16 ~4.4×.
- Because the share is the same at every pulse energy, the gain should be largest where the model is least
  saturated: at 1 µJ. The 1 µJ line (already 0.85 of the measurement) should reach the data near ×2 and
  overshoot beyond, while 20–30 µJ stay short.
- The shape should barely change: more holes deepen the same lines. They do not broaden the lines or fill
  the valley.

If this holds, the test shows that no amount of EII fixes the model: the missing absorption is not a missing
number of holes, but a missing *kind* of absorber (the charge-state spread of `final-model-progression.md` §9).
`figs/eii_scale_summary.png` (d) compares the 20 µJ shapes directly.

---

## References

- D. Bote, F. Salvat, A. Jablonski, C. J. Powell, *Cross sections for ionization of K, L and M shells of atoms
  by impact of electrons and positrons with energies up to 1 GeV: Analytical formulas*, At. Data Nucl. Data
  Tables 95, 871 (2009).
- D. Bote, F. Salvat, *Calculations of inner-shell ionization by electron impact with the distorted-wave and
  plane-wave Born approximations*, Phys. Rev. A 77, 042701 (2008).
- X. Llovet, C. J. Powell, F. Salvat, A. Jablonski, *Cross sections for inner-shell ionization by electron
  impact*, J. Phys. Chem. Ref. Data 43, 013102 (2014): the review; Eqs. 65, 87–89 are X.3–X.5 here.
- NIST, `BoteSalvatICX.jl` (github.com/usnistgov/BoteSalvatICX.jl): the coefficient tables used.
- Q. Y. van den Berg et al., PRL 120, 055002 (2018): the Burgess–Chidichimo form of Part VII.
