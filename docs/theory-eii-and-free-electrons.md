# Part VII — Electron-impact ionisation by photo- and Auger electrons

Theory and back-of-envelope estimates only. No code or config was changed. Companion plan:
`docs/eii-free-electrons-implementation-plan.md`. Builds on Part VI
(`docs/theory-middlemen-and-pathway-audit.md`). Read its §1 first: the dip-depth scaling there
(Eq. VI.2) is how every number below is turned into a statement about the dip.

Supersedes the stashed drafts from 2026-08-28 (`git show 'stash@{0}':docs/theory-electron-impact-ionization.md`
and `...:docs/eii-implementation-plan.md`). §8 lists what is kept from them and what changes.

---

## 0. Summary

1. **Electron sources.** Every absorbed photon releases a 7.0–8.0 keV photoelectron. Every L-shell
   hole releases a ~0.9 keV Auger electron. Every 2s hole releases a 10–110 eV Coster–Kronig
   electron. The M-shell cascade adds ~10–40 eV electrons. Each 1s hole (made by resonant absorption)
   adds a ~7.1 keV KLL electron in 40% of cases.
2. **Cross section.** The Burgess–Chidichimo formula from the screenshot (van den Berg et al., PRL 120,
   055002), with XATOM binding energies, reproduces the `estimates-on-role-of-eii.pdf` numbers: at one
   7 keV electron per atom, 1/rate = 65 fs (2p₃/₂), 11 fs (3s), 2.3 fs (3p), 0.21 fs (3d), against the
   slide's 60/10/2/0.25 fs.
3. **"No thermalisation" holds for the shape of the distribution but not for the energy.** A 7 keV
   electron in solid Cu loses its energy in ~8.5 fs over ~290 nm of path (Joy–Luo stopping power). A
   0.9 keV Auger electron does so in ~1.1 fs. With fixed-energy groups, the L-shell EII yield depends
   on the length of the simulation window and comes out about 4× too high over 45 fs. A 4–6 group
   slowing-down ladder (still "a few discrete electron levels", but with downward flow) reproduces the
   continuous yields to within 2–7% (§4).
4. **Yields per 7 keV primary over its whole slowing-down:** 0.109 2p₃/₂, 0.053 2p₁/₂, 0.035 2s,
   0.89 3s, 4.7 3p, 63 3d ionisations. Auger electrons (0.9 keV) are below the L-shell thresholds and
   only ionise the M shell (16 3d each).
5. **Two very different effects on the dip:**
   - **L-shell EII** makes 2p holes without a photon: +0.13 Kα1-type and +0.064 Kα2-type holes per
     absorbed photon, i.e. **+25% on f** (Part VI Eq. VI.2) before losses. Half of it happens more than
     3.5 fs and about 150 nm (path) after the photoabsorption. The focus is 100 × 170 nm and the pulse
     8 fs, so the realised gain on the ln-dip is +10% (spatial factor 0.5) to +19% (no transport
     loss) in the estimator. It is linear in pulse energy, so it scales the whole curve rather than
     adding high-intensity absorption.
   - **M-shell EII** (tens of 3d ionisations per primary) turns neutral atoms into middlemen and base
     2p holes into satellites. It adds no 2p holes, so it cannot deepen the saturated line centre. It
     shifts resonances, so it broadens the dip and can make blue- or red-side absorption depending on
     the Kα shift sign (Part VI §3.4). It grows as F², so it produces threshold-like behaviour. In a
     metal, single valence-3d excitations are not localised, so this is an upper bound (§6.3).
6. **Estimated effect at 8048 eV (rate-equation estimator, absolute T; experiment 0.278 at 20 µJ):**
   starting from the feed-bug-fixed model (0.364 at 20 µJ, 0.367 at 30 µJ), L-shell EII gives
   0.359 / 0.363 (spatial factor 0.5) or 0.355 / 0.358 (factor 1, no transport loss). Combined with
   Part VI (middlemen + L2→L3 CK): 0.335 / 0.329. Adding M-shell EII at 0.5 × BCF moves absorption *off*
   the line centre (0.349 / 0.346, vs 0.346 / 0.341 for middlemen alone), as expected for a pure
   shift mechanism (§6.3).
7. **Verdict.** EII is real and not negligible, but on these numbers it is a 10–20% effect on the
   line-centre dip. It is not the factor ~2.5 the experiment needs. It is worth implementing *after*
   the Part VI middleman ladder, because M-shell EII feeds that ladder and L-shell EII feeds the
   existing blocks. It is mainly worth doing for the shape (broadening, blue-side onset), not the depth.

---

## 1. Where the free electrons come from

Birth energies from XATOM (neutral binding energies: 1s 8969, 2s 1099, 2p₁/₂ 972, 2p₃/₂ 952, 3s 129,
3p₁/₂ 88, 3p₃/₂ 85, 3d 16.5, 4s 8.5 eV; Auger kinetic energies read from the `-decay` tables):

| Source | Birth energy | Per absorbed photon (non-GRASP double-sat config) |
|---|---|---|
| photoelectron from 2s / 2p₁/₂ / 2p₃/₂ | 6949 / 7076 / 7096 eV | 0.42 / 0.12 / 0.26 |
| photoelectron from M shell | 7.9–8.0 keV | 0.21 |
| L₃-MM / L₂-MM Auger | 750–960 eV (mean ≈ 870) | ≈ 0.8 (one per 2p₃/₂, 2p₁/₂ and 2s→2p hole that decays by Auger) |
| L₁-L₂,₃M Coster–Kronig | 12–106 eV | 0.39 |
| M₂₃-M₄₅M₄₅ / M₁ super-CK | 6–95 eV (mostly ≈ 12–40) | ≈ 1–2 |
| KLL Auger | 6.9–7.2 keV | ≈ 0.6 × (resonant absorptions per photon) ≈ 0.05–0.1 at 8048 eV |
| KLM Auger | 8.0–8.2 keV | ≈ 0.01 |

So there are three natural discrete electron families: ~7–8 keV (photoelectrons plus KLL), ~0.9 keV
(L-shell Auger), and ≲100 eV (Coster–Kronig and M cascade). Only the first can ionise the L shell.

---

## 2. Cross section

$$
\sigma_{BCF}(\varepsilon)=\frac{\pi a_0^2}{(I_i/I_H)^2}\frac{\zeta_i I_i'}{R_0^4\,\varepsilon}
\Big\{\Big(c_1+\frac{c_2+c_3l}{n}\Big)\ln\frac{\varepsilon}{I_i'}
+R_0\Big(c_4+\frac{c_5+c_6l}{n}\Big)\Big(1-\frac{I_i'}{\varepsilon}\Big)
+R_0^{r_n}\Big(c_7+\frac{c_8+c_9l}{n}\Big)\Big(1-\frac{I_i'}{\varepsilon}\Big)^2\Big\}
\tag{VII.1}
$$

with {c₁…c₉} as in the screenshot, R₀ = I′/I = 1 (no continuum lowering; this also makes the r_n
exponent irrelevant), ζ_i the *current* occupation of subshell i, and I_i the XATOM binding energy of
the actual ion. The initial charge z enters only through I_i and ζ_i of that ion, so no extra factor
is needed. EII rate per target atom from a group of electrons with density n_e per atom:

$$
\Gamma^{EII}_i = \frac{n_e}{n}\; n\,\sigma_i(\varepsilon)\,v(\varepsilon)
\tag{VII.2}
$$

(n = `X.n` in nm⁻³, σ in nm², v in nm/fs, giving fs⁻¹). With n_e measured as "electrons per atom",
this has the same form as the existing `S_ion_Fi·J` terms.

Values (ζ for metallic 3d¹⁰4s¹; 3p and 3d summed over j):

| Subshell | I (eV) | σ(7 keV) (nm²) | 1/(nσv) at n_e = n, 7 keV | slide | σ(2 keV) (nm²) | σ(0.9 keV) (nm²) |
|---|---|---|---|---|---|---|
| 2s | 1099 | 1.34×10⁻⁶ | 184 fs | — | 1.75×10⁻⁶ | 0 (below threshold) |
| 2p₁/₂ | 972 | 1.83×10⁻⁶ | 135 fs | — | 2.99×10⁻⁶ | 0 |
| 2p₃/₂ | 952 | 3.77×10⁻⁶ | 65 fs | 60 fs | 6.26×10⁻⁶ | 0 |
| 3s | 129 | 2.16×10⁻⁵ | 11.4 fs | 10 fs | 5.5×10⁻⁵ | 9.1×10⁻⁵ |
| 3p | 86 | 1.06×10⁻⁴ | 2.3 fs | 2 fs | 2.9×10⁻⁴ | 5.1×10⁻⁴ |
| 3d | 16.5 | 1.17×10⁻³ | 0.21 fs | 0.25 fs | 3.5×10⁻³ | 6.8×10⁻³ |

Sanity check: the 3d value at 7 keV (1.2×10⁻³ nm²) is close to the *total* inelastic cross section
implied by the ~6–7 nm inelastic mean free path of 7 keV electrons in Cu (≈1.6×10⁻³ nm²). In a metal,
"3d ionisation" by a fast electron is therefore essentially every valence excitation, including
plasmons. That matters for §6.3.

---

## 3. How many free electrons there are

Photoabsorptions per atom at the front surface, beam centre (σ_tot = 5.86×10⁻⁷ nm², focus 100 × 170 nm
FWHM, photon energy 8048 eV):

| E_pulse | peak fluence (photons nm⁻²) | σΦ_peak | energy deposited (eV/atom) |
|---|---|---|---|
| 1 µJ | 4.0×10⁴ | 0.024 | 190 |
| 5 µJ | 2.0×10⁵ | 0.12 | 950 |
| 20 µJ | 8.1×10⁵ | 0.47 | 3800 |
| 30 µJ | 1.2×10⁶ | 0.71 | 5700 |

For comparison, stripping Cu to Cu¹¹⁺ (all of 4s and 3d) costs about 1.3 keV per atom. At ≳10 µJ the
focal volume is heated far past that within the pulse. The question is only how fast the energy gets
into bound-state ionisation.

Because each primary slows down in ~8 fs (§4), the density of ≥1 keV electrons peaks at about
0.3 per atom at 20 µJ (beam centre, pulse peak), against 0.47 if they never slowed down.

---

## 4. The "no thermalisation" assumption

Two different things are often both called thermalisation:

- **Maxwellisation** (electron–electron collisions reshaping the distribution into a Maxwellian): slow
  for keV electrons, tens of fs (the Si calculation on slide 5 of `estimates-on-role-of-eii.pdf`). The
  assumption is fine for this.
- **Slowing down** (energy loss to the lattice's bound and valence electrons, i.e. exactly the EII
  events plus plasmon and valence excitation): fast. From the Joy–Luo stopping power of Cu
  (J = 322 eV, k = 0.83):

  | Birth energy | S (eV/nm) | time to 1 keV | time to 50 eV | path to 50 eV |
  |---|---|---|---|---|
  | 8.0 keV | 13.6 | 8.7 fs | 9.9 fs | 354 nm |
  | 7.1 keV | 14.8 | 7.5 fs | 8.6 fs | 291 nm |
  | 0.94 keV | 50 | — | 1.1 fs | 13 nm |
  | 0.1 keV | 92 | — | 0.1 fs | 1 nm |

A fixed-energy group model with no energy loss therefore gets the L-shell yield wrong in a
window-dependent way. At 20 µJ over the 45 fs window it gives about 0.22 2p₃/₂ EII holes per atom at
the beam centre, against about 0.05 from the slowing-down budget. For the M shell it exhausts the 3d
shell within a few fs. It also breaks energy conservation, since each primary could ionise forever.

**Proposed form, keeping the spirit of "a few discrete levels":** a common log-spaced energy ladder
E₁ > E₂ > … > E_G (for example 6 groups from 8 keV to 30 eV). Electrons are born into the group
containing their birth energy. Group g empties into g+1 at the continuous-slowing-down rate

$$
k_g=\frac{S(E_g)\,v(E_g)}{E_g-E_{g+1}}
\tag{VII.3}
$$

and ionises subshell i at rate Eq. VII.2 while in the group. The last group feeds a thermal bath
(n_th, optionally T_e), which is only needed for M-shell collisional ionisation. Accuracy of this
discretisation against the continuous yields, for a 7.09 keV primary:

| Groups | 2p₃/₂ | 2p₁/₂ | 2s | 3p | 3d | time to leave keV range |
|---|---|---|---|---|---|---|
| 3 | 111% | 112% | 121% | 94% | 94% | 7.8 fs |
| 4 | 99% | 100% | 107% | 97% | 96% | 7.0 fs |
| 6 | 102% | 102% | 104% | 99% | 98% | 7.5 fs |
| 8 | 101% | 101% | 101% | 99% | 99% | 7.6 fs |

Six groups are enough. Each group is one scalar per (x, y) pixel, like `rho_2s`.

Primary depletion: in this ladder the primary loses energy through S(E), which already includes the
energy spent on ionisation. The "non-depleting primary" approximation of the stashed draft is
therefore not needed and should not be used.

---

## 5. Ionisation yields and their timing

Ionisations per primary over the full slowing-down (continuous yields, no transport loss):

| Primary | 2s | 2p₁/₂ | 2p₃/₂ | 3s | 3p | 3d |
|---|---|---|---|---|---|---|
| 2p photoelectron, 7.09 keV | 0.035 | 0.053 | 0.109 | 0.89 | 4.7 | 63 |
| 2s photoelectron, 6.95 keV | 0.034 | 0.051 | 0.106 | 0.88 | 4.7 | 62 |
| M-shell photoelectron, 7.95 keV | 0.042 | 0.061 | 0.127 | 0.99 | 5.2 | 68 |
| KLL Auger, 7.1 keV | 0.035 | 0.053 | 0.109 | 0.89 | 4.8 | 63 |
| L₃-MM Auger, 0.9 keV | 0 | 0 | 0 | 0.11 | 0.75 | 16 |
| L₁ CK, ~0.1 keV | 0 | 0 | 0 | 0 | 0.003 | 3.6 |
| M super-CK, ~35 eV | 0 | 0 | 0 | 0 | 0 | 0.8 |

Energy check: 63 3d ionisations × (16.5 eV binding + secondary kinetic energy) ≈ 7 keV. The 3d number
is really "most of the primary's energy ends up in the valence shell", not a count of on-site vacancies.

Per absorbed photon, EII adds:

- Kα1-type 2p₃/₂ holes: 0.109 + 0.035 × 0.613 (2s→L3 satellites) = **0.130**, against f = 0.516 from
  photoabsorption (**+25%**);
- Kα2-type holes: 0.053 + 0.035 × 0.313 = **0.064**, against 0.248 (**+26%**).

Timing of the 2p₃/₂ EII from a 7.09 keV primary: 25 / 50 / 75 / 90% complete after 1.8 / 3.5 / 5.2 /
6.3 fs, at 83 / 156 / 218 / 250 nm of path, at electron energies 5.8 / 4.4 / 3.0 / 2.0 keV.

---

## 6. What EII does to the dip

### 6.1 L-shell EII (new 2p holes)

The new holes land on whichever atom is hit, mostly neutral atoms. They enter the base L3/L2 or 2s
populations with an even sublevel spread (a fast projectile does not pick a sublevel). Two losses
reduce the +25% gain:

- **Transport.** From a screened-Rutherford estimate, the transport mean free path is ≈100 nm at
  7 keV and ≈40 nm at 2 keV. The first ~100 nm of the path is nearly straight and, for 2s/2p
  photoelectrons (dipole distribution), preferentially along the polarisation, i.e. the 170 nm axis.
  The median EII event is therefore ~100–150 nm from its origin, comparable to the focus. Spatial
  retention: g_sp ≈ 0.4–0.6.
- **Timing.** Half the EII holes appear more than 3.5 fs after the photoabsorption that launched the
  electron. With an 8 fs FWHM pulse, holes made after the peak see less resonant flux. This is
  already inside the estimator, which uses a 5 fs slowing-down clock.

The estimator runs g_sp = 0.5 and 1.

### 6.2 L-shell EII of an already-holed atom

This gives 2p⁻² and similar states: within a 2p hole's 1 fs life, the chance is n_e·nσv·τ ≈
0.3/65 ≈ 0.5%. Negligible.

### 6.3 M-shell EII (spectators and middlemen)

In the atomic picture, at 20 µJ (beam centre, pulse peak), the 3d EII rate per atom is about
0.3/0.21 fs ≈ 1.4 fs⁻¹. A base 2p hole would then acquire a 3d spectator within its own lifetime, and
every neutral atom in the focus would become a middleman before the pulse peak. Even at 1 µJ the
cumulative 3d ionisation is σΦ × 63 ≈ 1.5 per atom at the centre. That is a strong effect, with three
properties:

- It adds no 2p holes, so it cannot raise the saturated line-centre dip (Part VI §1.4).
- It moves resonances by ~1 eV per 3d hole (sign: Part VI §3.4). Because it grows as F², it gives
  threshold-like broadening. That is qualitatively what slides 5–6 show on the blue side (if the shift
  is blue) and also at 1 µJ, where the measured dip is about twice as wide as the model's.
- In metallic Cu, single d-band holes made by a fast electron delocalise in ~0.3 fs and are screened.
  Only multiply ionised sites (on-site U ≈ 8 eV), or a heated, disordered lattice late in the pulse,
  behave atomically. The atomic BCF rate is therefore an upper bound for localised spectator
  production early in the pulse. Treat the M-shell EII strength as a bracketing parameter
  (0 to 1 × BCF).

Two feed shapes are needed, as the stashed draft correctly argued: M-shell EII of a *core-holed* atom
preserves the core hole's sublevel (like `sigma_Ka1_from_2p`), whereas M-shell EII of a neutral atom
or middleman is a scalar ladder step.

### 6.4 Estimates

Rate-equation estimator of Part VI §7. EII is on top of the feed-bug-fixed model; "middlemen" means
the Part VI lumped pool with its satellites at −2.7 or +2.7 eV. T(8048) and T(8028) at 5 / 20 / 30 µJ:

| Variant | T(8048) | T(8028) | line ln-dip at 20 µJ (vs own wing) |
|---|---|---|---|
| feed bug fixed | 0.367 / 0.364 / 0.367 | 0.388 / 0.384 / 0.388 | 0.159 |
| + L-shell EII, spatial factor 0.5 | 0.364 / 0.359 / 0.363 | 0.386 / 0.381 / 0.385 | 0.174 (+10%) |
| + L-shell EII, spatial factor 1 | 0.360 / 0.355 / 0.358 | 0.384 / 0.379 / 0.382 | 0.189 (+19%) |
| middlemen + L2 CK + L-shell EII (0.5) | 0.355 / 0.335 / 0.329 | 0.382 / 0.362 / 0.356 | 0.193 |
| same + M-shell EII (0.5 × BCF), shift −2.7 | 0.361 / 0.349 / 0.346 | 0.384 / 0.370 / 0.365 | 0.153 |
| same + M-shell EII (0.5 × BCF), shift +2.7 | 0.360 / 0.348 / 0.344 | 0.385 / 0.370 / 0.366 | 0.157 |

With M-shell EII, T(8044) at 20 µJ goes 0.361 → 0.355 (red shift), or T(8055) 0.385 → 0.380 and
T(8035) 0.388 → 0.387 (blue shift). The integrated excess absorption does not grow; it drops
slightly, 3.9 → 3.5 eV. That is the behaviour of a redistribution, not a new source of absorbers. The estimator's single lumped shift cannot spread absorption as far as the
experiment's ±10 eV. A resolved ladder with the per-hole shifts of Part VI §3.4 could, if M-shell
stripping is real in the metal.

---

## 7. Consistency checks EII makes possible

- **Electron number:** Σ(holes created) = Σ_g n_g + n_th at every (x, y, t). This is exact
  bookkeeping, so any mismatch is a bug.
- **Energy:** Σ(birth energies) = Σ_g n_g E_g + Σ(ionisation energies spent) + (energy in the thermal
  bath). This is the check the non-depleting design could not pass.
- **Reduction:** all EII cross sections set to 0 reproduces the no-EII run bit-for-bit.
- **Yields:** a single electron injected into an otherwise empty grid reproduces §5's table within the
  discretisation error in §4.

---

## 8. Relation to the stashed 2026-08-28 attempt

Kept:

- the BCF sourcing (XATOM's `-impact_ionization_total_cross_section` crashed; it is not needed);
- n_e as a dimensionless "electrons per atom";
- the sublevel-preserving feed for spectator EII on core-holed atoms;
- the six-production-channel bookkeeping lesson (a config flag can add a production channel);
- the electron-number check.

Changed:

- Fixed birth energies with a phenomenological τ_th = 10 fs and a non-depleting primary are replaced
  by the slowing-down ladder (§4). With τ_th = 10 fs and no depletion, the L-shell yield per primary
  was 10 fs/65 fs ≈ 0.15, roughly right by coincidence; the M-shell yield was unbounded.
- The stashed "Phase B" (n-resolved ground ladder) is the same object as Part VI's middleman ladder.
  Photoionisation cascades and EII should feed one ladder, not two.
- The stash's validated magnitudes came from `Cu-seed-satellite-eii.yaml` (8 × 8 pixels over
  ±1000 nm, 0.1 fs pulse, the grid memory flags as under-resolved), so they are not comparable with
  the mono configs.

---

## 9. Open questions

- The M-shell EII strength in the metal (0 to 1 × BCF). This is the largest uncertainty, and it
  controls the broadening, not the depth.
- Transport: whether a geometric factor is enough or a per-group transverse diffusion term
  (D ≈ vλ_tr/3 ≈ 1700 nm²/fs at 7 keV) is needed. The (x, y) grid pitch is ~36 nm, so diffusion is
  resolvable.
- Continuum lowering (R₀ < 1) in the heated focus lowers M-shell thresholds further. It matters for
  M-shell EII only.
- KLL electrons exist only when the probe is resonant, so L-shell EII feedback is slightly stronger
  on the lines than off them. This is a small (≈10%) correction to the +25%.
