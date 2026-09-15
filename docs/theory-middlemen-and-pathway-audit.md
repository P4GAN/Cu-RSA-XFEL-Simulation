# Part VI — Middleman states and a pathway audit

Theory, audit and back-of-envelope estimates only. No code or config was changed. Companion plan:
`docs/middlemen-implementation-plan.md`. Electron-impact ionisation is covered separately in
`docs/theory-eii-and-free-electrons.md` (Part VII).

Sources used for the numbers (all reproducible, 2026-09-14):

- XATOM (`xatom -s Cu -relativity`, the binary `xatom/xatom_tools.py` already calls): neutral
  photoabsorption at 8047.91 eV, `-decay` tables for `1s1`, `2s1`, `2p1,0`, `2p0,1`, `3s1`,
  `3p0,1`, `3p1,0`, `3d0,1`, `3d0,2`, `3d1,1`, `2p0,1_3d0,1`, `2p0,1_3d0,2`, `2p0,1_3p0,1`,
  `1s1_3d0,1`, and ΔSCF Kα shifts for a set of spectator configurations.
- The saved mono sweep `data/production_sweep_mono_24437094/` (configs `Cu-seed-mono-SASE-original`
  and `Cu-seed-mono-SASE-double-satellite`, 10 shots per point). The user's figure "Double-satellite
  pathways and 2p1/2 holes" is the second of these.
- `data/transmittance_seeded_uJ.csv` and slides 5, 6 and 8 of `docs/Results_RSA_seeded.pdf`.
- A rate-equation estimator written for this audit (§7). It reproduces the saved Maxwell–Bloch
  T(8048 eV) to within 0.01 at 1–30 µJ, and is used only for *differences* between variants.

---

## 0. Summary

1. **The saturated dip has a simple ceiling, and the model sits on it.** Once the Kα resonance is
   saturated (≳2 µJ here), each 2p₃/₂ hole can remove at most η ≈ 0.39 resonant photons
   (half of the 2p₃/₂ sublevels are dark to linearly polarised light). The line-centre dip is then
   ln(T_wing/T_line) ≈ η·f·κL, where f is the number of Kα1-resonant 2p₃/₂-type holes made per
   absorbed photon (0.52 in the current model) and κL ≈ 0.90 is the non-resonant optical depth
   (§1). This reproduces both saved configs to about 20%.
2. **The experiment is far above that ceiling.** At 20–30 µJ the measured line-centre ln-dip is
   0.36–0.40 against the model's 0.11–0.16. The spectrally integrated excess absorption,
   ∫ln(T₀.₁µJ/T)dE, is 14.7 eV at 20 µJ and 18 eV at 30 µJ in the experiment, against 2.1–2.7 eV in
   the model: 6–8× too little, and growing with pulse energy only in the experiment (§1.3).
3. **Audit: nearly every absorbed photon ends in a 3d⁻ⁿ ion within about 1 fs, and the model deletes
   it.** XATOM gives the 2p₃/₂-hole Auger products as 3d⁻² 33%, 3p⁻¹3d⁻¹ 31%, 3p⁻² 26%, 3s-containing
   8%. 3p and 3s holes then collapse to 3d⁻² in about 0.2 fs, and 3d holes have no Auger channel at
   all. Every tracked decay (L3, L2, K, 2s remainder, every satellite) currently has no destination,
   so those atoms stop absorbing (spurious bleaching of the wings) and are never photoionised again
   into satellite 2p holes. These ions are the "middlemen" (§3).
4. **One population-creating bug.** The double-satellite `feed_from` makes 1.1–2.5 daughter atoms per
   decayed 3p± parent, where the physical number is 0.53–0.72. The cause is that the parents' widths
   were reduced by the feed rates. It inflates the current dip (§2.3).
5. **Magnitudes: T at 8048 eV** (rate-equation estimator, §7; single additions are each applied on
   top of "feed bug fixed"):

   | Variant | 5 µJ | 20 µJ | 30 µJ | T(8000) at 30 µJ |
   |---|---|---|---|---|
   | **experiment** (digitised) | 0.320 | 0.278 | 0.277 | 0.403 |
   | current model, saved MB runs | 0.352 | 0.351 | 0.358 | 0.415 |
   | estimator, current model | 0.358 | 0.351 | 0.353 | 0.433 |
   | feed bug fixed (§2.3) | 0.367 | 0.364 | 0.367 | 0.436 |
   | + middlemen, satellite shift −2.7 / +2.7 / 0 eV | 0.363 / 0.363 / 0.362 | 0.346 / 0.346 / 0.344 | 0.341 / 0.340 / 0.338 | 0.406 |
   | + L2→L3M45 Coster–Kronig, f₂₃ = 0.41 (§4) | 0.363 | 0.358 | 0.361 | 0.436 |
   | + L-shell EII, spatial factor 0.5 / 1 (Part VII) | 0.364 / 0.360 | 0.359 / 0.355 | 0.363 / 0.358 | 0.438 |
   | middlemen + L2 CK + L-shell EII (0.5) | 0.355 | 0.335 | 0.329 | 0.406 |
   | same + M-shell EII (0.5 × BCF) | 0.361 | 0.349 | 0.346 | 0.406 |
   | dark-state mixing 2 fs⁻¹ alone (§1.5) | 0.358 | 0.348 | 0.350 | 0.436 |
   | middlemen + L2 CK + L-shell EII + dark-state mixing 2 fs⁻¹ (§1.5) | 0.344 | 0.318 | 0.310 | 0.406 |

   Middlemen are the largest single addition. They remove the unphysical rise of the wings
   (T(8000 eV) stays at 0.406–0.407 instead of rising to 0.436 at 30 µJ) and lower the line T by
   0.02–0.03 at 20–30 µJ. Because the saturated lines are power-broadened to ~7 eV, the sign of their
   Kα shift hardly matters at the line centre; it matters for the shape (§6.2). None of the additions
   closes the gap on its own, because photoionising a middleman produces the same number of 2p holes
   per absorbed photon as photoionising a neutral atom (§1.4). Together, the depth-adding pathways
   (middlemen + L2 CK + L-shell EII) close 34–42% of the Kα1 gap at 20–30 µJ, measured from the
   bug-fixed baseline. With dark-state mixing added, 53–63%. Nothing here touches the Kα2 depth
   (0.36 vs 0.31 measured) or the experiment's near-flat absorption between and beside the lines
   (8015–8065 eV).
6. **Recommendation, in order:** fix the feed bug; add a middleman pool (one scalar is enough for the
   dip depth, a short ladder only for the shape) routed from every current "vanish" channel; add the
   L2→L3M45 Coster–Kronig feed. In parallel, run the cheap dark-state-mixing test (plan step 5). It is
   the one lever found that raises η rather than f, and in the estimator it is as large as the
   middlemen. EII (Part VII) comes after the middleman pool.
7. **What none of this explains:** the experiment's Kα2 depth, and its near-flat absorption over
   8015–8065 eV at ≥20 µJ (T ≈ 0.30–0.33 between and beside the lines, where the model stays at
   0.38–0.40). That needs resonances spread by ±10 eV, and more of them than photoabsorption makes.
   Candidates: M-shell stripping in the heated focus, with per-hole shifts summing to ~10 eV; or
   something outside the 1s←2p picture.

---

## 1. What sets the depth of the dip

### 1.1 The saturated-RSA limit

Take one resonant species with a 2p-hole lower manifold L (degeneracy g_L) and a 1s-hole upper
manifold U (g_U). Holes are made isotropically at rate P per atom. For j=3/2→1/2 (Kα1-type),
g_L = 2g_U. With linear polarisation, only g_U of the g_L lower states couple to the field (quantise
along the polarisation: the π transitions connect m=±1/2 only, so m=±3/2 are dark). For
j=1/2→1/2 (Kα2-type) every lower state is bright.

In the strongly saturated limit (W ≫ Γ), bright-lower and upper populations equalise. Every decay
of U is then replaced by one stimulated absorption, so the photon absorption rate is Γ_U·U. Solving
the bright-pair balance, with Kα1 radiative decay returning half of its population to bright states
(Clebsch–Gordan weights 1/3 + 1/6 from each K sublevel):

$$
\eta_{3/2}=\frac12\,\frac{\Gamma_K}{\Gamma_{L3}+\Gamma_K-\Gamma_{r1}/2}=0.392,\qquad
\eta_{1/2}=\frac{\Gamma_K}{\Gamma_{L2}+\Gamma_K-\Gamma_{r2}}=0.640
\tag{VI.1}
$$

photons per created hole (config widths). If the 2p₃/₂ sublevels were mixed faster than the hole
decays, all six states would share the population and η₃/₂ would rise to 0.643 (§1.5).

Summing over species resonant at the probe, with f_s the number of species-s holes made per absorbed
photon and O_s ≤ 1 its spectral overlap with the probe:

$$
\ln\frac{T_{wing}}{T_{line}}\ \approx\ \kappa_{PI}L\sum_s \eta_s f_s O_s ,\qquad \kappa_{PI}L\approx0.90
\tag{VI.2}
$$

The dip saturates in pulse energy because both the numerator (resonant absorption) and κ_PI scale
with the number of photoabsorption events. Only f, η and O matter; fluence, pulse duration and spot
size drop out once saturated.

### 1.2 Check against the saved runs (20 µJ, 8048 eV)

| Config | f (Kα1-type) | Eq. VI.2 (all resonant) | Saved MB run |
|---|---|---|---|
| `original` (2p₃/₂ only) | 0.259 | 0.091 | 0.073 |
| `double-satellite` | 0.516 + daughters | 0.18–0.20 | 0.158 |

The MB runs sit at about 80% of the ideal ceiling. The remainder is the unsaturated transverse edge,
the far end of the foil, and the pulse wings. Going from `original` to `double-satellite` doubles f
and roughly doubles the ln-dip at all pulse energies (1.9× at 1 µJ, 2.2× at 5 and 20 µJ). The
scaling holds in the linear regime too, since there ln-dip ∝ ⟨ρ₂ₚ⟩ ∝ f.

`κL = 0.90` rather than `nσ_tot·20 µm = 0.967`: the solver reads the transmitted field after 13 of its
14 absorbing z-steps (the off-by-one documented in `Sample._evaluate_n_level_3D_lean`), so the
effective foil is 18.6 µm. This is also why the cold T is 0.407 rather than 0.380. The measured cold
T of 0.415 therefore suggests the configured σ_tot is about 10% high, or the foil is thinner than
nominal; worth checking independently.

### 1.3 What the experiment requires

Slides 5–6 of `Results_RSA_seeded.pdf` (single-shot T vs pulse energy at fixed photon energy) show
three behaviours the current model lacks:

- At the line centres, T keeps falling roughly as log(E_pulse) from 1 to 50 µJ: 0.38 → 0.33 → 0.30 →
  0.28 at Kα1 for 1, 5, 10, 30 µJ. The model is flat above about 5 µJ.
- On the **blue** side of each line (+4 to +15 eV: 8032–8035 and 8052–8065 eV) there is a steeper,
  threshold-like absorption with onset around 3–10 µJ. At 8032–8035 eV it reaches T ≈ 0.30, deeper
  than at Kα2 itself. A steep onset is the signature of a two-step process (absorber ∝ F²).
- Weaker absorption with a higher threshold extends to 8090 eV (up to +40 eV above Kα1).

Quantitatively:

| | Exp. peak ln-dip | Model peak ln-dip | Exp. area (eV) | Model area (eV) |
|---|---|---|---|---|
| 1 µJ | 0.090 | 0.069 | 2.0 | 0.75 |
| 5 µJ | 0.229 | 0.128 | 6.5 | 1.9 |
| 20 µJ | 0.361 | 0.129 | 14.7 | 2.1–2.7 |
| 30 µJ | 0.365 | 0.111 | 18.1 | 1.3–2.4 |

(area = ∫ln(T₀.₁µJ/T) dE over 8000–8100 eV; experiment from the hand-digitised CSV, so ±20% at best.)

Via Eq. VI.2, the experimental ceiling needs η·f ≈ 0.42. With η₃/₂ = 0.39 that is f ≈ 1.1 Kα1-resonant
2p₃/₂ holes per absorbed photon. **No rerouting of photoabsorption can give that**, because each
absorbed photon makes at most one inner-shell hole. Getting there needs holes that cost no photon
(EII, Part VII), a larger η (§1.5), or absorption that is not 1s←2p at all.

### 1.4 Figure of merit for a new pathway

Using Eq. VI.2, any proposed pathway can be scored by Δf, the extra resonant 2p-type holes per
absorbed photon, times its overlap O with the probe. Two classes behave differently:

- **Linear pathways** (extra holes ∝ photons absorbed): L2→L3M45 Coster–Kronig, EII of the L-shell,
  KLM completeness. Their relative effect is roughly independent of pulse energy.
- **Sequential pathways** (a species made by an earlier absorption, then photoionised again:
  absorbers ∝ F²): middlemen, M-shell EII precursors. Their effect grows with pulse energy.

Middlemen are sequential, but the atom that is photoionised a second time has the same branching as
a neutral atom. So at a fixed number of photoabsorption events they add **no** holes. Their effect on
the line-centre dip comes from two places: (a) they keep absorbing, so the number of photoabsorption
events does not bleach away (in the current model deleted atoms absorb nothing); (b) their satellite
holes resonate at shifted energies, which broadens the dip and deepens it wherever the shifts pile
up.

### 1.5 The dark-state ceiling (outside the two proposals, flagged because it is large)

η₃/₂ = 0.39 assumes the m=±3/2 2p₃/₂ holes stay dark for their whole 1 fs life. In an isolated atom
nothing mixes them. In a satellite state with an open 3d shell, however, the 2p–3d exchange
interaction is about 1–2 eV, so the 2p hole's m_j precesses on a sub-femtosecond timescale. With full
mixing η₃/₂ = 0.643 (+64%). Mixing only matters when the bright states are depleted, i.e. above
saturation, so it would lift the ceiling exactly where the model and experiment part ways (≥2 µJ) and
leave 0.1–1 µJ untouched. The satellite blocks reuse the base `Tijs` (spectator approximation), so
the current model cannot show this. A phenomenological test is cheap: a Lindblad dephasing term that
relaxes each L3 manifold towards isotropy at rate γ_mix ≈ 1–2 fs⁻¹ (plan step 5).

Estimator check (γ_mix = 2 fs⁻¹ on every Kα1-type manifold, on the bug-fixed model): T(8048) = 0.382
/ 0.358 / 0.348 / 0.350 at 1 / 5 / 20 / 30 µJ, against 0.385 / 0.367 / 0.364 / 0.367 without mixing.
Kα2 is unchanged (no dark states). That is +27% on the 20 µJ line ln-dip. It is as large as the
middlemen and larger than L-shell EII, from one physical parameter, and it acts only above
saturation as the experiment requires.

---

## 2. Pathway audit

### 2.1 Where one absorbed photon goes (non-GRASP double-satellite config)

| First hole | Share of σ_tot = 5.86×10⁻⁷ nm² | Tracked as | Kα1-type 2p₃/₂ holes per absorbed photon |
|---|---|---|---|
| 2s | 41.8% | `rho_2s` | 0.256 (via L1-L3M Coster–Kronig into the 4 satellites) |
| 2p₃/₂ | 25.9% | base L3 | 0.259 |
| 2p₁/₂ | 11.8% | base L2 | 0 (Kα2-type: 0.118 + 0.131 via 2s) |
| 3s/3p/3d/4s | 20.5% | `rho_other` | 0 |

XATOM's own neutral split at 8047.91 eV is 2s 47.6%, 2p₃/₂ 26.5%, 2p₁/₂ 13.4%, 3s 6.8%, 3p 5.1%,
4s 0.4%, 3d 0.2% (σ_tot = 5.15×10⁻⁷ nm²). The config's "other" (1.20×10⁻⁷) is about twice XATOM's
3s+3p+3d+4s (0.64×10⁻⁷), because it is defined as "paper total minus the tracked shells".

### 2.2 Every outflow of every tracked state

"✗ vanish" means the population leaves the model: it is in no population, absorbs nothing and is
never photoionised again. Branching fractions are XATOM unless stated. Magnitudes are per absorbed
photon, or relative to the parent population. J·σ·τ numbers are at the 20 µJ peak flux
(9.5×10⁴ photons nm⁻² fs⁻¹).

| Tracked state | Process | Physical product | Branch | In model | Verdict |
|---|---|---|---|---|---|
| ground | photoionisation | 2p₃/₂, 2p₁/₂, 2s, M-shell hole | 26/12/42/20% | ✓ | complete (1s is below the K edge) |
| base L3 (2p₃/₂⁻¹) | Auger, Γ_L3 | 3d⁻² 33%, 3p⁻¹3d⁻¹ 31%, 3p⁻² 26%, 3s-containing 8%, 4s 1% → after M-cascade 3d⁻²…3d⁻⁵ | ~99% of decays | ✗ vanish | **missing → middlemen** |
| | L fluorescence | 3d⁻¹, 3s⁻¹ | 1.0% | ✗ vanish | negligible |
| | further PI (σ₂ = 4.15×10⁻⁷) | 2p⁻², 2s⁻¹2p⁻¹ (Kα shifts +27…+37 eV, 0.4 fs lifetimes); 2p⁻¹M⁻¹ | J σ₂ τ_L ≈ 4% of L3 holes | ✗ vanish (M part ✓ via `sigma_Ka1_from_2p`) | low value at mono energies |
| base K (1s⁻¹) | Kα1/Kα2 radiative | L3 / L2 | 26% / 14% (config) | ✓ | |
| | Kβ radiative | 3p⁻¹ → 3d⁻² | 5.4% | ✗ vanish | → middlemen |
| | KLL Auger | 2p⁻² 27%, 2s⁻¹2p⁻¹ 10%, 2s⁻² 3% | 40% | ✗ vanish | → middlemen after L-Auger (resonances at +35 eV irrelevant for mono) |
| | KLM Auger | 2p₃/₂⁻¹M⁻¹ 5.3%, 2p₁/₂⁻¹M⁻¹ 2.6%, 2s⁻¹M⁻¹ 2.1% | 9.9% | 3d±/3p± parts ✓ (`Gamma_A_K_eV`); 2p3s and 2sM ✗ | small |
| base L2 (2p₁/₂⁻¹) | L2-L3M45 Coster–Kronig | 2p₃/₂⁻¹3d⁻¹ = the 3d± satellites' L3k | 0 in the atom; ≈0.36–0.41 in the metal (Γ_L2 1.04 config vs 0.67 XATOM) | ✗ vanish | **missing connection into existing blocks** |
| | Auger | 3d⁻² 34%, 3p3d 30%, 3p⁻² 24%, … | rest | ✗ vanish | → middlemen |
| | L2-L3N1 Coster–Kronig | 2p₃/₂⁻¹4s⁻¹ (Kα shift +0.24 eV) | 2% | ✗ vanish | tiny; effectively a base L3 hole |
| 2s (L1) | L1-L2,3M Coster–Kronig | 2p⁻¹3d⁻¹, 2p⁻¹3p⁻¹ | 92.6% | ✓ satellites | |
| | L1-L2,3N1 Coster–Kronig | 2p⁻¹4s⁻¹ | 1.7% | ✗ vanish | tiny; base-like 2p hole |
| | L1-MM Auger | M-shell double holes | 5.8% | ✗ vanish | → middlemen |
| other (M⁻¹) | own decay | 3s → 3p3d/3p4s/3d4s, 3p → 3d⁻² (both ≈0.2 fs); 3d stable | 100% | **no decay term at all** | "other" *is* the n≈2 middleman pool |
| | PI of 2p₃/₂/2p₁/₂/2s | 2p⁻¹3d⁻ⁿ satellites | 26/12/42% of its PI | ✗ vanish (σ₂_other) | **missing → satellites** |
| single sats L3k | core Auger, Γ_L | 3d⁻³…3d⁻⁶ | ~100% for 3d±; ~25% for 3p± | ✗ vanish | → middlemen |
| | 3p± spectator super-CK | 2p⁻¹3d⁻² (double sats) | 66% (3p+), 69% (3p−) of the physical width | ✓ but **non-conserving** | **bug, §2.3** |
| | M-shell PI of the spectator ion | double sats | J σ τ ≈ 0.6% (3s+3p); 10⁻⁴ (3d) | ✗ | negligible (the "single → double by PI" example) |
| L2k | L2k-L3kM45 Coster–Kronig (metal) | double-sat L3k | as base L2 | ✗ | missing, second order |
| U_k | Kα radiative | L3k / L2k | | ✓ | |
| | Auger | double core, 2p⁻¹3d⁻¹M⁻¹ | | ✗ vanish | → middlemen |
| double sats | core Auger | 3d⁻⁴…3d⁻⁶ | | ✗ vanish | → middlemen |
| middlemen 3d⁻ⁿ | (not tracked) | PI 2p₃/₂ → 2p₃/₂⁻¹3d⁻ⁿ; PI 2s → CK → 2p⁻¹3d⁻⁽ⁿ⁺¹⁾; PI M → 3d⁻⁽ⁿ⁺²⁾ | same fractions as neutral | — | **the main missing pathway** |

Reading the table: the single largest omission is not a missing branch but a missing *sink*. Every
decay channel ends in the same few configurations (3d⁻², 3d⁻³, 3d⁻⁴, …), and those ions keep
absorbing and keep making 2p holes for the rest of the pulse.

### 2.3 Bug: double-satellite `feed_from` creates population

`feed_diag_satellite_block` adds Γ_feed·ρ_parent to each daughter and never subtracts it from the
parent. The parent loses population only through its own `Mij` diagonal, which is the configured
`Gamma_L_eV` / `Gamma_K_eV` / `Gamma_L2_eV`. The double-spectator plan (§3) and theory doc §12.7
reduced those widths by the feed rates ("0.887 + 1.745 = 2.632"). That is backwards. The parent's
width must be the **total** (2.632 eV for 2p₃/₂⁻¹3p₃/₂⁻¹), with the feed a branch of it. That is how
the 2s pathway is already written: full Γ_L1 = 8.13 eV decay, plus additive satellite feeds.

Direct check (unit population in one parent manifold, zero field, the functions in `Model.py` on the
shipped config):

| Parent | Configured outflow (eV) | Daughters gain (eV) | Daughters per decayed parent | Physical |
|---|---|---|---|---|
| 3p+ lower | 0.893 | 1.740 | **1.95** | 0.66 |
| 3p+ upper | 1.813 | 2.039 | **1.12** | 0.53 |
| 3p+ L2 | 0.984 | 1.740 | **1.77** | 0.64 |
| 3p− lower | 0.990 | 2.190 | **2.21** | 0.69 |
| 3p− upper | 1.835 | 2.530 | **1.38** | 0.58 |
| 3p− L2 | 0.868 | 2.190 | **2.52** | 0.72 |

The shipped double-satellite configs (`Cu-seed-*double-satellite*.yaml`, including `-grasp`, where
all three widths are set to the base stand-ins 0.67/1.37/0.65 eV) all have this. Two consequences:
daughters are over-produced by about 3×, and the 3p± parents live about 3× too long. Both deepen the
dip. Estimated effect at 8048 eV: ΔT = +0.009 at 5 µJ, +0.013 at 20 µJ, +0.014 at 30 µJ when fixed
(estimator). A paired Maxwell–Bloch shot with the same SASE seed is described in §6.3.

Fix: set each parent's width to its un-carved total (config value + Σ feed-out). Add a
construction-time check in `XLO_sim.__init__` that no daughter's summed feed from a parent manifold
exceeds that manifold's decay rate. Correct the §12.7 paragraph of the theory doc. The trace
diagnostic `total_population_t_last` would have caught this, since it must be non-increasing.

### 2.4 "other" is a dead end

`rho_other` is fed by M-shell photoionisation, loses population only by further photoionisation into
nothing, and has no decay. Physically, 3s and 3p holes collapse to 3d⁻² (or 3d⁻¹4s⁻¹) within about
0.2 fs, so "other" is exactly the n ≈ 2 middleman population, minus its ability to make satellites.
In the middleman scheme below, `rho_other` is absorbed into the ladder.

---

## 3. Middleman states

### 3.1 What they are

A middleman is an ion with M-shell holes and no L- or K-shell hole. XATOM gives:

- **3d holes are Auger-stable.** `3d0,1`, `3d0,2` and `3d1,1` have no Auger channel (filling 3d from
  4s releases too little energy to eject anything).
- **3p holes are not.** M₂₃-M₄₅M₄₅ super-Coster–Kronig: Γ = 2.79 eV (3p₃/₂), 3.29 eV (3p₁/₂),
  giving 3d⁻² 91% and 3d⁻¹4s⁻¹ 9%. Lifetime about 0.2 fs.
- **3s holes are not.** Γ = 3.06 eV, giving 3p3d 46%, 3p4s 30%, 3d4s 16%, 3d⁻² 8%, then the 3p cascade.

So one femtosecond after any L-shell event the atom is in a pure 3d⁻ⁿ (plus 4s) configuration, and
it stays there for the rest of the pulse. Taking 4s as refilled from the conduction band, the natural
state label is n, the number of 3d holes.

### 3.2 Production: n distribution per decay

The table applies the XATOM branchings, then collapses every 3p hole to +2 3d holes and every 3s hole
to about +3.

| Decaying state | n = 1 | n = 2 | n = 3 | n = 4 | n ≥ 5 | Mean n |
|---|---|---|---|---|---|---|
| base L3 (Auger) | 0.01 | 0.37 | 0.33 | 0.24 | 0.05 | 2.9 |
| base L2 (Auger) | 0.01 | 0.38 | 0.33 | 0.22 | 0.06 | 2.9 |
| 2s L1-MM remainder | — | ≈0.2 | ≈0.3 | ≈0.3 | ≈0.2 | ≈3.5 |
| K → KLL → 2p⁻² → 2 × L-Auger | — | — | — | 0.1 | 0.9 | ≈6 |
| single satellite `3d±` L3k (core Auger) | — | — | 0.37 | 0.33 | 0.30 | 3.9 |
| double satellite (core Auger) | — | — | — | 0.37 | 0.63 | 4.9 |
| "other" (direct M-shell PI; 3% is 4s → n = 0) | 0.13 | 0.54 | 0.23 | 0.07 | — | 2.1 |

A second photoabsorption on a middleman adds another 2–3 holes, so n drifts upwards through the
pulse. At 20 µJ the mean n of the middleman pool reaches about 3–4 by the end of the pulse.

### 3.3 Absorption and branching of middlemen

At 8 keV the photoabsorption is dominated by the L shell, which is unaffected by 3d holes, so the
cross section and branching are essentially those of the neutral atom. XATOM `-pcs` at 8047.91 eV:

| Ion | σ_tot (10⁻⁷ nm²) | 2p₃/₂ share | 2s share |
|---|---|---|---|
| neutral | 5.149 | 0.265 | 0.476 |
| 3d⁻¹ | 5.168 | 0.264 | 0.474 |
| 3d⁻² | 5.191 | 0.262 | 0.472 |
| 3d⁻³ | 5.218 | 0.261 | 0.470 |
| 3d⁻⁴ | 5.248 | 0.259 | 0.467 |
| 3d⁻⁶ | 5.318 | 0.256 | 0.462 |

(the 2p/2s partials are constant to 0.2%; the small rise is the M shell). Deleting these ions, as the
model does now, removes 100% of their absorption. Middlemen should absorb with σ_tot and branch as
in §2.1, but into satellite configurations:

- 2p₃/₂ PI → 2p₃/₂⁻¹3d⁻ⁿ (Kα1-type satellite, shifted δ₁(n))
- 2p₁/₂ PI → 2p₁/₂⁻¹3d⁻ⁿ (Kα2-type satellite)
- 2s PI → 2s⁻¹3d⁻ⁿ → (8.1 eV) Coster–Kronig → 2p⁻¹3d⁻⁽ⁿ⁺¹⁾ or 2p⁻¹3p⁻¹3d⁻ⁿ → …
- M-shell PI → 3d⁻⁽ⁿ⁺²⁾ (stays in the ladder)

### 3.4 Where their satellites resonate

ΔSCF Kα1 shifts relative to the bare line (XATOM, configuration averages):

| Spectator | 3d⁻¹ | 3d⁻² | 3d⁻³ | 3d⁻⁴ | 3d⁻⁵ | 3d⁻⁶ | 3p⁻¹ | 3s⁻¹ | 4s⁻¹ | 2s⁻¹ | 2p₃/₂⁻¹ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| δ(Kα1) (eV) | −0.80 | −1.70 | −2.65 | −3.62 | −4.57 | −5.47 | +2.8 | +2.5 | +0.24 | +26.9 | +35.4 |

Kα2-type shifts agree to within 0.03 eV. The per-hole 3d shift is about −0.9 eV in XATOM. The GRASP
values in the `-grasp` configs put the dominant multiplet line of a single 3d spectator on the
**blue** side (+1.13 eV for 3d₅/₂, +0.34 eV for 3d₃/₂). The sign matters a great deal here:

- If 3d holes red-shift Kα, middleman satellites fill 8040–8046 and 8020–8026 eV.
- If they blue-shift, they fill 8050–8060 and 8030–8040 eV. That is where the experiment shows its
  steep, threshold-like absorption (§1.3).

The experiment's blue-side onset is therefore a direct, if indirect, test of the sign. Both signs are
run in §6.

### 3.5 Solid-state caveats

- A single 3d hole in metallic Cu sits in a d band about 3 eV wide and can hop away in about
  0.2–0.5 fs. Two or more 3d holes on one site are localised by the on-site repulsion U ≈ 8 eV (this
  is why the Cu L₃M₄₅M₄₅ Auger spectrum is atomic-like). The Auger-cascade middlemen are n ≥ 2, so
  the atomic picture is reasonable for them. For n = 1 it is questionable, but n = 1 is almost only
  made by direct 3d photoionisation (0.2% of photoabsorption).
- 4s holes are refilled from the conduction band on the femtosecond scale. Ignore them, as the model
  already does.
- L2-L3M45 Coster–Kronig is closed in the free atom (XATOM Γ_L2 = 0.67 eV) but open in the metal:
  the config's literature Γ_L2 = 1.04 eV includes it. Whichever width is used, the destination has to
  match, and it currently does not (§4).

### 3.6 Bang for the buck

1. **One lumped middleman scalar** routed from every vanish channel, absorbing with σ_tot and
   photoionised into satellites. It captures the de-bleaching and about 80% of the satellite feed.
   Cost: 1–2 scalars.
2. **A short ladder M₂, M₃, M₄, M₅₊** (4 scalars). This resolves the shift distribution, which is
   what decides whether middlemen deepen the line centre or broaden the dip. It maps onto existing
   blocks for n = 1 (3d± channels) and n = 2 (double channels); n ≥ 3 needs a decision (§5).
3. **Separate 3p⁻¹ or 3s⁻¹ middleman populations: not worth it.** At 0.2 fs lifetimes their chance of
   being photoionised before collapsing is J·σ·τ ≈ 10⁻³.

---

## 4. Missing connections into states that already exist (no new state)

| Connection | Feed | Magnitude | Worth it? |
|---|---|---|---|
| base L2 → 3d± L3k (L2-L3M45 CK, metal) | even spread, rate f₂₃Γ_L2, split 3d₅/₂ : 3d₃/₂ ≈ 6 : 4 | 11.8% × f₂₃ ≈ 4–5% of photoabsorption as Kα1-type holes (+9% on f) | yes, one feed term |
| 2s → 2p4s CK as base L3/L2 | even spread | 0.7% of photoabsorption (+1.5% on f) | trivial to add |
| K → 2p₃/₂⁻¹3s⁻¹ (KL₃M₁) | as `Gamma_A_K_eV` | 1.2% of K decays | no (no 3s-spectator block) |
| satellite L2k → double-sat L3k (CK) | even spread | second order | later |
| single sat → double sat by photoionising the spectator | sublevel-preserving | J σ τ ≈ 10⁻⁴…6×10⁻³ | no |
| satellite/base 2p hole → 2p⁻² by further PI | — | ≈4% of holes at 20 µJ, resonant at +35 eV | no for mono near Kα |

---

## 5. A system for adding middlemen (concept; details in the plan)

Two ideas keep the scheme extensible without new density-matrix blocks:

1. **One generic incoherent transfer table** replaces per-mechanism special cases. Each entry is
   `{from: state/manifold, to: state/manifold, rate_eV | sigma_nm2 (× J), sublevels: even | preserve}`.
   The existing `Gamma_A_2s_eV`, `Gamma_A_K_eV`, `feed_from` and `sigma_Ka1_from_*` terms are all
   special cases of it. "Vanish" becomes an explicit destination (`middleman: auto`) whose n
   distribution comes from an XATOM cascade matrix.
2. **An XATOM-generated cascade matrix** C[parent → n] (the table in §3.2) written by a new
   `xatom_tools.middleman_parameters()`, so adding a parent or refining the branching is regenerated,
   not hand-edited.

Mapping middleman satellites onto existing blocks:

| n (3d holes) | Existing block | Shift used vs XATOM | Error |
|---|---|---|---|
| 1 | `3d+`, `3d-` | −0.8 (or GRASP +1.1/+0.3) | none |
| 2 | `3d+3d+`, `3d-3d+`, `3d-3d-` | −1.7 | none |
| 3, 4, 5+ | option A: lump into the double-satellite blocks | −1.7 vs −2.7…−4.6 | 1–3 eV too close to the line (upper bound on line-centre depth) |
| | option B: treat as non-resonant (count as absorbers only) | — | lower bound |
| | option C: one extra lumped "3d⁻ⁿ⁺" block at −3.5 eV | — | best, but one extra 8-level block |

Running A and B brackets the answer. If the bracket is wide (§6 says it is at 20–30 µJ), option C is
worth its cost of one block.

---

## 6. Back-of-envelope magnitudes

### 6.1 Analytic: how much bleaching the model has

In the current model every photoionised atom disappears after about 1 fs. The fraction missing when
a photon arrives, averaged over the pulse, the Gaussian focus and depth, is

$$
B\approx\frac{\langle\sigma\Phi\rangle}{2}\cdot\frac{1}{2}\cdot\frac{1-T}{\kappa L},\qquad
\sigma\Phi_{peak}=0.024,\ 0.12,\ 0.47,\ 0.71\ \text{at}\ 1,\ 5,\ 20,\ 30\ \mu\text{J}
$$

(pulse average 1/2, fluence-weighted transverse average 1/2, depth average (1−T)/κL ≈ 0.66). That
gives B ≈ 0.3%, 2%, 8% and 12%. Restoring those atoms raises κ_PI, and with it the saturated
resonant absorption, by the same fraction. This is why the middlemen matter at 20–30 µJ and not
below 5 µJ, and why the current wings rise with pulse energy (0.407 → 0.415–0.419 at 30 µJ) while the
experimental wings do not.

### 6.2 Rate-equation estimator

T at selected photon energies (eV), the integrated excess absorption A = ∫ln(T₀.₁µJ/T)dE over
8000–8100 eV, and the line-centre ln-dip against the variant's own T(8000):

| 20 µJ | 8020 | 8028 | 8035 | 8044 | 8048 | 8055 | 8060 | 8000 | A (eV) | ln-dip |
|---|---|---|---|---|---|---|---|---|---|---|
| experiment | 0.322 | 0.313 | 0.330 | 0.281 | 0.278 | 0.317 | 0.350 | 0.404 | 14.7 | 0.37 |
| estimator, current model | 0.406 | 0.373 | 0.403 | 0.374 | 0.351 | 0.401 | 0.414 | 0.424 | −0.0 | 0.19 |
| feed bug fixed | 0.414 | 0.384 | 0.409 | 0.388 | 0.364 | 0.407 | 0.418 | 0.426 | −1.4 | 0.16 |
| + middlemen (−2.7 eV) | 0.394 | 0.366 | 0.390 | 0.367 | 0.346 | 0.388 | 0.398 | 0.406 | 3.5 | 0.16 |
| + middlemen (+2.7 eV) | 0.395 | 0.366 | 0.390 | 0.369 | 0.346 | 0.387 | 0.398 | 0.406 | 3.5 | 0.16 |
| + middlemen, non-resonant satellites (option B) | 0.395 | 0.367 | 0.391 | 0.371 | 0.348 | 0.389 | 0.399 | 0.406 | — | 0.15 |
| middlemen + L2 CK + L-shell EII (V5c) | 0.393 | 0.362 | 0.388 | 0.361 | 0.335 | 0.385 | 0.397 | 0.406 | 3.9 | 0.19 |
| … + M-shell EII, shift −2.7 eV | 0.393 | 0.370 | 0.390 | 0.355 | 0.349 | 0.391 | 0.399 | 0.406 | 3.5 | 0.15 |
| … + M-shell EII, shift +2.7 eV | 0.398 | 0.370 | 0.387 | 0.374 | 0.348 | 0.380 | 0.395 | 0.407 | 3.5 | 0.16 |

| T(8048) / T(8028) | 1 µJ | 5 µJ | 20 µJ | 30 µJ |
|---|---|---|---|---|
| experiment | 0.365 / 0.386 | 0.320 / 0.351 | 0.278 / 0.313 | 0.277 / 0.305 |
| feed bug fixed | 0.385 / 0.400 | 0.367 / 0.388 | 0.364 / 0.384 | 0.367 / 0.388 |
| middlemen + L2 CK + L-shell EII (V5c) | 0.380 / 0.398 | 0.355 / 0.382 | 0.335 / 0.362 | 0.329 / 0.356 |
| V5c + dark-state mixing 2 fs⁻¹ (V7) | 0.376 / 0.398 | 0.344 / 0.381 | 0.318 / 0.362 | 0.310 / 0.356 |
| dark-state mixing alone (V6) | 0.382 / 0.400 | 0.358 / 0.388 | 0.348 / 0.384 | 0.350 / 0.388 |

What the estimator says:

1. **The feed bug** deepens the current dip by 0.009–0.014 in T at ≥5 µJ, and adds 0.03 to the
   line ln-dip at 20 µJ.
2. **Middlemen fix the sign of the trend.** Without them the integrated excess absorption *falls*
   with pulse energy, because the wings bleach faster than the dip grows. With them it grows
   (1.6 → 3.5 → 4.2 eV at 5/20/30 µJ), as in the experiment (6.5 → 14.7 → 18.1 eV), but at about a
   quarter of the size. The line-centre ln-dip against the wing barely changes (0.16). The option-B
   bracket shows why: with the middlemen's satellites pushed off resonance, T(8048) is 0.348 / 0.344
   at 20 / 30 µJ, against 0.344 / 0.338 with fully resonant satellites. About 80% of the middleman
   effect at the line is the restored non-resonant photoabsorption. The satellites add the expected
   η·f ≈ 0.2 on top of it. So options A and B of §5 differ by < 0.006 in T at the line, and the
   ladder resolution (plan step 3) matters for the shape only.
3. **The shift sign moves absorption to one side.** With M-shell EII converting neutral atoms into
   middlemen, the −2.7 eV case deepens 8044 eV (0.355 vs 0.367) and the +2.7 eV case deepens 8055 eV
   (0.380 vs 0.388) and 8035 eV. The experiment has its steep threshold absorption on the blue side.
4. **The best combination of depth-adding pathways** (V5c: middlemen + L2→L3 CK + L-shell EII) closes
   about 34% (20 µJ) to 42% (30 µJ) of the line-centre gap from the bug-fixed baseline: 0.364 → 0.335
   against 0.278 measured. Adding dark-state mixing (V7) takes it to 0.318, which is 53% at 20 µJ and
   63% at 30 µJ. Mixing is the only change that raises the Kα1 ln-dip against the wing substantially
   (0.16 → 0.25). It does nothing at Kα2.
5. **What remains unexplained.** About ×4 in integrated absorption, and absorption 8–12 eV away from
   both lines (8020, 8035, 8055–8060 eV) that is almost as strong as at the lines. The latter needs
   resonances spread over ±10 eV. That is a multi-hole spread (3d⁻ⁿ with n ≫ 3, i.e. M-shell
   stripping) far beyond the single ±2.7 eV lumped shift used here, and it still requires more
   absorbers per photon than photoabsorption supplies.

### 6.3 Paired Maxwell–Bloch shot for the feed bug

One mono-SASE shot (seed 0, 20 µJ, 8048 eV, production grid) of `Cu-seed-mono-SASE-double-satellite`
as shipped, and the same with the 3p± widths restored (plan step 0), run at t-grid 3000 and 7 × 7
pixels to fit on a laptop:

| | T(8048), 20 µJ |
|---|---|
| as shipped (carved widths) | 0.349 |
| widths restored (population-conserving) | 0.363 |
| **difference** | **+0.014** |
| estimator prediction | 0.351 → 0.364 (+0.013) |
| saved 10-shot production sweep, as shipped | 0.351 |

The Maxwell–Bloch solver confirms the estimator's size for the bug. Every double-satellite result
so far makes the Kα1 dip about 0.014 too deep at 20 µJ.

---

## 7. The estimator (method and limits)

`toy_rsa.py` (session scratchpad; offered for `scripts/` in the plan doc) is a z-marching rate-equation
model with every species of the double-satellite config. Each species has a Kα1-type (j=3/2, half
dark) and a Kα2-type (j=1/2) resonant system. Absorption cross sections are computed from the Kα
radiative widths as a Lorentzian convolved with a 1 eV Gaussian (the mono). It uses the real focus
(Gauss–Laguerre rings), an 8 fs Gaussian pulse, and the solver's effective 18.6 µm thickness.

Against the saved MB runs at 8048 eV it gives 0.381/0.358/0.351/0.353 vs 0.373/0.352/0.351/0.358
(1/5/20/30 µJ). It is about 0.01 shallower at Kα2 and bleaches the wings more strongly than MB
(T(8000) 0.424 vs 0.411 at 20 µJ). It has no coherences, no diffraction and no SASE spikes, so only
differences between variants are quoted.

---

## 8. Open questions

- The sign of the per-3d-hole Kα shift (XATOM config-average −0.9 eV vs GRASP dominant line
  +0.3…+1.1 eV). This decides whether middlemen deepen the red or the blue side. A GRASP 2p⁻¹3d⁻ⁿ →
  1s⁻¹3d⁻ⁿ multiplet run for n = 2–4 would settle it.
- Solid-state f₂₃ for Cu: bracket 0 (atom) to 0.41 (width difference); a literature value would pin
  it.
- Localisation of n = 1 middlemen in the metal (§3.5). This hardly matters, because Auger-cascade
  middlemen are n ≥ 2.
- Whether the experiment's transmitted-beam divergence (slides 2–3: "divergence becomes large after
  transmitting copper foil") lets part of the beam miss the photodiode at high pulse energy. A
  refraction estimate (resonant δn ~ 2×10⁻⁷, 0.1 mrad deflection vs ~1 mrad natural divergence) says
  this is small, but it is a possible systematic on exactly the energies where the model fails.
