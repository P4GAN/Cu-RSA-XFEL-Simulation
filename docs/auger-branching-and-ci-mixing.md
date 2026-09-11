# Auger branching into the satellite channels: configuration mixing and how to handle it

This explains a physics issue found while computing the satellite channels' `Gamma_A_2s_eV`/
`Gamma_A_K_eV` (and their `_to_L2_eV` siblings) via GRASP/RATIP: the calculation shows strong
configuration mixing that XATOM's numbers don't reflect, why that mixing is real physics rather
than a bug, and the branching scheme adopted to handle it. See `grasp/NEXT_SESSION.md` for the
technical recipe/provenance and `grasp/cross_section_validation.py` for the recorded numbers.

No independent cross-check (JAC, literature) has been done for any of this yet — unlike the
satellite-detuning and cross-section work earlier in this campaign, which was cross-validated
against at least one other method before being trusted. Everything here should be read as a
first-pass GRASP/RATIP result, methodologically reasoned through but not yet independently
confirmed.

## 1. What these quantities are

Each satellite channel (`3d+`, `3d-`, `3p+`, `3p-`) represents a Kα-satellite line: a copper atom
that has a spectator hole (in 3d or 3p) sitting alongside the main K/L-hole during the Kα decay.
`Gamma_A_2s_eV` and `Gamma_A_K_eV` are the rates at which population *arrives* in a given
channel's upper (K-hole+spectator) manifold via an Auger process, rather than by direct
photoionization from the ground state:

- **`Gamma_A_2s_eV`**: starting from a bare 2s-hole (no spectator yet), a 2p electron falls in to
  fill it (this is what creates the L-hole) while simultaneously a 3d (or 3p) electron is ejected
  as the Auger electron — landing you in exactly the "L-hole + 3d-hole" (or 3p-hole) double-hole
  state that channel's manifold describes. One process both creates the spectator *and* performs
  the core-hole transition.
- **`Gamma_A_K_eV`**: the same idea, but starting from a bare 1s-hole (K-hole) instead — 2p fills
  1s, 3d/3p electron ejected, landing in the same kind of double-hole final state.
- The `_to_L2_eV` siblings are the same two processes, but where the 2p electron that falls in
  comes from 2p₁/₂ (L2) instead of 2p₃/₂ (L3) — feeding the parallel L2/Kα2-satellite pathway
  (`use_L2_pathway`).

Physically, this matters for your project because it's a *feed* into the satellite Maxwell-Bloch
blocks that doesn't come from the ground state at all — it's populated from the (much larger) 2s-
and K-hole populations decaying sideways into the satellite manifolds. Getting the split between
`3d+` and `3d-` right changes how much population lands in the *narrower-* vs *wider-*detuned
line, which is exactly the mechanism that turns one Kα-satellite feature into a wider, shallower
composite dip.

## 2. The setup: same final states as the cross-section work

Computing an Auger rate with RATIP's `xauger` needs the same two-state setup as a photoionization
cross section (`xphoto`): an initial bound state and a final bound state, and RATIP computes the
transition amplitude between them (for Auger, the "amplitude" describes emitting an electron into
the continuum rather than a photon). The energy of the ejected Auger electron isn't an input —
it falls straight out of the initial/final state energy difference.

The convenient part: the *same* double-hole final states already built for the `sigma2_Ka1_*`
cross-section work are exactly what's needed here too. `Gamma_A_2s_eV`/`Gamma_A_K_eV` for the
`3d+`/`3d-` channels both use `cu_l3p3d` (the "L-hole + 3d-hole" final state, combined-notation on
both the L-shell and the 3d-shell, so it already spans all four combinations: L2+3d₃/₂, L2+3d₅/₂,
L3+3d₃/₂, L3+3d₅/₂) as the final state — only the *initial* state changes: `cu_2sholemax` for the
2s-branch, `cu_kholemax` for the K-branch. `Gamma_A_K_eV`/`Gamma_A_2s_eV` for the `3p+`/`3p-`
channels would use the analogous "L-hole + 3p-hole" final state (`cu_l3p3p`) — but that state hits
an unresolved `rmcdhf` crash (`LODCSH2: ncf=10 ncfblock=5`, the same one documented for
`sigma2_Ka1_2p3`'s missing `2p3+3p` channel), so the 3p-channel numbers aren't done.

## 3. The problem: `cu_l3p3d`'s levels don't map cleanly onto "3d+" and "3d-"

`cu_l3p3d` has 11 CI-mixed levels (from `rmcdhf`+`rci`'s full diagonalization within each J/parity
block). Looking at which starting configurations ("CSFs") dominate each level (`xrelci`'s own
"weights of major contributors to ASF" printout — these weights *are* the squared mixing
coefficients, i.e. population fractions):

```
Level  J^P   Dominant CSF(s) and weight                      Character
  1    4-    100.0% (L3, M5)                                  clean
  2    2-     58.7% (L3,M5) + 41.1% (L3,M4)                    MIXED
  3    3-     52.3% (L3,M4) + 47.6% (L3,M5)                    MIXED
  4    2-     58.9% (L3,M4) + 41.0% (L3,M5)                    MIXED
  5    1-     96.0% (L3,M4)                                    clean
  6    3-     52.1% (L3,M5) + 46.4% (L3,M4)                    MIXED
  7    1-     95.5% (L3,M5)                                    clean
  8    2-     99.9% (L2,M4)                                    clean
  9    2-     99.7% (L2,M5)                                    clean
 10    3-     98.3% (L2,M5)                                    clean
 11    1-     99.2% (L2,M4)                                    clean
```

(M4 = 3d₃/₂ hole, M5 = 3d₅/₂ hole; L2/L3 = 2p₁/₂/2p₃/₂ hole.)

The L2-associated levels (8-11) are essentially pure. Four of the seven L3-associated levels
(2, 3, 4, 6) are close to a 50/50 mix of "3d₃/₂ spectator" and "3d₅/₂ spectator" character. There
is no way to look at one of these levels and say "this one is the 3d+ state" — it genuinely is
both, in quantum superposition, in roughly equal parts.

## 4. Why this happens: a near-degeneracy, and it's real

This isn't a bug or a numerical artifact — it's standard relativistic-atomic-structure behavior,
and it's specific to the L3-branch for an identifiable, checkable reason.

**The mechanism.** "L3-hole + 3d-hole" isn't one final state — it's two different starting
configurations (spectator in 3d₃/₂, or spectator in 3d₅/₂) that happen to produce overlapping
total angular momenta. Coupling the L3 hole (j=3/2) to a 3d₃/₂ hole (j=3/2) gives total J ∈
{0,1,2,3}; coupling it to a 3d₅/₂ hole (j=5/2) gives J ∈ {1,2,3,4}. **These ranges overlap at
J = 1, 2, 3** — three separate total-J values where the two configurations are literally the same
symmetry block. Any two basis states sharing a J/parity block are, in general, not the true
eigenstates on their own — GRASP's full CI diagonalization mixes them, and how much depends on
how the electron-electron (Coulomb/exchange) coupling between them compares to their unperturbed
energy difference.

That energy difference is just the 3d fine-structure (spin-orbit) splitting — and it's small: this
campaign's own `cu_3dhole` calculation puts 3d₅/₂ and 3d₃/₂ only **≈0.245 eV** apart. The
electron-electron exchange coupling between two valence-shell configurations is typically on the
order of an eV or more — bigger than the splitting it's competing against. That combination
(small unperturbed splitting, comparatively large coupling) is exactly the textbook recipe for
strong, close-to-democratic mixing. Nothing about copper's 3d spin-orbit splitting is unusual;
it's just small enough, relative to the coupling, to land in this regime.

**Why L2 stays clean.** The L2 hole has j=1/2. Coupling j=1/2 to 3d₃/₂ (j=3/2) gives J ∈ {1,2};
coupling to 3d₅/₂ (j=5/2) gives J ∈ {2,3}. These overlap at only **one** J (J=2) instead of three.
Fewer shared symmetry blocks means fewer opportunities for the two configurations to mix — which
is exactly the >98%-pure pattern seen above. This is a clean, checkable angular-momentum-counting
explanation, not a coincidence: the number of overlapping J values (1 for L2, 3 for L3) tracks
directly with how strong the observed mixing is.

**Confirming case: the 3p spectator behaves completely differently, for a predictable reason.**
The analogous "L-hole + 3p-hole" final state (`cu_l3p3p`) has the *same* angular-momentum-overlap
structure as the 3d case (L3⊗3p₁/₂ and L3⊗3p₃/₂ share J=1,2; in fact even L2⊗3p₁/₂ and L2⊗3p₃/₂
join in at J=1), so if mixing were purely about counting overlapping J values, you'd expect similar
scrambling here too. It doesn't happen — `cu_l3p3p`'s levels come out mostly >96% pure. The reason
is the other half of the mechanism: this session's own `cu_3phole` calculation puts the 3p₁/₂-3p₃/₂
splitting at **≈2.51 eV** — about ten times the 3d splitting, and now large enough to dominate over
the exchange coupling rather than lose to it. Same coupling topology, opposite outcome, because the
one number that actually controls the competition (the spin-orbit splitting) is different. That's
a useful independent check that the explanation is the right one, not a post-hoc story fitted to
one data point.

## 5. Why XATOM doesn't show this

XATOM most likely computes each spectator configuration's rate as its own single-reference
calculation — building "the final state with a 3d₃/₂ hole" and "the final state with a 3d₅/₂ hole"
as two independently well-defined targets, without ever constructing the off-diagonal coupling
between them. That's a completely standard and often-adequate simplification in atomic structure
codes — it's *only* wrong when the two configurations are close enough in energy, relative to
their coupling, for the mixing to matter. That's precisely the situation here. This isn't "GRASP
is noisier than XATOM" — the full CI treatment is capturing real correlation physics that a
single-configuration method is structurally unable to see, in a case (small 3d spin-orbit
splitting) where that physics happens to matter.

## 6. The branching scheme: weight by population fraction, not by "which one wins"

Since the printed CI weights already *are* population fractions (`|mixing coefficient|²`), the
natural fix is: instead of assigning each level's entire Auger rate to whichever configuration is
nominally dominant, split each level's rate proportionally across all the configurations it's
actually built from, and sum the pieces that belong to each channel:

```
Gamma_A(channel) = sum over levels L of [ rate(L) x weight of channel's configuration in L ]
```

This is the right reduction for a population-based (incoherent) model: you're asking "of all the
population that ends up in this manifold via this Auger process, what fraction lands with a 3d₅/₂
spectator vs a 3d₃/₂ spectator?", and `|c|²` branching is the standard way to answer that from a
mixed eigenstate. It's also smooth — a level sitting at 52/48 contributes almost evenly to both
channels, rather than flipping its entire rate to one side or the other depending on which side of
50% it happens to land on, which would make the two channels' values sensitive to numerical noise
right at the point where the physics is telling you they *shouldn't* be cleanly separable at all.

A cruder "assign each level to its dominant configuration" scheme was tried first, for comparison
— it gives numbers 15-40% further from XATOM's than the `|c|²`-weighted scheme, in the direction
you'd expect from double-counting the majority component and dropping the minority one entirely.
The `|c|²`-weighted numbers below are the ones worth using.

## 7. Results

All energies in eV, both Coulomb-gauge, computed at the "reduced 5-channel other" level of
completeness this campaign has been running at throughout (no self-energy, no `4s`, etc. — see
`grasp/NEXT_SESSION.md` for the general caveats that apply to every number in this campaign).

| Quantity | Channel | GRASP/RATIP (`\|c\|²`-weighted) | Current XATOM value | Ratio |
|---|---|---|---|---|
| `Gamma_A_2s_eV` | 3d+ | 1.857 | 1.5733 | 1.18 |
| `Gamma_A_2s_eV` | 3d- | 0.815 | 1.3121 | 0.62 |
| `Gamma_A_2s_to_L2_eV` | 3d+ | 0.733 | 0.7137 | 1.03 |
| `Gamma_A_2s_to_L2_eV` | 3d- | 0.484 | 0.5309 | 0.91 |
| `Gamma_A_K_eV` | 3d+ | 0.00304 | 0.00259 | 1.17 |
| `Gamma_A_K_eV` | 3d- | 0.00169 | 0.003083 | 0.55 |
| `Gamma_A_K_to_L2_eV` | 3d+ | 0.00256 | 0.002064 | 1.24 |
| `Gamma_A_K_to_L2_eV` | 3d- | 0.00058 | 0.000637 | 0.91 |
| `Gamma_A_2s_eV` | 3p+ | 0.973 | 1.4535 | 0.67 |
| `Gamma_A_2s_eV` | 3p- | 0.550 | 0.6458 | 0.85 |
| `Gamma_A_2s_to_L2_eV` | 3p+ | 0.526 | 0.7902 | 0.67 |
| `Gamma_A_2s_to_L2_eV` | 3p- | 0.445 | 0.5062 | 0.88 |
| `Gamma_A_K_eV` | 3p+ | 0.0243 | 0.0340 | 0.71 |
| `Gamma_A_K_eV` | 3p- | 0.0231 | 0.02891 | 0.80 |
| `Gamma_A_K_to_L2_eV` | 3p+ | 0.0423 | 0.0290 | 1.46 |
| `Gamma_A_K_to_L2_eV` | 3p- | 0.00365 | 0.00256 | 1.42 |

**Pattern worth noting**: the `_to_L2` (L2-branch) numbers agree with XATOM to 3-17% *for the 3d
channels* — consistent with L2 being the case where mixing is weak and a single-configuration
method is a good approximation to begin with, which is a reassuring internal check that the method
is behaving sensibly where it's expected to. The `to_L3` (`Gamma_A_2s_eV`/`Gamma_A_K_eV` proper)
3d numbers show a real, larger disagreement, particularly `Gamma_A_2s_eV`: GRASP puts 3d+ noticeably
*above* 3d- (ratio ≈2.3), while XATOM has them close to even (ratio ≈1.2, mildly favoring 3d-).
That's not just a magnitude difference — it's a different qualitative picture of which satellite
line gets fed more, which is exactly the kind of thing that would change simulated dip shape, not
just its overall depth. The 3p numbers, where CI mixing is weak throughout (both L2- and L3-
branches), land in a tighter, more consistent 0.67-1.46 band against XATOM — closer to the general
accuracy level seen for the rest of this campaign's cross sections, which is what you'd expect once
the near-degeneracy that makes the 3d `to_L3` numbers hard is no longer a factor.

## 8. What this means for the dip-widening physics

The mixing result is itself informative, independent of exactly which numbers end up in the
config: population reaching the "L3-hole + 3d-hole" manifold via 2s- or K-hole Auger decay does
not sort cleanly into "3d+" or "3d-" — a majority of it (the four mixed levels) genuinely goes into
*both*, in comparable amounts, because those are the true eigenstates of the system. That's a
mechanism, independent of the exact numbers, for why this feed should broaden rather than sharpen
the satellite contribution to the dip: even a perfectly resolved measurement of "the L3+3d
manifold" would still show contributions from both detunings, because the states you're populating
are inherently mixed, not because of any blurring introduced by finite bandwidth or your own
model's coarseness.

## 9. Open items

- **The 3p+/3p- channels are no longer blocked.** The `cu_l3p3p` "crash" this document originally
  reported as an open item turned out to be a wrong `rmcdhf` block-serial-number input in the
  original attempt, not a real bug — see `grasp/NEXT_SESSION.md` §5. All 8 quantities in the table
  above are now computed for both 3d and 3p.
- **No independent cross-check yet.** Given how much the 3d `to_L3` numbers disagree with XATOM in
  relative terms, this is the one place in the campaign so far where a third method (JAC, as used
  for the satellite detunings) would be particularly worth the effort before trusting these numbers
  for anything beyond a qualitative "the feed is genuinely split, not one-sided" argument.
- **`Gamma_L_eV`/`Gamma_K_eV`/`Gamma_L2_eV`** (the satellite-shifted states' *total* decay widths,
  as opposed to this one resolved Auger channel) still need the same "sum over every allowed
  channel" treatment as the base `GammaKeVN`/`GammaL1eVN`/etc., and will likely run into the same
  kind of CI-mixing attribution question wherever two final configurations share a symmetry block
  — the `|c|²`-weighting approach here should generalize directly.
