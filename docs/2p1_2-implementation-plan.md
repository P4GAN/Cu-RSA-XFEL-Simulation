# Planning: 2p$_{1/2}$ (L2, Kα2) pathway — corrected derivation and implementation

This document supersedes the implementation reasoning in
`theory-and-2s-satellite-pathways.md` Part III (§15–22), which shipped two real bugs, found and
fixed in sequence:

1. The field coupling for the new 2p$_{1/2}$ sublevels was silently routed through the wrong half
   of the Maxwell–Bloch field (`Omega_plus` swapped with `Omega_minus`) because of how `Model.py`
   builds its interaction Hamiltonian from raw array-index comparisons rather than physical state
   role (§2–3). That bug produced spurious *gain* at the Kα2 detuning in the transmitted spectrum,
   contradicting real SACLA XFEL data (unpublished, shared by the user) showing Kα1 **and** Kα2
   both as genuine, deepening absorption dips ("reverse saturable absorption") at every tested
   pulse energy.
2. Fixing (1) turned the spurious gain into a genuine absorption dip, but on the wrong side of the
   spectrum (+20 eV instead of −20 eV) — a second, independent sign error in the detuning matrix
   $\Delta_{ij}$, exposed only once (1) was fixed (§7).

This document works the problem from first principles — including the part that was skipped the
first time — so the fix is understood, not just found by trial and error. Part III of the theory
doc should be corrected to reference this document once the fix is merged.

## 1. What actually needs to be true

Three independent things have to be correct simultaneously for a new dipole-coupled manifold to
behave physically in this codebase:

1. **The angular/CG matrix elements** ($T_{ij\sigma}$, $G_{ij}$) — magnitude *and* relative sign
   between sublevels.
2. **The detuning** ($\Delta_{ij}$) — which residual frequency, relative to the Kα1 rotating frame,
   each coherence picks up.
3. **Which physical field component drives which matrix element** — i.e., does `Model.py`'s
   `Hint[i,j]` couple to `Omega_plus` or `Omega_minus`, and is that choice consistent with which of
   $i,j$ is the energetically higher state.

Section 18 of the theory doc did (1) carefully and correctly. Section 19 did (2) correctly. **Item
(3) was never checked** — it was implicitly assumed to come along for free by "extending the base
block," and it doesn't. This document re-derives (3) from scratch and re-verifies (1) and (2)
independently.

## 2. Re-deriving item (3): which field couples to which matrix element

Start from the true (lab-frame) dipole interaction for a single two-level pair, ground/lower state
$|g\rangle$ and excited/upper state $|e\rangle$ ($E_e>E_g$, $\hbar\omega_{eg}=E_e-E_g$):

$$H_{int}(t) = -\vec d\cdot\vec E(t), \qquad \vec d = d_{eg}|e\rangle\langle g| + d_{ge}|g\rangle\langle e|, \quad d_{ge}=d_{eg}^*.$$

Write the field in the standard slowly-varying-envelope form relative to a carrier $\omega_0$:

$$\vec E(t) = \varepsilon(t)\,e^{-i\omega_0 t} + \varepsilon^*(t)\,e^{+i\omega_0 t}.$$

Move to the interaction picture w.r.t. $H_0=\hbar\omega_{eg}|e\rangle\langle e|$
(so $|e\rangle\langle g|\to e^{+i\omega_{eg}t}|e\rangle\langle g|$), and apply the rotating-wave
approximation (drop terms oscillating at $\pm(\omega_{eg}+\omega_0)$, keep only the
near-resonant $\omega_{eg}-\omega_0$ combination — which is *exactly zero* when the atom is on
resonance with the rotating frame, i.e. $\omega_0=\omega_{eg}$, as is the case for the base
K↔L3 pair in this code):

$$H_{int}^{(RWA)}[e,g] = -d_{eg}\,\varepsilon(t), \qquad H_{int}^{(RWA)}[g,e] = -d_{ge}\,\varepsilon^*(t).$$

**This is the rule, and it is unconditional:** the matrix element with the *upper* state as the row
index couples to $\varepsilon(t)$ (this code's `Omega_plus`); the matrix element with the *lower*
state as the row index couples to $\varepsilon^*(t)$ (`Omega_minus`). It has nothing to do with
which array index is numerically larger — that's only a bookkeeping choice.

### 2.1 How the existing code encodes this rule

`XLO_sim.py` builds:
```python
Tijs_plus  = Tijs * Hij     # Hij[i,j] = 1 iff i > j  (np.tril(ones, -1))
Tijs_minus = Tijs * Hij.T   # i.e. 1 iff i < j
```
and `Model.py`'s `Hint[i,j] = 1j*(Tijs_plus[i,j,s]*Omega_plus[s] + Tijs_minus[i,j,s]*Omega_minus[s])`.

Since `Tijs` is set symmetrically (`Tijs[i,j,s]=Tijs[j,i,s]`, required by Hermiticity of a real
dipole operator), exactly one of `Tijs_plus[i,j]`/`Tijs_minus[i,j]` survives for any $i\neq j$,
selected purely by **whether $i>j$ as raw array indices** — not by whether $i$ is the physically
upper state.

For the base block this is invisible because it happens to be true by construction: K sits at
indices 4,5 and L3 sits at 0–3, so "$i>j$" and "$i$ is physically upper (K)" coincide for every
K↔L3 pair. **The `Hij[i,j]=\mathbb 1[i>j]` masking is only a correct proxy for "$i$ is upper" when
every upper-manifold index is larger than every lower-manifold index it couples to.**

### 2.2 Where it breaks

The Part III implementation appended 2p$_{1/2}$ at indices 6,7 — *after* K (4,5) — because K's
indices were already hardcoded elsewhere and couldn't be moved without touching the whole base
block. That makes L2 (physically **lower** than K) have a **larger** raw index than K. For the
K↔L2 pair, `i>j` is now true when $i=$L2 (the physically lower state), backwards from what
§2's rule requires. Concretely: `Tijs_plus[i_p, 4, 1]` (L2 row, K column) survived the masking and
got the `Omega_plus` coupling, when physically it's `Tijs_plus[4, i_p, 1]` (K row, L2 column) that
should. **This is a real bug**, not a sign/phase-convention ambiguity: it swaps which field
component drives absorption vs. emission-like dynamics on the K↔L2 pair, changing the sign of the
net stimulated response.

Confirmed two ways:
- Directly, by tracing `Hij[4,7]` (index 4 = K, 7 = L2 upper-$m$ sublevel): `4<7` so `Hij[4,7]=0`,
  meaning `Tijs_plus[4,7,*]` is **structurally forced to zero regardless of the value stored in
  `Tijs[4,7,*]`** — no value-only fix (sign flip, σ-index swap) can repair this, which is exactly
  why two independent empirical attempts at value-level fixes (documented in the chat history) had
  *zero* effect on the simulated spectrum.
- Numerically: the buggy code produced a spurious ~5% *gain* peak at −19.3 eV (should be
  Kα2's detuning, −19.93 eV) that got **bigger**, not smaller, when the competing K↔L3 coupling was
  artificially weakened — inconsistent with an interference effect, consistent with the K↔L2 pair
  independently having its stimulated-response sign backwards.

## 3. The fix: role-based masking, not index-based masking

Replace the index-comparison mask with a mask built from **physical role**:

```python
ei_upper = self.ei_K                    # true for every "upper" (1s-hole) sublevel
ei_lower = self.ei_L3 + self.ei_L2      # true for every "lower" (2p-hole) sublevel, any manifold
role_mask_upper_lower = np.outer(ei_upper, ei_lower)   # [i,j]=1 iff i upper, j lower
self.Tijs_plus  = np.einsum('ijs, ij->ijs', self.Tijs, role_mask_upper_lower)
self.Tijs_minus = np.einsum('ijs, ij->ijs', self.Tijs, role_mask_upper_lower.T)
```

**Why this is safe:** whenever `ei_L2` is all-zero (`use_L2_pathway=False` — every existing config,
including every satellite-channel config, since L2 and satellites are mutually exclusive and
satellite channels reuse these same `Tijs_plus`/`Tijs_minus`), `role_mask_upper_lower` reduces to
exactly `ei_K[i]*ei_L3[j]`, which is provably identical to the old `Hij[i,j]` for every $(i,j)$
pair where `Tijs` is actually nonzero (verified by direct enumeration of all 8 base-block entries
and the 2-level block's 2 entries — the only places `Hij` and the role mask disagree are
same-manifold pairs like K↔K or L3↔L3, where `Tijs` is identically zero anyway, since there is no
dipole coupling between degenerate sublevels of the same $j$-manifold). Confirmed numerically:
`Tijs_plus`/`Tijs_minus` are bit-for-bit identical to the pre-fix values for every non-L2 config
tested (`Cu-seed.yaml`, `Cu-seed-2-level.yaml`, `Cu-seed-satellite.yaml`).

This also fixes `Model.py::Omega_source_regular` (the field *emitted/absorbed by* the medium)
automatically, since it's built generically from `X.Tijs_plus`/`X.Tijs_minus` rather than
re-deriving the masking itself.

**Why this is the *general* fix, not a patch:** it doesn't matter what array index a future
manifold occupies — correctness now depends only on `ei_upper`/`ei_lower` being populated
correctly, which every manifold-construction block already does for its own diagonal-population
bookkeeping anyway.

## 4. Re-verifying items (1) and (2) independently

### 4.1 $T_{ij\sigma}$, $G_{ij}$ (Clebsch–Gordan)

Re-derived via `sympy.physics.wigner` (`wigner_3j`, `wigner_6j`), using the standard
Wigner–Eckart reduced-matrix-element formula for a single-electron dipole operator between
$j_i=1/2$ ($1s$) and $j_f\in\{1/2,3/2\}$ ($2p$), with Condon–Shortley phases:

$$\langle l_f j_f\|C^{(1)}\|l_i j_i\rangle = (-1)^{j_f+l_i+3/2}\sqrt{(2j_f+1)(2j_i+1)}\;
\begin{Bmatrix}l_f&j_f&1/2\\j_i&l_i&1\end{Bmatrix}\langle l_f\|C^{(1)}\|l_i\rangle.$$

**Validation against the existing (trusted) 2p$_{3/2}$ block:** applying this formula to all four
$m$-resolved 1s↔2p$_{3/2}$ transitions reproduces the codebase's existing `Tijs` values for that
block **exactly**, sign included (all 8 entries, both magnitude and sign) — confirming the formula
and phase convention match what's already validated in production.

**Applied to 2p$_{1/2}$:**
$$
\langle 1s(m{=}{-}\tfrac12)|T_{q=+1}|2p_{1/2}(m{=}\tfrac12)\rangle = -\sqrt2/3, \qquad
\langle 1s(m{=}\tfrac12)|T_{q=-1}|2p_{1/2}(m{=}{-}\tfrac12)\rangle = +\sqrt2/3.
$$

The two $m$-branches carry **opposite relative sign** — unlike 2p$_{3/2}$'s branches, which are
same-sign. This is a genuine angular-momentum-algebra fact ($j=l-\tfrac12$ manifolds pick up a
sign the $j=l+\tfrac12$ manifold doesn't), not a free convention. The original Part III
implementation used $+\sqrt2/3$ for both branches; this has been corrected to $-\sqrt2/3$ for the
$m{=}{-}\tfrac12\to m{=}\tfrac12$ branch.

**Caveat, checked directly:** this specific sign only matters if the codebase's field decomposition
ever lets the two branches interfere (e.g. a non-symmetric seed polarization). For the seed models
tested here it provably doesn't change any observable (confirmed by flipping it and re-running —
bit-for-bit identical output), because each branch only couples to one other level with no
competing amplitude at the same frequency to interfere with. It is still corrected here because (a)
it *is* the textbook-correct value and (b) a future seed field or extension (elliptical
polarization, additional coupled levels) could make it observable, and there's no reason to leave a
known-wrong constant in the code once identified.

$G_{ij}$ (branching ratios) were **not** part of the bug. Independently verified two ways:
- $T_{ij\sigma}^2=G_{ij}$ self-consistency, matching the same pattern the base 2p$_{3/2}$ block
  already satisfies for every entry.
- Total branching out of each K sublevel sums to 1 and splits **exactly 2:1** between L3 and L2
  (e.g. from $m=-\tfrac12$: $1/3+2/9+1/9=2/3$ to L3, $2/9+1/9=1/3$ to L2), matching the physical
  Kα1:Kα2 intensity ratio independently of the config's empirical $\Gamma_{rK\alpha1}/\Gamma_{rK\alpha2}=1.94$.

### 4.2 $\Delta_{ij}$ (detuning) — revisited in §8

The first pass at this document claimed $\Delta_{ij}$ was independently correct, based on it being
the residual rotating-frame frequency $\omega_{K\alpha2}-\omega_0$ and on the spurious feature (at
the time) sitting on the correct side of the spectrum. That check was **confounded**: it implicitly
assumed a specific mapping between "which coherence has which residual frequency" and "which side
of the FFT'd spectrum the feature appears on," without verifying that mapping against the actual
field-sourcing code (`Omega_source_regular`) — which, at the time, was still using the *buggy*
`Tijs_plus`/`Tijs_minus` routing from §2–3. Once that routing was fixed, the same $\Delta_{ij}$
produced a genuine absorption dip (confirming §3's fix worked) but on the **wrong side** (+20 eV
instead of −20 eV) — see §8 for the corrected, routing-fix-aware re-derivation and the resulting
sign correction.

### 4.3 Everything else audited for the same class of bug

Checked every other array that couples two different manifolds, specifically for hidden
index-order dependence (the same failure mode as §2–3):

| Array | Directional risk? | Verdict |
|---|---|---|
| `Gamma_sp_Gij` (spontaneous-emission feed) | Used directly as `feed[i] += Gij[i,s]*rho[s,s]`, no index-comparison masking | **Safe** — direction is whatever's explicitly coded (`Gij[i_p,4]` means "feed into L2 from K", written correctly) |
| `Mij` (dephasing) | Built via `outer(a,b)+outer(b,a)` | **Safe** — manifestly symmetric by construction |
| `Delta_ij` | Not masked by `Hij`, but its sign must be consistent with which coherence the (fixed) routing actually sources | **Bug, fixed** (§8) — a second, independent sign error, exposed only after §3's fix |
| `S_ground_Fi`, `S_ion_Fi`, `gamma_ion` | Per-level scalars, no $(i,j)$ pairing | **Not applicable** — no ordering to get wrong |
| `Tijs_plus`/`Tijs_minus` | Built via `Hij[i,j]=\mathbb1[i>j]` | **Bug, fixed** (§3) |

## 5. On array ordering going forward

The role-based mask (§3) makes this a non-issue for L2 specifically and for any future manifold,
**provided** every manifold's `ei_*` indicator array is correctly populated (which every
manifold-construction block must do anyway, for its own diagonal bookkeeping) — array position no
longer matters. This is the recommended pattern for any future coupled manifold: don't rely on
`Hij`/raw index comparisons anywhere; always build couplings from `ei_upper`/`ei_lower`-style
physical-role masks. If a reader is tempted to "just reorder the array so K is last," resist it —
K's index is hardcoded in many places in the base-block construction, and reordering doesn't fix
the underlying fragility, it just moves which future extension will trip over it. The general fix
means it now doesn't matter which order manifolds are appended in.

## 6. First validation pass (before §8's fix)

1. **Regression, all non-L2 configs**: `Tijs_plus`/`Tijs_minus` bit-for-bit identical to pre-fix
   values for `Cu-seed.yaml`, `Cu-seed-2-level.yaml`, `Cu-seed-satellite.yaml`. Zero behavior change
   for any config that doesn't use L2. (This holds after §8's fix too — `Delta_ij` is identically
   zero whenever `use_L2_pathway=False`.)
2. **Spurious gain peak eliminated**: the ~5% gain peak at −19.3 eV in `Cu-seed-L2.yaml`'s
   transmitted spectrum was gone after §3's fix — but a re-run with a wider window revealed it had
   been replaced by a genuine **dip on the wrong side**, at +20 eV instead of −20 eV. That
   observation is what motivated §8.

## 7. A second bug the routing fix exposed: $\Delta_{ij}$'s sign

With §3's fix in place, `Cu-seed-L2.yaml`'s transmitted spectrum showed a real, localized
absorption dip — the right *kind* of feature (confirming §3 fixed the physical mechanism) — but
centered at **+20 eV**, the mirror image of where Kα2 (20 eV *below* Kα1) should produce it.

This traces back to §4.2's confounded check. Re-deriving it properly, tracking the sign all the way
through to the FFT convention actually used by `tools.SF_spectrum_w`:

1. **The FFT convention** (`Optics.py::my_fft_phased`, wrapping `np.fft.fftn`) is the standard
   $F(\omega)=\sum_t f(t)\,e^{-i\omega t}$. A time-domain term $f(t)\propto e^{+i\Omega t}$
   ($\Omega>0$) therefore produces an FFT peak at **positive** $\omega=+\Omega$ (no extra sign flip
   anywhere in `nd_kspace`/`fftfreq`/the eV-conversion coefficients — checked directly).
2. **What sources `Omega_plus`** (`Model.py::Omega_source_regular`):
   `Omega_plus_source[s] = Σ_ij Tijs_minus[i,j,s]·ρ_hermitian[j,i]`. After §3's fix,
   `Tijs_minus[i,j]` survives only for $i=$lower, $j=$upper — so for the K↔L2 pair the surviving
   term is `Tijs_minus[L2,K,s]·ρ_hermitian[K,L2]`, i.e. `Omega_plus` is sourced by (dominantly)
   $\rho_{K,L2}(t)$.
3. **How $\rho_{K,L2}(t)$ oscillates**: from `Model.py`'s off-diagonal equation,
   $d\rho_{ij}/dt \supset -i\,\Delta_{ij}[i,j]\,\rho_{ij}$, so $\rho_{K,L2}(t)\propto
   e^{-i\,\Delta_{ij}[K,L2]\,t}$ (and $\rho_{hermitian}[K,L2]$ inherits the same frequency, since
   $\Delta_{ij}$'s built-in antisymmetry makes $\overline{\rho_{L2,K}}$ oscillate the same way).
4. **Putting it together**: `Omega_plus`$(t)\propto e^{-i\,\Delta_{ij}[K,L2]\,t}$, i.e.
   $\Omega=-\Delta_{ij}[K,L2]$ in step 1's notation, so the FFT peak lands at
   $\omega=-\Delta_{ij}[K,L2]$.

For the peak to land at the physically correct $\omega=-\Delta\omega_{L2-L3}$ (negative — Kα2 below
Kα1), step 4 requires $\Delta_{ij}[K,L2]=+\Delta\omega_{L2-L3}$ — the **opposite sign** from what
`f_detuning = +\Delta\omega_{L2-L3}\cdot ei_{L2}$ was producing
($\Delta_{ij}[K,L2]=f[K]-f[L2]=-\Delta\omega_{L2-L3}$). **Fix:** flip the sign of `f_detuning`:

```python
f_detuning = -self.DeltaomegaL2mL3A * self.ei_L2 if self.use_L2_pathway else np.zeros(self.nlevel)
```

This is a single, well-isolated sign flip, identically zero (no-op) whenever `use_L2_pathway=False`
— confirmed by re-checking `Delta_ij` is all-zero for every non-L2 config after this change.

**Why this wasn't caught earlier:** §4.2's original check only verified $\Delta_{ij}$'s value
matched a physical-sounding formula (residual rotating-frame frequency), not that the formula's
sign convention was consistent with which specific coherence the *rest of the code* — specifically
the field-sourcing step, which depends on the routing fixed in §3 — actually uses. Two independent
bugs (§3's routing, §7's detuning sign) happened to partially mask each other: before §3's fix, the
wrong routing sourced `Omega_plus` from a different (wrong) coherence that, combined with the
(also, it turns out, wrong-signed) $\Delta_{ij}$, happened to place the spurious feature on the
correct *side* of the spectrum even though its physical *character* (gain vs. absorption) was
wrong. Fixing one bug in isolation exposed the other. The lesson generalizes: when a numerical
result matches expectation on one axis (frequency side) but not another (gain vs. absorption sign),
don't stop checking once the first axis matches — trace the *whole* causal chain (here: EOM → which
coherence sources which field component → FFT convention) rather than pattern-matching against a
single expected feature.

## 8. Full validation, after both fixes

1. **Regression, all non-L2 configs**: `Tijs_plus`/`Tijs_minus` and `Delta_ij` bit-for-bit identical
   to pre-fix values (`Delta_ij` all-zero) for `Cu-seed.yaml`, `Cu-seed-2-level.yaml`,
   `Cu-seed-satellite.yaml`.
2. **Spectral shape, `Cu-seed-L2.yaml`**: two clean, localized absorption dips — one at
   $\omega\approx-0.09$ eV (Kα1, depth to ${\sim}0.23$ against a ${\sim}0.37$ local baseline) and
   one at $\omega\approx-19.3$ eV (Kα2, depth to ${\sim}0.32$), matching the expected
   $-\Delta\omega_{L2-L3}=-19.93$ eV to within the feature's own width. The mirror-image region
   at $+20$ eV is now smooth/featureless (consistent with the seed pulse's own broad envelope, not
   a resonance). This is the qualitatively correct picture — two genuine absorption resonances,
   correctly positioned, of the same character (dips) — matching the real SACLA data's Kα1+Kα2
   reverse-saturable-absorption structure far better than either intermediate (buggy) state did.

## 9. Open items / not yet resolved

- **Quantitative comparison to SACLA data** is still open — this repo's single-Gaussian-pulse test
  config (`Cu-seed-L2.yaml`, nominal FWHM 0.1 fs) is not a SASE pulse and isn't averaged over shots;
  matching dip *depth*/*width*/intensity-scaling quantitatively would need a more realistic,
  multi-shot SASE-averaged simulation, not just the qualitative shape check done here.
- **Ground-state $m$-branching into 2p$_{1/2}$** is still a 50/50 placeholder (theory doc §20's
  existing caveat, unchanged by this investigation).
- **$S_{ion,Fi}$ for L2** uses a flat `sigma2_Ka1_2p1` across both $m$-sublevels, unlike the base
  block's $m$-resolved asymmetric factors (e.g. 0.75/1.25 for K, 0.70/0.83/1.06/1.41 for L3) — not
  independently re-derived, flagged as a simplification consistent with the ground-state branching
  caveat above.
- **The satellite channels' own `Delta_k`/`sign_ij_block` sign** was not re-derived with the same
  rigor as §8 here (out of scope for this session — satellite channels don't touch the code path
  that was buggy). Given §8 shows this class of sign error is easy to introduce and easy to miss
  with a superficial check, it would be worth applying the same "trace the whole causal chain
  through `Omega_source_regular` and the FFT convention" method to the satellite mechanism too, as
  a follow-up, rather than assuming its existing validation (theory doc §14) was equally rigorous
  on this specific point.
