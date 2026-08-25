# Planning: double-spectator-hole satellite channels (3d+3d+, 3d-3d+, 3d-3d-)

This document plans and records the implementation of a third generation of satellite channels,
extending `docs/theory-and-2s-satellite-pathways.md` Part II (single-spectator Kα1-satellites,
e.g. `2p+3d+`) one hole deeper, to double-M-shell-spectator states `2p+3d+3d+`, `2p+3d-3d+`,
`2p+3d-3d-`. Numbers here are traceable to three independent calculations done in the investigating
session (a hand-derived σ·Φ estimate, a stochastic `xatom_mc_fly` Monte Carlo cascade, and a
deterministic rate-equation solve over the user-supplied `xatom/transitions_dict_relativistic.json`
+ `xatom/f_line_dict_relativistic.json` databases), which agree with each other to within a factor
of ~2.

## 1. Motivation and the production mechanism actually used

The naive route to a double-spectator state — further-photoionizing an already-populated
single-spectator channel's own spectator hole (generalizing §12.4's "further-ionization loss") —
is **not** how these states get populated in practice. It's cross-section-limited (σ ~ 10⁻¹⁰–10⁻⁹
nm²) against the weak seed/Kα1 field flux, and predicts peak populations around 10⁻⁷–10⁻⁸ —
negligible. Two independent checks confirmed a different, dominant mechanism instead:

- **`xatom_mc_fly`** (a stochastic multi-photon/Auger cascade code, same tool as
  `xatom/F_lines_XATOM.ipynb`), run at this repo's actual toy-grid pulse parameters (400
  trajectories), showed 0.5% of trajectories transiently visiting a 2p-hole + double-3d-hole
  configuration — three to four orders of magnitude above the σ·Φ estimate. Tracing one hit
  trajectory showed the mechanism directly: a 2s-hole Auger-decays into `2p+3p+` (the existing
  `3p+` single-spectator channel), and **that 3p spectator hole itself then Auger-decays a second
  time**, ejecting a second M-shell electron while the 2p₃/₂ hole sits untouched as a bystander —
  landing on `2p+3d-3d+`.
- Running `xatom -hole 3p0,1 -decay` (bare 3p₃/₂ hole) confirmed this isn't a fluke: **>90% of a
  bare 3p-hole's total decay width goes into ejecting a second 3d electron** (M₂₃-M₄,₅M₄,₅
  Coster-Kronig). By contrast, `xatom -hole 3d0,1 -decay` (bare 3d-hole) has **no Auger channel at
  all** — filling a 3d vacancy from 4s only releases ~7.8 eV, ~0.6 eV short of the ~8.5 eV needed
  to eject a second weakly-bound electron, so a lone 3d spectator is Auger-stable. This is why the
  `3d+`/`3d-` channels don't self-feed a double-3d state, but `3p+`/`3p-` do.

This means the physically dominant, and by far the cheapest to implement, production route is:
**redirect a fraction of the `3p+`/`3p-` single-spectator channels' own `Gamma_L_eV` decay
(currently 100% generic loss in the code) into three new channels**, using XATOM's own
already-computed branching ratios for that redirection. No new cross-section-driven feed term, no
new precursor population, no pump-flux plumbing is needed.

## 2. Which channels, and how much population

Running `xatom -hole 2p0,1_3p0,1 -decay` / `-hole 2p0,1_3p1,0 -decay` directly (the `3p+`/`3p-`
channels' own `L_k` configurations) gives every Auger line, split by whether the *spectator*
decays (2p-hole survives, feeds a double-spectator state) or the *2p-hole itself* decays (ends the
channel's own coherent Kα1-satellite emission — this remains generic loss, unchanged). Cross-checked
against the independently-generated `transitions_dict_relativistic.json`'s own `A`-type rows for
the same two configurations — agreement to 3 significant figures on every branching fraction below.

| Target channel | Detuning vs Kα1 | Γ_L (eV) | Γ_K (eV) | fed from `3p+` (eV) | fed from `3p-` (eV) |
|---|---|---|---|---|---|
| `3d+3d+` | −1.70 eV | 0.572 | 1.553 | 0.637 | 0.046 |
| `3d-3d+` | −1.69 eV | 0.603 | 1.553 | 1.027 | 1.688 |
| `3d-3d-` | −1.68 eV | 0.628 | 1.553 | 0.081 | 0.460 |

(Detunings are ~2.1× the corresponding single-spectator channel's own detuning — e.g. `3d+` alone
is −0.80 eV — consistent with an additive "one more screening charge" picture and independently
reproduced by the `f_line_dict_relativistic.json` energies, −1.61 eV there vs −1.70 eV from the
live calibrated calculation; the ~5% difference is method noise between the two XATOM invocation
styles, not a discrepancy worth chasing further.)

A deterministic rate-equation solve (reproducing `F_lines_XATOM.ipynb`'s own `dp/dt = rate_matrix
@ p` approach, but run to completion over the full 1340-configuration database at this repo's
SASE-scale pulse parameters) gives peak populations for these exact configurations directly,
independent of the branching-ratio arithmetic above:

| Configuration | Peak population | vs. single-spectator `3d+` (4.40e-3 in the same calc) |
|---|---|---|
| `2p+3d-3d+` (mixed) | 2.83e-3 | 64% |
| `2p+3d+3d+` (pure) | 1.16e-3 | 26% |
| `2p+3d-3d-` (pure) | 4.13e-4 | 9% |

All three are **comparable in order of magnitude to the existing single-spectator channels**, not
a small correction — this is the headline result that makes implementing them worthwhile. `3d-3d+`
(the mixed channel) dominates; `3d-3d-` is the smallest of the three but still non-trivial. Triple-
and higher-hole states appear in the same database at comparable-or-smaller populations
(e.g. one triple-3d config reaches 5.5e-4) but are out of scope here — see §7.

## 3. Architecture: reuse `satellite_channels`, don't duplicate it

The existing per-channel machinery (`XLO_sim.py`'s `satellite_channel_params` construction,
`Model.py`'s `MB_satellite_block_regular`/`feed_diag_satellite_block`, `Sample.py`'s marching loop,
`Model.absorption`, `Omega_source_regular`) is already fully generic over "a list of local 6-level
blocks, each with its own `Mij`/`Delta_ij`/feed". A double-spectator channel is structurally
*exactly* that — same `Tijs`/`Gij` (spectator approximation, reused verbatim, same argument as
every other channel in Part II) — the **only** things that differ are (a) its `Mij` uses its own
`Gamma_L_eV`/`Gamma_K_eV`, and (b) **where its feed comes from**.

So: double-spectator entries are appended into the *same* `satellite_channels` list (internally),
distinguished by a new optional per-entry key, `feed_from`, instead of the existing
`Gamma_A_2s_eV`/`sigma_Ka1_from_2p`/`sigma_Ka1_from_1s` keys:

```yaml
satellite_channels:
- name: 3p+
  ...                    # existing keys unchanged
  Gamma_L_eV: 0.887       # WAS 2.6325 -- carved out: 1.745 (sum of feed_from Gamma_feed_eV
                          # values below across all double-satellite channels fed from 3p+)
- name: 3p-
  ...
  Gamma_L_eV: 0.985       # WAS 3.1793 -- carved out: 2.194

double_satellite_channels:
- name: 3d+3d+
  detuning_eV: -1.70
  Gamma_L_eV: 0.572
  Gamma_K_eV: 1.553
  feed_from:
  - channel: 3p+
    Gamma_feed_eV: 0.637
  - channel: 3p-
    Gamma_feed_eV: 0.046
- name: 3d-3d+
  ...
- name: 3d-3d-
  ...
```

`XLO_sim.py` loads `double_satellite_channels` and appends its entries onto the same internal
`self.satellite_channels` list (so `X.satellite_channel_params`, `rho_sat_ijtxyz`, and every
downstream consumer stay a single flat list — no new arrays, no new Sample.py loop). Each entry's
`feed_from` is resolved to a list of `(parent_index, Gamma_feed_fs)` pairs at construction time
(parent looked up by name in `self.satellite_channels`, must appear *before* the double-satellite
entry in Python-dict iteration order — enforced, not assumed).

**The carve-out is exactly the established pattern** (§12.7 of the theory doc, and how
`sigma2_Ka1_1s`/`sigma2_Ka1_2p3` were already reduced when `satellite_channels` was first added):
the fed fraction must be *subtracted* from `3p+`/`3p-`'s own `Gamma_L_eV`, or that population
budget is double-counted (once as generic loss via the parent's own `Mij`, once again as explicit
double-satellite feed). Verified: `0.887 + 1.745 = 2.632` (matches `3p+`'s original/bare total decay
width exactly); `0.985 + 2.194 = 3.179` (matches `3p-`'s).

### Feed mechanism (`Model.py`)

`feed_diag_satellite_block(X, chan, rho_2s_xy, rho_base_ijxy, rho_sat_ijxy, J_Omega_minus_xy,
J_Omega_plus_xy)` gains one new parameter, `rho_sat_ijxy` (the *full* list of every satellite
channel's current local block, pre-update — same timestep convention as `rho_base_ijxy`/`rho_2s_xy`
already use). When `chan.feed_from` is non-empty:

```
feed[0:4] = sum_over_parents( (Gamma_feed_fs / 4) * diag(rho_sat_ijxy[parent_index])[0:4].sum() )
```

spread evenly over the lower/`L_k`-manifold's 4 msublevels — same "no known angular dependence to
do otherwise" convention Eq. S2 already uses for the 2s-Auger feed, and physically apt here too
(the ejected/filling Auger electrons are M-shell, not L-shell, so there's no new information about
which of the *parent's own* 4 already-populated `2p+` msublevels is more likely to end up in which
of the *child's* 4 `2p+` msublevels — spread evenly). No feed into the upper (`U_k`, `1sX_kX_k'`)
manifold in v1 (§7).

When `chan.feed_from` is empty/absent, behavior is **byte-for-byte identical** to the existing
code path (the four original channels are unaffected structurally; only their `Gamma_L_eV` values
change in the YAML, which is data, not code).

### `Sample.py`

`MB_satellite_block_regular`'s call site (both `_evaluate_n_level_3D_full` and `_lean`) passes the
*whole* pre-update `rho_sat_ijxy` list into every channel's own `d_rho_sat_it` computation — trivial
addition, one extra list in the `params` tuple, computed once per `it` before any channel's rho is
updated (already the existing ordering: `d_rho_sat_it` is a list comprehension over all channels
*before* the `rho_sat_ijxy[k] = rho_sat_ijxy[k] + d_rho_sat_it[k]` update loop, so every channel
(parent or child) reads every other channel's pre-update value regardless of list order — no new
ordering dependency introduced).

### `xatom_tools.py`

Two new functions, both reusing 100% of the existing XATOM-invocation/parsing machinery:

- `double_spectator_channel_parameters(name, spectator_pair, Ka1_energy_eV)` — generalizes
  `satellite_channel_parameters` to a *pair* of spectator fragments (e.g. `('3d+','3d+')`),
  building hole configs like `2p0,1_3d0,1_3d0,1`... — **correction, see §4** — and returning
  `detuning_eV`/`Gamma_L_eV`/`Gamma_K_eV` via the same `transition_energy_eV`/
  `state_total_decay_width_eV` calls already used everywhere else, just with double-hole XATOM
  fragments (`3d0,2` for `3d+3d+`, `3d1,1` for `3d-3d+`, `3d2,0` for `3d-3d-` — XATOM's own
  `"nl<n_->,<n_+>"` notation, confirmed to work directly, no new XATOM-side support needed).
- `spectator_self_auger_feed_eV(parent_spectator, target_pair)` — runs `-decay` on
  `2p0,1_{parent_spectator}` and returns the partial rate(s) for rows where the initial hole is the
  *spectator's own label* (not `2p+`) and the two final labels match `target_pair` (either order).
  This is what fills in `feed_from`'s `Gamma_feed_eV` values; also returns the *sum* over all such
  rows so the YAML carve-out comment can be generated/checked automatically.

## 4. XATOM hole-config detail

A double-spectator `L_k`/`U_k` pair's hole config is **not** `2p0,1_3d0,1_3d0,1` (repeating the
single-hole fragment) — XATOM's notation packs *all* holes in one subshell into one fragment
(confirmed empirically in the investigating session): `3d0,2` = 2 holes in 3d₅/₂ (`3d+3d+`),
`3d1,1` = 1 hole in 3d₃/₂ + 1 in 3d₅/₂ (`3d-3d+`), `3d2,0` = 2 holes in 3d₃/₂ (`3d-3d-`). So:
`lower_hole = f'2p0,1_{sd}'`, `upper_hole = f'1s1_{sd}'`, with `sd` one of `{'3d0,2', '3d1,1',
'3d2,0'}` directly — no generalization of the fragment-building logic needed, just a new
`DOUBLE_SPECTATOR_HOLE` lookup dict mirroring `SPECTATOR_HOLE`.

## 5. Consistency checks

- `double_satellite_channels: []` (absent) must reproduce prior behavior exactly — the new
  `feed_from`-aware branch in `feed_diag_satellite_block` is dead code with no channels to trigger
  it, and the parent channels' `Gamma_L_eV` values only change if the YAML is edited to carve them
  out, which only happens in configs that also define the new channels.
- Carve-out budget: `Gamma_L_eV(3p+)_new + sum(Gamma_feed_eV from 3p+) == Gamma_L_eV(3p+)_bare`
  (2.632 eV), same for `3p-` (3.179 eV) — checked by hand above; `XLO_sim.py` should assert this at
  construction time (loudly, not silently) given how easy it is to get a carve-out wrong (§12.7's
  own warning, borne out already once in this codebase's history for `GammaA_L1_to_L3M45eVN`).
- Population trace (sum over ground/other/2s/base-block/all single- and double-satellite channels)
  must stay ≤ 1 and decrease only via documented untracked-loss channels, exactly the existing
  Part II/§14 check, now extended one tier deeper.
- With a double-satellite channel's own `Gamma_L_eV`/`Gamma_K_eV` set equal to its parent's
  (unphysical, but a useful debugging limit), its coherent response should be structurally
  identical to a regular single-spectator channel's — it *is* one, just fed differently.

## 6. What this does NOT cover (deferred)

- **Triple-and-higher-hole states**: the rate-equation ranking (§2) shows some triple-3d
  configurations at comparable population to the double-3d states here, but are out of scope for
  this pass — a natural next step once these three are validated against real data.
- Mixed `3p+3d`-spectator double-hole states (e.g. `2p+3p+3d+`) were checked in the earlier
  back-of-envelope pass and found smaller/less cleanly separated than the pure-3d-double states
  prioritized here; not implemented.
- (§9 below: the L2k extension and further-ionization loss, both originally listed here, are now
  implemented.)

## 7. Upper-manifold (`U_k`) self-feed — implemented

`1s3p+`'s own 3p spectator hole **does** Coster-Kronig-decay a second time, exactly like its
`2p+3p+` counterpart, landing on `1s3d+3d+`-type states while the 1s core hole survives — confirmed
directly via `xatom -hole 1s1_3p0,1 -decay` / `-hole 1s1_3p1,0 -decay`:

| Target channel | fed from `1s3p+` (eV) | fed from `1s3p-` (eV) |
|---|---|---|
| `3d+3d+` | 0.7303 | 0.0499 |
| `3d-3d+` | 1.2235 | 1.9699 |
| `3d-3d-` | 0.0852 | 0.5104 |

Branching into pure double-3d states is smaller than the lower-manifold case (53–58% of
`Gamma_K_eV`, vs. 66–69% of `Gamma_L_eV`) — expected, since the 1s core hole's own decay (KLL/KLM
Auger, `Gamma_K_eV` overall being much larger than `Gamma_L_eV`) is a faster competing channel than
the 2p+ core hole's own decay was. Still a large, non-negligible fraction, and confirmed nonzero,
so implemented rather than dropped.

**Mechanism** (`Model.py`'s `feed_diag_satellite_block`): `chan.feed_from` entries are now
`(parent_index, Gamma_feed_fs, manifold)` triples. `manifold='lower'` (default, omitted from the
YAML) feeds the child's own lower (`ei_L3_sat`, local 0–3) manifold from the parent's lower-manifold
population, spread via `auger_weight = ei_L3_sat/sum(ei_L3_sat)` — unchanged from §3.
`manifold='upper'` instead reads the parent's *upper* (`ei_K` local, indices 4–5) population and
spreads it via the analogous `auger_weight_K = ei_K_local/sum(ei_K_local)` onto the child's own
upper manifold. Both cases write through the same `feed[:nlevel_base] += einsum('i,xy->ixy',
dst_weight, ...)` line — `dst_weight` is zero outside its own manifold's indices by construction,
so no separate slicing is needed.

**Carve-out**: exactly the same pattern as §3, but against `Gamma_K_eV` instead of `Gamma_L_eV`.
`3p+`: $1.8134 + 2.0390 = 3.8524$ eV (bare). `3p-`: $1.8347 + 2.5302 = 4.3649$ eV (bare). Both
verified exact (config's pre-existing `Gamma_K_eV` values). YAML entries needing the upper manifold
carry an explicit `manifold: upper` key; entries without it default to `'lower'`
(`XLO_sim.py` raises if any other string is given).

Verified end-to-end on `config/base/Cu-seed-double-satellite.yaml`: `U_k` populations for the three
double-satellite channels come out to roughly 2–3% of their own `L_k` population (e.g. `3d+3d+`:
$U_k=8.1\times10^{-6}$ vs $L_k=3.1\times10^{-4}$) — small but nonzero, consistent with `U_k` being a
short-lived intermediate that rapidly decays (via the channel's own coherent Kα1-satellite dynamics,
reusing `Tijs`/`Gij` verbatim as always) into its own `L_k`, the same qualitative pattern the base
block's 1s-hole (K) population has relative to its 2p-hole (L3) population. Population trace and
regression checks (§5) unaffected — `1.005570` max vs. `1.005525` without the upper-manifold feed,
both consistent with the same small pre-existing RK4 numerical drift the *unmodified* code already
has on this grid (checked: `Cu-seed-satellite.yaml`, no double-satellite channels at all, shows
`1.006546` on the same grid).

## 8. `3p-` bare-hole vs. channel-specific branching — cross-checked

`spectator_self_auger_feed_eV` (and therefore every number in §2's table and this section's) always
uses the *channel-specific* hole config (`2p0,1_3pX,Y` / `1s1_3pX,Y`, i.e. with the other core hole
present) — never the bare spectator-only hole. Confirmed this is the right choice, and quantified
how much it matters, by comparing against `xatom -hole 3p1,0 -decay` (bare 3p₃/₂⁻¹... i.e. 3p₁/₂
hole, no 2p+/1s spectator):

| | bare `3p1,0` | channel-specific `2p0,1_3p1,0` |
|---|---|---|
| total width | 3.293 eV | 3.1793 eV (config `Gamma_L_eV`, pre-carve) |
| → pure double-3d fraction | 91.6% | 68.9% |

Same qualitative pattern already found for `3p+` (bare: 90.8%, channel-specific: 66.1%) — both
spin-orbit sub-manifolds lose ~23–25 percentage points of branching to the *core* hole's own
competing decay once it's actually present, a good internal consistency check on the whole
mechanism. Using the bare-hole numbers instead (as a cruder, cheaper-to-compute approximation)
would have overestimated every `Gamma_feed_eV` value in §2's table by roughly this amount — the
implementation correctly uses the more expensive but more accurate channel-specific calls.

## 9. 2p1/2-satellite (L2k) extension — implemented

Each double-satellite channel now gets the full Part IV treatment one tier deeper: an own local
L2k (2p1/2+XX) manifold at indices 6–7 (`X.satellite_nlevel` grows to 8 exactly as it already does
for the single-spectator channels), detuned by its own `detuning_eV_L2_split`, with its own
`Gamma_L2_eV` width. **Feed mechanism**: identical in kind to §3/§7 — not a new cross-section or
2s-Auger term, but a `manifold='L2'` `feed_from` entry redirecting part of the *parent* channel's
own `Gamma_L2_eV` decay (its own L2k = 2p1/2+X configuration's spectator hole, self-Auger-decaying
a second time while the 2p1/2 core hole survives).

**A clean, unplanned-for confirmation of the spectator approximation**: running
`xatom -hole 2p1,0_3p0,1 -decay` / `-hole 2p1,0_3p1,0 -decay` (the L2k configurations of the `3p+`/
`3p-` channels) gives Auger branching into double-3d states that is *numerically identical, to 4
decimal places*, to the corresponding lower-manifold (`2p0,1_3pX,Y`) branching computed in §2 —
e.g. `3p+ -> 3d-+3d+` is 1.0241 eV whether the accompanying core hole is `2p3/2` or `2p1/2`. This
makes physical sense (the M-shell spectator's own decay shouldn't resolve the fine-structure detail
of a distant core hole) but wasn't assumed going in — it was checked, and held exactly rather than
approximately.

| Target channel | `detuning_eV_L2_split` | `Gamma_L2_eV` | fed from `3p+`'s L2k (eV) | fed from `3p-`'s L2k (eV) |
|---|---|---|---|---|
| `3d+3d+` | 21.26 eV | 0.6518 | 0.6350 | 0.0463 |
| `3d-3d+` | 21.26 eV | 0.6094 | 1.0241 | 1.6841 |
| `3d-3d-` | 21.26 eV | 0.5550 | 0.0806 | 0.4594 |

Carve-out: `3p+`'s own `Gamma_L2_eV` (bare 2.72362 eV, matching the pre-existing
`Cu-seed-satellite.yaml` value exactly) becomes $0.98392 + 1.7397 = 2.72362$; `3p-`'s (bare
3.05808 eV) becomes $0.86828 + 2.1898 = 3.05808$ — both exact, same pattern as §3/§7.

**Code**: `Model.py`'s `feed_diag_satellite_block` gains a `manifold == 'L2'` branch reading
`parent_rho[nlevel_base:nlevel_base+n_L2]` (the parent's own L2k diagonal) and writing into
`feed[nlevel_base:nlevel_base+n_L2]` (the child's own L2k diagonal) — both parent and child share
the same `nlevel_base`/`n_L2` since they're both members of the same flat `satellite_channels`
list with the same `X.satellite_nlevel`. `XLO_sim.py`'s required-keys check for the L2 extension is
now conditional: double-satellite (`feed_from`-bearing) entries only need
`detuning_eV_L2_split`/`Gamma_L2_eV` (their own intrinsic properties), not
`Gamma_A_2s_to_L2_eV`/`sigma_Ka1_from_2p1` (the old feed-mechanism keys, irrelevant here and
defaulted to 0). A `manifold: 'L2'` entry without `use_L2_pathway: True` raises at construction —
that manifold's index range wouldn't exist on either side.

Verified on `config/base/Cu-seed-double-satellite.yaml` (now `use_L2_pathway: True` throughout, no
longer the workaround it was in §5): every channel shows nonzero `L_k`/`U_k`/`L2k` populations, the
L2k contribution is substantial (e.g. `3d-3d+`: L2k $4.8\times10^{-4}$ vs. `L_k`
$7.2\times10^{-4}$ — the same order of magnitude, not a small correction), and population trace
stays in the same numerical-drift envelope as before (`1.007387` vs. the unmodified
`Cu-seed-satellite.yaml`'s own pre-existing `1.006546` on the same grid — see §5).

## 10. Further-ionization loss — implemented

`sigma_ion_from_2p`/`sigma_ion_from_1s` for each double-satellite channel, via the same
`total_photoionization_cross_section_nm2` every other channel already uses (a `-pcs` call on the
double-hole configuration itself — fast, unlike the `-decay` calls on 3p-spectator configs that
drive this document's other slow steps). All three channels come out numerically close to each
other and to the existing single-spectator channels' own values ($5.04\times10^{-7}$/
$5.74\times10^{-7}$ nm² for 2p-/1s-manifold further ionization, vs. $5.01\times10^{-7}$/
$5.72\times10^{-7}$ for the `3d+` channel) — the spectator approximation holding here too, as
expected (removing yet another electron is dominated by the core-hole-adjacent shells, largely
insensitive to which specific M-shell electron is already missing).
