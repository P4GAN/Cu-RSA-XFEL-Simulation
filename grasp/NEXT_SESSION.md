# GRASP/RATIP campaign — handoff for the next session

**Goal for the next session**: recompute *every* atomic-physics parameter in
`config/base/Cu-seed-SASE-double-satellite.yaml` / `Cu-seed-mono-SASE-double-satellite.yaml`
(and by extension the other `Cu-seed*.yaml` variants, which share the same base parameters) via
GRASP2018 + RATIP-2012 Dirac-Fock, replacing the XATOM-derived values wherever GRASP/RATIP can
reach them. This document is the complete technical briefing to pick that up cold. The
user-facing companion (`docs/grasp-ratip-explained.md`) covers the same ground in plain language
— read this one for the actual recipe.

**Read `grasp/README.md` and `grasp/grasp_tools.py`'s module docstring first** — they're shorter
and cover the same build/recipe basics. This document goes further: full parameter inventory,
worked prompt sequences for every GRASP/RATIP tool used so far, and the untried categories.

## 0. Scope: what actually needs to land in the config, and what doesn't

The target config files (`Cu-seed-SASE-double-satellite.yaml`, `Cu-seed-mono-SASE-double-
satellite.yaml`) have `use_2s_pathway: true` and `use_L2_pathway: true` — 2s and 2p_1/2 each get
their own explicit population/cross-section treatment, separate from the "other"/auxiliary
catch-all. Two scope decisions follow from this, both settled (don't re-litigate them):

- **No pump-energy values.** `sigma1_pump_*`/`sigma2_pump_*` keys don't exist in these files at
  all. The pump-energy calculations done so far were purely to validate the method against the
  original paper's published table — not part of what needs computing here.
- **"Other"/auxiliary = 5 channels, not 7.** Since 2s and 2p_1/2 have their own keys
  (`sigma1_Ka1_2s`, `sigma1_Ka1_2p1`, `sigma2_Ka1_2s`, `sigma2_Ka1_2p1`), every `*_other` value
  actually going into these config files means 3s + 3p_1/2 + 3p_3/2 + 3d_3/2 + 3d_5/2 only (no
  2s, no 2p_1/2, no 4s — this reduced atomic model has no 4s electron). This is DIFFERENT from
  the 7-channel sum used earlier to validate against the original (pre-split) paper table — see
  §1 for why that's not a contradiction.

## 1. What's validated already (don't redo this)

Everything below is checked into `grasp/` and has been independently re-verified to reproduce
these exact numbers (see `grasp/cross_section_validation.py`, runs standalone; and
`grasp/satellite_detuning_analysis.py`, needs the external working directory).

**Satellite detunings** (bound-bound, via `rbiotransform`+`rtransition`) — all 4 single-spectator
channels (3d+, 3d-, 3p+, 3p-) cross-validated against JAC and literature. XATOM had these wrong
(sign-flipped for 3d-/3p-, ~2x too small for 3d+/3p+). The 3 double-3d-spectator channels are
also done but lower-confidence (heavier CSF mixing) and Kα1-side only.

**Cross sections** (via `xphoto`) — cross-checked against the original paper's Table I
(`Cu-seed-original.yaml`'s `sigma1_pump_*`/`sigma2_pump_*`/`sigma1_Ka1_*`/`sigma2_Ka1_*`) purely
**as a validation of the method** — 9 of 11 entries matched, most to <1%. **This was a one-time
check that the GRASP/RATIP pipeline reproduces a known-good published table; it is not part of
what needs to end up in the actual target config files** (see §0 above) and does not need
redoing or extending. In particular: **the `_pump_*` keys don't exist at all** in
`Cu-seed-SASE-double-satellite.yaml`/`Cu-seed-mono-SASE-double-satellite.yaml` (they're specific
to the older pump-probe config variant) — don't compute any pump-energy cross section for the
actual recompute, only for a future spot-check against the paper if one is ever wanted again.

**Scope correction, read carefully — the "other"/auxiliary definition is context-dependent:**
during the paper-validation check (§ above), matching `Cu-seed-original.yaml`'s `sigma1_Ka1_other`
needed a 7-channel sum (2s + 2p_1/2 + 3s + 3p_1/2 + 3p_3/2 + 3d_3/2 + 3d_5/2), because that
original config predates `use_2s_pathway`/`use_L2_pathway` and its "other" catch-all still
includes 2s and 2p_1/2. **The actual target config files already split 2s and 2p_1/2 out into
their own explicit keys** (`sigma1_Ka1_2s`, `sigma1_Ka1_2p1`, `sigma2_Ka1_2s`,
`sigma2_Ka1_2p1`) — so for every `*_other` value that actually needs writing into those files,
**use the 5-channel sum instead** (3s + 3p_1/2 + 3p_3/2 + 3d_3/2 + 3d_5/2, no 2s/2p_1/2, no 4s —
this reduced model's ground state has no 4s electron). This 5-channel value is already computed
for `sigma1_Ka1_other` (6.840×10⁻⁸ nm², see §2a) — it just needs writing into the config; it was
never "wrong", it answers a different (and, for the actual recompute, the *correct*) question
than the 7-channel paper-validation number. See `grasp/cross_section_validation.py`'s module
docstring for both numbers side by side with this distinction spelled out.

## 2. Full parameter inventory — what's left

### 2a. Base (non-satellite) parameters

`sigma1_pump_*`/`sigma2_pump_*` are excluded entirely per §0 — not listed below.

| Key | Status |
|---|---|
| `sigma1_Ka1_2s` | **Done, ready to write in.** 2.837×10⁻⁷ nm² |
| `sigma1_Ka1_2p1` | **Done, ready to write in.** 8.208×10⁻⁸ nm² (this key only exists once `use_L2_pathway` is on, which it is in the target files) |
| `sigma1_Ka1_2p3` | **Done, ready to write in.** 1.521×10⁻⁷ nm² (validated to <1% against the paper) |
| `sigma1_Ka1_other` | **Done, ready to write in — use the 5-channel value, 6.840×10⁻⁸ nm², not the 7-channel 4.342×10⁻⁷ used for paper validation** (see §0/§1). |
| `sigma2_Ka1_1s` | **Done, ready to write in.** 5.672×10⁻⁷ nm², complete 5/5 channel sum, validated to 87% against the paper (a believable level of agreement for a 5-channel summed quantity). |
| `sigma2_Ka1_2p3` | **Done, complete 5/5 channels.** 4.151×10⁻⁷ nm² — matches the original-paper GRASP/RATIP value (4.15×10⁻⁷) to 0.03%. The missing `2p3+3p` channel is no longer missing: the `cu_l3p3p` "crash" (§5) turned out to be a wrong `rmcdhf` block-serial-number input sequence in the original attempt, not a real bug — rebuilt cleanly and added (8.809×10⁻⁶ au), improving this from 94% to 99.97% agreement. |
| `sigma2_Ka1_2s` | **Done, ready to write in.** 3.142×10⁻⁷ nm² (config currently has the XATOM value 4.05×10⁻⁷, ratio 0.776 — see `grasp/cross_section_validation.py`'s `SIGMA2_2S_CHANNELS_AU`). Built `cu_2sholemax` (new maximal-peel 2s-hole reference, seeded from `cu_2shole.w`) plus 4 new final double-hole states (`cu_2s2p`/`cu_2s3s`/`cu_2s3p`/`cu_2s3d`, all converged with 2s frozen after hitting the same `rmcdhf` "Method 2 unable to solve" failure the recipe warns about). `2s+1s` confirmed E1/threshold-forbidden at 8047.91 eV (0 lines initialized by `xphoto`, not just assumed) — matches the existing `2p3+1s` precedent. Gauge agreement 97.7%, in line with the rest of this campaign's sigma2_* values. |
| `sigma2_Ka1_2p1` | **Done, complete 5/5 channels.** 5.125×10⁻⁷ nm² (config currently has the XATOM value 4.6903×10⁻⁷, ratio 1.093). Confirmed the existing `cu_l3p1s`/`cu_l3p2s`/`cu_l3p2pf`/`cu_l3p3d` final-state files (built for `sigma2_Ka1_2p3`) already use combined (2p-,2p) notation and are literally reusable as-is (e.g. `cu_l3p1s.c` == `cu_1s2p.c`) — no new SCF builds needed, just `xphoto` re-paired against `cu_lholemax`'s level 2 (L2, J=1/2-, -1616.98 Ha) instead of level 1 (L3). `2p1+1s` confirmed forbidden (0 lines) like `2p3+1s`. `2p1+3p` (1.169×10⁻⁵ au) added once `cu_l3p3p` was fixed — note the ratio got slightly *worse* (1.023→1.093) going from 4/5 to 5/5 channels; the complete number is the one to trust, the earlier partial agreement was partly coincidental. Gauge agreement 97.0%. |
| `sigma2_Ka1_other` | **Done, complete for 3p1/2/3p3/2/3d3/2/3d5/2 (4/5 subshells), 3s still missing one channel.** 5.184×10⁻⁷ nm² (config has the XATOM value 5.0613×10⁻⁷, ratio 1.024 — excellent). Sigma1-share-weighted average of the 5 auxiliary subshells' own total further-ionization cross sections, computed via `cu_3shole`/`cu_3phole`/`cu_3dhole` (all already existed from earlier `sigma1_Ka1_other` work; `cu_3phole`/`cu_3dhole` needed the same `xrelci` `'*'`-stripping fix `cu_lholemax` needed). 6 new final double-hole states built (`cu_3s3p`/`cu_3s3d`/`cu_3pfurther`/`cu_3dfurther`/`cu_3p3d`), plus heavy reuse of existing 1s/2s/2p-family finals. The `2p+3p` channel for 3p1/2/3p3/2 (previously missing, thought crash-blocked) is now included — `cu_l3p3p`'s "crash" was a wrong `rmcdhf` input sequence, fixed (see §5); adding it moved the overall ratio from 0.864 to 1.024. One gap remains: 3s is missing its own `3s+3s_further` (full 3s depletion) channel — `xphoto` exits 0 but silently stops mid-run without ever reaching the results section, a genuinely new failure mode this campaign hadn't hit (reproduced twice identically). See `grasp/cross_section_validation.py`'s `SIGMA2_OTHER5_AU` for full detail. |
| `GammaKeVN`, `GammaL1eVN`, `GammaL2eVN`, `GammaL3eVN` | **Done, ready to write in (with caveats).** These are *total* (radiative + Auger) decay widths, not a single line's A-coefficient — turned out to be almost entirely Auger for L1/L2/L3 (radiative negligible/not attempted there) and a genuine mix for K. Computed via `xauger` summed over every energetically-allowed double-hole final state reachable from each base hole (heavy reuse of final states already built for `sigma2_Ka1_*`/`Gamma_A_*` above — see `grasp/cross_section_validation.py`'s `GAMMA_TOTAL_WIDTH_CHANNELS_AU`): `GammaKeVN`=1.366 eV (config 1.49, ratio 0.917; missing KL1L1 — same silent-`xauger`-stop bug as `sigma2_Ka1_3s`'s `3s+3s_further` — and Kbeta radiative), `GammaL1eVN`=7.196 eV (config 8.13, ratio 0.885; missing only the confirmed-forbidden L1→L2 Coster-Kronig), `GammaL3eVN`=0.670 eV (config 0.61, ratio 1.098, essentially complete), `GammaL2eVN`=0.646 eV (config 1.04, ratio 0.621 — tried the obvious L2→L3-Coster-Kronig-plus-M-ejection channel via the existing `cu_l3p3s`/`cu_l3p3p`/`cu_l3p3d`, confirmed energetically forbidden for all three, so the gap isn't an obvious missing channel; note the theory doc itself flags `GammaL2eVN` as a *calibrated* XATOM estimate, not first-principles, so some of this gap may be on XATOM's side). |
| `GammarKalpha1eVN`, `GammarKalpha2eVN` | **Done, ready to write in.** Read directly off the already-existing `k_base3.l23_base3.ct` (Γ = ħ·A_coulomb): Kalpha1 (K→L3) = 0.3933 eV (config 0.39375, ratio 0.999); Kalpha2 (K→L2) = 0.2016 eV (config 0.202668, ratio 0.995). See `grasp/cross_section_validation.py`'s `GAMMA_R_KALPHA_EV`. |
| `GammaA_L1_to_L3M45eVN` | **Deliberately 0.0, don't touch.** This generic feed was fully carved out and replaced by the 4 satellite_channels' own explicit `Gamma_A_2s_eV` (see §2b below, now partially computed via `xauger`) — the physical process this key used to describe is literally the same setup used there (`cu_2sholemax` → `cu_l3p3d`/`cu_l3p3p`), just resolved per-channel now instead of as one generic number. |
| `hwKalpha1N`, `hwKalpha2N` | **Deliberately not touched.** These are experimental reference values (8047.91, 8027.98 eV) used as the calibration anchor for every detuning in this campaign. GRASP's own absolute energies have a known ~5 eV systematic offset from experiment (missing higher-order QED/correlation) — do NOT replace these with raw GRASP output. |

### 2b. Satellite channels (×4: 3d+, 3d-, 3p+, 3p-)

| Key | Status |
|---|---|
| `detuning_eV` | Done for all 4 |
| `detuning_eV_L2_split` | Done for all 4 |
| `Gamma_A_2s_eV`, `Gamma_A_K_eV` | **Done for all 4 channels (3d+, 3d-, 3p+, 3p-).** `xauger` confirmed working (first-ever use this campaign, no crash). 3d channels: `cu_2sholemax`/`cu_kholemax` (initial) paired with `cu_l3p3d` (final) — `Gamma_A_2s_eV`(3d+)=1.857 eV, (3d-)=0.815 eV; `Gamma_A_K_eV`(3d+)=0.00304 eV, (3d-)=0.00169 eV. 3p channels: same initials paired with `cu_l3p3p` (final, rebuilt after its earlier crash turned out to be a wrong `rmcdhf` input sequence, not a real bug — see §5) — `Gamma_A_2s_eV`(3p+)=0.973 eV, (3p-)=0.550 eV; `Gamma_A_K_eV`(3p+)=0.0243 eV, (3p-)=0.0231 eV. **Important**: `cu_l3p3d`'s levels show strong, real CI mixing between 3d3/2/3d5/2 spectator character for the L3-associated levels (near-degenerate 3d fine structure vs. exchange coupling); `cu_l3p3p`'s levels are much cleaner (3p1/2-3p3/2 splitting ~2.51 eV, ~10x the 3d splitting, no longer competitive with the exchange coupling) — read `docs/auger-branching-and-ci-mixing.md` before using or extending any of these; they use a `\|c\|²`-weighted branching scheme (`grasp/cross_section_validation.py`'s `gamma_a_branching()`), not simple line/level selection. No independent cross-check done yet (unlike everything else in this campaign) — worth a JAC comparison given the 3d `to_L3` numbers disagree with XATOM's *relative* ordering, not just magnitude. |
| `Gamma_L_eV`, `Gamma_K_eV`, `Gamma_L2_eV` | **Done, by deliberate approximation — user decided to defer the rigorous triple-hole treatment.** Attempted the real triple-hole calculation (spectator preserved, L3-hole decaying via 3s+3p, i.e. a "3s+3p+3d-spectator" final state) — `rcsfgenerate`/`rangular`/`rwfnestimate`/`rmcdhf` all converged cleanly on a 23-CSF/5-block case, but `xrelci` corrupted 12 of the 23 levels (nonsensical energies, mislabeled with a spurious integer "J=2" that doesn't exist in the actual CSF list) reproducibly on two independent rebuilds — a genuine, deterministic `xrelci` parser limitation for this complexity class, not an input mistake (contrast the `cu_l3p3p` "crash" in §5, which *was* just a wrong input sequence). Falling back to the plain `rmcdhf` mixing file (skipping Breit/QED) then failed differently — `xauger` can't read that file format directly, confirming RATIP's mixing-file reader needs the `xrelci`-refined format specifically. Given XATOM's own per-channel `Gamma_L_eV`/`Gamma_K_eV` values already exceed the old "spectator approximation" default (they vary per channel, not one constant), and a semi-quantitative estimate (weighting each base-width channel's contribution by how much its donor-shell electron count shrinks with a spectator present) put the effect at only ~10% — smaller than several other method-to-method discrepancies already found this campaign — **the user decided to use this session's GRASP *base* (non-satellite) total widths as the stand-in instead of chasing the triple-hole fix further**: `Gamma_L_eV`=0.6699 (=`GammaL3eVN`), `Gamma_K_eV`=1.3659 (=`GammaKeVN`), `Gamma_L2_eV`=0.6457 (=`GammaL2eVN`) for all 4 satellite channels. Written into both `-grasp` config files. Revisit only if the ~10% estimate needs firming up — would need either fixing the `xrelci` bug (likely a `-g -fcheck=all`/`lldb` job) or a from-scratch alternative construction of the triple-hole CSF list. |
| `Gamma_A_2s_to_L2_eV`, `Gamma_A_K_to_L2_eV` | **Done for all 4 channels**, via the same `cu_l3p3d`/`cu_l3p3p` runs as above. 3d: `Gamma_A_2s_to_L2_eV`(3d+)=0.733 eV, (3d-)=0.484 eV; `Gamma_A_K_to_L2_eV`(3d+)=0.00256 eV, (3d-)=0.00058 eV — these agree with XATOM to 3-17%, much better than the to-L3 numbers, consistent with the L2-branch being the case where CI mixing is weak (see the doc). 3p: `Gamma_A_2s_to_L2_eV`(3p+)=0.526 eV, (3p-)=0.445 eV; `Gamma_A_K_to_L2_eV`(3p+)=0.0423 eV, (3p-)=0.00365 eV. |
| `sigma_Ka1_from_2p`, `sigma_Ka1_from_1s`, `sigma_Ka1_from_2p1` | **Tooling problem resolved, but the physics came out wrong — do not use these numbers.** The `xphoto` crash pairing `cu_lholemax`/`l23_3dsat2` (see the long debugging trail this row used to describe) turned out to be specific to `l23_3dsat2`'s own files (built via the older GRASP2018 `rci` pathway): substituting my own independently-built, physically-identical `cu_l3p3d` (via RATIP-native `rcsfgenerate`+`rmcdhf`+`xrelci`) in its place ran `xphoto` cleanly with no crash at all — no `rbiotransform` needed after all. **But** re-reading `xatom_tools.py`'s `spectator_ionization_cross_section_nm2` docstring precisely (after computing numbers) revealed I had the wrong physical picture: this is XATOM's per-subshell "sudden approximation" photoabsorption cross section from a single-hole parent (a `-pcs`-style single quantity), not a `\|c\|²`-weighted sum over `cu_l3p3d`'s several final levels the way I computed it. Sure enough, comparing against XATOM: the L3-branch (`sigma_Ka1_from_2p`) came out suspiciously close (ratio 0.94/0.62 for 3d+/3d-) but the L2-branch (`sigma_Ka1_from_2p1`) came out ~200x too small and the K-branch (`sigma_Ka1_from_1s`, via `cu_kholemax`→`cu_1s3d`) came out ~1000x too big — inconsistency of that scale means the L3-branch match was very likely coincidental, not evidence the method is right. Also worth noting: XATOM's own `sigma_Ka1_from_2p1` value is numerically *identical* to `sigma_Ka1_from_2p` in the config, which may mean XATOM doesn't compute this independently for the L2 branch either. Left as XATOM in both `-grasp` configs — computing this properly needs figuring out XATOM's exact "-pcs" single-subshell convention and replicating it with a single dominant-line `xphoto` calculation, not a multi-level sum. |
| `sigma_ion_from_2p`, `sigma_ion_from_1s` | **Done, by the same deliberate base-value approximation as `Gamma_L_eV`/`Gamma_K_eV` above.** Confirmed via `xatom_tools.py`'s own `total_photoionization_cross_section_nm2` docstring ("pass the double-hole config itself... used to populate S_ion_Fi") that this is exactly the same triple-hole total-further-ionization quantity as those widths, just a cross section instead of a width — the same "needs triple-hole final states, deferred per the user's decision" reasoning applies directly. Using `sigma2_Ka1_2p3`=4.1513e-7 (for `sigma_ion_from_2p`) and `sigma2_Ka1_1s`=5.6716e-7 (for `sigma_ion_from_1s`) for all 4 satellite channels. |

### 2c. Double-satellite channels (×3: 3d+3d+, 3d-3d+, 3d-3d-)

| Key | Status |
|---|---|
| `detuning_eV` | Done for all 3, lower confidence (see §1) |
| `detuning_eV_L2_split` | **Done.** 3d+3d+: 24.057 eV, 3d-3d+: 22.245 eV, 3d-3d-: 17.022 eV — in the same range as the single-spectator L2-splits (14.5-25.1 eV), a good sanity check. Turned out tractable despite the heavy spectator-pair mixing on the lower state (no CSF exceeds ~90%, some levels near-democratically mixed across 3-4 CSFs): the L2-vs-L3 *manifold* separation itself is clean (level counts per manifold match the CSF counts in `l23_3d3dsat.c` exactly), so `satellite_detuning_analysis.py`'s existing upper-group filtering just needed an additional lower-state-manifold filter (`_3D3D_LOWER_MANIFOLD`, `double_3d_l2_splits()`) — no new GRASP calculation needed, pure analysis of the existing `.ct`/`.csum` files. |
| `Gamma_L_eV`, `Gamma_K_eV`, `Gamma_L2_eV` | **Done, same base-width approximation as §2b's entry** (0.6699/1.3659/0.6457 eV for all 3 double-satellite channels) — the double-spectator case would need an even harder *quadruple*-hole final state, so the same user decision to defer applies here even more strongly. |
| `feed_from` (6 entries: 2 parent channels × 3 manifolds) | **Not started at all, no recipe yet.** These are Coster-Kronig branching ratios — the rate at which a 3p+/3p- *single*-satellite state's spectator itself Auger-decays, feeding population into the corresponding *double*-3d-satellite state. Needs `xauger` applied to a spectator-Auger-decay channel specifically (spectator ionizes via autoionization, not the core hole) — a new physical process not touched by anything done so far. |
| `sigma_ion_from_2p`, `sigma_ion_from_1s` | **Done, same base-value approximation as §2b's entry** (4.1513e-7 / 5.6716e-7 for all 3 double-satellite channels) — the double-spectator case would need an even harder quadruple-hole final state, so deferring applies here even more strongly, same as `Gamma_L_eV`/`Gamma_K_eV`/`Gamma_L2_eV` above. |

## 3. Directory map

| What | Where |
|---|---|
| GRASP2018 source+build | `/Users/parkinpham/Programming/grasp` (built from github.com/compas/grasp) |
| RATIP-2012 source+build (patched, see §4) | `/Users/parkinpham/Programming/ratip-2012` |
| RATIP-2012 pre-patch binaries (backup) | `/Users/parkinpham/Programming/ratip-2012-original-binaries-backup` |
| GRASP/RATIP working directory — every `.c`/`.w`/`.mix`/`.ct`/`.sum` file from this campaign | `/Users/parkinpham/Programming/grasp_cu_run` |
| This repo's checked-in tooling | `grasp/` (`grasp_tools.py`, `cross_section_validation.py`, `satellite_detuning_analysis.py`, `README.md`) |
| JAC (independent 3rd cross-check, used early in the campaign) | `jac/cu_kshell_properties.jl` in this repo; ad-hoc Julia test scripts were in a session scratchpad, not preserved |

`grasp_cu_run` is NOT part of this repo (hundreds of binary intermediate files, depends on
GRASP2018/RATIP-2012 being built) — but it is the actual data backing every number in
`grasp/cross_section_validation.py`, and everything needed to extend this campaign (converged
orbitals to seed new calculations from, existing harmonized reference states) is already sitting
there. Start from it rather than rebuilding from scratch.

## 4. RATIP-2012 patches already applied (in `/Users/parkinpham/Programming/ratip-2012`)

Three sites of the same bug (`present(x) .and. x` isn't guaranteed to short-circuit in Fortran,
so gfortran evaluates `x` even when the caller omitted it → segfault reading an unbound optional
argument) were fixed:
- `rabs_angular.f90:1151` (`wigner_9j_symbol`'s `no_triangels`) — nested `if(present) then;
  if(x) ...; end if`.
- `rabs_csl.f90` (`set_configuration_scheme`'s `append`) — precomputed local flag
  (`append_is_true = present(append); if (append_is_true) append_is_true = append`) since this
  site has an `else` branch that a naive nested-if would break.
- `rabs_xl.f90` ×2 (`XL_Coulomb_coefficients`'s `Slater_integrals`) — same precomputed-flag fix,
  shared between both occurrences.

Plus one unrelated bug: `rabs_photo.f90`'s format label `2` had `5xa4` (missing comma) instead of
`5x,a4` — hard runtime error the first time that exact format gets used for output. Fixed.

All 8 RATIP tools were rebuilt in place after patching (`xauger`, `xcesd`, `xeinstein`, `xphoto`,
`xrec`, `xrelci`, `xreos`, `xtoolbox`) — **`xphoto`, `xrelci`, and now `xauger` too have been
exercised since the rebuild and all work cleanly** (`xauger` needed for essentially everything in
§2's Auger-rate rows: confirmed working first try, no crash, no debugging needed — the concern
below about budgeting debug time turned out unnecessary for `xauger` itself; the remaining
`xauger`-dependent work's difficulty is in physics attribution — see §2b's `Gamma_A_2s_eV` row and
`docs/auger-branching-and-ci-mixing.md` — not in the tool being broken). `xcesd`/`xeinstein`/
`xrec`/`xreos`/`xtoolbox` remain fully untested.

## 5. Known gaps / caveats to carry forward

- **K-edge threshold calibration** (only matters if pump-energy values ever get revisited — not
  relevant to the current recompute scope, §0): my Breit+VP (no self-energy) K-edge threshold sits
  ~150 eV above wherever the original paper's calculation puts it. The cross-section shape and
  magnitude both check out once evaluated clear of threshold (confirmed via an energy scan, not
  checked into the repo). Adding self-energy to the `xrelci` step (answer `y` instead of `n` to
  "Estimate contributions from self-energy?") might close some of this gap — untried, since
  self-energy was skipped throughout this campaign for speed.
- **RESOLVED: the `cu_l3p3p` (2p3+3p double-hole, 4-block/10-CSF) "crash"** (`LODCSH2: ncf=10
  ncfblock=5`, an `ERROR STOP` in the CSF loader) was not a real GRASP bug — rebuilding the exact
  same CSF list (`1s(2,*)2s(2,*)2p(5,*)3s(2,*)3p(5,*)3d(10,*)`, confirmed byte-identical to the
  original `cu_l3p3p.c`) and running `rmcdhf` with the correct 4-block serial-number sequence
  (`1 2` / `1 2 3 4` / `1 2 3` / `1`, then weight scheme `1`) converged with no issue. The original
  attempt (whoever/whenever it was) almost certainly had a block-count mismatch in its own input —
  exactly the kind of mistake made and caught several times this session for other multi-block
  cases (see `cu_2s3s`/`cu_3p3d`'s own build logs). This single fix completed `sigma2_Ka1_2p3` to
  5/5 (99.97% agreement, up from 94%), `sigma2_Ka1_2p1` to 5/5, `sigma2_Ka1_other`'s 3p1/2/3p3/2
  legs, and unblocked `Gamma_A_2s_eV`/`Gamma_A_K_eV`/their `_to_L2_eV` siblings for the 3p+/3p-
  channels — worth remembering before writing off any other "crash" in this campaign as a real
  bug without first checking the input sequence carefully.
- **`xauger` turned out to work cleanly on the first real attempt** — no repeat of the `xphoto`
  segfault saga. The hard part for `Gamma_A_*` work isn't the tool, it's *physics attribution*:
  the final double-hole states (e.g. `cu_l3p3d`) show strong, real CI mixing between spectator
  sub-levels (3d3/2 vs 3d5/2) for levels that share a total-J block with more than one physical
  configuration — see `docs/auger-branching-and-ci-mixing.md` for the full explanation and the
  `\|c\|²`-weighted branching scheme adopted. Expect this same attribution question (not a new
  tool bug) for every remaining `Gamma_A_*`/`feed_from`/total-decay-width item below. `feed_from`
  specifically is a genuinely new physical process (spectator autoionization, not core-hole decay)
  not yet attempted at all.
- **Total decay widths** (`Gamma_L_eV` etc., both base and per-satellite) need summing A-coefficients
  over *every* allowed lower state from a given hole, not just the dominant Kα-type line already
  computed. This means additional `rtransition` runs against final states not yet built (e.g. the
  K-hole's decay isn't just K→L3; it's K→L3 dominant plus weaker K→L1/K→M-shell channels too) —
  scope this out per-state before diving in; it may be a non-trivial fraction of the total width
  from channels barely touched so far.

## 6. Reference: XATOM's own definition of these quantities

`xatom/xatom_tools.py` and `docs/theory-and-2s-satellite-pathways.md` are the authoritative
source for exactly what physical quantity each config key represents (the GRASP/RATIP campaign's
job is to reproduce the *same* quantity via an independent method, not to redefine it). In
particular:
- `other_state_parameters()`'s docstring has the exact weighting convention for `sigma2_*_other`
  (sigma1-share-weighted average over the 6 XATOM `OTHER_HOLE` subshells — the "not a branching
  decomposition, just an effective single-scalar rate" explanation is worth rereading before
  building the GRASP/RATIP equivalent).
- `SPECTATOR_HOLE`, `DOUBLE_SPECTATOR_HOLE`, `OTHER_HOLE` dicts define the exact hole-config
  labeling convention (3d0,1 / 3d1,0 etc.) that this campaign's own channel names (3d+, 3d-, ...)
  were chosen to match 1:1.
- Section 12.7 of the theory doc explains which new-channel quantities *do* need a carve-out from
  an existing generic term (the 2s-Auger `Gamma_A_2s_eV`/`GammaA_L1_to_L3M45eVN` pair, and the
  double-satellite `Gamma_L_eV`/`Gamma_K_eV`/`Gamma_L2_eV` redirection) and which don't
  (`sigma_Ka1_from_2p/1s/2p1` vs. `sigma2_Ka1_2p3/1s/2p1` — **corrected 2026-09**: these are
  independently-measured different-process quantities, not a verified split of one total, so
  `sigma2_Ka1_*` is written in full/un-subtracted; see the theory doc and both `-grasp` configs'
  own comments for the reasoning). Read §12.7 before writing any newly-computed GRASP/RATIP value
  into the config files — the distinction determines whether a value goes in as-is or net of a
  sibling channel's own rate.

## 7. Recipe quick-reference

Full recipe detail (exact prompt sequences, gotchas) is in `grasp_tools.py`'s module docstring
and its inline function docstrings — read those before starting. Condensed checklist for a new
single-hole or satellite-type calculation:

1. `rcsfgenerate`: reference config string using the **combined non-relativistic notation** for
   any shell whose hole-count varies (e.g. `3d(9,*)` for one hole, `3p(5,*)` for one hole,
   `2p(4,*)` for two holes) — NOT hand-split into `nl-`/`nl` pieces, which either explodes into
   spurious CSFs (if marked active) or needs correct hand-derivation of every j-sublevel
   combination (error-prone). Even (integer J) vs odd (half-integer J) 2J-bounds must match the
   electron count parity, or `rcsfgenerate` loops retrying.
2. Build ONE **maximal-peel** reference per initial-state "family" (every shell `*`, `Core
   subshells:` left empty) so it harmonizes against every possible final state without
   per-comparison peel bookkeeping — see `cu_groundmax`/`cu_kholemax`/`cu_lholemax` in the working
   directory for worked examples.
3. `rangular` → `rwfnestimate` (seed from a related already-converged `.w` file, option `1`) →
   `rmcdhf`. Multi-block ASF-selection prompt count is empirical per case (see `grasp_tools.py`).
   If `rmcdhf` fails with `Method 2 unable to solve for <orbital>` / `Number of nodes counted: 1,
   Correct number: 0`, restrict "Enter orbitals to be varied" to exclude the specific
   same-symmetry-pair orbital that's failing (freeze it at the seed value) rather than answering
   `*` for "vary all".
4. For a **transition energy/rate** (satellite detunings): `rci` (Breit+VP+mass-shift via GRASP2018,
   `y/y/n/n` then the ASF pattern) on both states, then `rbiotransform`+`rtransition`.
5. For a **cross section**: `xrelci` (RATIP's own CI, `y/y/y/n/n/y/y/n` for Breit+VP) on both
   states instead of GRASP2018's `rci` — strip `rcsfgenerate`'s `*` block separators first
   (`grep -v '^ \*$'`) if the file has more than one symmetry block, `xrelci`'s CSL reader can't
   handle them. Then `xphoto` between the two `.mix` files.
6. **Always** `rm -f` the exact output filenames before rerunning `xrelci`/`xphoto` — both refuse
   to overwrite ("as new" error), and a stale leftover silently desyncs every subsequent prompt
   in ways that look like unrelated bugs.
7. Sum individual (not RATIP's "Total", which accumulates across the whole run) cross sections
   filtered by initial level via `grasp_tools.sum_initial_level_cross_sections` for any `sigma2_*`
   (or Auger-rate, once `xauger` is in play) total-over-channels quantity.
8. Convert `xphoto`'s a.u. cross sections to nm² via `grasp_tools.au_to_nm2`.
