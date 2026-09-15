# Middlemen and missing pathways: implementation plan

Companion to `docs/theory-middlemen-and-pathway-audit.md` (Part VI). Section numbers "VI §x" refer to
that document.

> **Implementation status (2026-09-15): steps 0, 1, 2, 4 and 5 are implemented; step 3 (the M₁…M₅₊
> ladder) and option C are not.** Where the code lives:
>
> - Step 0: every `config/base/*double-satellite*.yaml` (and `-duration`) carries the 3p± total widths;
>   `XLO_sim._build_pathway_extensions` raises if a manifold's summed `feed_from` exceeds its width.
> - Bookkeeping: `_build_pathway_extensions` precomputes `decay_untracked_base` / `S_untracked_base`,
>   and per channel `chan.decay_untracked` / `chan.S_untracked`, i.e. every level's outflow with no
>   other destination. Negative entries (a feed exceeding its source) raise.
> - Steps 1–2: `use_middlemen` + `middlemen: {targets, twos_ck, sigma_Ka1_total}`;
>   `Model.middleman_gain_loss` / `MB_middleman_regular`; the routing into the targets is in
>   `feed_diag_satellite_block`; absorption is in `Model.absorption(..., rho_mid_xyz)`; with
>   `use_middlemen`, "other" is no longer pumped. Outputs: `X.rho_mid_txyz`, and `rho_mid_t_last`
>   (included in `total_population_t_last`).
> - Step 4: `L2_CK_feed: {rate_eV, targets}` and `GammaA_L1_to_L2eVN`.
> - Step 5: `L3_sublevel_mixing_fs_inv` (base block) and `L3_sublevel_mixing_satellite_fs_inv`,
>   a trace-preserving depolarising Lindblad term in `_MB_nlevel_regular_core` (also in rate-equation
>   mode).
> - `Sample._step_increments` holds every per-step RK4 call and is shared by the full and lean loops.
> - `tools.verify_code` (used by `run_intensity_sweep.py` and now also `run_mono_sweep.py`) refuses
>   configs with these flags on a stale import and writes them into `*.provenance.txt`.
> - Example configs: `config/base/Cu-seed-mono-SASE-middlemen.yaml` (steps 1, 2 and 4) and
>   `...-middlemen-eii.yaml` (+ Part VII).
>
> Validation results are in the "Validation" section at the end.

Goal: route every decay and ionisation channel that currently leaves the model ("vanish") into
explicit populations, let those populations keep absorbing and be photoionised into the existing
satellite blocks, and add the missing feeds into existing blocks. Constraint: no new density-matrix
blocks. Only scalar (x, y) populations are added, plus one optional extra block discussed in step 3.

Expected payoff at 8048 eV (estimator, VI §6): the bug fix makes the 20 µJ dip 0.013 shallower;
middlemen then make it 0.018–0.026 deeper at 20–30 µJ and remove the unphysical rise of the wings;
L2→L3M45 CK adds another 0.006. The step-5 dark-state test is worth −0.016 on its own. With
L-shell EII (Part VII), everything together reaches T(8048, 20 µJ) ≈ 0.318, against 0.351 now and
0.278 measured.

---

## Step 0 — make the double-satellite feed conserve population (do first)

This is a correctness fix, independent of everything else (VI §2.3).

**Config.** In every config with `double_satellite_channels`, set each parent's widths back to their
totals (configured value + Σ `Gamma_feed_eV` out of that manifold):

| Config family | Parent | `Gamma_L_eV` | `Gamma_K_eV` | `Gamma_L2_eV` |
|---|---|---|---|---|
| non-GRASP (`Cu-seed-*double-satellite.yaml`) | 3p+ | 0.893 → 2.633 | 1.813 → 3.852 | 0.984 → 2.724 |
| | 3p− | 0.990 → 3.179 | 1.835 → 4.365 | 0.868 → 3.058 |
| GRASP (`*-grasp.yaml`, base stand-ins) | 3p+ | 0.670 → 2.410 | 1.366 → 3.405 | 0.646 → 2.386 |
| | 3p− | 0.670 → 2.860 | 1.366 → 3.896 | 0.646 → 2.836 |

(The non-GRASP totals are exactly the XATOM `2p0,1_3p0,1`, `1s1_3p0,1`, `2p1,0_3p0,1` widths. For
GRASP, either keep the stand-in + feed, as listed, or switch the 3p± parents to the XATOM totals.
The 3p spectator hole's own ~2 eV super-Coster–Kronig width is not a "spectator approximation"
detail you can drop.)

**Code.** Add a construction-time budget check in `XLO_sim.__init__`, next to the existing
`Gamma_A_K` budget check. For every parent channel and manifold, Σ feed-out rate ≤ that manifold's
`Mij` diagonal; raise `ValueError` otherwise. It would have caught this bug.

**Docs.** Correct theory doc §12.7's third bullet and `double-spectator-satellite-implementation-plan.md`
§3 and §5 ("carve-out"). The rule is the 2s rule: the parent decays at its full width, and the feed is
a branch of it.

**Check.** Paired runs, same seed, before and after (VI §6.3). Then re-run the 20/30 µJ mono points.
Every double-satellite sweep made so far over-states the dip by about 0.01 in T at ≥5 µJ.

---

## Step 1 — a middleman pool that catches every "vanish"

### State

`rho_mid_xy`: one scalar per (x, y) pixel. The ladder version in step 3 replaces it with a short list.
It absorbs "other": `rho_other` becomes the same object (VI §2.4). Keep the name `rho_other` as an
alias for one release so existing plots do not break.

### Feed: the untracked outflow of every tracked population

Precompute in `XLO_sim.__init__`, per block, a vector `untracked_out[i]` (fs⁻¹, one per local level):

```
untracked_out[i] = Mij[i, i]
                 − Σ_j Gamma_sp_Gij[j, i]                 # radiative return inside the block
                 − Σ_(tracked feeds out of level i)        # Gamma_A_K*, feed_from, L2 CK (step 4), ...
```

The runtime feed into the pool is then

```
d rho_mid/dt  +=  Σ_blocks Σ_i untracked_out[i] · Re rho_ii            (Auger / non-radiative losses)
               +  Σ_blocks Σ_i (S_ion_Fi[0,i] J− + S_ion_Fi[1,i] J+) · Re rho_ii   (further PI of ions)
               +  (Γ_L1 − Σ_k Γ_A,k − Σ_k Γ_A2,k) · rho_2s + S_2s_F·J · rho_2s
               +  S_ground_Fi[:, other]·J · rho_ground                 (the old "other" pump)
```

`untracked_out` must come out ≥ 0 for every level. Assert it: a negative entry is exactly the step-0
bug. With step 1 in place, `total_population_t_last` in `tools.compute_run_outputs` should be 1 to RK4
accuracy for the first time. Make that a hard check in the validation script, not just a diagnostic.

### Absorption

In `Model.absorption`, add `n · σ_mid · rho_mid` with `σ_mid` = the ground total by default (config
key `sigma_Ka1_middleman`; XATOM says 1.00–1.03 × ground for 3d⁻¹…3d⁻⁶, VI §3.3). Replace the
`rho_other · S_other_F` term with this.

### Where the code goes

| File | Change |
|---|---|
| `XLO_sim.py` | build `untracked_out` per block (base: from `Mij`, `Gamma_sp_Gij`, `auger_feeding_matrix`; satellites: from `chan.Mij`, `chan.Gamma_sp_Gij`, `feed_from`, `Gamma_A_K`); `sigma_Ka1_middleman`; flag `use_middlemen` (default False) |
| `Model.py` | `MB_middleman_regular(t, rho_mid_xy, params)` with params = (X, rho_ground_xy, rho_2s_xy, rho_base_ijxy, rho_sat_ijxy list, J−, J+); `absorption(...)` gains `rho_mid` |
| `Sample.py` | one extra RK4 call per `it` in **both** `_evaluate_n_level_3D_full` and `_lean`, from pre-update values (same convention as `rho_2s`); lean keeps a 2-slot buffer for the absorption window |
| `tools.compute_run_outputs` | `rho_mid_t_last`. Also fix the corner-pixel read of ground/other/2s in lean mode while touching it (see memory note `project_sweep_coherence_outputs_broken`) |

Alone, this step is the de-bleaching effect: the wings stop rising, and the line T drops by
0.01–0.02 at 20–30 µJ even before middlemen make satellites. Photoionisation out of `rho_mid` goes
nowhere in this step, so for now it is a self-loop that only adds absorption.

---

## Step 2 — photoionise middlemen into the satellite blocks

For each photoionisation channel of the middleman pool:

| Channel | Share | Destination |
|---|---|---|
| 2p₃/₂ | 25.9% | L3k manifold of the target satellite block(s); sublevel weights = `S_ground_Fi[:, 0:4]` pattern (same PI physics as ground) |
| 2p₁/₂ | 11.8% | L2k manifold of the target block(s) |
| 2s | 41.8% | instantaneous CK (the 2s hole lives 0.08 fs, J·σ·τ ≈ 10⁻⁶): Γ_A,k/Γ_L1 into the L3k of the target block for n+1, Γ_A2,k/Γ_L1 into its L2k, remainder back to the pool |
| M shell | 20.5% | back to the pool (n → n+2) |

This is `feed_diag_satellite_block` gaining one more additive term of the same shape as the existing
ground pump in `feed_diag_base_block`:

```python
# chan.mid_share: this channel's share of middleman 2p holes (config/XATOM, see step 3 mapping)
feed[:4] += chan.mid_share * np.einsum('i,xy->ixy', X.S_ground_Fi[0, :4], J_minus * rho_mid_xy)
feed[:4] += chan.mid_share * np.einsum('i,xy->ixy', X.S_ground_Fi[1, :4], J_plus  * rho_mid_xy)
feed[6:8] += chan.mid_share * X.sigma1_Ka1_2p1 * 0.5 * (J_minus + J_plus) * rho_mid_xy   # L2k
```

and the pool loses the same amounts. Target blocks with a single lumped pool: split the pool's 2p holes
over the three double-satellite channels by statistical weight (3d+3d+ : 3d−3d+ : 3d−3d− =
15 : 24 : 6). This is option A in VI §5, and gives an upper bound on line-centre depth.

---

## Step 3 — resolve the pool into a ladder M₁…M₅₊

Only needed if step 2 shows the bracket (option A vs B) is wide. On the estimator it is narrow at the
line centre (±2.7 eV shifts change T(8048) by < 0.002, because the saturated lines are power-broadened
to ~7 eV), but it matters for the *shape*: the blue-side onset of VI §1.3.

**Cascade matrix.** Add `xatom_tools.middleman_cascade(parent_hole_config)`. It parses the parent's
`-decay` Auger table, then recursively collapses every M-shell hole with the XATOM branchings from VI
§3.1 (3p → 3d⁻² 91% / 3d⁻¹4s⁻¹ 9%; 3s → per its table; 4s treated as refilled), and returns P(n)
for n = 1…N. Parents: `2p0,1`, `2p1,0`, `1s1` (non-radiative part, via `2p0,2` etc.), `2s1` (MM
remainder), every satellite L3k/U_k/L2k configuration (their spectators add to n). Cache it in
`xatom/middleman_cascade.json` and emit YAML with a new `xatom/print_middleman_parameters.py`
(same pattern as `print_satellite_parameters.py`).

**Config.**

```yaml
use_middlemen: true
middlemen:
  n_max: 5                      # M5 is the overflow bin
  sigma_Ka1_total: [5.168e-7, 5.191e-7, 5.218e-7, 5.248e-7, 5.318e-7]   # XATOM, n = 1..5+
  cascade:                      # P(n | source); generated, not hand-written
    base_L3: [0.01, 0.37, 0.33, 0.24, 0.05]
    base_L2: [...]
    base_K_nonrad: [...]
    2s_remainder: [...]
    '3d+': {lower: [...], upper: [...], L2: [...]}
    ...
  targets:                      # which existing block receives 2p holes made on M_n
    1: {'3d+': 0.6, '3d-': 0.4}
    2: {'3d+3d+': 0.333, '3d-3d+': 0.533, '3d-3d-': 0.133}
    3: lumped_double            # option A; 'none' = option B
    4: lumped_double
    5: lumped_double
```

**Option C (only if A and B disagree too much):** one extra 8-level block `3d-n+` at the XATOM
n ≈ 3.5 shift (−3.2 eV, or its GRASP counterpart), receiving n ≥ 3. Cost: one more
`MB_satellite_block_regular` call per RK4 substep. The blocks already dominate runtime, so this is
about +12%.

---

## Step 4 — missing feeds into existing blocks (no new state)

| Feed | Config key | Default | Rate / fraction |
|---|---|---|---|
| base L2 → `3d+`/`3d-` L3k (L₂-L₃M₄₅ CK, metal) | `GammaCK_L2_L3M45eVN` | 0 | 0.37–0.43 eV (config Γ_L2 − XATOM atomic Γ_L2), split 0.6 : 0.4, even sublevel spread. **No carve-out**: Γ_L2 = 1.04 already includes it. |
| satellite L2k → double-sat L3k (same CK) | `GammaCK_L2k_L3kM45_eV` per channel | 0 | same rate; destination by the 3d count rule |
| 2s → base L3 / L2 via 2p4s CK | `Gamma_A_2s_to_base_eV`, `..._to_base_L2_eV` | 0 | 0.090 / 0.051 eV (XATOM `2s0 → 2p± 4s0`) |
| K → 2p⁻¹3s⁻¹ (KL₃M₁, KL₂M₁) | — | — | skip (no 3s-spectator block; 1.2% of K decays) |

The L2 CK feed reuses the `Gamma_A_K` code path in `feed_diag_satellite_block`, with the source
switched from the K-hole diagonal to the base L2 diagonal (indices 6, 7).

---

## Step 5 (test, optional) — dark-state ceiling

VI §1.5: add `L3_sublevel_mixing_fs_inv` (default 0). In `_MB_nlevel_regular_core` it adds, inside
each block's L3 manifold, a relaxation of the 4×4 L3 sub-block towards Tr(ρ_LL)·𝟙/4:

```
drho[a, b] += -gamma_mix * (rho[a, b] - delta_ab * trace_L3 / 4)     for a, b in L3 manifold
drho[a, k] += -0.5 * gamma_mix * rho[a, k]                           for a in L3, k in K (coherence damping)
```

This preserves the trace. Run the 5/20/30 µJ mono points at γ = 0.5, 1, 2 fs⁻¹. The estimator predicts
+27% on the Kα1 ln-dip at γ = 2 fs⁻¹ (T(8048, 20 µJ) 0.364 → 0.348) and nothing at Kα2. If MB agrees,
the physics question becomes how fast the 2p–3d exchange mixes the 2p hole in the satellite
configurations. That is a GRASP question, and the only lever found here that
raises η instead of f.

---

## Validation

1. **Reduction:** with `use_middlemen: false` and all new keys absent, output is bit-for-bit
   identical to step 0.
2. **Trace:** with step 1 on, `total_population_t_last` is 1 within RK4 truncation at every t.
   Before step 1 it is < 1 by exactly the vanished fraction, which gives a free cross-check of
   `untracked_out`.
3. **Full vs lean lockstep** for every new population (same as for the satellite blocks).
4. **Estimator cross-check:** `toy_rsa.py` variant V2 predicted T(8048) = 0.346 (20 µJ), 0.341
   (30 µJ), and wings flat at 0.406. The MB result should be within ~0.01 of these relative shifts.
5. **Cluster provenance:** port the `verify_code` / provenance guard from `run_intensity_sweep.py` to
   `run_mono_sweep.py` before the first cluster run with `use_middlemen` (memory note
   `project_cluster_stale_code_new_flags`: new flags have been silently ignored three times).

### Results (2026-09-15)

1. **Reduction.** With every new flag off, the double-satellite, `original` and rate-equation configs
   match the pre-extension code (a worktree of commit ff828c4, on feed-fixed copies of the configs),
   in lean and full mode. T agrees to 6 digits, and the worst relative difference over all saved
   outputs is 4.8×10⁻¹⁶. The `rho_eg_*` coherence outputs are excluded because they are pure
   round-off in both versions. The result is round-off level, not bit-for-bit: the numba kernel is
   compiled with `fastmath`, and the new terms change the order of its sums.
2. **Trace.** With middlemen on, `total_population_t_last` stays within [0.99970, 1.00003] in the
   MB shots below (dt = 0.015 fs). On a small test grid the deviation shrinks with the step:
   1×10⁻³ at dt = 0.03 fs, 1.5×10⁻⁴ at 0.0075 fs. Without middlemen the trace drops to 0.868, the
   fraction that vanishes.
3. **Full vs lean** (all extensions on, including EII and mixing) agree to 4×10⁻¹⁰ relative in the
   transmitted spectrum and 6×10⁻⁸ in the middleman population. Lean mode keeps the pool's z buffer
   in float32.
4. **MB vs estimator.** One mono shot at 20 µJ and 8048 eV (seed 0, tgrid 3000, 7×7 pixels). Each row
   adds to the one above, except the last. ΔT is relative to the feed-fixed baseline:

   | Variant | MB T | MB ΔT | Estimator ΔT |
   |---|---|---|---|
   | feed bug fixed (baseline) | 0.3632 | — | — |
   | + middlemen, option A (double-satellite targets) | 0.3446 | −0.0186 | −0.019 |
   | + middlemen, option B (non-resonant), instead of A | 0.3479 | −0.0153 | −0.016 |
   | + middlemen (A) + L2 CK + 2s→bare-2p CK | 0.3396 | −0.0236 | −0.025 |
   | + L-shell EII, spatial factor 0.5 | 0.3341 | −0.0291 | −0.029 |
   | + dark-state mixing 2 fs⁻¹, base and satellite blocks | 0.3202 | −0.0430 | −0.046 |
   | dark-state mixing 2 fs⁻¹ alone | 0.3508 | −0.0124 | −0.016 |

   Every row lands within 0.004 of the estimator. The digitised experiment is at 0.278, so the
   no-free-parameter additions (through EII) close 34% of the gap at 20 µJ, and adding the mixing
   closes 50%.
5. **Provenance.** `tools.verify_code` guards both `run_intensity_sweep.py` and `run_mono_sweep.py`.

The follow-up sweeps (SASE intensity and mono, five variants from the baseline to full mixing) are
`scripts/generate_pathway_sweeps.sh` → `scripts/submit_pathway_sweep_{sase,mono}.sh`.

## Cost

Step 1 adds one scalar RK4 call and a handful of `einsum`s per `it`, and step 3 adds four more. The
per-block kernel calls (8 blocks × 4 RK4 substeps) dominate runtime, so steps 1–4 are a few per cent.
Option C is about +12%.

## Order of work

Step 0 → re-run the mono sweep points at 5/20/30 µJ → step 1 → step 2 (option A and B runs) →
step 4 → decide on step 3 and option C from the A/B bracket and the blue-side shape → step 5 test.
Part VII (EII) comes after step 2, because M-shell EII feeds the same pool.

## Estimator script

`toy_rsa.py` (session scratchpad) could go into `scripts/estimate_rsa_pathways.py`. It runs a variant
at 24 photon energies and 5 pulse energies in 1.5–3 minutes on a laptop and is useful for triage before
spending cluster time.
