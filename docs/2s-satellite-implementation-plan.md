# Implementation plan: explicit 2s-hole satellite pathways

## Context

`docs/theory-and-2s-satellite-pathways.md` (Part II) works out the physics for three new
double-hole states — $2p^+3d^+$, $2p^+3d^-$, $2p^+3p^+$, coherently coupled to $1s3d^+$, $1s3d^-$,
$1s3p^+$ — fed by 2s(L1)-hole Auger decay and by direct spectator-shell photoionization of the
existing 1s-hole/2p-hole populations. These currently get folded into the generic, undetuned
2p$_{3/2}$ population (`Model.py::MB_nlevel_regular`'s `auger_feeding_matrix`); the goal is to give
each one its own detuned, coherent Maxwell–Bloch dynamics against the shared Kα1 field.

Per the latest decision, each satellite pair uses the **same sublevel-resolved 6-level structure**
as the base Kα1 block (not a 2-level reduction) — reusing the existing $T_{ij\sigma}$ (Eq. 25) and
$G_{ij}$ (Eq. 17) tensors verbatim. This plan is the code-level counterpart of that document.

## Architecture

**Unify at the kernel level, not the storage level.** The base 6-level block and the three new
satellite blocks all evolve under the same equation (theory doc Eq. S1, which reduces to the
existing Eq. 4 realisation when detuning $\Delta=0$) — same $T_{ij\sigma}$, $G_{ij}$, and (by
default) the same $\Gamma_{L3},\Gamma_K$. So all four blocks should run through **one generalized
numba kernel**, parametrized by:
- a per-level, per-$xy$ **feed array** `feed_diag_ixy` (shape `(6, nx, ny)`) — replaces the
  kernel's current hardcoded "ground pump + 2s-Auger" construction; computed by a small
  block-specific Python function *outside* the kernel and passed in already-summed, exactly the
  way `rho_ground_xy`/`rho_2s_xy` are already passed in today as frozen-for-this-RK4-step inputs.
- a scalar **detuning** `Delta` and a precomputed **sign pattern** `sign_ij` (reusing
  `X.Hij - X.Hij.T`, no new convention) — adds the `-iΔ·sign_ij·ρ_ij` term (theory doc §4) that the
  current code omits because it's always zero for the base block.

Everything else in the kernel (Rabi/Hint commutator, `Mij` decay, `Gamma_sp_Gij` spontaneous
feed-back, ionization loss) is **reused unmodified** per block — for satellite channels these
default to the base system's own `Mij`/`Gamma_sp_Gij` (spectator approximation, theory doc §10),
only rebuilt if a channel overrides $\Gamma_{L,k}/\Gamma_{K,k}$.

**Keep the base block's storage/naming exactly as today** (`X.rho_ijtxyz`, `MB_nlevel_regular`,
etc.) — `Plot.py`, `tools.py`, and the notebooks all read these names directly, and there's no need
to touch that surface for this task. Satellite blocks get **new, separate, list-based storage**
(`X.rho_sat_ijtxyz`, one `(6,6,tgrid,xgrid,ygrid,zgrid)` array per channel) so adding a 4th channel
later is a one-line config addition, not a code change. With `satellite_channels` empty/absent, all
new loops are no-ops and the base block's kernel call reproduces today's exact formula (`Delta=0`,
`feed_diag` built to equal the current inline expression) — this is the safety net for verification.

Field-source and opacity contributions are **summed across blocks at the call site**
(`Sample.py`), not inside `Model.py`'s shared functions — `Omega_source_regular` and the ionization
part of `absorption` need no internal changes, just one extra call per satellite channel whose
result gets added in.

This keeps the design open for genuinely dissimilar future pathways (e.g. a
ground→2p$_{1/2}$→3d3d→2p$_{3/2}$3d3d chain): a new block just needs its own `feed_diag` function
(possibly sourced from another *satellite* block's diagonal instead of the base block's — the
feed function doesn't care) and, if its internal structure isn't the same 6-level shape, its own
$T_{ij\sigma}/M_{ij}$ built the same way `nlevel==6`'s are built today. The kernel itself is already
shape-generic (numba, sized off `rho_ijxy.shape[0]`), so that generalization is cheap when needed —
not built now, to avoid speculative machinery for a case that isn't concrete yet.

## Config schema (new)

A structured list (not flat `sigma1_pump_2s`-style keys — with 4+ new numbers per channel, a flat
per-channel key explosion would be harder to extend than a list of dicts):

```yaml
satellite_channels:
  - name: "3d+"                      # spectator label; lower=2p+{name}, upper=1s{name}
    detuning_eV: 0.0                 # E_upper - E_lower - hw_Ka1N (Δ_k, theory doc §10)
    Gamma_A_2s_eV: 3.04               # Γ_A(2s→lower), §12.1(a) — pre-split, net of any bookkeeping subtraction (§12.7)
    sigma_pump_from_2p: 0.0          # σ(2p→lower), pump field, §12.1(b)
    sigma_Ka1_from_2p: 0.0           # σ(2p→lower), Kα1/seed field
    sigma_pump_from_1s: 0.0          # σ(1s→upper), pump field, §12.2
    sigma_Ka1_from_1s: 0.0           # σ(1s→upper), Kα1/seed field
    # optional overrides (default: reuse base Γ_L3/Γ_K/Mij, §10 widths assumption):
    # Gamma_L_eV: ...
    # Gamma_K_eV: ...
  - name: "3d-"
    ...
  - name: "3p+"
    ...
```
Absent key ⇒ `[]` (no behaviour change, matches how `use_2s_pathway` etc. already default). The
§12.7 double-counting subtraction (from `sigma2_*_2p3`, `sigma2_*_1s`, `GammaA_L1_to_L3M45eVN`) is
done **manually in the YAML**, documented with a subtraction comment — mirroring the existing
`sigma1_pump_other: 1.36e-7 # 3.23e-7 - 1.87e-7` convention — rather than auto-subtracted in code
(keeps the numbers auditable, matches existing repo style, avoids hidden coupling between config
entries).

## File-by-file changes

**`XLO_sim/XLO_sim.py` (`__init__`)**
- Parse `self.satellite_channels` (raw YAML list, default `[]`).
- Build `self.sign_ij_block = X.Hij - X.Hij.T` once (shared 6×6 array; only matters when `Delta≠0`).
- For each channel, build a small holder (e.g. `types.SimpleNamespace`) with: `name`, `Delta_fs`
  (`detuning_eV/self.hbar`), `Gamma_A_fs`, `S_feed_2p` and `S_feed_1s` (3-vectors: pump/Ka1-/Ka1+,
  same value repeated for ± per the existing `S_2s_F`-style convention), and `Mij`/`Gamma_sp_Gij`
  (reuse `self.Mij`/`self.Gamma_sp_Gij` unless the channel overrides widths, in which case rebuild
  with the exact formula already used for `self.Mij`). Store the list as
  `self.satellite_channel_params`.

**`XLO_sim/Model.py`**
- Extend `_MB_nlevel_regular_core`: replace the
  `(S_ground_Fi0, S_ground_Fif, auger_feeding_diag, rho_ground_xy, rho_2s_xy)` argument cluster with
  `(feed_diag_ixy, Delta, sign_ij)`. Inside the existing fused loop: `i==j` branch adds
  `feed_diag_ixy[i,x,y]` (in place of today's inline pump computation); `i!=j` branch adds
  `-1j*Delta*sign_ij[i,j]*rho_ijxy[i,j,x,y]`. The `Gamma_sp_Gij`-driven spontaneous-feed sum, `Hint`
  commutator, `Mij` decay, and ionization terms are untouched.
- Add `feed_diag_base_block(X, rho_ground_xy, rho_2s_xy, J_P_xy, J_Om_xy, J_Op_xy)`: reproduces
  *exactly* today's inline formula from `MB_nlevel_regular` (ground pump via `S_ground_Fi` + 2s-Auger
  via `auger_feeding_matrix`), so the base block's numerics are unchanged after the refactor.
- Add `feed_diag_satellite_block(chan, rho_2s_xy, rho_base_ijxy, J_P_xy, J_Om_xy, J_Op_xy)`:
  builds the `(6,nx,ny)` array — uniform `e_L3/4`-weighted spread of `Gamma_A_fs*rho_2s_xy` into
  indices 0–3 (Eq. S2), plus sublevel-preserving `chan.S_feed_2p·J_F` × `rho_base_ijxy[i,i]` into
  indices 0–3 (Eq. S3), plus sublevel-preserving `chan.S_feed_1s·J_F` × `rho_base_ijxy[i,i]` into
  indices 4–5 (Eq. S4).
- Update `MB_nlevel_regular` to call `feed_diag_base_block` and pass `Delta=0.0,
  sign_ij=X.sign_ij_block` into the extended core — signature change only, identical result.
- Add `MB_satellite_block_regular(t, rho_ijxy, params)`: same shape as `MB_nlevel_regular`, pulls
  `chan.Mij/Gamma_sp_Gij` (falls back to `X.Tijs_plus/minus`, unchanged) and calls
  `feed_diag_satellite_block` + the extended core with `Delta=chan.Delta_fs`.
- `Omega_source_regular` and `absorption`: no internal changes — called once per block, results
  summed at the call site (see `Sample.py`). `absorption`'s ionization term gains an optional
  per-channel `S_ion_Fi`-like input (§12.4 of the theory doc), defaulting to all-zero (no-op) until
  triple-ionization data exists.

**`XLO_sim/Sample.py`**
- `init_n_level_3D`: allocate `rho_sat_ijtxyz = [np.zeros_like(rho_ijtxyz) for _ in
  X.satellite_channel_params]` (empty list ⇒ no extra memory).
- `evaluate_n_level_3D`, inner `it` loop: after the existing base-block RK4 step, loop over
  `enumerate(X.satellite_channel_params)`, RK4-step each `rho_sat_ijxy[k]` via
  `MB_satellite_block_regular`, store into `rho_sat_ijtxyz[k][it,:,:,:,:,iz]`.
- Field-source accumulation: after the existing `Omega_source_regular(rho_ijtxyz[...], ...)` call,
  add `Omega_source_regular(rho_sat_ijtxyz[k][...], ...)` for each channel into the same
  `Omega_pstxy` update.
- Absorption calls (`Model.absorption`, for both `pump` and `field` modes): pass the per-channel
  diagonal populations/optional ionization cross-sections through so their opacity is counted
  (theory doc Eq. S8).
- `MB_2s_regular`, `MB_other_regular`, `MB_ground_regular`: **no changes** — the Auger *destination*
  bookkeeping lives entirely in `feed_diag_*`, not in the source-population equations, and ground
  pump accounting is unaffected (theory doc §12.7).

## Rollout / verification

1. Land the refactor with `satellite_channels: []` in all existing configs. Run the existing
   `Cu-seed.yaml` (or `example_script.py`) before/after and diff `rho_ijtxyz`/`Omega_pstxyz` — must
   match today's output (this is the main correctness gate for the kernel-signature change).
2. Add one channel with `Delta=0`, `Gamma_A_2s_eV` set equal to (a fraction of) the existing
   `GammaA_L1_to_L3M45eVN`, all `sigma_*` at 0, and the old `auger_feeding_matrix`'s corresponding
   share turned down by the same amount — check the satellite's $L_k$ population growth matches
   what the old lumped path used to produce, isolating the Auger-feed wiring.
3. With everything else zero, set a small nonzero `Delta` on one channel and confirm its coherence
   free-precesses at the expected frequency (small standalone script exercising
   `_MB_nlevel_regular_core` directly, not a full 3D run) — isolates the new detuning term.
4. Once real per-channel numbers are available: check the population/energy budget identities from
   theory doc §14 (Auger rates sum to `GammaL1eVN`; total population only drops via documented
   untracked-loss channels).

## Explicitly out of scope for this pass

- `Plot.py` support for the new states (no plotting requested; the new arrays are available on `X`
  for ad hoc inspection in notebooks in the meantime).
- Auto-subtracting §12.7's double-counted cross sections/rates in code (kept manual/YAML-documented).
- Triple-ionization loss cross sections for the satellite blocks (§12.4) — default zero.
- A generalized multi-shape/multi-hop block registry — the three requested channels are all the same
  6-level shape fed from the same two source blocks (base + 2s), so the simple list-of-channels
  approach above covers them; revisit only when a genuinely differently-shaped or chained pathway
  (like the ground→2p$_{1/2}$→… example) is actually being added.
