# XLO-sim theory: current model and 2s-hole satellite pathways

This document has two parts:

- **Part I** restates the theory and numerical scheme in `XLO-sim_equations.pdf`, cross-checked
  against what `XLO_sim/*.py` actually implements (the code has drifted slightly ahead of the PDF —
  notably the 2s-hole state — and that drift is called out explicitly). Stochastic/noise terms
  (PDF Eqs. 5, 6, the noise terms in Eqs. 4/23/27/31/32) are **omitted throughout**, per scope.
- **Part II** works out the physics and equations needed to explicitly track three new "satellite"
  double-hole states fed by 2s(L1)-hole decay: $2p^+3d^+$, $2p^+3d^-$, $2p^+3p^+$, coherently coupled
  to $1s3d^+$, $1s3d^-$, $1s3p^+$ respectively. Assumptions are flagged inline in **⚠ boxes** — please
  sanity-check these before implementation.

Equation numbers `(1)`–`(37)` refer to the PDF. New equations here are numbered `(M*)` (existing
code behaviour not in the PDF) and `(S*)` (new satellite-pathway physics).

---

## Part I — Existing theory and numerical scheme

### 1. Physical picture

A copper target is exposed to an intense pump (XFEL) pulse and a co-propagating Kα1-resonant seed
(or noise-seeded ASE) field. The pump photoionizes Cu 1s/2p/2s/valence shells fast enough that the
target's own opacity changes during the pulse (reverse saturable absorption). K-shell (1s)
photoionization creates a population inversion between the 1s-hole ("K") and 2p$_{3/2}$-hole ("L3")
manifolds, which is the gain medium for stimulated Kα1 (2p$_{3/2}\to$1s) emission. The seed/ASE field
and the ionic density matrix are propagated self-consistently (Maxwell–Bloch) through the sample
volume.

### 2. State space

| State | Symbol | Sublevels tracked | Nature |
|---|---|---|---|
| Neutral ground | $\rho^{(ground)}$ | 1 (scalar) | incoherent population |
| 1s hole (K) | $\rho_{ij}$, $i,j\in\{4,5\}$ | 2: $m=\mp\tfrac12$ | density-matrix block, coherent |
| 2p$_{3/2}$ hole (L3) | $\rho_{ij}$, $i,j\in\{0..3\}$ | 4: $m=-\tfrac32..\tfrac32$ | density-matrix block, coherent |
| 2s hole (L1) | $\rho^{(2s)}$ | 1 (scalar) | incoherent population |
| Other | $\rho^{(other)}$ | 1 (scalar) | catch-all sink for all other Cu$^+$ configurations |

The 6-level block (2p$_{3/2}$: indices 0–3, 1s: indices 4–5) is a single density matrix $\rho_{ij}$
with coherences allowed *within* the block (that's the actual "Maxwell–Bloch" physics — stimulated
Kα1 emission/absorption lives in the 2p$_{3/2}$↔1s coherences). Ground, 2s, and "other" are
population-only scalars with no tracked coherence to anything.

A reduced **`nlevel=2`** mode also exists in the code: the 2p$_{3/2}$ and 1s manifolds are each
collapsed to a single level (sublevel/polarization structure ignored), with an isotropic
effective dipole coupling. Part II originally proposed reusing this same reduction for the new
satellite pairs, but that plan was dropped (§10) in favour of giving each satellite pair the full
sublevel-resolved structure. It's still worth having the exact numbers on hand, since the $2/3$
branching constant they encode reappears throughout Part II (`Model.py`, `nlevel==2` branch):
$$
T_{01\sigma}=T_{10\sigma}=\sqrt{2/9}\ \ (\forall\sigma),\qquad G_{01}=\tfrac23 . \tag{M0}
$$
$G_{01}=2/3$ is not arbitrary: $\Gamma_{sp}=\Gamma_{rK\alpha1}+\Gamma_{rK\alpha2}$ is, by construction,
the *combined* Kα1+Kα2 radiative rate (PDF, note under Eq. 9), but only the Kα1 (2p$_{3/2}$)
channel is dynamically tracked. $\Gamma_{rK\alpha1}/\Gamma_{sp}=0.39375/0.596418=0.6602\approx 2/3$
and $\Gamma_{rK\alpha2}/\Gamma_{sp}\approx 1/3$. So $\Gamma_{sp}\cdot G_{01}=\Gamma_{rK\alpha1}$
exactly: of the population leaving the 1s-hole state radiatively, $2/3$ (the Kα1 fraction) is fed
back into the tracked 2p$_{3/2}$ population, and the remaining $1/3$ (Kα2, into the untracked
2p$_{1/2}$/L2) is lost from the model. The same accounting holds sublevel-by-sublevel in the 6-level
matrix ($\sum_i G_{i,4}=\sum_i G_{i,5}=2/3$, PDF Eq. 17). **This is the key mechanism Part II reuses**
for the new coherent pairs.

Energy-level cartoon (decay direction, not to scale):

```
 higher energy   1s-hole (K)        <- created by direct photoionization (pump term)
      |            |  \
      |     Γ_sp∙G  |   \ Γ_K−Γ_sp∙G  (non-radiative / untracked, e.g. KLL Auger)
      |            v    v
      |          2p-hole (L3)  --Γ_L3--> untracked ("other"-like sink, no explicit destination)
      |            ^
      |     Γ_A(L1→L3M45)  (Auger, currently lumped evenly over all 4 L3 sublevels)
      |            |
 lower energy    2s-hole (L1)  --(Γ_L1 − Γ_A)--> untracked (remainder of 2s decay, no destination)
                    ^
             ground-state photoionization (pump term)
```

### 3. Field equations (PDF Eqs. 1–3, 31–33; noise dropped)

Paraxial propagation of the emitted/seed Kα1 field ($\Omega_\sigma^{(\pm)}$, in Rabi-frequency units)
and the pump field ($\mathcal P$), $\sigma\in\{-1,+1\}$ circular polarizations:
$$
\left[\frac{\partial}{\partial z}\mp\frac{i}{2k_0}\left(\frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial y^2}\right)\right]\Omega_\sigma^{(\pm)}
= -\frac{\kappa_{\Omega_\sigma}}{2}\Omega_\sigma^{(\pm)} \pm i\,\frac{3}{8\pi}\lambda^2\Gamma_{sp}\,n\,T^{(\mp)}_{i'j'\sigma}\frac{\rho_{j'i'}+\rho^*_{i'j'}}{2}
\tag{31,32}
$$
$$
\left[\frac{\partial}{\partial z}-\frac{i}{2k_P}\left(\frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial y^2}\right)\right]\mathcal P = -\frac{\kappa_{\mathcal P}}{2}\mathcal P \tag{33}
$$
The classical polarization source term is exactly PDF Eq. 9's flux relation run in reverse: it's
the coherent-emission analogue of spontaneous $\Gamma_{sp}$ decay, driven by the off-diagonal
coherences $\rho_{j'i'}$. Photon flux and field-Rabi-frequency conversions (used for cross-section
bookkeeping):
$$
J_{\Omega_\sigma}=\frac{\Omega_\sigma^{(+)}\Omega_\sigma^{(-)}}{\frac{3}{8\pi}\lambda^2\Gamma_{sp}},\qquad J_{\mathcal P}=\frac{2\epsilon_0 c|\mathcal P|^2}{\hbar\omega_{\mathcal P}}. \tag{9,10}
$$

### 4. Atomic master equation — general form (PDF Eq. 4, noise dropped)

$$
\partial_t\rho_{ij}=-i\Big(\tfrac{E_i-E_j}{\hbar}-\mathrm{sign}_{ij}\,\omega_0\Big)\rho_{ij}
-\Big(\sum_s\tfrac{\gamma_{si}+\gamma_{sj}}{2}\Big)\rho_{ij}+\delta_{ij}\sum_s\gamma_{is}\rho_{ss}
+i\sum_{\sigma,s}\rho_{sj}T_{is\sigma}\!\left[H_{is}\Omega_\sigma^{(+)}+H_{si}\Omega_\sigma^{(-)}\right]
-i\sum_{\sigma,s}\rho_{is}T_{sj\sigma}\!\left[H_{sj}\Omega_\sigma^{(+)}+H_{js}\Omega_\sigma^{(-)}\right]
\tag{4}
$$
with $\mathrm{sign}_{ij}=\pm1/0$ and $H_{ij}=1$ iff $E_i>E_j$ (else 0) as in PDF Eq. 7. **Important
and easy to miss**: the first term is a *detuning* term — it vanishes only when $E_i-E_j=\hbar\omega_0\,\mathrm{sign}_{ij}$
exactly. In the current Cu-Kα1 implementation this term does **not appear anywhere in the code**
(`Model.py::_MB_nlevel_regular_core` has no such phase term) because every tracked $(i,j)$ pair
(all 2p$_{3/2}$↔1s sublevel combinations) is *exactly resonant* with $\omega_0$ by construction — the
rotating frame is defined at the Kα1 frequency, and there's no fine structure within a shell. **Part
II needs to reintroduce this term** for the satellite pairs, since they are detuned by construction.

### 5. Cu-Kα1 realisation (6-level block)

Decomposition into pump/decay, ionization, and coherent parts (PDF Eq. 11, noise dropped):
$$
\partial_t\rho_{ij}=\Big(\partial_t\rho_{ij}\Big)_{pump/decay}+\Big(\partial_t\rho_{ij}\Big)_{ion.}+\Big(\partial_t\rho_{ij}\Big)_{M.\text{-}B.} \tag{11}
$$

**Pump/decay** (PDF Eqs. 12–16):
$$
\Big(\partial_t\rho_{ij}\Big)_{pump/decay}=-M_{ij}\rho_{ij}+\delta_{ij}\Big[p^{(pump)}_i+\Gamma_{sp}G_{is'}\rho_{s's'}\Big],\qquad
M_{ij}=\Gamma_{L3}e^{(L3)}_ie^{(L3)}_j+\Gamma_K e^{(K)}_ie^{(K)}_j+\tfrac{\Gamma_{L3}+\Gamma_K}{2}\big(e^{(L3)}_ie^{(K)}_j+e^{(K)}_ie^{(L3)}_j\big)
\tag{12,13}
$$
$$
p^{(pump)}_i=\rho^{(ground)}J_{F'}S^{(ground)}_{F'i},\qquad e^{(L3)}=(1,1,1,1,0,0),\ \ e^{(K)}=(0,0,0,0,1,1) \tag{14,18}
$$
$S^{(ground)}_{Fi}$ (PDF Eq. 15) carries the angular branching fractions (e.g. $0.27,0.23,\dots$) for
photoionization from ground into each 2p$_{3/2}$/1s sublevel; $G_{ij}$ (PDF Eq. 17) is the
spontaneous-emission branching matrix discussed above (§2). The code's `Mij` adds a configurable
`additional_dephasing` on top of $(\Gamma_{L3}+\Gamma_K)/2$ for the off-diagonal (coherence) dephasing
rate — a pure-dephasing knob not present in the PDF (`XLO_sim.py:143`).

**Ionization** (further photoionization of an already-singly-ionized level; PDF Eqs. 20–22) — pure
loss, no explicit destination state:
$$
\Big(\partial_t\rho_{ij}\Big)_{ion.}=-\tfrac12\big(\gamma^{(ion.)}_i+\gamma^{(ion.)}_j\big)\rho_{ij},\qquad \gamma^{(ion.)}_i=S^{(ion.)}_{F'i}J_{F'} \tag{20,21}
$$

**Maxwell–Bloch (coherent)** (PDF Eqs. 23–26):
$$
\Big(\partial_t\rho_{ij}\Big)_{M.\text{-}B.}=i\big(H^{int.}_{is'}\rho_{s'j}-\rho_{is'}H^{int.}_{s'j}\big),\qquad
H^{int.}_{ij}=T^{(+)}_{ij\sigma'}\Omega^{(+)}_{\sigma'}+T^{(-)}_{ij\sigma'}\Omega^{(-)}_{\sigma'},\qquad T^{(\pm)}_{ij\sigma}=T_{ij\sigma}H_{ij/ji} \tag{23,24,26}
$$
$T_{ij\sigma}$ (PDF Eq. 25) is the dipole-ratio tensor; every nonzero entry couples exactly one
$(2p_{3/2}\ m, 1s\ m')$ pair to exactly one circular polarization ($\Delta m=\mp1$ selection rule).

### 6. Ground / other populations, opacity (PDF Eqs. 28–30, 34–35)

$$
\partial_t\rho^{(ground)}=-\rho^{(ground)}S^{(ground)}_{F'i'}J_{F'},\qquad
\partial_t\rho^{(other)}=-\rho^{(other)}S^{(other)}_{F'}J_{F'}+\rho^{(ground)}S^{(ground)}_{F',7}J_{F'} \tag{28,29}
$$
$$
\kappa_F=n\Big(\rho^{(ground)}S^{(ground)}_{Fi'}+\rho^{(other)}S^{(other)}_F+\rho_{i'i'}S^{(ion.)}_{Fi'}+\sigma^{(compound)}_F\Big) \tag{34}
$$

### 7. Existing code addition not in the PDF: the 2s (L1) hole state

The PDF predates the 2s-hole state entirely. As currently implemented (`Model.py::MB_2s_regular`,
`XLO_sim.py:277-284`), $\rho^{(2s)}$ is a fifth scalar population:
$$
\partial_t\rho^{(2s)}=\underbrace{\rho^{(ground)}S^{(ground)}_{F',2s}J_{F'}}_{\text{pump from ground}}\ \underbrace{-\ \rho^{(2s)}S^{(2s)}_{F'}J_{F'}}_{\text{further ionization, pure loss}}\ \underbrace{-\ \Gamma_{L1}\rho^{(2s)}}_{\text{total 2s-hole decay}} \tag{M1}
$$
Of the total decay rate $\Gamma_{L1}$ (`GammaL1eVN`=8.13 eV), only a **part**,
$\Gamma_A^{(L1\to L3M45)}$ (`GammaA_L1_to_L3M45eVN`=5.07 eV), is explicitly fed back into the
2p$_{3/2}$ population — spread **evenly across all 4 msublevels**, i.e. treated as an incoherent,
undetuned source term added directly into the same `pump_term` as the ground-state pump:
$$
\Big(\partial_t\rho_{ii}\Big)_{Auger} = \frac{e^{(L3)}_i}{\sum_{i'}e^{(L3)}_{i'}}\,\Gamma_A^{(L1\to L3M45)}\,\rho^{(2s)},\qquad i\in\{0,1,2,3\} \tag{M2}
$$
The remaining $\Gamma_{L1}-\Gamma_A^{(L1\to L3M45)}=3.06\text{ eV}$ of 2s-hole decay is **pure loss**
(no destination) — this budget corresponds physically to the unmodeled L1→L2 Coster–Kronig,
L1→L3M23 (3p-spectator) Auger, and other channels. This is exactly the number Part II needs to
partition explicitly.

A `use_2s_pathway` flag exists: when `False`, the Auger-feed matrix and $S^{(2s)}_F$ are zeroed
*and* the ground→2s pump cross section (`S_ground_Fi[:,-2]`) is also zeroed — i.e. it's a full
kill-switch, not just a decoupling of the Auger feed.

**Bookkeeping pattern worth noting** (directly relevant to Part II): when the 2s channel was carved
out, the "other" cross sections were *derived by subtraction* from a previously-measured total,
e.g. `config/base/Cu-seed.yaml`: `sigma1_pump_other: 1.36e-7 # 3.23e-7 - 1.87e-7`. Any new explicit
channel must be split out of whatever generic bucket currently contains it the same way, or its
cross section/rate will be double-counted.

### 8. Numerical scheme

The grid is $(t,x,y,z)$ — time and transverse coordinates resolved at every $z$-slice, $z$ is the
outer/propagation loop (`Sample.py::evaluate_n_level_3D`). Per $z$-step:

1. **Atomic update (inner `t` loop, fixed-step classical RK4, `tools.RK45_step`)** — despite the
   name this is *not* an adaptive/embedded RK45; it's a plain 4-stage RK4 with fixed $\Delta t$,
   applied independently to: the 6×6 (or 2×2) coherent block (`MB_nlevel_regular`), and the scalar
   ODEs for ground, other, 2s (`MB_ground_regular`, `MB_other_regular`, `MB_2s_regular`) — all
   driven by the *same* $\Omega_\sigma(t)$, $J_F(t)$ at the current $z$.
2. **Field propagation (Strang/symmetric split-step in $z$)** — for each of $\mathcal P$ and
   $\Omega_\sigma^{(\pm)}$, `Optics.py::Fresnel_propagator_with_absorption` applies
   $e^{-\kappa\Delta z/4}$ (half-step absorption) → FFT → free-space drift kernel
   $e^{-iz\pi\lambda(k_x^2+k_y^2)}$ (+ optional hard $k$-space low-pass, `is_kfilter`) → IFFT →
   $e^{-\kappa\Delta z/4}$ (other half-step absorption). $\kappa$ is evaluated from the *density
   matrix at the current $z$* (Eq. 34-type absorption, computed by `Model.py::absorption`).
3. **Coherent source injection** — after propagation, the classical polarization source
   (`Omega_source_regular`, the RHS of Eqs. 31/32 minus the $-\kappa\Omega/2$ term) is added as an
   explicit Euler step, $\Omega_\sigma \mathrel{+}= \Delta z\cdot(\text{source})$, using the
   *post-propagation* $z$-slice density matrix.
4. Diffraction of the transverse profile is handled entirely in the FFT step (zero-padded by
   `xpad`,`ypad` to suppress wraparound); the pump field is propagated the same way but scalar
   (no polarization tensor).
5. Circular ($\Omega_{-1},\Omega_{+1}$) ↔ linear (x,y) field bases are related by a fixed unitary
   `transform_matrix` (`tools.linear_to_circular`/`circular_to_linear`); all atomic dynamics run in
   the circular basis since $T_{ij\sigma}$ is diagonal there.

This is a **fully deterministic, mean-field (no-noise) split-step Maxwell–Bloch solver** — the
stochastic seeding described by PDF Eqs. 5,6,27 exists in the theory but (per scope, and also
matching `is_use_stochastic` typically being disabled) is not covered here.

---

## Part II — New: explicit 2s-hole satellite pathways

### 9. Motivation

The lumped treatment (§7/Eq. M2) assumes every Auger daughter of the 2s hole behaves like an
ordinary 2p$_{3/2}$ hole — same transition frequency, same coupling — because it dumps the fed
population straight into the resonant $\rho_{ii}$ diagonal. Physically, L1→L3M45 and L1→L3M23 Auger
decay leaves the ion with **two** simultaneous holes (2p$_{3/2}$ *and* a 3d or 3p spectator), and the
subsequent 2p→1s radiative decay of *that* configuration is measurably detuned from plain Kα1 (the
spectator hole shifts both levels via reduced screening) — this is the well-known Kα
satellite/hypersatellite structure (KL$_3$M$_{4,5}$, KL$_3$M$_{2,3}$ in Siegbahn/IUPAC notation).
Modeling this explicitly means these satellites get their own coherent Maxwell–Bloch dynamics,
detuned from and coupled to the *same* shared field $\Omega_\sigma$, rather than being folded into
the main line.

### 10. Notation and scope

> **⚠ Assumption (notation).** We read the user's "+/−" as the standard X-ray spin-orbit shorthand
> $nl^+\equiv nl_{j=l+1/2}$, $nl^-\equiv nl_{j=l-1/2}$. So $2p^+=2p_{3/2}$ (the only 2p level this
> code ever tracks — $2p^-=2p_{1/2}$/L2 is not modeled, consistent with Kα2 also not being tracked
> coherently, §2), $3d^+=3d_{5/2}$, $3d^-=3d_{3/2}$, $3p^+=3p_{3/2}$. 1s has no fine structure so it
> carries no suffix. **Please confirm** — if "+/−" instead labels something else (e.g. two
> empirically-resolved satellite components rather than literal $j$-sublevels of the spectator), the
> equations below are unaffected structurally, only the Auger-branching justification in §12.1
> changes.

Three channels, indexed $k=1,2,3$, spectator label $X_k\in\{3d^+,3d^-,3p^+\}$:

| $k$ | Lower state $L_k$ (L3-like) | Upper state $U_k$ (K-like) | Spectator shell |
|---|---|---|---|
| 1 | $2p^+3d^+$ | $1s3d^+$ | $3d_{5/2}$ |
| 2 | $2p^+3d^-$ | $1s3d^-$ | $3d_{3/2}$ |
| 3 | $2p^+3p^+$ | $1s3p^+$ | $3p_{3/2}$ |

> **⚠ Assumption (which state is "upper"/"lower").** By analogy with the base system (§2: 1s-hole
> is upper/K-like, decays radiatively into 2p-hole/L3-like — confirmed from the *direction* of the
> $G_{ij}$ branching matrix, §2), and because the existing Auger feed (Eq. M2) populates the
> **2p-containing** configuration directly (L1→L3M: the 2s vacancy is filled by a 2p electron,
> leaving 2p+spectator holes) — $2p^+X_k$ must be the state fed by 2s-Auger decay, i.e. the
> **lower/L3-like** member of the pair. $1sX_k$ is then upper/K-like, and (symmetric with the base
> system, where the 1s-hole is populated by direct photoionization of neutral ground) must be
> populated by direct photoionization of the spectator shell **from the already-1s-holed ion** — this
> is exactly the "photoionization from 1s to 1s3p+" example in the prompt (§12.2).

> **Design decision (full sublevel structure, not a 2-level reduction).** Each satellite pair is
> now modeled with the **same sublevel-resolved 6-level structure as the base Kα1 block** — 4
> msublevels for $L_k$ ($2p^+X_k$, $m=-\tfrac32..\tfrac32$) and 2 msublevels for $U_k$ ($1sX_k$,
> $m=\mp\tfrac12$) — reusing $T_{ij\sigma}$ (Eq. 25) and $G_{ij}$ (Eq. 17) **exactly as-is**, per
> channel. This replaces an earlier plan to use an isolated 2-level reduction per pair (Eq. M0-style):
> that reduction requires guessing an effective coupling/branching that reproduces the real 6-level
> polarization and branching behaviour — exactly the kind of thing that's easy to get subtly wrong
> in practice — whereas reusing the base tensors outright is both simpler and exact under the
> spectator approximation (widths box below), at negligible extra compute cost (density-matrix size
> is not the bottleneck here). **No new coupling/branching parameters are needed** — every channel
> inherits Eq. 25 and Eq. 17 verbatim; only per-channel *rates* (detuning, widths, feed cross
> sections) are new.

> **⚠ Assumption (widths).** Default $\Gamma_{L,k}=\Gamma_{L3}$, $\Gamma_{K,k}=\Gamma_K$ (spectator
> approximation: the M-shell hole is a bystander to the L3/K core-hole decay). Real Auger phase
> space is mildly reduced by the pre-existing spectator vacancy, but that's a second-order
> correction; flagging as override-able rather than modeling it.

Level scheme (detunings not to scale; every box below is itself a sublevel manifold — 4-wide for
the $2p^+X_k$ boxes, 2-wide for the $1sX_k$ boxes, exactly like the base K/L3 manifolds):

```
1s-hole (K, 2 sublevels)  ---Γ_K, Γ_sp∙2/3 (Kα1)--->  2p-hole (L3, 4 sublevels)   [Δ = 0, existing system]

1s3d+ (2 sublevels)  ---Γ_K, Γ_sp∙2/3, detuned Δ_1--->  2p+3d+ (4 sublevels)   <--Γ_A(2s→L3M4)--  2s-hole
1s3d- (2 sublevels)  ---Γ_K, Γ_sp∙2/3, detuned Δ_2--->  2p+3d- (4 sublevels)   <--Γ_A(2s→L3M5)--  2s-hole
1s3p+ (2 sublevels)  ---Γ_K, Γ_sp∙2/3, detuned Δ_3--->  2p+3p+ (4 sublevels)   <--Γ_A(2s→L3M2/3)--  2s-hole
  ^                                                        ^
  |  photoionize 3d/3p spectator, m-preserving (§12.2)     |  photoionize 3d/3p spectator, m-preserving (§12.1)
  |  from 1s-hole sublevels                                |  from 2p-hole sublevels
1s-hole population                                       2p-hole population
```

### 11. General block formalism

Each satellite channel $k$ gets its own **6-level block**, locally indexed exactly like the base
system (local $0..3$ = $2p^+X_k$ msublevels, local $4,5$ = $1sX_k$ msublevels), with zero coherence
to any other block or to the base block (different electron configurations ⇒ zero dipole overlap —
only *populations* cross between blocks, via Auger/photoionization). Eq. 4 — with the detuning term
reinstated (§4) — applies verbatim inside each block:
$$
\partial_t\rho^{(k)}_{ij}= -i\Delta_k\,\mathrm{sign}_{ij}\,\rho^{(k)}_{ij}
-\Big(\sum_s\tfrac{\gamma^{(k)}_{si}+\gamma^{(k)}_{sj}}{2}\Big)\rho^{(k)}_{ij}+\delta_{ij}\sum_s\gamma^{(k)}_{is}\rho^{(k)}_{ss}
+i\sum_{\sigma,s}\rho^{(k)}_{sj}T_{is\sigma}\!\left[H_{is}\Omega_\sigma^{(+)}+H_{si}\Omega_\sigma^{(-)}\right]
-i\sum_{\sigma,s}\rho^{(k)}_{is}T_{sj\sigma}\!\left[H_{sj}\Omega_\sigma^{(+)}+H_{js}\Omega_\sigma^{(-)}\right]
+\delta_{ij}F^{(k)}_i(\vec r,t),\qquad i,j\in\{0..5\}_{\text{local to }k}
\tag{S1}
$$
with $\mathrm{sign}_{ij}=+1$ for local-K→local-L3 ($i\in\{4,5\},j\in\{0..3\}$), $-1$ for the reverse,
$0$ on the diagonal — same convention as Eq. 7, applied within the block. $T_{ij\sigma}$, $H_{ij}$
are the **unmodified base tensors** (Eq. 25, Eq. 7, Eq. 18's $e^{(L3)},e^{(K)}$ reused verbatim as
each block's local $e^{(L)},e^{(K)}$); $\Delta_k$ is the only structurally new term relative to the
base system's realisation (Eq. 11–24, which implicitly has $\Delta=0$ throughout, §4).
$F^{(k)}_i(\vec r,t)$ collects all external population feed into local level $i$, derived below.

### 12. Elementary processes

#### 12.1 Feed into the lower manifold $L_k$ ($2p^+X_k$, local indices 0–3)

**(a) 2s-hole Auger decay**, generalizing Eq. M2 — spread evenly over the 4 msublevels (same
convention as the existing code; no known angular dependence to do otherwise):
$$
F^{(k)}_{i,\text{Auger}}=\frac{e^{(L)}_i}{\sum_{i'}e^{(L)}_{i'}}\,\Gamma_A^{(2s\to L_k)}\,\rho^{(2s)},\qquad i\in\{0,1,2,3\},\quad e^{(L)}=(1,1,1,1,0,0)\ \text{(local)}
\tag{S2}
$$
> **⚠ Assumption (Auger branching).** $\Gamma_A^{(L1\to L3M45)}$=5.07 eV currently covers *both*
> $3d^+$ and $3d^-$ combined; there is currently **no rate at all** for L1→L3M23 (3p spectator) — the
> code's only 2s Auger channel is M45. Proposed default split by spectator statistical weight
> $g=2j+1$: $\Gamma_A^{(2s\to 2p^+3d^+)}=\tfrac{6}{10}\Gamma_A^{(L1\to L3M45)}=3.04\text{ eV}$,
> $\Gamma_A^{(2s\to 2p^+3d^-)}=\tfrac{4}{10}\Gamma_A^{(L1\to L3M45)}=2.03\text{ eV}$. For 3p, a new
> total L1→L3M23 rate must be supplied (not derivable from the current config) and, of that,
> $\tfrac{4}{6}$ (statistical weight of $3p_{3/2}$ vs $3p_{1/2}$) assigned to $\Gamma_A^{(2s\to
> 2p^+3p^+)}$; the $3p^-$ remainder stays in the generic/lumped bucket (§12.6) since $3p^-$ isn't
> being modeled "for now." These are rough degeneracy-weighted placeholders, not measured partial
> rates — replace if better values exist. This choice is orthogonal to the 6-level-vs-2-level
> decision above (it's about *which* msublevels get the Auger population, not how many are tracked).

**(b) Direct spectator photoionization from the existing 2p-hole population** — unlike the Auger
feed, ionizing a spectator M-shell electron doesn't touch the core-hole's own angular momentum, so
this is a **sublevel-preserving** map (base 2p-hole sublevel $i$ feeds satellite sublevel $i$
directly), not an even spread:
$$
F^{(k)}_{i,\text{ion}}=\rho_{ii}(\vec r,t)\Big[\sigma^{(2p\to L_k)}_{\mathcal P}J_{\mathcal P}+\sigma^{(2p\to L_k)}_{\mathcal E}\big(J_{\Omega_{-1}}+J_{\Omega_{+1}}\big)\Big],\qquad i\in\{0,1,2,3\}
\tag{S3}
$$
where $\rho_{ii}$ is the **base block's** 2p-hole diagonal population at msublevel $i$ (matching
index). This distinction (uniform-spread Auger feed vs. sublevel-preserving photoionization feed)
is only visible once sublevels are resolved — a direct benefit of dropping the 2-level reduction.
As before, this is a *transfer*, not a pure loss like `S_ion_Fi` — see §12.7 for the double-counting
fix this requires.

#### 12.2 Feed into the upper manifold $U_k$ ($1sX_k$, local indices 4–5) — the requested worked example

Sublevel-preserving spectator photoionization from the existing 1s-hole population, structurally
identical to (S3):
$$
F^{(k)}_{i,\text{ion}}=\rho_{ii}(\vec r,t)\Big[\sigma^{(1s\to U_k)}_{\mathcal P}J_{\mathcal P}+\sigma^{(1s\to U_k)}_{\mathcal E}\big(J_{\Omega_{-1}}+J_{\Omega_{+1}}\big)\Big],\qquad i\in\{4,5\}
\tag{S4}
$$
Concretely for $k=3$ ($U_3=1s3p^+$, the requested "photoionization from 1s to $1s3p^+$" example),
written per 1s msublevel ($i=4$: $m=-\tfrac12$, $i=5$: $m=+\tfrac12$):
$$
\Big(\partial_t\rho^{(3)}_{ii}\Big)_{feed}=\rho_{ii}(\vec r,t)\Big[\sigma^{(1s\to 1s3p^+)}_{\mathcal P}J_{\mathcal P}(\vec r,t)+\sigma^{(1s\to 1s3p^+)}_{\mathcal E}\big(J_{\Omega_{-1}}(\vec r,t)+J_{\Omega_{+1}}(\vec r,t)\big)\Big],\qquad i\in\{4,5\}
\tag{S4'}
$$
where $\rho_{ii}$ ($i=4,5$) is the **base block's** 1s-hole diagonal population. No analogous
single-photon feed from *ground* into $U_k$ or $L_k$ directly is included — creating either
double-hole configuration in one photon from neutral ground is a two-electron (shake-off-type)
process, cross section assumed negligible. Flag if this should be included.

#### 12.3 Coherent Maxwell–Bloch coupling and radiative decay $U_k\to L_k$

Fully reuses the base branching structure (Eq. 12's second term / Eq. 17), per channel:
$$
F^{(k)}_{i,\text{rad}}=\Gamma_{sp}\sum_{s\in\{4,5\}}G_{is}\,\rho^{(k)}_{ss},\qquad i\in\{0,1,2,3\}
\tag{S5}
$$
and the decay superoperator (identical in form to Eq. 13, block-local $e^{(L)},e^{(K)}$ exactly as
in Eq. 18):
$$
M^{(k)}_{ij}=\Gamma_{L,k}\,e^{(L)}_ie^{(L)}_j+\Gamma_{K,k}\,e^{(K)}_ie^{(K)}_j+\tfrac{\Gamma_{L,k}+\Gamma_{K,k}}{2}\big(e^{(L)}_ie^{(K)}_j+e^{(K)}_ie^{(L)}_j\big)+\Gamma^{(add)}_k \tag{S6}
$$
($\Gamma^{(add)}_k$ an optional per-channel extra pure dephasing, mirroring `additional_dephasing`.)
As in the base system, $\Gamma_{sp}\sum_iG_{is}=\tfrac23\Gamma_{sp}=\Gamma_{rK\alpha1}$ for any source
sublevel $s$; the remaining $\tfrac13\Gamma_{sp}$ fraction of $U_k$'s radiative decay is the
satellite's own untracked-L2 analogue ($1sX_k\to 2p^-X_k$, not modeled) — pure loss, same
bookkeeping role as the base system's dropped Kα2 branch.

#### 12.4 Further-ionization loss (optional, deferred)

Structurally identical to Eqs. 20–22, applied per block; new cross sections
$\sigma^{(L_k\to\cdot)}_F,\sigma^{(U_k\to\cdot)}_F$ for triple-ionization of the double-hole states.
**Default to 0** (no data) unless supplied; purely a loss term, contributing to opacity only
(§12.6), same as the base `S_ion_Fi` treatment.

#### 12.5 Field source term (generalizes Eqs. 31/32)

All blocks share the same physical field mode, so the polarization source sums over blocks, reusing
the **same** $T_{ij\sigma}$ tensor for every block (no per-channel coupling parameter):
$$
\left(\text{source}\right)_\sigma^{(\pm)} = \pm i\,\tfrac{3}{8\pi}\lambda^2\Gamma_{sp}\,n\left[T^{(\mp)}_{i'j'\sigma}\tfrac{\rho_{j'i'}+\rho^*_{i'j'}}{2}+\sum_{k=1}^{3}T^{(\mp)}_{i'j'\sigma}\tfrac{\rho^{(k)}_{j'i'}+\rho^{(k)*}_{i'j'}}{2}\right] \tag{S7}
$$
Each block's contribution naturally oscillates at its own $\Delta_k$ relative to the rotating frame
(carried by $\rho^{(k)}_{ij}(t)$'s own equation of motion, Eq. S1) — no special handling needed
beyond including the detuning term.

#### 12.6 Opacity (generalizes Eq. 34)

$$
\kappa_F=n\Big(\rho^{(ground)}S^{(ground)}_{Fi'}+\rho^{(other)}S^{(other)}_F+\rho^{(2s)}S^{(2s)}_F
+\rho_{i'i'}S^{(ion.)}_{Fi'}+\sum_{k=1}^{3}\sum_{i'\in\{0..5\}}\rho^{(k)}_{i'i'}S^{(ion,k)}_{Fi'}+\sigma^{(compound)}_F\Big) \tag{S8}
$$
$S^{(ion,k)}$ includes **both** the transfer cross sections (S3)/(S4) — these are real photon
absorption events and must count toward opacity even though they populate a tracked state rather
than vanishing — **and** the optional §12.4 loss cross sections.

#### 12.7 Double-counting adjustments to existing terms

Every new channel above is carved out of population/opacity that the current code already accounts
for generically. Each must be *subtracted* from its current generic home, mirroring the
`sigma1_pump_other: ... # 3.23e-7 - 1.87e-7` pattern (§7). (The sublevel-vs-scalar distinction
doesn't change *what* gets subtracted, only how the transferred population is distributed once
inside a channel — S2 vs. S3/S4.)

- $\Gamma_A^{(2s\to 2p^+3d^+)}+\Gamma_A^{(2s\to 2p^+3d^-)}$ **replaces** (not adds to)
  $\Gamma_A^{(L1\to L3M45)}$'s contribution to the generic Eq. M2 feed — i.e. once these two channels
  are explicit, Eq. M2's generic feed should drop the M45 rate entirely (it would otherwise land in
  *both* the generic 2p-hole population and the new $2p^+3d^\pm$ populations).
- The new L1→L3M23 total rate, minus $\Gamma_A^{(2s\to2p^+3p^+)}$, is a *new addition* to the generic
  Eq. M2 feed (it was previously part of the unmodeled $\Gamma_{L1}-\Gamma_A^{(L1\to L3M45)}=3.06$ eV
  remainder, §7) — the generic bucket's implicit rate should increase by (3p− share) and the total
  budget $\Gamma_{L1}$ must still be respected.
- $\sigma^{(2p\to L_k)}_F$ (S3) must be subtracted from the base 2p-hole's generic further-ionization
  cross section (`sigma2_*_2p3`, feeding the base `S_ion_Fi` row) so the *total* opacity/loss rate
  of the 2p-hole population is unchanged — only its fate (untracked loss vs. explicit $L_k$
  population) changes.
- $\sigma^{(1s\to U_k)}_F$ (S4) must likewise be subtracted from `sigma2_*_1s`.
- Ground-state pump accounting (Eq. 28/PDF) is **unaffected** — no new ground-state channel is added
  in this iteration (§12.2).

### 13. Parameter inventory

New physical inputs required (none derivable from the current repo — all need atomic-structure data
or literature values). Reusing the base $T_{ij\sigma}$/$G_{ij}$ tensors verbatim means **no new
coupling or branching-ratio parameters are needed** — only rates:

| Quantity | Count | Notes |
|---|---|---|
| Detunings $\Delta_k$ | 3 | satellite transition energy − Kα1 energy, converted via $/\hbar$ |
| Auger split $\Gamma_A^{(2s\to L_k)}$ | 3 (2 from splitting M45, 1 new L3M23 total + split) | §12.1(a); statistical-weight defaults proposed |
| Spectator photoionization $\sigma^{(2p\to L_k)}_{\mathcal P,\mathcal E}$ | 6 (2 fields × 3 channels) | §12.1(b), sublevel-preserving |
| Spectator photoionization $\sigma^{(1s\to U_k)}_{\mathcal P,\mathcal E}$ | 6 (2 fields × 3 channels) | §12.2, sublevel-preserving |
| Widths $\Gamma_{L,k},\Gamma_{K,k}$ | optional, default = $\Gamma_{L3},\Gamma_K$ | spectator approximation |
| Further-ionization loss $\sigma^{(L_k\to\cdot)},\sigma^{(U_k\to\cdot)}$ | optional, default 0 | §12.4 |

### 14. Consistency checks to run once implemented

- With all three channels' rates/cross-sections set to zero and `use_2s_pathway` behaviour otherwise
  unchanged, the simulation must reproduce **current** results exactly (pure refactor limit).
- Total population, summed over ground/other/2s/base-block(6)/all three satellite-blocks(6 each),
  should only decrease via the *documented* untracked-loss channels (§7 remainder, §12.3's $1/3$
  Kα2-analogue, §12.4 if enabled) — any other drift indicates a double-counted or missing term from
  §12.7.
- $\Gamma_A^{(2s\to 2p^+3d^+)}+\Gamma_A^{(2s\to 2p^+3d^-)}+\Gamma_A^{(2s\to2p^+3p^+)}+(\text{3p}^-\text{ remainder})+(\text{other untracked}) = \Gamma_{L1}$ exactly (budget check, §7/§12.6).
- With all three $\Delta_k=0$, each satellite block's coherent response should be structurally
  identical to the base block's, differing only by whatever feed rate is injected — a useful
  debugging limit distinct from the "all rates zero" check above.

---

## Part III — New: 2p$_{1/2}$ (L2, Kα2) pathway

### 15. Motivation and scope

The base system (§2) tracks only 2p$_{3/2}$ (L3): the code's own comment on the L3 opacity/decay
budget states plainly that the Kα2 branch ($1/3$ of $\Gamma_{sp}$, decaying into 2p$_{1/2}$/L2) "is
lost from the model." This section makes that branch explicit: the 2p$_{1/2}$ hole manifold gets
its own coherent Maxwell–Bloch dynamics, coupled to the field at the (~20 eV lower) Kα2 energy,
instead of being pure loss. Equation numbers here use prefix `(K*)`. Satellite-of-L2 physics (a
2p$_{1/2}$+spectator double hole, analogous to Part II) is explicitly **out of scope** for this
section — see §19.

### 16. Decision: one shared field, not a second one

**Design decision.** Kα1 and Kα2 photons are represented as **one shared field** $\Omega_\sigma^{(\pm)}(\vec r,\tau)$
(the same field already used throughout Parts I–II), with the Kα2 coherences carrying an additional
detuning relative to the existing Kα1 rotating frame — **not** a second, independent field with its
own carrier wavevector and propagation equations.

This is a physical statement, not a numerical convenience: a real photodetector measuring the
Kα1+Kα2 emission sees one electromagnetic field with two spectral components beating against each
other, exactly the quantity a single time-domain envelope $\Omega_\sigma(\vec r,\tau)$ correctly
represents (the same mechanism Eq. S7 already uses to sum multiple *satellite* coherences into one
shared field — Kα2 is simply one more term in that same sum, Eq. K5 below). A second field would
duplicate the entire propagation pipeline (diffraction, absorption, zero-padding) for a field 99.75%
degenerate with the first ($\Delta E_{K\alpha1,K\alpha2}/\hbar\omega_0\approx0.25\%$), and the
codebase already anticipated the shared-field approach: `XLO_sim.py` computes
`DeltaomegaL2mL3A = (hwKalpha1N - hwKalpha2N)/hbar` and `f05Kalpha12A`, both present since before
this section but **used nowhere until now** — a clear breadcrumb for the intended design.

> **⚠ Numerical consequence.** Sharing one field means $\Omega_\sigma$'s envelope must now resolve
> a ~20 eV beat frequency (period $2\pi\hbar/\Delta E_{K\alpha}\approx0.21$ fs), an order of
> magnitude faster than the satellite channels' 1–3 eV detunings (§13's parameter inventory). This
> tightens the existing $\Delta t$-resolution requirement from the numerical-scheme discussion
> (docs numerics review) — worth re-checking $\Delta t\cdot\Delta E_{K\alpha}/\hbar$ stays
> comfortably resolved (≳10–20 samples/period) whenever `tgrid`/`tmax` are changed.

### 17. Structural consequence: this extends the base block, it is not a new block

Unlike the satellite pathways (Part II), 2p$_{1/2}$ shares its **entire 1s (K) population** with
the existing 2p$_{3/2}$ (L3) block — there is only one physical 1s-hole state in the ion, decaying
via *either* Kα1 or Kα2, not two independent 1s populations. A satellite-style separate 6-level
block (its own local K manifold, Part II §11) would therefore be wrong: it would silently duplicate
the 1s population instead of sharing it, and coherences $\rho_{2p_{1/2},1s}$ could not be
represented at all in either block's own array.

The only structurally correct implementation is to **extend the base block itself**: local indices
$0..3$ = 2p$_{3/2}$ (unchanged), $4,5$ = 1s (unchanged), and **new** local indices $6,7$ = 2p$_{1/2}$
$(m=-\tfrac12,\tfrac12)$, i.e. `nlevel` grows from 6 to 8 when this pathway is enabled
(`XLO_sim.py`: `self.nlevel = nlevel_base + 2`). This is implemented as a genuine extension, not an
independent block: it reuses **every** existing base-block code path in `Sample.py` and
`Model.py::absorption` unchanged (they are already generic in `nlevel`), the only new code is the
tensor construction in `XLO_sim.py::__init__` below.

`use_L2_pathway: True` requires `nlevel: 6` in the YAML (the base value before extension) and is
currently **mutually exclusive with `satellite_channels`** (§19).

### 18. Dipole tensor and branching matrix: Clebsch–Gordan derivation

$T_{ij\sigma}$ (Eq. 25) and $G_{ij}$ (Eq. 17) for 2p$_{3/2}\leftrightarrow$1s were given, not
derived, in the PDF. For 2p$_{1/2}\leftrightarrow$1s (both $j=1/2$), reusing them is not an option —
the msublevel structure and Clebsch–Gordan coefficients genuinely differ — so this section derives
them from the Wigner–Eckart theorem, cross-checked three independent ways.

**Radial/reduced-matrix-element ratio.** Using the standard 6j-symbol identity relating
$\langle 1s\|D\|2p_j\rangle$ for $j=3/2$ and $j=1/2$ (same radial integral, LS/statistical
approximation — consistent with this repo's own $\Gamma_{rK\alpha1}/\Gamma_{rK\alpha2}=1.943$ being
close to the pure-CG prediction of exactly 2, PDF's existing constants):
$$
|\langle 1/2\|D\|3/2\rangle|^2 \,/\, |\langle 1/2\|D\|1/2\rangle|^2 = 2,
$$
reproducing the config's own $\Gamma_{rK\alpha1}\!:\!\Gamma_{rK\alpha2}\approx2\!:\!1$ ratio exactly
from angular-momentum algebra alone — a strong consistency check on the derivation below.

**Angular part.** For $j=1/2\to j'=1/2$ dipole coupling, the standard closed-form CG coefficients
$\langle j,m;1,q|j,m+q\rangle$ (same-$j$ rank-1 coupling) give exactly two allowed
$(m,\sigma,m')$ triples with $m,m'\in\{-\tfrac12,\tfrac12\}$ — compare to 2p$_{3/2}$'s four:
$$
1s(m\!=\!-\tfrac12) \xrightarrow{\sigma=\text{idx }1} 2p_{1/2}(m\!=\!\tfrac12), \qquad
1s(m\!=\!\tfrac12) \xrightarrow{\sigma=\text{idx }0} 2p_{1/2}(m\!=\!-\tfrac12),
\tag{K1}
$$
using this code's own established $\sigma$-index$\leftrightarrow\Delta m$ convention (index 0
$\leftrightarrow\Delta m(1s{-}2p)=+1$, index 1 $\leftrightarrow-1$, read off the existing
2p$_{3/2}$ `Tijs` entries). $|CG|^2=2/3$ for each (verified via unitarity: the complementary
$j'=3/2$ channel then carries the remaining $1/3$, matching the well-known $1/2\otimes1$
Clebsch–Gordan table). A third, $\sigma$-index-0/1 channel with $q=0$ ($m'=m$, weight $1/3$) exists
but — like the base block — is **not** part of $T_{ij\sigma}$ (paraxial $\sigma=\pm1$ only), only of
$G_{ij}$ (isotropic, all $q$).

**Self-consistency check against the existing base tensor.** For 2p$_{3/2}$, every nonzero
$T_{ij\sigma}$ satisfies $T_{ij\sigma}^2=G_{ij}$ exactly (verified against all four existing
entries: $(1/\sqrt3)^2=1/3=G_{04}$, $(1/3)^2=1/9=G_{14}$, etc.) — the paraxial $\sigma=\pm1$
channels are the *only* decay path for those specific $(i,j)$ pairs, so $G_{ij}$ (properly
normalized to $\Gamma_{rK\alpha1}/\Gamma_{sp}=2/3$ per source, by construction) reduces to the bare
angular weight. Applying the same logic with the pure-CG total ($1/3$ per source, not the empirical
$\Gamma_{rK\alpha2}/\Gamma_{sp}=0.340$, for consistency with how the 2p$_{3/2}$ matrix is *not*
individually re-normalized to its own empirical branching either) gives, using local indices
$6=2p_{1/2}(m{=}-\tfrac12)$, $7=2p_{1/2}(m{=}\tfrac12)$, $4,5=1s(m{=}-\tfrac12,\tfrac12)$:
$$
T_{7,4,\sigma=1}=T_{4,7,\sigma=1}=T_{6,5,\sigma=0}=T_{5,6,\sigma=0}=\sqrt{2}/3, \qquad T^2=2/9,
\tag{K2}
$$
$$
G_{7,4}=G_{6,5}=\tfrac{2}{9},\qquad G_{6,4}=G_{7,5}=\tfrac{1}{9}
\tag{K3}
$$
— exactly the same $1/9,2/9$ building-block fractions the existing 2p$_{3/2}$ matrix already uses
(Eq. 17), now arranged for 2 msublevels instead of 4; unitarity check
$\sum_{L3\text{ or }L2}G_{i,4} = 2/3+1/3=1$ per 1s source ✓.

### 19. Generalized detuning: $\Delta_{ij}$ matrix, not a scalar

Reusing the satellite mechanism's scalar $\Delta_k$ (Eq. S1) does not work here: with three
manifolds now sharing one block (K, L3, L2), a single scalar cannot express both the (zero) K↔L3
detuning and the nonzero K↔L2 *and* L3↔L2 detunings simultaneously. Generalizing to a per-level
vector $f_i$ ("intrinsic detuning of level $i$'s manifold from the shared Kα1 rotating frame"):
$$
f_i = \begin{cases}\Delta\omega_{L2-L3}=(\omega_{K\alpha1}-\omega_{K\alpha2}) & i\in\text{L2 (2p}_{1/2}\text{)}\\ 0 & i\in\text{K or L3}\end{cases},
\qquad
\Delta_{ij} = f_i - f_j,
\tag{K4}
$$
replacing the base-block's $\Delta{\cdot}\text{sign}_{ij}$ term (§4) in the general Eq. 4 form.
**Verified exact equivalence** to the existing satellite mechanism: for $f_i=\Delta_k\, e^{(K)}_i$
(a satellite's single-manifold case), $f_i-f_j\equiv\Delta_k\,\text{sign}_{ij}$ for every $(i,j)$ —
so Eq. K4 is a strict generalization, not a new formalism; the code (`_MB_nlevel_regular_core`) now
takes a single precomputed $\Delta_{ij}$ matrix parameter for both the base/L2 block and each
satellite channel (`chan.Delta_ij = Delta_k \cdot \text{sign\_ij\_block}`, precomputed once in
`XLO_sim.py` rather than combined inside the numba core).

With $E_{L3}=0,\,E_K=\hbar\omega_{K\alpha1}$ as reference, Eq. K4 follows from the standard
Eq. 4 form applied to all three pairs; note $\Delta_{ij}(K,L2)=\Delta_{ij}(L3,L2)=-\Delta\omega_{L2-L3}$
— both reduce to the same value because both are measured against the same K↔L3-resonant frame.

> **⚠ Scope note.** `use_L2_pathway` and `satellite_channels` are currently **mutually exclusive**
> (`XLO_sim.py` raises `ValueError` if both are set) — the satellite blocks reuse the base
> $T_{ij\sigma}/G_{ij}$ tensors verbatim (Part II §10), which only cover 2p$_{3/2}$↔1s; a
> 2p$_{1/2}$-spectator satellite channel (e.g. $2p^-3d^+$) would need its own new tensor derivation,
> analogous to §18 but combined with a spectator hole, and is deferred.

### 20. New physical inputs

| Quantity | Value / source | Notes |
|---|---|---|
| $\Delta\omega_{L2-L3}$ | `DeltaomegaL2mL3A`, already computed from `hwKalpha1N`,`hwKalpha2N` | no new input needed |
| $\Gamma_{L2}$ (`GammaL2eVN`) | new config key, no default | e.g. via `xatom_tools.state_total_decay_width_eV('2p1,0')`, calibrated against `GammaL3eVN`'s own XATOM/literature ratio the same way satellite `Gamma_L_eV` is (XATOM gives $\Gamma_{L2}=0.666$, $\Gamma_{L3}=0.635$ eV — ratio $1.049\times$ `GammaL3eVN`$=0.61\to0.640$ eV) |
| $\sigma^{(ground\to2p_{1/2})}_{\mathcal E}$ (`sigma1_Ka1_2p1`) | new config key, no default | XATOM ground-state `-pcs`, `2p-` row (≈$0.51\times$`sigma1_Ka1_2p3`) |
| $\sigma^{(2p_{1/2}\to\cdot)}_{\mathcal E}$ (`sigma2_Ka1_2p1`) | new config key, no default | XATOM `-pcs` on hole config `2p1,0`, total cross section |

> **⚠ Assumption (m-resolved ground-state branching).** Unlike $T_{ij\sigma}/G_{ij}$ (§18, pure
> bound-state Clebsch–Gordan algebra), the *photoionization* branching into specific
> 2p$_{1/2}$ msublevels from neutral ground state involves the outgoing photoelectron's partial
> waves (radial, not purely angular, physics) — not independently re-derived here. An even 50/50
> split across the 2 msublevels is used as a placeholder (`XLO_sim.py`), unlike 2p$_{3/2}$'s
> XATOM/literature-sourced $\{0.12,0.18,0.28,0.42\}$-type pattern (§2). Flag if a proper m-resolved
> calculation should replace this.

### 21. Numerical scheme — no changes needed

Because §17 implements this as an extension of the existing base block rather than a new one,
**every** downstream code path (`Sample.py`'s z/t marching, `Model.py::absorption`,
`Model.py::Omega_source_regular`) already generalizes automatically: they are written generically
in `X.nlevel`/`X.Tijs`/`X.S_ground_Fi` etc., so an 8-level array flows through unchanged. In
particular, the 2p$_{1/2}$↔1s coherences contribute to the **same** shared field source sum
(Eq. S7-style, now with an implicit fourth term) with **no separate call**:
$$
\left(\text{source}\right)_\sigma^{(\pm)} \mathrel{+}= \pm i\,\tfrac{3}{8\pi}\lambda^2\Gamma_{sp}\,n\,T^{(\mp)}_{i'j'\sigma}\tfrac{\rho_{j'i'}+\rho^*_{i'j'}}{2}\Big|_{i',j'\in\{6,7\}\times\{4,5\}}
\tag{K5}
$$
— this is literally the *same* `Omega_source_regular(X, rho_ijxy)` call already used for the base
block, applied to the now-8×8 `rho_ijxy`, realizing §16's "one shared field" decision directly.
Verified end-to-end (both `keep_z_history=True/False` paths): `nlevel` correctly becomes 8, results
are finite, population trace stays $\le1$, and the base-block-only (`use_L2_pathway: False`) path
reproduces prior results bit-for-bit (regression-tested).

### 22. Consistency checks

- With `use_L2_pathway: False` (default/absent), results must reproduce prior behavior exactly —
  verified (identical `max|Ω|` at the sample exit, before and after this section's refactor of the
  detuning mechanism, §19).
- With `use_L2_pathway: True`, population summed over all 8 base-block sublevels (now including
  2p$_{1/2}$) plus ground/other/2s should still only decrease via the same documented untracked-loss
  channels as before (§7) — the L2 extension redistributes *where* the previously-lost $1/3$
  Kα2 fraction goes, it does not add a new loss channel.
- $\Gamma_{L2}$ from XATOM should be checked against $\Gamma_{L3}$'s own literature/XATOM ratio
  rather than trusted in absolute terms (§20) — both this repo's `GammaL3eVN` and XATOM's own
  $\Gamma_{L3}$ prediction differ by a few percent, a discrepancy that should carry through
  consistently to $\Gamma_{L2}$.
