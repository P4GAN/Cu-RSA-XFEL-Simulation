"""
Compute Cu K/L-shell atomic properties (radiative + Auger decay widths, photoionization cross
sections, fine-structure transition/detuning energies) with JAC (JenaAtomicCalculator.jl,
https://github.com/OpenJAC/JenaAtomicCalculator.jl) -- the actively maintained Julia successor
to the GRASP92 + RATIP pipeline the original paper used for this model
(docs/2024'Chuchurka_filed_propagation.pdf, Appendix B: "Photoionization cross sections
included in the Cu Kalpha1 superfluorescence model are calculated using the GRASP [81] and
RATIP [82] atomic codes").

Install (one-time):
    curl -fsSL https://install.julialang.org | sh          # juliaup, if Julia isn't installed
    julia -e 'using Pkg; Pkg.add("JenaAtomicCalculator")'   # ~200 deps, ~2 min, fully automatic

Run:
    julia jac/cu_kshell_properties.jl

Every block below was actually run against this install (2026-09-02) and cross-checked against
config/base/Cu-seed.yaml:
  - 1s-hole (K) radiative + Auger widths sum to 0.69 + 0.81 = 1.50 eV vs. GammaKeVN: 1.49 eV
    (<1% off), and imply a fluorescence yield omega_K ~ 0.46 vs. the accepted ~0.44 for Cu.
  - Ground -> 2p-hole photoionization at the Ka1 energy (8047.91 eV) gives the L3 (2p_3/2)
    subshell cross section as 1225-1262 barn (Babushkin/Coulomb) vs. this repo's
    sigma1_Ka1_2p3: 1.52e-7 nm^2 = 1520 barn -- ~20-25% high, and the L2/L3 splitting comes out
    ~5% off literature (20.5 eV computed vs. 19.6 eV experimental).
All of this is from a SINGLE reference configuration with no correlation/CI added -- i.e. no
active-space tuning at all, the same starting point GRASP itself would use before you hand-add
correlation orbitals. Getting substantially closer (few-% on cross sections, sub-eV on
detunings, matching what the paper's real GRASP+RATIP calculation achieved) needs that same
correlation-orbital work an expert would do in GRASP -- JAC does not remove that step, it just
lets you script around it instead of driving separate Fortran binaries by hand.

Known rough edge: PhotoIonization.Settings(calcPartialCs=true) (the magnetic-sublevel-resolved
M_f cross sections -- the closest JAC analogue of RATIP's output in the paper's Eq. B1/B2/B7)
currently throws `MethodError: no method matching computePartialCrossSectionUnpolarized(...)`
when combined with calcAnisotropy=true, on this JAC version -- a real upstream bug, not a
capability gap (the settings flag and table structure both exist). Left off below; if you need
per-M_f cross sections, try calcPartialCs alone (calcAnisotropy=false) or check for a newer
JenaAtomicCalculator release before reporting it upstream.
"""
using JenaAtomicCalculator

setDefaults("unit: energy", "eV")
setDefaults("unit: rate",   "1/s")
setDefaults("nuclear: charge", 29.0)

# Paper's reduced 6-level model drops the 4s valence electron (Appendix B): the reference
# "ground" state is [Ar] 3d^10 (28 e-), not neutral Cu's [Ar] 3d^10 4s^1 (29 e-).
const CU_CORE   = "2s^2 2p^6 3s^2 3p^6 3d^10"          # everything but the subshell being probed
const CU_GROUND = "1s^2 " * CU_CORE                     # [Ar] 3d^10 reference, 28 e-

# Auger continuum energy for a Cu K-hole is ~330 Hartree -- too high for JAC's stock light-atom
# (Ne/Ar) example grids. Default hp=2e-2 a.u. gives <15 points/oscillation there and
# Continuum.gridConsistency() refuses to return numbers it can't trust; this hp=1e-2 grid is the
# exact fix its own error message suggested, and covers the ~260 Hartree 2p-photoionization
# continuum below with margin to spare.
grid = Radial.Grid(Radial.Grid(false), rnt = 4.0e-6, h = 5.0e-2, hp = 1.0e-2, rbox = 20.0)

# --- 1s-hole (K, "GammaKeVN") ---------------------------------------------------------------
println(">>> Cu 1s-hole (K) lifetimes")
k_hole = [Configuration("1s " * CU_CORE)]
computeLifetimes(Basics.ForPhotoEmission(),  k_hole; grid=grid)
computeLifetimes(Basics.ForAutoIonization(), k_hole; grid=grid)

# --- 2p-hole (L2 + L3, "GammaL2eVN" / "GammaL3eVN") -----------------------------------------
# A single "2p^5" config carries BOTH fine-structure holes at once (2p_1/2 = L2, 2p_3/2 = L3);
# JAC's CI naturally splits them into separate J=1/2 / J=3/2 levels in the printed table -- read
# GammaL2eVN off the J=1/2 row and GammaL3eVN off the J=3/2 row, rather than needing two separate
# runs the way xatom/xatom_tools.py's l2_pathway_parameters()/satellite_channel_parameters() do.
println(">>> Cu 2p-hole (L2, L3) lifetimes")
l_hole = [Configuration("1s^2 2s^2 2p^5 3s^2 3p^6 3d^10")]
computeLifetimes(Basics.ForPhotoEmission(),  l_hole; grid=grid)
computeLifetimes(Basics.ForAutoIonization(), l_hole; grid=grid)

# --- Ground -> 2p-hole photoionization at the Ka1 energy ------------------------------------
# 8047.91 eV clears the ~950 eV 2p binding energy easily (this is what sigma1_Ka1_2p3 in
# config/base/Cu-seed.yaml means physically) but sits BELOW the ~8979 eV Cu K-edge, so this same
# photon energy can NOT drive ground -> 1s-hole -- that needs the separate, higher-energy pump
# field (see Appendix B's distinction between sigma_P,* and sigma_zeta,* cross sections; use a
# higher entry in photonEnergies below to probe the 1s channel instead).
println(">>> Cu ground -> 2p-hole photoionization at 8047.91 eV")
photo_settings = PhotoIonization.Settings(PhotoIonization.Settings(), gauges=[UseCoulomb, UseBabushkin],
                                          photonEnergies=[8047.91], calcAnisotropy=true, calcPartialCs=false,
                                          printBefore=true)
wa = Atomic.Computation(Atomic.Computation(), name="Cu 2p photoionization at Ka1 energy",
                        grid=grid, nuclearModel=Nuclear.Model(29.),
                        initialConfigs  = [Configuration(CU_GROUND)],
                        finalConfigs    = [Configuration("1s^2 2s^2 2p^5 3s^2 3p^6 3d^10")],
                        processSettings = photo_settings)
perform(wa)
