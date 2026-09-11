"""
Cross-section and satellite-detuning values obtained from the GRASP2018 + RATIP-2012 Dirac-Fock
pipeline (see grasp_tools.py for the method/unit-conversion background), and their comparison
against config/base/Cu-seed-original.yaml.

The GRASP/RATIP calculations themselves are interactive Fortran runs against files in an external
scratch working directory (not part of this repo -- see grasp_tools.py's "External setup" section
to reproduce them); this script records the resulting numbers as data (so the provenance/method
for each value is documented next to it) and reproduces the unit-converted comparison against the
config. Run directly (`python cross_section_validation.py`) to print the summary tables.

IMPORTANT -- two different "other"/auxiliary scopes, don't mix them up (see grasp/NEXT_SESSION.md
section 0 for the full explanation): the 7-channel sum (2s + 2p_1/2 + 3s + 3p + 3d) below is used
ONLY to validate against Cu-seed-original.yaml's Table I, which predates the 2s/2p_1/2 split. The
actual `use_2s_pathway`/`use_L2_pathway`-enabled target config files (Cu-seed-SASE-double-
satellite.yaml etc.) need the narrower 5-channel sum instead (3s + 3p + 3d only, no 2s/2p_1/2,
since those have their own separate keys there) -- see SIGMA1_OTHER5_AU below, which is exactly
that 5-channel value and is what should actually be written into sigma1_Ka1_other/sigma2_Ka1_other
for those files. Pump-energy values (sigma1_pump_*/sigma2_pump_*) were likewise computed only to
validate against the original paper's table -- those keys don't exist in the satellite-extended
target config files at all, don't compute new ones for that recompute.
"""

from grasp_tools import au_to_nm2

# ---------------------------------------------------------------------------------------------
# Satellite transition energies (bound-bound, via rcsfgenerate+rmcdhf+rci+rbiotransform+
# rtransition), dominant line per channel selected by A_coulomb * (2J+1) [* purity for the
# lower-confidence double-3d channels]. All energies in eV.
# ---------------------------------------------------------------------------------------------

BASE_KA1_EV = 8053.094  # ground-detuning reference: no-spectator K-hole -> L3-hole
BASE_KA2_EV = 8033.078  # same, -> L2-hole

# (channel, Ka1-satellite energy eV, Ka2-satellite energy eV or None if not computed)
SATELLITE_LINES_EV = {
    "3d+": (8054.224, 8031.681),  # hole in 3d_5/2
    "3d-": (8053.437, 8030.946),  # hole in 3d_3/2
    "3p+": (8054.880, 8029.788),  # hole in 3p_3/2
    "3p-": (8051.580, 8037.056),  # hole in 3p_1/2 -- the standout: +3.98 eV Ka2-satellite detuning
    # Double-3d-spectator channels: lower confidence (much more CSF mixing in the lower/core-hole
    # manifold than the single-spectator cases, see the memory note on this). detuning_eV_L2_split
    # now done too (satellite_detuning_analysis.py's double_3d_l2_splits()) -- the L2/L3 manifold
    # separation on the lower state turned out to be clean even though the spectator-pair
    # sub-assignment within each manifold isn't, so Ka1/Ka2 could be extracted the same way as the
    # Ka1-only value was, just filtering by lower-state manifold too.
    "3d+3d+": (8054.685, 8030.628),
    "3d-3d+": (8052.178, 8029.933),
    "3d-3d-": (8050.667, 8033.645),
}


# GammarKalpha1eVN/GammarKalpha2eVN: radiative partial widths for the Kalpha1 (K-hole -> L3-hole)
# and Kalpha2 (K-hole -> L2-hole) lines specifically -- Gamma = hbar * A_coulomb, read directly off
# k_base3.l23_base3.ct (already computed as a side effect of the satellite-detuning reference
# energies, no new calculation needed). Excellent agreement with the existing XATOM values.
HBAR_EV_S = 6.582119569e-16
GAMMA_R_KALPHA_EV = {
    "Kalpha1": HBAR_EV_S * 5.97553e14,  # 0.393317 eV vs XATOM's 0.39375 (ratio 0.9989)
    "Kalpha2": HBAR_EV_S * 3.06279e14,  # 0.201596 eV vs XATOM's 0.202668 (ratio 0.9947)
}


def satellite_detunings():
    """detuning_eV / detuning_eV_L2_split for every satellite_channels/double_satellite_channels
    entry, matching the fields actually written into config/base/Cu-seed*-double-satellite.yaml.
    """
    out = {}
    for name, (ka1, ka2) in SATELLITE_LINES_EV.items():
        entry = {"detuning_eV": round(ka1 - BASE_KA1_EV, 4)}
        if ka2 is not None:
            entry["detuning_eV_L2_split"] = round(ka1 - ka2, 4)
        out[name] = entry
    return out


# ---------------------------------------------------------------------------------------------
# Photoionization cross sections (xphoto), in atomic units (a0^2), Coulomb gauge. This is the
# full set from Table I of the original paper (Chuchurka et al.) -- see its notation: superscript
# (g)/(i) = from ground state / ionization cross section of the state named; subscript P/Omega =
# pump (9000 eV) / Kalpha1 field (8047.91 eV) photon energy; final/initial-hole label 1s, 2p
# (meaning 2p_3/2), or "a" (auxiliary/other = everything except 1s and 2p_3/2, i.e. 2s + 2p_1/2 +
# 3s + 3p + 3d in this original, pre-use_2s_pathway/use_L2_pathway model -- confirmed by getting
# both sigma1_pump_other and sigma1_Ka1_other to <0.2% against the paper only once 2s/2p1/2 were
# included in the sum, not just 3s/3p/3d).
#
# Config key <-> paper parameter:
#   sigma1_pump_1s    = sigma_P,1s^(g)     sigma2_pump_1s    = sigma_P,1s^(i)
#   sigma1_pump_2p3   = sigma_P,2p^(g)     sigma2_pump_2p3   = sigma_P,2p^(i)
#   sigma1_pump_other = sigma_P,a^(g)      sigma2_pump_other = sigma_P^(a)
#   sigma1_Ka1_2p3    = sigma_Omega,2p^(g) sigma2_Ka1_1s     = sigma_Omega,1s^(i)
#   sigma1_Ka1_other  = sigma_Omega,a^(g)  sigma2_Ka1_2p3    = sigma_Omega,2p^(i)
#                                          sigma2_Ka1_other  = sigma_Omega^(a)
# (there is no sigma1_Ka1_1s / sigma_Omega,1s^(g): 8047.91 eV is below the K-edge, ground-state
# absorption can't create a 1s hole there.)
# ---------------------------------------------------------------------------------------------

# sigma1_*_2p3: ground-state photoionization cross section into a 2p_3/2 hole. Single xphoto
# line each (cu_groundmax.c -> cu_lholemax's L3/J=3/2 level).
SIGMA1_2P3_AU = {"pump": 3.701e-5, "Ka1": 5.433e-5}

# sigma1_*_2s / _2p1: the two "auxiliary" channels that got their own explicit config keys once
# use_2s_pathway/use_L2_pathway existed, but are part of the original model's "other" sum.
SIGMA1_2S_AU = {"pump": 7.740e-5, "Ka1": 1.013e-4}
SIGMA1_2P1_AU = {"pump": 2.002e-5, "Ka1": 2.931e-5}

# The 5 "other" subshells this reduced (no-4s) atomic model can represent: 3s, 3p_1/2, 3p_3/2,
# 3d_3/2, 3d_5/2. XATOM's own OTHER_HOLE set (xatom_tools.py) also includes 4s, which this
# model's ground state doesn't have an electron in -- omitted here, and XATOM's own per-subshell
# breakdown (xatom/recomputed_parameters.json) shows 4s is a small (~3%) contributor, so this is
# a minor, understood gap, not a missing large piece.
SIGMA1_OTHER5_AU = {
    "pump": {"3s": 1.059e-5, "3p-": 2.469e-6, "3p+": 4.520e-6, "3d-": 8.960e-8, "3d+": 1.270e-7},
    "Ka1": {"3s": 1.396e-5, "3p-": 3.563e-6, "3p+": 6.551e-6, "3d-": 1.460e-7, "3d+": 2.073e-7},
}


def sigma1_other_au(energy):
    """Full original-model "auxiliary" sum: 2s + 2p_1/2 + the 5 "other" subshells."""
    return SIGMA1_2S_AU[energy] + SIGMA1_2P1_AU[energy] + sum(SIGMA1_OTHER5_AU[energy].values())


# sigma2_*_1s / _2p3: TOTAL further-ionization cross section from an already-hole ion, at the
# given photon energy -- summed over every remaining-shell double-hole final state (RATIP's own
# "Total" column can't be used for this since it accumulates across everything in one xphoto run,
# not just the channels belonging to one initial level -- see sum_initial_level_cross_sections).
SIGMA2_1S_CHANNELS_AU = {
    "pump": {"1s+2s": 8.348e-5, "1s+3s": 1.1772e-5, "1s+2p": 4.6694e-5,
             "1s+3p": 5.7902e-6, "1s+3d": 2.8389e-7},
    "Ka1": {"1s+2s": 2.736e-5 + 8.246e-5, "1s+3s": 3.832e-6 + 1.163e-5,
            "1s+2p": 6.647e-6 + 1.895e-5 + 4.280e-5, "1s+3p": 8.323e-7 + 2.351e-6 + 5.211e-6,
            "1s+3d": 1.544e-7 + 1.005e-7 + 2.075e-7},
}

SIGMA2_2P3_CHANNELS_AU = {
    # 2p3+1s is E1-forbidden at both energies (see grasp_tools.py note). 2p3+3p (Ka1 only) was
    # blocked by what looked like an unresolved rmcdhf CSF-loader crash (LODCSH2: ncf=10
    # ncfblock=5) -- turned out to just be a wrong block-serial-number input sequence in the
    # original attempt (the same kind of mistake made and fixed several times this session for
    # other multi-block cases), not a real GRASP bug. Rebuilt cu_l3p3p cleanly and it converges
    # fine; see grasp/NEXT_SESSION.md. pump-energy 2p3+3p still not computed (out of scope, see
    # module docstring -- pump values were only ever needed for the one-time paper validation).
    "pump": {"2p3+1s": 0.0, "2p3+2s": 5.9252e-5, "2p3+3s": 8.42791e-6,
             "2p3+3d": 2.159743e-7, "2p3+2p_further": 3.43596e-5},
    "Ka1": {"2p3+1s": 0.0, "2p3+2s": 7.781e-5, "2p3+3s": 1.108709e-5,
            "2p3+3d": 3.525956e-7, "2p3+2p_further": 5.018610e-5, "2p3+3p": 8.80874e-6},
}

# sigma2_Ka1_2s (2s-hole ion's own total further-ionization cross section, Ka1 energy only --
# no sigma2_pump_2s needed per NEXT_SESSION.md scope). Initial state: cu_2sholemax (new maximal-
# peel 2s-hole reference built this session, seeded from cu_2shole.w). Final double-hole states
# built fresh this session (cu_2s2p/cu_2s3s/cu_2s3p/cu_2s3d.c/.w/_full.mix in the working
# directory), all converged with 2s frozen (rmcdhf hit the same "Method 2 unable to solve"
# failure the recipe warns about -- fixed the same way, excluding 2s from "orbitals to be
# varied"). 2s+1s (ejecting the deep 1s electron from a 2s-hole ion) is energetically forbidden
# at 8047.91 eV (1s binding energy ~8979 eV) -- confirmed by an explicit xphoto run against the
# already-existing cu_1s2s final state returning 0 initialized lines, not just assumed; recorded
# as 0.0 here matching the existing 2p3+1s precedent (SIGMA2_2P3_CHANNELS_AU above), which is
# forbidden for the same reason.
SIGMA2_2S_CHANNELS_AU = {
    "Ka1": {"2s+1s": 0.0, "2s+2p": 8.5631e-5, "2s+3s": 1.5274e-5,
            "2s+3p": 1.08366e-5, "2s+3d": 4.7375e-7},
}

# sigma2_Ka1_2p1 (2p_1/2/L2-hole ion's own total further-ionization cross section, Ka1 only).
# Initial state: level 2 of the EXISTING cu_lholemax (2-level L-hole reference: level 1 = L3
# J=3/2- at -1617.72 Ha, level 2 = L2 J=1/2- at -1616.98 Ha, per xrelci_lholemax_full.log -- no
# rebuild needed, just re-selected the other level). Final states: the SAME already-existing
# cu_l3p1s/cu_l3p2s/cu_l3p2pf/cu_l3p3d files used for sigma2_Ka1_2p3 -- confirmed by inspection
# these already use combined (2p-,2p) notation on the L-shell hole (e.g. cu_l3p1s.c is literally
# identical to cu_1s2p.c), so they cover both L2- and L3-initiated channels already; only new
# xphoto pairings were needed (paired against level 2 instead of level 1), no new SCF builds.
# l3p1s (2p1+1s) is 0 lines / E1-threshold-forbidden, same as 2p3+1s. l3p3p (2p1+3p) is NOT
# included -- the existing cu_l3p3p.c reference hits the same unresolved rmcdhf CSF-loader crash
# documented for sigma2_Ka1_2p3 (LODCSH2: ncf=10 ncfblock=5, confirmed by re-checking
# cu_l3p3p_rmcdhf.log), so this is a 4/5-channel sum with the same ~4% expected gap as
# sigma2_Ka1_2p3. Gauge agreement 97.0%.
SIGMA2_2P1_CHANNELS_AU = {
    # 2p1+3p (via the fixed cu_l3p3p, see sigma2_Ka1_2p3's own note above) completes this to 5/5;
    # previously 4/5 with a coincidentally-close ratio (1.023) that got slightly worse (1.093) once
    # complete -- the complete number is the trustworthy one, not the earlier partial agreement.
    "Ka1": {"2p1+1s": 0.0, "2p1+2s": 1.0349e-4, "2p1+2p_further": 6.7352e-5,
            "2p1+3d": 4.770054e-7, "2p1+3p": 1.168721e-5},
}

# sigma2_Ka1_other: sigma1-share-weighted average (xatom_tools.other_state_parameters convention,
# see grasp/NEXT_SESSION.md §6) of the 5 auxiliary subshells' (3s, 3p_1/2, 3p_3/2, 3d_3/2, 3d_5/2)
# OWN total further-ionization cross sections at Ka1 energy -- the single biggest remaining piece
# of this campaign, now done. Initial states: cu_3shole (1-level, already existed and converged
# from earlier sigma1_Ka1_other work), cu_3phole (2-level: level 1 = 3p_3/2 J=3/2- at -1649.161 Ha,
# level 2 = 3p_1/2 J=1/2- at -1649.069 Ha -- already existed but its xrelci run had never been
# rerun with the '*' block-separator stripped, so it crashed the same way cu_lholemax's did; fixed
# identically), cu_3dhole (2-level: level 1 = 3d_5/2 J=5/2+ at -1651.948 Ha, level 2 = 3d_3/2
# J=3/2+ at -1651.939 Ha, same fix). Final double-hole states: reused cu_1s3s/cu_2s3s/cu_l3p3s
# (3s channels), cu_1s3p/cu_2s3p (3p channels), cu_1s3d/cu_2s3d/cu_l3p3d (3d channels) built
# earlier this session/campaign, plus 6 new final states built for this piece specifically:
# cu_3s3p, cu_3s3d (mirror the cu_2sXX pattern exactly), cu_3p3d (combined-combined 3p+3d, 11
# CSFs/5 blocks -- converged fine despite similar complexity to the crashed cu_l3p3p), cu_3pfurther
# (double 3p-hole via non-relativistic combined notation "3p(4,*)", 5 CSFs/3 blocks, J=0,1,2),
# cu_3dfurther (double 3d-hole via "3d(8,*)", 9 CSFs/5 blocks, J=0..4). All 1s+X channels are
# E1/threshold-forbidden (0 lines), consistent with the 1s+2s/2p3+1s/2p1+1s precedent.
#
# One known gap remains (down from two): 3s is missing the "3s further" (full 3s depletion,
# ejecting the last 3s electron) channel -- xphoto ran cleanly (exit 0) but silently stopped after
# the first continuum-orbital convergence block without ever reaching the "Results for PI
# line"/summary section -- reproduced twice identically, not a transient issue. This specific edge
# case (single-CSF J=0 final ion with one fewer declared subshell than the initial state, since
# rcsfgenerate drops 0-occupation subshells entirely) was never hit earlier in the campaign. Given
# the analogous "X+X_further" channels elsewhere are non-negligible (e.g. 2p3+2p_further was
# ~15-20% of sigma2_Ka1_2p3), this gap is a real caveat, not a rounding footnote -- flagged for a
# follow-up debugging session (would need the same lldb/-fcheck=all treatment the original xphoto
# segfault got).
#
# The 3p-/3p+ "2p+3p" gap that used to be here is FIXED: the cu_l3p3p reference that looked like
# it hit an unresolved rmcdhf CSF-loader crash (LODCSH2 ncf=10/ncfblock=5) turned out to just have
# a wrong block-serial-number input sequence in the original attempt -- rebuilt cleanly (see
# grasp/NEXT_SESSION.md), and reused for sigma2_Ka1_2p3's own missing 2p3+3p channel too, which
# now matches the original-paper GRASP/RATIP value to 0.03% (was 94% with the channel missing).
SIGMA2_OTHER5_AU = {
    "Ka1": {
        "3s": 1.950564e-4,   # 5/6 channels -- missing 3s+3s_further (see caveat above)
        "3p-": 2.072977e-4,  # 5/5 channels, complete (2p+3p added via the fixed cu_l3p3p)
        "3p+": 1.551334e-4,  # 5/5 channels, complete
        "3d-": 1.543184e-4,  # 6/6 channels, complete
        "3d+": 1.042625e-4,  # 6/6 channels, complete
    },
}


def sigma2_other_au(energy):
    """sigma1-share-weighted average of the 5 SIGMA2_OTHER5_AU subshells, using SIGMA1_OTHER5_AU
    as weights (same convention as xatom_tools.other_state_parameters)."""
    s1 = SIGMA1_OTHER5_AU[energy]
    s2 = SIGMA2_OTHER5_AU[energy]
    s1_total = sum(s1.values())
    return sum(s1[k] * s2[k] for k in s1) / s1_total

# sigma2_pump_other / sigma2_Ka1_other (sigma_P^(a) / sigma_Omega^(a)): NOT YET COMPUTED. This is
# the total further-ionization rate averaged over all 5 auxiliary hole types (2s, 2p_1/2, 3s, 3p,
# 3d) -- roughly 5x the work of one sigma2_*_1s value, since each of those 5 needs its own set of
# double-hole final states built from scratch. sigma2_Ka1_2s (one of the 5) is now done -- see
# SIGMA2_2S_CHANNELS_AU above.
CONFIG_VALUES_NM2 = {
    "sigma1_pump_1s": 2.53e-6,  # near-threshold at my own K-edge -- see the module docstring note
    "sigma1_pump_2p3": 1.04e-7,
    "sigma1_pump_other": 3.23e-7,
    "sigma1_Ka1_2p3": 1.52e-7,
    "sigma1_Ka1_other": 4.34e-7,
    "sigma2_pump_1s": 4.75e-7,
    "sigma2_pump_2p3": 3.02e-7,
    "sigma2_Ka1_1s": 6.53e-7,
    "sigma2_Ka1_2p3": 4.15e-7,
    "sigma2_Ka1_2s": 4.05e-7,  # config's current (XATOM-derived) value, for comparison
    "sigma2_Ka1_2p1": 4.6903e-7,  # config's current (XATOM-derived) value, for comparison (5/5 channels)
    "sigma2_pump_other": 3.27e-7,  # not yet computed on my side
    "sigma2_Ka1_other": 5.0613e-7,  # config's current (XATOM-derived) value, for comparison
}


def cross_section_summary():
    """Print every computed cross section against its config/base/Cu-seed-original.yaml value."""
    rows = {}  # config_key -> mine_nm2 (or None if not computed)
    rows["sigma1_pump_2p3"] = au_to_nm2(SIGMA1_2P3_AU["pump"])
    rows["sigma1_Ka1_2p3"] = au_to_nm2(SIGMA1_2P3_AU["Ka1"])
    rows["sigma1_pump_other"] = au_to_nm2(sigma1_other_au("pump"))
    rows["sigma1_Ka1_other"] = au_to_nm2(sigma1_other_au("Ka1"))
    rows["sigma2_pump_1s"] = au_to_nm2(sum(SIGMA2_1S_CHANNELS_AU["pump"].values()))
    rows["sigma2_Ka1_1s"] = au_to_nm2(sum(SIGMA2_1S_CHANNELS_AU["Ka1"].values()))
    rows["sigma2_pump_2p3"] = au_to_nm2(sum(SIGMA2_2P3_CHANNELS_AU["pump"].values()))
    rows["sigma2_Ka1_2p3"] = au_to_nm2(sum(SIGMA2_2P3_CHANNELS_AU["Ka1"].values()))
    rows["sigma2_Ka1_2s"] = au_to_nm2(sum(SIGMA2_2S_CHANNELS_AU["Ka1"].values()))
    rows["sigma2_Ka1_2p1"] = au_to_nm2(sum(SIGMA2_2P1_CHANNELS_AU["Ka1"].values()))
    rows["sigma2_Ka1_other"] = au_to_nm2(sigma2_other_au("Ka1"))
    rows["sigma1_pump_1s"] = None  # near-threshold; see energy-scan note in the paper trail above
    rows["sigma2_pump_other"] = None

    print(f"{'quantity':22s} {'mine (nm^2)':>13s} {'config (nm^2)':>14s} {'ratio':>8s}")
    for key, cfg in CONFIG_VALUES_NM2.items():
        mine = rows.get(key)
        mine_str = f"{mine:13.4e}" if mine is not None else "  not computed"
        ratio_str = f"{mine / cfg:8.4f}" if mine is not None else "     n/a"
        print(f"{key:22s} {mine_str} {cfg:14.4e} {ratio_str}")


# ---------------------------------------------------------------------------------------------
# Auger feed into the satellite channels (Gamma_A_2s_eV / Gamma_A_K_eV / their _to_L2_eV
# siblings), via RATIP's xauger -- untested at the start of this campaign, now confirmed working.
#
# IMPORTANT -- see docs/auger-branching-and-ci-mixing.md for the full physics explanation before
# using or extending these numbers. Short version: the final state (cu_l3p3d, an "L-hole +
# 3d-hole" double-hole state with combined notation on both the L-shell and 3d-shell) has strong
# configuration mixing between "L3+3d_3/2" and "L3+3d_5/2" character (4 of its 7 L3-associated
# levels are close to 50/50 mixed) -- a real physical near-degeneracy (3d spin-orbit splitting
# ~0.245 eV is small compared to the L3-3d exchange coupling), not a numerical artifact. XATOM's
# numbers don't show this because it almost certainly treats each spectator configuration as an
# independent single-reference calculation, never building the off-diagonal coupling between them.
# The L2-associated levels stay >98% pure (only one, not three, overlapping total-J value for the
# L2(j=1/2) x 3d coupling), which is why the _to_L2_eV numbers below agree with XATOM much better
# than the to-L3 ones do.
#
# GAMMA_A_CSF_ID maps cu_l3p3d's global CSF index (1-11, counting through all 4 J-blocks in file
# order) to its physical (L-shell hole, M-shell hole) character; GAMMA_A_LEVEL_WEIGHTS is xrelci's
# own "weights of major contributors to ASF" (already |mixing coefficient|^2, i.e. population
# fractions) for each of cu_l3p3d's 11 levels -- a property of cu_l3p3d's own CI diagonalization,
# independent of which initial state (cu_2sholemax or cu_kholemax) the Auger rate is computed
# from. gamma_a_branching() combines these with a per-level total-rate dict (from xauger's
# "Individual and total Auger rates" table) to get the |c|^2-weighted branching into each of the
# 4 (L-shell, M-shell) buckets -- see the module docstring section 6 of the doc above for why
# this scheme (not "assign each level to its dominant configuration") was chosen.
#
# 3p+/3p- channels use the analogous "L-hole+3p-hole" final state (cu_l3p3p) -- this looked like
# it hit an unresolved rmcdhf LODCSH2 crash (same as sigma2_Ka1_2p3's missing 2p3+3p channel), but
# turned out to just be a wrong block-serial-number input sequence in the original attempt; fixed
# and reused here (see grasp/NEXT_SESSION.md). The 3p case shows MUCH weaker CI mixing than 3d
# (levels mostly >96% pure) -- consistent with the mixing mechanism in the doc above: this
# session's own cu_3phole calculation puts the 3p1/2-3p3/2 splitting at ~2.51 eV, about 10x the 3d
# splitting (~0.245 eV), so it's no longer competitive with the exchange coupling. Still branched
# the same |c|^2-weighted way for consistency, even though it barely matters here.
# ---------------------------------------------------------------------------------------------

HARTREE_TO_EV = 27.211386

GAMMA_A_3D_CSF_ID = {1: "L2+M4", 2: "L3+M4", 3: "L3+M5", 4: "L2+M4", 5: "L2+M5", 6: "L3+M4",
                      7: "L3+M5", 8: "L2+M5", 9: "L3+M4", 10: "L3+M5", 11: "L3+M5"}

GAMMA_A_3D_LEVEL_WEIGHTS = {
    1: {11: 1.00000},
    2: {7: 0.58667, 6: 0.41127, 5: 0.00192, 4: 0.00014},
    3: {9: 0.52275, 10: 0.47550, 8: 0.00175},
    4: {6: 0.58850, 7: 0.41035, 4: 0.00061, 5: 0.00054},
    5: {2: 0.96028, 3: 0.03892, 1: 0.00079},
    6: {10: 0.52083, 9: 0.46399, 8: 0.01518},
    7: {3: 0.95466, 2: 0.03772, 1: 0.00762},
    8: {4: 0.99898, 7: 0.00058, 5: 0.00031, 6: 0.00013},
    9: {5: 0.99723, 7: 0.00240, 4: 0.00027, 6: 0.00010},
    10: {8: 0.98308, 9: 0.01325, 10: 0.00367},
    11: {1: 0.99158, 3: 0.00641, 2: 0.00200},
}

# xauger "Individual and total Auger rates" per-level rates (a.u., Coulomb), cu_l3p3d as final
# state, initial state as labeled.
GAMMA_A_3D_LEVEL_RATES_AU = {
    "2sholemax": {1: 3.622e-2, 2: 4.876e-3, 3: 1.409e-2, 4: 4.627e-3, 5: 3.657e-3, 6: 2.990e-2,
                  7: 4.955e-3, 8: 1.384e-2, 9: 5.447e-3, 10: 2.138e-2, 11: 3.934e-3},
    "kholemax": {1: 4.505e-5, 2: 4.720e-6, 3: 1.738e-5, 4: 2.353e-6, 5: 9.663e-7, 6: 1.021e-4,
                 7: 1.139e-6, 8: 2.050e-5, 9: 1.704e-6, 10: 9.245e-5, 11: 9.873e-7},
}

# cu_l3p3p CSF -> (L-shell, 3p-shell) character (P- = 3p1/2 hole, P+ = 3p3/2 hole)
GAMMA_A_3P_CSF_ID = {1: "L2+P-", 2: "L3+P+", 3: "L2+P-", 4: "L2+P+", 5: "L3+P-", 6: "L3+P+",
                      7: "L2+P+", 8: "L3+P-", 9: "L3+P+", 10: "L3+P+"}

GAMMA_A_3P_LEVEL_WEIGHTS = {
    1: {6: 0.98408, 5: 0.01098, 4: 0.00399, 3: 0.00095},
    2: {10: 1.00000},
    3: {9: 0.98988, 8: 0.00778, 7: 0.00235},
    4: {2: 0.98384, 1: 0.01616},
    5: {5: 0.98743, 6: 0.01037, 4: 0.00218, 3: 0.00003},
    6: {8: 0.96201, 7: 0.03187, 9: 0.00612},
    7: {4: 0.99382, 6: 0.00455, 5: 0.00159, 3: 0.00004},
    8: {7: 0.96579, 8: 0.03021, 9: 0.00400},
    9: {3: 0.99898, 6: 0.00100, 4: 0.00002, 5: 0.00000},
    10: {1: 0.98384, 2: 0.01616},
}

GAMMA_A_3P_LEVEL_RATES_AU = {
    "2sholemax": {1: 4.441e-3, 2: 1.893e-2, 3: 3.169e-3, 4: 9.079e-3, 5: 5.996e-3,
                  6: 1.436e-2, 7: 6.478e-3, 8: 1.284e-2, 9: 8.272e-3, 10: 8.055e-3},
    "kholemax": {1: 7.216e-7, 2: 4.614e-6, 3: 7.422e-4, 4: 1.420e-4, 5: 1.621e-6,
                 6: 8.249e-4, 7: 5.551e-7, 8: 1.581e-3, 9: 1.703e-6, 10: 1.321e-4},
}


def gamma_a_branching(initial, csf_id, level_weights, level_rates_au):
    """|c|^2-weighted Auger branching (eV) into each (L-shell, M/P-shell) bucket, for the given
    initial state ('2sholemax' or 'kholemax') and final-state composition/rate tables."""
    rates = level_rates_au[initial]
    buckets = {}
    for level, rate in rates.items():
        for csf, weight in level_weights[level].items():
            bucket = csf_id[csf]
            buckets[bucket] = buckets.get(bucket, 0.0) + weight * rate
    return {k: v * HARTREE_TO_EV for k, v in buckets.items()}


GAMMA_A_2S_EV = gamma_a_branching("2sholemax", GAMMA_A_3D_CSF_ID, GAMMA_A_3D_LEVEL_WEIGHTS,
                                   GAMMA_A_3D_LEVEL_RATES_AU)
GAMMA_A_K_EV = gamma_a_branching("kholemax", GAMMA_A_3D_CSF_ID, GAMMA_A_3D_LEVEL_WEIGHTS,
                                  GAMMA_A_3D_LEVEL_RATES_AU)
GAMMA_A_2S_3P_EV = gamma_a_branching("2sholemax", GAMMA_A_3P_CSF_ID, GAMMA_A_3P_LEVEL_WEIGHTS,
                                      GAMMA_A_3P_LEVEL_RATES_AU)
GAMMA_A_K_3P_EV = gamma_a_branching("kholemax", GAMMA_A_3P_CSF_ID, GAMMA_A_3P_LEVEL_WEIGHTS,
                                     GAMMA_A_3P_LEVEL_RATES_AU)

GAMMA_A_CONFIG_EV = {
    ("Gamma_A_2s_eV", "3d+"): 1.5733, ("Gamma_A_2s_eV", "3d-"): 1.3121,
    ("Gamma_A_2s_to_L2_eV", "3d+"): 0.7137, ("Gamma_A_2s_to_L2_eV", "3d-"): 0.5309,
    ("Gamma_A_K_eV", "3d+"): 0.00259, ("Gamma_A_K_eV", "3d-"): 0.003083,
    ("Gamma_A_K_to_L2_eV", "3d+"): 0.002064, ("Gamma_A_K_to_L2_eV", "3d-"): 0.000637,
    ("Gamma_A_2s_eV", "3p+"): 1.4535, ("Gamma_A_2s_eV", "3p-"): 0.6458,
    ("Gamma_A_2s_to_L2_eV", "3p+"): 0.7902, ("Gamma_A_2s_to_L2_eV", "3p-"): 0.5062,
    ("Gamma_A_K_eV", "3p+"): 0.034001, ("Gamma_A_K_eV", "3p-"): 0.028913,
    ("Gamma_A_K_to_L2_eV", "3p+"): 0.028999, ("Gamma_A_K_to_L2_eV", "3p-"): 0.002563,
}


def gamma_a_summary():
    """Print the Auger-branching results against their config (XATOM) values."""
    rows = {
        ("Gamma_A_2s_eV", "3d+"): GAMMA_A_2S_EV["L3+M5"],
        ("Gamma_A_2s_eV", "3d-"): GAMMA_A_2S_EV["L3+M4"],
        ("Gamma_A_2s_to_L2_eV", "3d+"): GAMMA_A_2S_EV["L2+M5"],
        ("Gamma_A_2s_to_L2_eV", "3d-"): GAMMA_A_2S_EV["L2+M4"],
        ("Gamma_A_K_eV", "3d+"): GAMMA_A_K_EV["L3+M5"],
        ("Gamma_A_K_eV", "3d-"): GAMMA_A_K_EV["L3+M4"],
        ("Gamma_A_K_to_L2_eV", "3d+"): GAMMA_A_K_EV["L2+M5"],
        ("Gamma_A_K_to_L2_eV", "3d-"): GAMMA_A_K_EV["L2+M4"],
        ("Gamma_A_2s_eV", "3p+"): GAMMA_A_2S_3P_EV["L3+P+"],
        ("Gamma_A_2s_eV", "3p-"): GAMMA_A_2S_3P_EV["L3+P-"],
        ("Gamma_A_2s_to_L2_eV", "3p+"): GAMMA_A_2S_3P_EV["L2+P+"],
        ("Gamma_A_2s_to_L2_eV", "3p-"): GAMMA_A_2S_3P_EV["L2+P-"],
        ("Gamma_A_K_eV", "3p+"): GAMMA_A_K_3P_EV["L3+P+"],
        ("Gamma_A_K_eV", "3p-"): GAMMA_A_K_3P_EV["L3+P-"],
        ("Gamma_A_K_to_L2_eV", "3p+"): GAMMA_A_K_3P_EV["L2+P+"],
        ("Gamma_A_K_to_L2_eV", "3p-"): GAMMA_A_K_3P_EV["L2+P-"],
    }
    print(f"{'quantity':22s} {'channel':8s} {'mine (eV)':>10s} {'config (eV)':>12s} {'ratio':>8s}")
    for key, cfg in GAMMA_A_CONFIG_EV.items():
        mine = rows[key]
        print(f"{key[0]:22s} {key[1]:8s} {mine:10.5f} {cfg:12.5f} {mine / cfg:8.4f}")


# ---------------------------------------------------------------------------------------------
# Total (radiative + Auger) decay widths of the 4 base hole states -- GammaKeVN/GammaL1eVN/
# GammaL2eVN/GammaL3eVN. Radiative part: hbar*A_coulomb summed over every allowed rtransition
# line (only Kalpha1/Kalpha2 done for K -- Kbeta (K->M2/M3) not attempted, see gap below).
# Auger part: xauger, summed (not |c|^2-branched -- a TOTAL width doesn't care which channel
# population ends up in) over every energetically-allowed double-hole final state reachable from
# each base hole, reusing final states built for sigma2_Ka1_*/Gamma_A_* above wherever possible.
#
# GammaKeVN: Auger channels are every double-hole final state reachable from a bare K-hole via
# "one L-shell electron fills 1s, another electron (L- or M-shell) is ejected" -- KL1L2/L3
# (cu_2s2p), KL1M1/M2,3/M4,5 (cu_2s3s/2s3p/2s3d), KL2L2/L2L3/L3L3 (cu_l3p2pf), KL2,3M1 (cu_l3p3s),
# KL2,3M2,3 (cu_l3p3p), KL2,3M4,5 (cu_l3p3d). KL1L1 (final state cu_2sempty, 2s fully depleted) is
# NOT computed -- xauger exits 0 but silently stops after the first channel without reaching the
# results section, the SAME failure mode as sigma2_Ka1_3s's missing "3s+3s_further" channel (see
# that note above) -- confirms this is a general RATIP edge case (single-CSF J=0 final state
# missing a subshell from its declared list because rcsfgenerate drops 0-occupation subshells),
# not specific to xphoto. Radiative part uses only Kalpha1+Kalpha2 (Kbeta/K->M2,M3 not attempted).
#
# GammaL1eVN: Auger channels are every double-hole state reachable from cu_2sholemax via "2p
# electron fills 2s, M-shell electron ejected" -- L1->L3M1 (cu_l3p3s), L1->L3M2,3 (cu_l3p3p),
# L1->L3M4,5 (cu_l3p3d) [this last one is exactly the pre-per-channel-split GammaA_L1_to_L3M45eVN
# the theory doc describes]. L1->L2L3_further (cu_l3p2pf, "2p fills 2s, another 2p electron
# ejected") is confirmed energetically FORBIDDEN (0 transitions initialized -- 2s alone doesn't
# have enough excitation energy to create a full double-L-shell-hole), not missing due to a bug.
# No radiative contribution attempted (L-shell hole fluorescence yield is negligible; theory doc's
# own GammaL1eVN=8.13eV number is itself understood to be almost entirely Auger).
#
# GammaL2eVN/GammaL3eVN: Auger channels are the M-shell double-hole states reachable via "one
# M-shell electron fills the L-hole, another M-shell electron is ejected" -- reuses cu_3s3p/
# cu_3s3d/cu_3p3d/cu_3pfurther/cu_3dfurther wholesale from the sigma2_Ka1_other work. Also tried,
# for L2 only, the Coster-Kronig channel "L3 electron fills L2 (L2 is ~20eV more bound than L3,
# so this releases energy), M-shell electron ejected" via cu_l3p3s/cu_l3p3p/cu_l3p3d as final
# states -- confirmed energetically forbidden for all three (0 transitions each): the relevant
# levels of those final states sit ~15-40 eV *above* L2's own energy once the second-hole
# relaxation energy is properly accounted for (not the naive "L2-L3 splitting minus M binding"
# estimate), so this specific channel doesn't fill the ~40% gap between the Auger-only L2 total
# and its (itself only "calibrated", not first-principles, per the theory doc) config value.
# No radiative contribution attempted for either.
HARTREE_TO_EV_WIDTH = 27.211386  # same constant as HARTREE_TO_EV above, named separately for clarity here

GAMMA_TOTAL_WIDTH_CHANNELS_AU = {
    "K": {  # 8/9 channels -- missing KL1L1 (cu_2sempty, silent xauger stop)
        "KL1L2/L3 (2s2p)": 6.6340634139746219E-003,
        "KL1M1 (2s3s)": 5.2732612910215524E-004,
        "KL1M2/M3 (2s3p)": 8.2384956128803014E-004,
        "KL1M4/M5 (2s3d)": 3.7517174801750128E-005,
        "KL2L2/L2L3/L3L3 (l3p2pf)": 1.5875264147618404E-002,
        "KL2/L3M1 (l3p3s)": 7.1657331292122666E-004,
        "KL2/L3M2/M3 (l3p3p)": 3.4309831607442116E-003,
        "KL2/L3M4/M5 (l3p3d)": 2.8932479377504568E-004,
    },
    "L1": {  # complete except the energetically-forbidden L1->L2L3_further
        "L1->L3M1 (l3p3s)": 2.9884273770622690E-002,
        "L1->L3M2/M3 (l3p3p)": 9.1632045952098992E-002,
        "L1->L3M4/M5 (l3p3d)": 1.4291529624288574E-001,
    },
    "L3": {  # complete for the M-shell-filling channel set
        "3s3p": 1.671595e-03, "3s3d": 3.279300e-04, "3p3d": 7.538070e-03,
        "3pfurther": 6.660100e-03, "3dfurther": 8.419470e-03,
    },
    "L2": {  # complete for the M-shell-filling channel set; CK-to-L3 channel confirmed forbidden
        "3s3p": 1.647073e-03, "3s3d": 2.958500e-04, "3p3d": 7.309453e-03,
        "3pfurther": 6.554900e-03, "3dfurther": 7.920310e-03,
    },
}

GAMMA_TOTAL_RADIATIVE_EV = {
    "K": GAMMA_R_KALPHA_EV["Kalpha1"] + GAMMA_R_KALPHA_EV["Kalpha2"],  # Kbeta not attempted
    "L1": 0.0, "L2": 0.0, "L3": 0.0,  # not attempted (negligible for L1; not attempted for L2/L3)
}

GAMMA_TOTAL_WIDTH_CONFIG_EV = {"K": 1.49, "L1": 8.13, "L2": 1.04, "L3": 0.61}


def gamma_total_width_summary():
    """Print GammaKeVN/GammaL1eVN/GammaL2eVN/GammaL3eVN (Auger + radiative where computed)."""
    print(f"{'state':6s} {'Auger (eV)':>12s} {'radiative (eV)':>15s} {'total (eV)':>12s} "
          f"{'config (eV)':>12s} {'ratio':>8s}")
    for state, channels in GAMMA_TOTAL_WIDTH_CHANNELS_AU.items():
        auger_ev = sum(channels.values()) * HARTREE_TO_EV_WIDTH
        rad_ev = GAMMA_TOTAL_RADIATIVE_EV[state]
        total_ev = auger_ev + rad_ev
        cfg = GAMMA_TOTAL_WIDTH_CONFIG_EV[state]
        print(f"{state:6s} {auger_ev:12.4f} {rad_ev:15.4f} {total_ev:12.4f} {cfg:12.4f} "
              f"{total_ev / cfg:8.4f}")


if __name__ == "__main__":
    print("=== Satellite detunings (eV) ===")
    for name, entry in satellite_detunings().items():
        print(f"  {name:8s} {entry}")
    print()
    print("=== Cross sections ===")
    cross_section_summary()
    print()
    print("=== Auger branching into satellite channels (see docs/auger-branching-and-ci-mixing.md) ===")
    gamma_a_summary()
    print()
    print("=== Total decay widths (base hole states) ===")
    gamma_total_width_summary()
