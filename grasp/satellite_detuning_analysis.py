"""
Dominant-line selection for the satellite-channel Kalpha1/Kalpha2-satellite detunings in
grasp/cross_section_validation.py's SATELLITE_LINES_EV, from the raw GRASP2018 `rtransition`
.ct files.

Each satellite channel's upper (K-hole+spectator) and lower (L-hole+spectator) CSL files mix
several physically-distinct configurations (different spectator-hole j-sublevel, and for the
lower state, different core-hole manifold L2 vs L3) into shared J/parity symmetry blocks -- see
grasp_tools.py's "External setup" for why (rcsfgenerate's combined-notation trick explores every
j-sublevel placement of a hole count automatically, rather than needing separate hand-built
references per sublevel). This module records each level's dominant physical character (read off
the "Weights of major contributors to ASF" section of the corresponding `rci` .csum file) and
uses grasp_tools.dominant_line to pick out the one transition per channel that represents that
channel's actual Kalpha-satellite line.

Composition data below (upper_char/lower_char dicts) was read by hand from the .csum files in the
external GRASP working directory (see grasp_tools.py) -- it is not re-derived automatically here,
since re-parsing the "Weights of major contributors to ASF" section into structured hole-type
labels needs the physical CSF ordering from the corresponding .c file, which isn't preserved in a
simply-parseable form. Rerunning any of this from scratch means reading those files again.

Run directly to reproduce every dominant-line pick and cross-check against
cross_section_validation.py's recorded values.
"""

from grasp_tools import parse_ct, dominant_line

# ---------------------------------------------------------------------------------------------
# 3d+/3d- single-spectator channel (k_3dsat2.c / l23_3dsat2.c)
# ---------------------------------------------------------------------------------------------

# Upper (K-hole+3d spectator): (J, local level) -> dominant spectator character.
# Purity: J=1 and J=3 blocks are single-CSF (pure by construction); J=2 has two nearly-pure
# levels (99.89% each way).
_3D_UPPER_CHAR = {
    (1.0, 1): "3d-",
    (2.0, 1): "3d+",
    (2.0, 2): "3d-",
    (3.0, 1): "3d+",
}

# Lower (L-hole+3d spectator): (J, local level) -> (manifold, spectator, purity). J=2/J=3 levels
# 1-2 are genuinely mixed (~58-77% one CSF) -- this is why dominant-line-by-A-coefficient matters
# here rather than just picking "the purest level" for each character.
_3D_LOWER_CHAR = {
    (0.0, 1): ("L3", "3d-", 1.00),
    (1.0, 1): ("L3", "3d-", 0.960), (1.0, 2): ("L3", "3d+", 0.955), (1.0, 3): ("L2", "3d-", 0.992),
    (2.0, 1): ("L3", "3d+", 0.587), (2.0, 2): ("L3", "3d-", 0.589),
    (2.0, 3): ("L2", "3d-", 0.999), (2.0, 4): ("L2", "3d+", 0.997),
    (3.0, 1): ("L3", "3d-", 0.523), (3.0, 2): ("L3", "3d+", 0.521), (3.0, 3): ("L2", "3d+", 0.983),
    (4.0, 1): ("L3", "3d+", 1.00),
}

# ---------------------------------------------------------------------------------------------
# 3p+/3p- single-spectator channel (k_3psat.c / l23_3psat.c)
# ---------------------------------------------------------------------------------------------

_3P_UPPER_CHAR = {
    (0.0, 1): "3p-",
    (1.0, 1): "3p+", (1.0, 2): "3p-",
    (2.0, 1): "3p+",
}

_3P_LOWER_CHAR = {
    (0.0, 1): ("L3", "3p+", 0.941), (0.0, 2): ("L2", "3p-", 0.941),
    (1.0, 1): ("L3", "3p+", 0.962), (1.0, 2): ("L3", "3p-", 0.937),
    (1.0, 3): ("L2", "3p-", 0.998), (1.0, 4): ("L2", "3p+", 0.966),
    (2.0, 1): ("L3", "3p-", 0.696), (2.0, 2): ("L3", "3p+", 0.722), (2.0, 3): ("L2", "3p+", 0.910),
    (3.0, 1): ("L3", "3p+", 1.00),
}


def _extract_single_spectator(ct_path, upper_char, lower_char):
    """Shared logic for the 3d and 3p single-spectator cases: group candidate lines by
    (manifold, spectator character), requiring the spectator to match on both sides (a radiative
    K-to-L transition shouldn't flip which electron is the spectator), then take the dominant
    line per group.
    """
    transitions = parse_ct(ct_path)
    groups = {}
    for t in transitions:
        u = upper_char.get((t["j1"], t["lev1"]))
        lo = lower_char.get((t["j2"], t["lev2"]))
        if u is None or lo is None:
            continue
        manifold, lo_spec, purity = lo
        if u != lo_spec:
            continue
        groups.setdefault((manifold, u), []).append(t)

    results = {}
    for key, lines in groups.items():
        results[key] = dominant_line(lines, purity_by_level=None)  # already pre-filtered by spectator match
    return results


# ---------------------------------------------------------------------------------------------
# Double-3d-spectator channels (k_3d3dsat.c / l23_3d3dsat.c) -- lower confidence, see the module
# docstring: the lower/core-hole manifold mixes much more heavily here (many levels 45-80% pure)
# than the single-spectator cases, so composition must be summed by hole-TYPE GROUP (several CSFs
# sharing the same physical hole-type but different intermediate sub-coupling are the same
# physical state) rather than by raw CSF index, and the candidate-line ranking is weighted by
# upper-level purity as an explicit confidence discount.
# ---------------------------------------------------------------------------------------------

_3D3D_UPPER_GROUP = {
    (0.5, 1): ("3d-3d+", 0.814), (0.5, 2): ("3d-3d-", 0.444), (0.5, 3): ("3d+3d+", 0.546),
    (1.5, 1): ("3d-3d-", 0.584), (1.5, 2): ("3d+3d+", 0.749),
    (1.5, 3): ("3d-3d+", 0.705), (1.5, 4): ("3d-3d+", 0.976),
    (2.5, 1): ("3d-3d+", 0.992), (2.5, 2): ("3d-3d-", 0.580),
    (2.5, 3): ("3d+3d+", 0.758), (2.5, 4): ("3d-3d+", 0.686),
    (3.5, 1): ("3d+3d+", 0.823), (3.5, 2): ("3d-3d+", 0.996), (3.5, 3): ("3d-3d+", 0.827),
    (4.5, 1): ("3d+3d+", 0.826), (4.5, 2): ("3d-3d+", 0.826),
}


_3D3D_LOWER_MANIFOLD = {
    # Lower state (l23_3d3dsat, L-hole + double-3d spectator): (J, local level) -> dominant
    # MANIFOLD only (L2 or L3) -- unlike the single-spectator _3D_LOWER_CHAR, this does NOT
    # attempt to resolve which specific spectator-pair (3d-3d-/3d-3d+/3d+3d+) each level belongs
    # to, because that's heavily mixed within each manifold (no single CSF exceeds ~90%, several
    # levels are close to democratically mixed across 3-4 CSFs). The L2-vs-L3 separation itself is
    # very clean, though: read off l23_3d3dsat.csum's "weights of major contributors to ASF"
    # (GRASP2018 rci's own convention -- raw mixing coefficients, square them for population
    # fractions, unlike RATIP xrelci's convention elsewhere in this campaign which already prints
    # squared weights), every level's dominant CSF belongs to the same manifold as its next few
    # contributors, and the L2/L3 level counts per block match the manifold's own CSF counts in
    # l23_3d3dsat.c exactly (e.g. block J=3/2 has 7 L3-type and 4 L2-type CSFs, and sure enough
    # exactly 7 levels come out L3-dominant and 4 L2-dominant). This is enough to compute
    # detuning_eV_L2_split (Ka1 - Ka2 per spectator-pair group) by additionally filtering
    # _extract_double_3d's existing upper-group candidate lines by lower-state manifold, without
    # needing the finer (and much less reliable) spectator-pair split on the lower side at all --
    # the upper state's own composition already fixes which physical spectator pair a transition
    # belongs to, since the spectator doesn't change identity mid-transition.
    (0.5, 1): "L3", (0.5, 2): "L3", (0.5, 3): "L3", (0.5, 4): "L3",
    (0.5, 5): "L2", (0.5, 6): "L2", (0.5, 7): "L2",
    (1.5, 1): "L3", (1.5, 2): "L3", (1.5, 3): "L3", (1.5, 4): "L3", (1.5, 5): "L3", (1.5, 6): "L3",
    (1.5, 7): "L3", (1.5, 8): "L2", (1.5, 9): "L2", (1.5, 10): "L2", (1.5, 11): "L2",
    (2.5, 1): "L3", (2.5, 2): "L3", (2.5, 3): "L3", (2.5, 4): "L3", (2.5, 5): "L3", (2.5, 6): "L3",
    (2.5, 7): "L3", (2.5, 8): "L2", (2.5, 9): "L2", (2.5, 10): "L2", (2.5, 11): "L2",
    (3.5, 1): "L3", (3.5, 2): "L3", (3.5, 3): "L3", (3.5, 4): "L3", (3.5, 5): "L3", (3.5, 6): "L3",
    (3.5, 7): "L2", (3.5, 8): "L2", (3.5, 9): "L2",
    (4.5, 1): "L3", (4.5, 2): "L3", (4.5, 3): "L3", (4.5, 4): "L2", (4.5, 5): "L2",
    (5.5, 1): "L3", (5.5, 2): "L3",
}


def _extract_double_3d(ct_path, lower_manifold=None):
    """lower_manifold=None (default): Ka1-only, as before (dominant line regardless of L2/L3).
    lower_manifold='L3' or 'L2': restrict to lines whose lower level is in that manifold, giving
    the Ka1 or Ka2 line specifically (used for detuning_eV_L2_split)."""
    transitions = parse_ct(ct_path)
    purity_lookup = {k: v[1] for k, v in _3D3D_UPPER_GROUP.items()}
    groups = {}
    for t in transitions:
        u = _3D3D_UPPER_GROUP.get((t["j1"], t["lev1"]))
        if u is None:
            continue
        if lower_manifold is not None:
            manifold = _3D3D_LOWER_MANIFOLD.get((t["j2"], t["lev2"]))
            if manifold != lower_manifold:
                continue
        group, _purity = u
        groups.setdefault(group, []).append(t)
    return {g: dominant_line(lines, purity_by_level=purity_lookup) for g, lines in groups.items()}


def double_3d_l2_splits(ct_path):
    """detuning_eV_L2_split (Ka1 - Ka2) for each of the 3 double-3d-spectator channels."""
    ka1 = _extract_double_3d(ct_path, lower_manifold="L3")
    ka2 = _extract_double_3d(ct_path, lower_manifold="L2")
    return {g: ka1[g]["E_eV"] - ka2[g]["E_eV"] for g in ka1 if g in ka2}


def run(grasp_dir="/Users/parkinpham/Programming/grasp_cu_run"):
    """Reproduce every dominant-line pick. Requires the external GRASP working directory (see
    grasp_tools.py) -- not runnable standalone from just this repo.
    """
    print("=== 3d+/3d- single-spectator ===")
    for key, t in sorted(_extract_single_spectator(
        f"{grasp_dir}/k_3dsat2.l23_3dsat2.ct", _3D_UPPER_CHAR, _3D_LOWER_CHAR
    ).items()):
        print(f"  {key}: E={t['E_eV']:.3f} eV  A_C={t['A_coulomb']:.3e}")

    print("=== 3p+/3p- single-spectator ===")
    for key, t in sorted(_extract_single_spectator(
        f"{grasp_dir}/k_3psat.l23_3psat.ct", _3P_UPPER_CHAR, _3P_LOWER_CHAR
    ).items()):
        print(f"  {key}: E={t['E_eV']:.3f} eV  A_C={t['A_coulomb']:.3e}")

    print("=== double-3d-spectator (Ka1-side only, lower confidence) ===")
    for key, t in sorted(_extract_double_3d(f"{grasp_dir}/k_3d3dsat.l23_3d3dsat.ct").items()):
        print(f"  {key}: E={t['E_eV']:.3f} eV  A_C={t['A_coulomb']:.3e}")

    print("=== double-3d-spectator detuning_eV_L2_split ===")
    for key, split in sorted(double_3d_l2_splits(f"{grasp_dir}/k_3d3dsat.l23_3d3dsat.ct").items()):
        print(f"  {key}: {split:.4f} eV")


if __name__ == "__main__":
    run()
