"""
Parsing and analysis helpers for the GRASP2018 + RATIP-2012 Dirac-Fock cross-check pipeline.

This complements xatom/xatom_tools.py with an independent atomic-structure method: full
Multiconfiguration Dirac-Hartree-Fock (GRASP2018) for wavefunctions/energies, then RATIP-2012
(xrelci for CI + Breit/QED, xphoto for photoionization amplitudes) for the actual transition
rates and cross sections. Used to validate config/base/Cu-seed-original.yaml's atomic-physics
parameters (satellite detunings, sigma1/sigma2 cross sections) against a second, independent
method, and in several cases to catch/fix outright bugs in XATOM's numbers (e.g. the 3d-/3p-
satellite detunings, which XATOM had sign-wrong or ~2x too small).

The actual GRASP/RATIP calculations (CSL files, orbitals, mixing coefficients) are NOT part of
this repo -- they're run externally in a scratch working directory (see the module docstring
below, "External setup", for the recipe), since GRASP2018 and RATIP-2012 are large Fortran
codebases that must be built from source and the working directory accumulates hundreds of
binary intermediate files per calculation. This module only parses/analyzes their text output.

External setup
--------------
- GRASP2018: https://github.com/compas/grasp, built with
  `cmake -DCMAKE_Fortran_FLAGS="-fallow-argument-mismatch -fallow-invalid-boz" .` (modern gfortran
  needs this for legacy argument-mismatch patterns) then `make install`.
- RATIP-2012 (S. Fritzsche): built with plain `gfortran -c`. As shipped it has a real bug in
  `rabs_angular.f90`'s `wigner_9j_symbol` (and 3 similar spots) -- `present(x) .and. x` isn't
  guaranteed to short-circuit in Fortran, so it segfaults whenever a caller omits that optional
  argument, which every `xphoto` run does. Fixed by splitting into
  `if (present(x)) then; if (x) ...; end if` (or a precomputed local flag where the original had
  an else-branch). See the GRASP+RATIP pipeline memory note for the full diagnosis if this needs
  redoing on a fresh RATIP checkout.
- Typical recipe for one transition/cross-section: `rcsfgenerate` (CSL) -> `rangular` ->
  `rwfnestimate` -> `rmcdhf` (GRASP2018 SCF) -> `xrelci` (RATIP's own CI, with Breit + vacuum
  polarization) -> `xphoto` (photoionization) or `rbiotransform`+`rtransition` (GRASP2018,
  bound-bound transition rates for satellite detunings instead of cross sections).
- Two RATIP-specific gotchas worth knowing before rerunning any of this:
  (1) `xphoto`/`xrelci` refuse to overwrite their own output files if they already exist from a
      previous attempt ("as new" file-open error) -- always `rm -f` the exact output names first.
  (2) `xrelci`'s CSL reader is older than GRASP2018's and can't parse the `*` block-separator line
      `rcsfgenerate` inserts between symmetry blocks -- strip it first:
      `grep -v '^ \\*$' name.c > name_xrelci.c` and feed *that* to `xrelci` (same `.w` orbital file
      works for both).

Unit convention
---------------
Cross sections throughout this repo (config/base/*.yaml's `sigma*` keys, xatom_tools.py) are in
nm^2. RATIP's `xphoto` reports cross sections in atomic units (a0^2). The conversion is the
standard atomic-physics one: 1 a.u. of cross section = a0^2 = 28.00285 Mb (megabarn), and
1 Mb = 1e-4 nm^2 (matching xatom_tools.py's own Mb -> nm^2 convention).
"""

import re

AU_TO_MB = 28.00285  # a0^2 in megabarns -- standard atomic-physics cross-section unit
MB_TO_NM2 = 1e-4  # 1 Mb = 1e-22 m^2 = 1e-4 nm^2
KAYS_TO_EV = 1.239841984e-4  # wavenumber (cm^-1) -> eV, for rtransition's .ct output


def au_to_nm2(sigma_au):
    """Convert a RATIP xphoto cross section (atomic units, a0^2) to nm^2."""
    return sigma_au * AU_TO_MB * MB_TO_NM2


def nm2_to_au(sigma_nm2):
    """Inverse of au_to_nm2 -- nm^2 to atomic units."""
    return sigma_nm2 / (AU_TO_MB * MB_TO_NM2)


def parse_ct(path):
    """
    Parse a GRASP2018 `rtransition` .ct file (bound-bound E1 transition amplitudes/rates between
    two CSL/mixing-coefficient pairs, e.g. a satellite-channel detuning calculation).

    Returns a list of dicts, one per transition line, with keys:
        lev1, j1, p1   -- upper-state local level index (within its J/parity block), J, parity
        lev2, j2, p2   -- same for the lower state
        E_kays, E_eV   -- transition energy (E_eV = E_kays * KAYS_TO_EV)
        A_coulomb      -- Einstein A coefficient, Coulomb gauge (s^-1)

    J values are floats (half-integers represented as e.g. 1.5), so dict keys built from them
    compare correctly against both `1` and `1.0`-style int/float lookups.
    """
    with open(path) as f:
        text = f.read()
    pat = re.compile(
        r"f1\s+(\d+)\s+(\d+(?:/2)?)\s*([+-])\s+f2\s+(\d+)\s+(\d+(?:/2)?)\s*([+-])"
        r"\s+([\d.]+)\s+C\s+([\d.D+-]+)"
    )

    def jval(s):
        return 0.5 * int(s[:-2]) if s.endswith("/2") else float(int(s))

    transitions = []
    for m in pat.finditer(text):
        lev1, j1, p1, lev2, j2, p2, e_kays, a_c = m.groups()
        e_kays = float(e_kays)
        transitions.append(
            {
                "lev1": int(lev1), "j1": jval(j1), "p1": p1,
                "lev2": int(lev2), "j2": jval(j2), "p2": p2,
                "E_kays": e_kays, "E_eV": e_kays * KAYS_TO_EV,
                "A_coulomb": float(a_c.replace("D", "E")),
            }
        )
    return transitions


def dominant_line(transitions, purity_by_level=None):
    """
    Pick "the" dominant transition among a set of candidate lines (e.g. every line between two
    hole-type-labeled level groups), ranked by Einstein-A weighted by upper-level degeneracy
    (2J+1) -- and, if `purity_by_level` is given, further discounted by how CSF-pure the upper
    level is (see the satellite-detuning composition note below), so a heavily mixed level with
    a large raw A doesn't win over a cleaner one just because of mixing.

    `purity_by_level`: optional dict mapping (j1, lev1) -> purity in [0, 1]. Missing entries
    default to purity 1.0.

    Composition note: for satellite-channel calculations, a level's dominant physical character
    (e.g. "3d+ spectator" vs "3d- spectator") isn't always >95% pure -- the lower (core-hole)
    manifold in particular can mix down to ~50-70% for the higher-J levels. Rather than requiring
    a clean composition label before picking a line, rank ALL candidate lines by this weight and
    trust the transition-rate physics to pick out the right one -- validated against JAC and
    literature for the 3d+ Kalpha1-satellite (this method landed on the correct line even though
    the naive "biggest raw A" choice picked a different, wrong one).

    Returns the transition dict with the largest weight, or None if `transitions` is empty.
    """
    if not transitions:
        return None
    def weight(t):
        purity = 1.0
        if purity_by_level is not None:
            purity = purity_by_level.get((t["j1"], t["lev1"]), 1.0)
        return t["A_coulomb"] * (2 * t["j1"] + 1) * purity
    return max(transitions, key=weight)


def sum_initial_level_cross_sections(sumfile, initial_level=1, gauge="coulomb"):
    """
    Sum the individual (not RATIP's own running "Total" column, which accumulates across
    whatever else is in the same xphoto run -- e.g. every energy in an energy scan, or every
    initial level if the initial-state CSL has more than one) photoionization cross sections in
    an `xphoto` .sum file, restricted to lines starting from a given initial-state level index.

    Use this to build a "total further-ionization" cross section (this repo's sigma2_* convention)
    by running one xphoto call per possible second-hole final state and summing across calls, or
    within one call whose final-state CSL already spans several final levels.

    gauge: 'coulomb' or 'babushkin'.

    Returns (total_au, lines_used) where lines_used is a list of (raw_line, cross_section_au)
    for provenance/debugging.
    """
    col = 4 if gauge == "coulomb" else 3  # 1-indexed position of the CS column after e-Energy
    with open(sumfile) as f:
        text = f.read()
    section = text.split("Individual and total photoionization cross sections")[1]
    section = section.split("Photoionization angular parameters")[0]
    row_pat = re.compile(
        r"\s*(\d+)\s*-\s*\d+\s+[\d/]+\s*[+-]\s+[\d/]+\s*[+-]\s+"
        r"([\d.]+E[+-]\d+)\s+([\d.]+E[+-]\d+)\s+([\d.]+E[+-]\d+)\s+([\d.]+E[+-]\d+|NaN)"
    )
    total = 0.0
    lines_used = []
    for line in section.splitlines():
        m = row_pat.match(line)
        if not m or int(m.group(1)) != initial_level:
            continue
        cs_str = m.group(1 + col)
        if cs_str == "NaN":
            continue  # numerically degenerate/ill-conditioned channel at this energy; skip
        cs = float(cs_str)
        total += cs
        lines_used.append((line.strip(), cs))
    return total, lines_used
