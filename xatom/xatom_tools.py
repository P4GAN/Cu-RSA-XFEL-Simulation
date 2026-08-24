"""
Tools for calling XATOM and extracting the atomic-structure parameters needed for the Cu Kalpha1
2s-hole satellite pathways in docs/theory-and-2s-satellite-pathways.md (Part II, section 13's
parameter inventory: detunings, Auger branching, spectator photoionization cross sections, widths,
and further-ionization loss cross sections).

Generalizes the ad hoc `run_xatom(...)` + hand-read-off-the-printout workflow in
../../v_XATOM_current/calculating_parameters.ipynb, and the regex-based total-energy-difference
approach in ../../v_XATOM_current/calculating_energies.ipynb, into reusable parsing functions plus
one top-level `satellite_channel_parameters(...)` that returns a dict matching the
`satellite_channels` YAML schema in config/base/Cu-seed-SASE.yaml directly.

XATOM's "nl<n_->,<n_+>" hole-count notation, confirmed against calculating_energies.ipynb's
paired probe calls (e.g. "2p0,1" vs "2p1,0" both tried, "1s1_3d0,1"/"2p0,1_3d0,1" used together):
"3d0,1" = 0 holes in 3d- (j=3/2) + 1 hole in 3d+ (j=5/2) -> a hole specifically in 3d+, and
"3d1,0" = a hole specifically in 3d-. This module always uses the explicit "0,1"/"1,0" single-hole
form, never the ambiguous "3d0,1"-with-larger-counts forms the notebook was still probing.

Pump-driven terms (theory doc's sigma_P * J_P in Eq. S3/S4) are intentionally NOT computed here:
there is currently no per-(t,z) pump photon flux threaded through Model.py's RK4 functions for any
block (base or satellite) to multiply such a cross section by, so a pump-only cross section would
be dead data. Only the seed/Kalpha1-field-driven terms XLO_sim.py actually consumes are produced.
"""

import os
import re
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache

XATOM_PATH = os.environ.get('XATOM_PATH', '/Users/parkinpham/Programming/xraypac/xatom/src')
HARTREE_TO_EV = 27.211386245988  # CODATA 2018

# Spectator-shell hole-config fragments for the four satellite channels (theory doc, S10/S12, plus
# 3p- which the doc's S12.1(a) originally left out "for now" -- structurally identical, just a
# fourth channel with the same 6-level-block treatment):
# k=1 -> 3d+ (3d_5/2), k=2 -> 3d- (3d_3/2), k=3 -> 3p+ (3p_3/2), k=4 -> 3p- (3p_1/2).
SPECTATOR_HOLE = {
    '3d+': '3d0,1',
    '3d-': '3d1,0',
    '3p+': '3p0,1',
    '3p-': '3p1,0',
}
BASE_UPPER_HOLE = '1s1'    # bare 1s hole (K), no spectator
BASE_LOWER_HOLE = '2p0,1'  # bare 2p+ hole (L3), no spectator


# ---------------------------------------------------------------------------
# Running XATOM
# ---------------------------------------------------------------------------

def run_xatom(hole_config='', photon_energy=None, decay=False, pcs=False, extra_args=()):
    """
    Run XATOM for neutral/ionized Cu and return its raw stdout text.

    Parameters
    ----------
    hole_config: str
        XATOM hole-configuration string, e.g. "2s1", "1s1_3d0,1". Empty string (default) runs
        the neutral ground-state atom (no -hole flag passed at all, matching what
        calculating_parameters.ipynb's `run_xatom("", ...)` calls relied on).
    photon_energy: float or None
        Photon energy in eV for the -PE flag (needed for -pcs output that reports absorption at a
        specific energy; not needed for -decay alone, since fluorescence/Auger rates and total
        decay widths are intrinsic to the ion, independent of any incident photon). None omits -PE.
    decay: bool
        Add -decay (fluorescence + Auger rates + total decay width/lifetime).
    pcs: bool
        Add -pcs (photoabsorption cross sections).
    extra_args: sequence of str
        Any additional raw CLI arguments, appended last.

    Returns
    -------
    str
        Raw stdout of the xatom run.
    """

    args = [f'{XATOM_PATH}/xatom', '-s', 'Cu', '-relativity']
    if hole_config:
        args += ['-hole', hole_config]
    if photon_energy is not None:
        args += ['-PE', str(photon_energy)]
    if decay:
        args += ['-decay']
    if pcs:
        args += ['-pcs']
    args += list(extra_args)

    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"xatom exited with code {result.returncode} for hole_config={hole_config!r}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout


@lru_cache(maxsize=None)
def run_xatom_cached(hole_config='', photon_energy=None, decay=False, pcs=False):
    """
    Cached wrapper around `run_xatom` (positional/keyword args only, all hashable) -- XATOM runs
    take real wall-clock time (seconds to tens of seconds, see calculating_parameters.ipynb's
    timing footers), and several of the convenience functions below need the *same* run
    (identical hole_config/photon_energy/flags) for more than one derived quantity, or share a
    parent configuration across multiple satellite channels (e.g. the "2s1" Auger table feeds all
    four channels' Gamma_A). Does not accept `extra_args` (not hashable as given) -- use
    `run_xatom` directly if needed.
    """
    return run_xatom(hole_config, photon_energy=photon_energy, decay=decay, pcs=pcs)


# ---------------------------------------------------------------------------
# Parsed result containers
# ---------------------------------------------------------------------------

@dataclass
class OrbitalEnergy:
    n_occ: int
    E_eV: float
    E0_eV: float
    Em_eV: float
    Ed_eV: float
    Eso_eV: float


@dataclass
class FluorescenceLine:
    initial: str
    final: str
    xray_E_eV: float
    rate_au: float
    R_int: float


@dataclass
class FluorescenceResult:
    lines: list = field(default_factory=list)
    total_rate_au: float = None
    lifetime_fs: float = None

    def find(self, initial, final):
        """Return the single line matching (initial, final), or None."""
        for line in self.lines:
            if line.initial == initial and line.final == final:
                return line
        return None


@dataclass
class AugerLine:
    initial: str
    final1: str
    final2: str
    KE_eV: float
    rate_au: float

    @property
    def rate_eV(self):
        return self.rate_au * HARTREE_TO_EV


@dataclass
class AugerResult:
    lines: list = field(default_factory=list)
    per_hole_rate_au: dict = field(default_factory=dict)  # {initial_hole: subtotal rate, a.u.}
    total_A_rate_au: float = None
    A_lifetime: tuple = None  # (value, unit)
    total_decay_rate_au: float = None
    total_decay_lifetime: tuple = None  # (value, unit)
    total_decay_width_eV: float = None

    def find(self, initial, final1, final2):
        """
        Return the single line matching `initial -> final1 + final2`, trying both orderings of
        the two final-state labels (XATOM's own ordering within a row is not alphabetical, e.g.
        "2s0 - 2p+ 3d+" but also rows like "2s0 - 3d- 3d+" appear -- match either order rather
        than assuming one).
        """
        for line in self.lines:
            if line.initial != initial:
                continue
            if (line.final1, line.final2) in ((final1, final2), (final2, final1)):
                return line
        return None


@dataclass
class PhotoabsorptionResult:
    photon_energy_eV: float
    per_subshell: dict = field(default_factory=dict)  # {subshell: (cs_Mb, cs_au)}
    total_cs_Mb: float = None
    total_cs_au: float = None


_FLOAT = r'[-+0-9.EeDd]+'  # XATOM output uses Fortran-style E-notation exponents throughout


def _to_float(token):
    return float(token.replace('D', 'E').replace('d', 'e'))


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_total_energy_rel(output):
    """
    Extract the fully relativistic total energy E(1) (eV) from the '|E_TOT_REL|  E(0)  E(1)'
    line. This is the *first-order* relativistic total energy -- distinct from the plain
    'Total Energy = ... [... eV]' line above it, which is non-relativistic and cannot
    distinguish e.g. 2p1/2 from 2p3/2 hole configurations (matches
    calculating_energies.ipynb::parse_total_energy_rel, verified against that notebook's probe
    calls).
    """
    match = re.search(r'\|E_TOT_REL\|\s+(' + _FLOAT + r')\s+(' + _FLOAT + r')', output)
    if match is None:
        raise ValueError('E_TOT_REL not found in XATOM output')
    return _to_float(match.group(2))


def parse_orbital_energies_rel(output):
    """
    Parse the 'Orbital energies with relativistic correction:' table into
    {subshell_label: OrbitalEnergy}, e.g. {'2p+': OrbitalEnergy(n_occ=4, E_eV=-1007.62, ...), ...}.

    Matched directly against the whole output rather than a sliced-out section: every "blank"
    line in XATOM's printout is actually a bare '#' (not an empty line), which makes section
    boundaries fragile to depend on; the row pattern itself (label, integer n_occ, then exactly
    5 floats, no ':') is distinctive enough not to collide with the plain (non-relativistic)
    'Orbital energies:' table above it (only 4 numeric columns) or any other section (all of
    which have a ':' in each data row).
    """
    if 'Orbital energies with relativistic correction' not in output:
        raise ValueError("'Orbital energies with relativistic correction' section not found")

    row_re = re.compile(
        r'^\s*#?\s*(\S+)\s+(\d+)\s+(' + _FLOAT + r')\s+(' + _FLOAT + r')\s+(' + _FLOAT + r')'
        r'\s+(' + _FLOAT + r')\s+(' + _FLOAT + r')\s*$',
        re.MULTILINE,
    )
    orbitals = {}
    for m in row_re.finditer(output):
        label, n_occ, E, E0, Em, Ed, Eso = m.groups()
        orbitals[label] = OrbitalEnergy(
            n_occ=int(n_occ), E_eV=_to_float(E), E0_eV=_to_float(E0),
            Em_eV=_to_float(Em), Ed_eV=_to_float(Ed), Eso_eV=_to_float(Eso),
        )
    if not orbitals:
        raise ValueError("no rows parsed from 'Orbital energies with relativistic correction' section")
    return orbitals


def parse_fluorescence(output):
    """Parse the 'Fluorescence (based on orbital energies):' section into a FluorescenceResult."""
    section_match = re.search(
        r'Fluorescence \(based on orbital energies\):\s*\n(.*?)Total F rate\s*=\s*(' + _FLOAT + r')'
        r'\s*a\.u\.\s*\(\s*lifetime\s*=\s*([\d.]+)\s*(\w+)\s*\)',
        output, re.DOTALL,
    )
    if section_match is None:
        raise ValueError("'Fluorescence' section not found (was -decay passed to run_xatom?)")
    body, total_rate, lifetime_val, lifetime_unit = section_match.groups()

    row_re = re.compile(
        r'^\s*#?\s*(\S+)\s*-\s*(\S+)\s*:\s+(' + _FLOAT + r')\s+(' + _FLOAT + r')\s+(' + _FLOAT + r')\s*$',
        re.MULTILINE,
    )
    lines = [
        FluorescenceLine(initial=i, final=f, xray_E_eV=_to_float(e), rate_au=_to_float(r), R_int=_to_float(ri))
        for i, f, e, r, ri in row_re.findall(body)
    ]
    return FluorescenceResult(
        lines=lines, total_rate_au=_to_float(total_rate), lifetime_fs=float(lifetime_val)
    )


def parse_auger(output):
    """Parse the 'Auger decay (based on orbital energies):' section into an AugerResult."""
    body_match = re.search(
        r'Auger decay \(based on orbital energies\):\s*\n(.*?)Total decay rate\s*=',
        output, re.DOTALL,
    )
    if body_match is None:
        raise ValueError("'Auger decay' section not found (was -decay passed to run_xatom?)")
    body = body_match.group(1)

    # Matched independently against the full output rather than chained onto the body slice:
    # every "blank" separator line in XATOM's printout is a bare '#' (not whitespace), so a
    # single regex demanding \s* between 'Total decay rate' and 'Total decay width' silently
    # fails to match across that '#'.
    total_decay_match = re.search(
        r'Total decay rate\s*=\s*(' + _FLOAT + r')\s*a\.u\.\s*\(\s*lifetime\s*=\s*([\d.]+)\s*(\w+)\s*\)', output
    )
    total_width_match = re.search(r'Total decay width\s*=\s*(' + _FLOAT + r')\s*eV', output)
    if total_decay_match is None or total_width_match is None:
        raise ValueError("'Total decay rate'/'Total decay width' lines not found")
    total_decay_rate, decay_lifetime_val, decay_lifetime_unit = total_decay_match.groups()
    (total_decay_width,) = total_width_match.groups()

    row_re = re.compile(
        r'^\s*#?\s*(\S+)\s*-\s*(\S+)\s+(\S+)\s*:\s+(' + _FLOAT + r')\s+(' + _FLOAT + r')\s*$',
        re.MULTILINE,
    )
    lines = [
        AugerLine(initial=i, final1=f1, final2=f2, KE_eV=_to_float(ke), rate_au=_to_float(r))
        for i, f1, f2, ke, r in row_re.findall(body)
    ]

    subtotal_re = re.compile(r'^\s*#?\s*(\S+)\s+hole A rate\s*=\s*(' + _FLOAT + r')\s*$', re.MULTILINE)
    per_hole_rate_au = {label: _to_float(rate) for label, rate in subtotal_re.findall(body)}

    total_A_match = re.search(
        r'Total A rate\s*=\s*(' + _FLOAT + r')\s*a\.u\.\s*\(\s*lifetime\s*=\s*([\d.]+)\s*(\w+)\s*\)', body
    )
    if total_A_match is None:
        raise ValueError("'Total A rate' line not found within Auger section")
    total_A_rate, A_lifetime_val, A_lifetime_unit = total_A_match.groups()

    return AugerResult(
        lines=lines,
        per_hole_rate_au=per_hole_rate_au,
        total_A_rate_au=_to_float(total_A_rate),
        A_lifetime=(float(A_lifetime_val), A_lifetime_unit),
        total_decay_rate_au=_to_float(total_decay_rate),
        total_decay_lifetime=(float(decay_lifetime_val), decay_lifetime_unit),
        total_decay_width_eV=_to_float(total_decay_width),
    )


def parse_photoabsorption(output):
    """Parse the 'Photoabsorption cross section:' section into a PhotoabsorptionResult."""
    body_match = re.search(
        r'Photoabsorption cross section:\s*\n(.*?)Total P cs\s*=', output, re.DOTALL
    )
    if body_match is None:
        raise ValueError("'Photoabsorption cross section' section not found (was -pcs and -PE passed?)")
    body = body_match.group(1)

    # Matched independently rather than chained onto the body slice for the same reason as in
    # parse_auger: literal keywords ('photon energy', 'Total P cs') can be preceded on their own
    # line by a bare '#' rather than whitespace, so a single \s*-joined pattern across lines
    # would silently fail to match.
    energy_match = re.search(r'photon energy\s*=\s*' + _FLOAT + r'\s*a\.u\.\s*=\s*(' + _FLOAT + r')\s*eV', output)
    total_match = re.search(r'Total P cs\s*=\s*(' + _FLOAT + r')\s+(' + _FLOAT + r')', output)
    if energy_match is None or total_match is None:
        raise ValueError("'photon energy ='/'Total P cs =' lines not found")
    (photon_energy,) = energy_match.groups()
    total_Mb, total_au = total_match.groups()

    row_re = re.compile(r'^\s*#?\s*(\S+)\s*:\s+(' + _FLOAT + r')\s+(' + _FLOAT + r')\s*$', re.MULTILINE)
    per_subshell = {label: (_to_float(mb), _to_float(au)) for label, mb, au in row_re.findall(body)}

    return PhotoabsorptionResult(
        photon_energy_eV=_to_float(photon_energy), per_subshell=per_subshell,
        total_cs_Mb=_to_float(total_Mb), total_cs_au=_to_float(total_au),
    )


# ---------------------------------------------------------------------------
# Mid-level convenience: total-energy-difference transition energies
# (reproduces calculating_energies.ipynb's validated Delta-SCF + calibration-shift approach)
# ---------------------------------------------------------------------------

def transition_energy_eV(initial_hole_config, final_hole_config):
    """
    XATOM's own (uncalibrated) prediction for the emitted-photon energy of
    initial_hole_config -> final_hole_config, as a total-energy difference (Delta-SCF) of the
    fully relativistic total energies. E.g. transition_energy_eV("1s1", "2p0,1") is XATOM's own
    Kalpha1 energy prediction, systematically offset from the experimental 8047.91 eV by the
    exchange-correlation functional's error -- see `satellite_detuning_eV` for the calibrated
    version that cancels this systematic error.
    """
    E_initial = parse_total_energy_rel(run_xatom_cached(initial_hole_config))
    E_final = parse_total_energy_rel(run_xatom_cached(final_hole_config))
    return E_initial - E_final


def satellite_detuning_eV(spectator, reference_transition=(BASE_UPPER_HOLE, BASE_LOWER_HOLE)):
    """
    Detuning Delta_k (eV) for one satellite channel (theory doc section 13: "satellite transition
    energy minus Kalpha1 energy"), computed as XATOM's *shift* between the satellite transition
    (1sX -> 2p+X) and the bare Kalpha1 transition (1s -> 2p+), rather than XATOM's absolute
    transition energies -- this cancels the exchange-correlation functional's systematic error in
    the absolute total energy, which the shift between two very similar configurations should not
    share (same trick as calculating_energies.ipynb::calibrated_satellite_energy, but returning
    the shift/detuning directly rather than an absolute calibrated line energy).

    Parameters
    ----------
    spectator: str
        One of the SPECTATOR_HOLE keys ('3d+', '3d-', '3p+', '3p-'), or a raw XATOM hole-config
        fragment for the spectator shell (e.g. '3d0,1').
    reference_transition: (str, str)
        (initial, final) hole-config pair for the undetuned reference transition. Defaults to the
        bare Kalpha1 transition (1s1 -> 2p0,1).

    Returns
    -------
    float
        Delta_k in eV. Divide by hbar (as XLO_sim.py does for `detuning_eV`) to get the fs^-1
        angular-frequency value the satellite_channels YAML schema expects.
    """
    spectator_config = SPECTATOR_HOLE.get(spectator, spectator)
    satellite_initial = f'{BASE_UPPER_HOLE}_{spectator_config}'
    satellite_final = f'{BASE_LOWER_HOLE}_{spectator_config}'

    ref_initial, ref_final = reference_transition
    return (
        transition_energy_eV(satellite_initial, satellite_final)
        - transition_energy_eV(ref_initial, ref_final)
    )


# ---------------------------------------------------------------------------
# Mid-level convenience: widths/lifetimes, Auger branching, photoionization cross sections
# ---------------------------------------------------------------------------

def state_total_decay_width_eV(hole_config):
    """
    Total decay width (eV, = hbar / lifetime) of the ion in the given hole configuration -- an
    intrinsic property of the ion, independent of any incident photon energy, so no -PE/-pcs is
    needed. Used for the satellite channels' Gamma_L_eV (width of the 2p+X lower/L-like state)
    and Gamma_K_eV (width of the 1sX upper/K-like state) -- theory doc section 10's "Assumption
    (widths)" box defaults these to the base Gamma_L3/Gamma_K (spectator approximation); this
    computes the double-hole state's *own* width directly instead of assuming the spectator is a
    pure bystander.
    """
    return parse_auger(run_xatom_cached(hole_config, decay=True)).total_decay_width_eV


def auger_partial_rate_eV(spectator, parent_hole_config='2s1'):
    """
    Gamma_A^(2s->L_k) (eV): the single resolved 2s-hole Auger channel feeding the 2p+X_k
    double-hole final state, read directly off the parent's Auger table -- no statistical-weight
    guessing needed (contrast theory doc section 12.1(a)'s placeholder g=2j+1 split, which this
    supersedes when XATOM is available). E.g. for spectator='3d+', looks up the
    "2s0 - 2p+ 3d+ :" row.
    """
    spectator_config = SPECTATOR_HOLE.get(spectator, spectator)
    auger = parse_auger(run_xatom_cached(parent_hole_config, decay=True))

    # XATOM labels a hole created in the "0,1"/"1,0" slot with the resulting subshell's own +/-
    # label (confirmed against calculating_parameters.ipynb output, e.g. "3d0,1" hole -> "3d+"
    # row) -- so the spectator's *label* (not its hole-count fragment) is what appears in the
    # Auger table.
    spectator_label = spectator if spectator in SPECTATOR_HOLE else _label_from_hole_fragment(spectator_config)
    line = auger.find(initial='2s0', final1='2p+', final2=spectator_label)
    if line is None:
        raise ValueError(
            f"no Auger line '2s0 - 2p+ {spectator_label}' found for parent {parent_hole_config!r}; "
            f"available finals: {[(l.final1, l.final2) for l in auger.lines if l.initial == '2s0']}"
        )
    return line.rate_eV


def _label_from_hole_fragment(hole_fragment):
    """'3d0,1' -> '3d+', '3d1,0' -> '3d-' (inverse of SPECTATOR_HOLE's convention)."""
    m = re.match(r'(\d[a-z])(\d),(\d)', hole_fragment)
    if m is None:
        raise ValueError(f"unrecognized XATOM hole-config fragment {hole_fragment!r}")
    shell, n_minus, n_plus = m.groups()
    if n_plus == '1':
        return shell + '+'
    if n_minus == '1':
        return shell + '-'
    raise ValueError(f"cannot infer +/- label from hole fragment {hole_fragment!r}")


def spectator_ionization_cross_section_nm2(parent_hole_config, spectator, photon_energy_eV):
    """
    sigma^(parent->spectator-ionized)_F (nm^2) at the given photon energy F, i.e. the
    photoionization cross section of the spectator shell computed *from* a given single-hole
    parent ion -- this is sigma^(2p->Lk) when parent_hole_config=BASE_LOWER_HOLE ('2p0,1', theory
    doc Eq. S3), or sigma^(1s->Uk) when parent_hole_config=BASE_UPPER_HOLE ('1s1', Eq. S4). XATOM
    reports cross sections in Mb (1 Mb = 1e-22 m^2 = 1e-4 nm^2), matching this repo's config
    convention (see config/base/Cu-seed.yaml's sigma1_*/sigma2_* values, all in nm^2).
    """
    spectator_label = spectator if spectator in SPECTATOR_HOLE else _label_from_hole_fragment(spectator)
    pa = parse_photoabsorption(run_xatom_cached(parent_hole_config, photon_energy=photon_energy_eV, pcs=True))
    if spectator_label not in pa.per_subshell:
        raise ValueError(
            f"subshell {spectator_label!r} not found in photoabsorption table for "
            f"parent={parent_hole_config!r}, E={photon_energy_eV} eV; available: {sorted(pa.per_subshell)}"
        )
    cs_Mb, _cs_au = pa.per_subshell[spectator_label]
    return cs_Mb * 1e-4  # Mb -> nm^2


def total_photoionization_cross_section_nm2(hole_config, photon_energy_eV):
    """
    Total photoionization cross section (nm^2, "Total P cs" summed over every subshell) of the
    ion in the given hole configuration, at the given photon energy -- this is the further-
    ionization loss cross section for *that specific configuration itself* (theory doc section
    12.4: sigma^(Lk->.)/sigma^(Uk->.), triple-ionization of the double-hole states), used to
    populate S_ion_Fi for a satellite channel. Pass the double-hole config itself (e.g.
    "2p0,1_3d0,1" for the 2p+3d+ lower/L-like manifold, "1s1_3d0,1" for the 1s3d+ upper/K-like
    manifold), not the spectator fragment alone.
    """
    pa = parse_photoabsorption(run_xatom_cached(hole_config, photon_energy=photon_energy_eV, pcs=True))
    return pa.total_cs_Mb * 1e-4  # Mb -> nm^2


# ---------------------------------------------------------------------------
# Top-level: one satellite_channels YAML entry per channel
# ---------------------------------------------------------------------------

def satellite_channel_parameters(name, spectator, Ka1_energy_eV=8047.91):
    """
    Assemble one full entry for the `satellite_channels` list in config/base/*.yaml (theory doc
    section 13's parameter inventory), for a single channel identified by its spectator shell.

    Parameters
    ----------
    name: str
        Channel name, e.g. "3d+" (stored verbatim into the YAML entry's 'name' field).
    spectator: str
        One of '3d+', '3d-', '3p+', '3p-' (SPECTATOR_HOLE keys).
    Ka1_energy_eV: float
        Reference (experimental) Kalpha1 diagram-line energy (eV), used both as the photon energy
        for the seed-field-driven photoionization cross-section lookups and as the calibration
        anchor for the detuning (default 8047.91 eV, matching hwKalpha1N in config/base/*.yaml).

    Returns
    -------
    dict
        {'name', 'detuning_eV', 'Gamma_A_2s_eV', 'Gamma_L_eV', 'Gamma_K_eV',
         'sigma_Ka1_from_2p', 'sigma_Ka1_from_1s', 'sigma_ion_from_2p', 'sigma_ion_from_1s'} --
        directly usable as one list entry under `satellite_channels:` in the YAML.
    """

    spectator_config = SPECTATOR_HOLE[spectator]
    lower_hole = f'{BASE_LOWER_HOLE}_{spectator_config}'  # 2p+X_k (local L manifold)
    upper_hole = f'{BASE_UPPER_HOLE}_{spectator_config}'  # 1sX_k  (local K manifold)

    return {
        'name': name,
        'detuning_eV': satellite_detuning_eV(spectator),
        'Gamma_A_2s_eV': auger_partial_rate_eV(spectator),
        'Gamma_L_eV': state_total_decay_width_eV(lower_hole),
        'Gamma_K_eV': state_total_decay_width_eV(upper_hole),
        'sigma_Ka1_from_2p': spectator_ionization_cross_section_nm2(BASE_LOWER_HOLE, spectator, Ka1_energy_eV),
        'sigma_Ka1_from_1s': spectator_ionization_cross_section_nm2(BASE_UPPER_HOLE, spectator, Ka1_energy_eV),
        'sigma_ion_from_2p': total_photoionization_cross_section_nm2(lower_hole, Ka1_energy_eV),
        'sigma_ion_from_1s': total_photoionization_cross_section_nm2(upper_hole, Ka1_energy_eV),
    }


def build_all_satellite_channels(Ka1_energy_eV=8047.91):
    """
    All four channels' parameters (3d+, 3d-, 3p+, 3p-), ready to assign directly to
    `satellite_channels:` in the YAML config.
    """
    return [
        satellite_channel_parameters(spectator, spectator, Ka1_energy_eV)
        for spectator in ('3d+', '3d-', '3p+', '3p-')
    ]
