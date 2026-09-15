"""
Electron-impact ionisation (EII) parameters for the free-electron slowing-down ladder
(docs/theory-eii-and-free-electrons.md, Part VII; docs/eii-free-electrons-implementation-plan.md).

Pure-Python formulas, evaluated once in `XLO_sim.__init__` from the `eii:` config block:

- `bcf_cross_section_nm2`: Burgess-Chidichimo EII cross section in the continuum-lowering form of van
  den Berg et al., PRL 120, 055002 (2018) (Part VII Eq. VII.1), with R0 = I'/I = 1 by default.
- `stopping_power_eV_nm`: Joy-Luo modified Bethe stopping power of solid Cu.
- `build_ladder`: log-spaced energy groups, the continuous-slowing-down transfer rate out of each
  group (Eq. VII.3), and per-group EII rates n*sigma*v at one free electron per atom (Eq. VII.2).

Units match the rest of the code: nm, fs, eV; cross sections in nm^2, rates in fs^-1.
"""

import numpy as np

BOHR_NM = 0.0529177210903
RYDBERG_EV = 13.605693122994
C_NM_FS = 299.792458
ME_C2_EV = 510998.95

# {c1..c9}, material- and charge-independent fit constants of the BCF formula
BCF_C = (1.0633, 0.6895, -0.4284, 1.6925, -1.7140, 2.2244, -0.5020, 0.5961, -0.1629)

# Neutral-Cu subshells as seen by a fast projectile: (binding energy eV, occupation zeta, n, l).
# Binding energies from XATOM (`xatom -s Cu -relativity`, 'Orbital energies with relativistic
# correction'); occupations for metallic 3d10 4s1. 3p and 3d are summed over j (the cross section is
# linear in zeta, and the j-splitting of the thresholds is irrelevant at keV energies).
DEFAULT_SUBSHELLS = {
    '2p3/2': (951.6, 4, 2, 1),
    '2p1/2': (972.2, 2, 2, 1),
    '2s': (1098.7, 2, 2, 0),
    '3s': (128.9, 2, 3, 0),
    '3p': (86.3, 6, 3, 1),
    '3d': (16.5, 10, 3, 2),
}
L_SHELL = ('2p3/2', '2p1/2', '2s')
M_SHELL = ('3s', '3p', '3d')


def bcf_cross_section_nm2(E_eV, I_eV, zeta, n, l, R0=1.0, rn=1.0):
    """Burgess-Chidichimo EII cross section (nm^2) of one subshell for projectile energy E_eV.

    With R0 = 1 (no continuum lowering, the default) the r_n exponent has no effect. The initial
    charge enters only through I_eV and zeta of the actual ion. Reproduces the
    estimates-on-role-of-eii slide: 1/(n sigma v) at 7 keV and one electron per atom = 65 / 11 /
    2.3 / 0.21 fs for 2p3/2 / 3s / 3p / 3d (slide: 60 / 10 / 2 / 0.25 fs).
    """
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = BCF_C
    E = np.asarray(E_eV, dtype=float)
    Ip = R0 * I_eV
    x = np.clip(Ip / E, 0.0, 1.0)
    bracket = ((c1 + (c2 + c3 * l) / n) * np.log(np.maximum(E / Ip, 1.0))
               + R0 * (c4 + (c5 + c6 * l) / n) * (1.0 - x)
               + R0 ** rn * (c7 + (c8 + c9 * l) / n) * (1.0 - x) ** 2)
    sigma = np.pi * BOHR_NM ** 2 / (I_eV / RYDBERG_EV) ** 2 * zeta * Ip / (R0 ** 4 * E) * bracket
    return np.where(E > Ip, sigma, 0.0)


def speed_nm_fs(E_eV):
    """Relativistic electron speed (nm/fs) at kinetic energy E_eV."""
    gamma = 1.0 + np.asarray(E_eV, dtype=float) / ME_C2_EV
    return C_NM_FS * np.sqrt(1.0 - 1.0 / gamma ** 2)


def stopping_power_eV_nm(E_eV, rho_g_cm3=8.96, Z=29, A=63.546, J_eV=322.0, k=0.83):
    """Joy-Luo modified Bethe stopping power (eV/nm), defaults for solid Cu.

    S = 78500 rho Z / (A E) ln(1.166 (E + k J) / J) [keV/cm, E in keV]; the k J term keeps it finite
    below the mean excitation energy J. Gives 7 keV -> 1 keV in ~7.3 fs and a ~290 nm path for a
    7.1 keV electron in Cu (Part VII section 4).
    """
    E_keV = np.asarray(E_eV, dtype=float) / 1e3
    J_keV = J_eV / 1e3
    S_keV_cm = 78500.0 * rho_g_cm3 * Z / (A * E_keV) * np.log(1.166 * (E_keV + k * J_keV) / J_keV)
    return S_keV_cm * 1e3 / 1e7


def build_ladder(n_atoms_nm3, n_groups=6, E_top_eV=7100.0, E_bottom_eV=30.0, subshells=None,
                 birth_energies_eV=None, stopping=None):
    """Discrete slowing-down ladder (Part VII section 4).

    Groups g = 0..n_groups-1 cover [E_bottom, E_top] log-spaced (group 0 highest); a final bin
    g = n_groups collects electrons below E_bottom (thermalised; no EII, no further transfer), so
    the sum over all n_groups+1 bins is exactly the number of free electrons ever produced.

    Returns a dict with
      E_edges (n_groups+1,), E_centres (n_groups,), k_down_fs (n_groups,) -- transfer rate g -> g+1,
      rates_fs {subshell: (n_groups,)} -- n sigma v at one free electron per atom,
      birth_group {name: g} for each entry of birth_energies_eV.
    """
    subshells = dict(DEFAULT_SUBSHELLS if subshells is None else subshells)
    stopping = {} if stopping is None else dict(stopping)
    birth = {'photo': 7090.0, 'KLL': 7100.0, 'LMM': 870.0, 'CK': 60.0}
    if birth_energies_eV:
        birth.update(birth_energies_eV)

    E_edges = np.geomspace(E_top_eV, E_bottom_eV, n_groups + 1)
    E_centres = np.sqrt(E_edges[:-1] * E_edges[1:])
    dE = E_edges[:-1] - E_edges[1:]
    k_down = stopping_power_eV_nm(E_centres, **stopping) * speed_nm_fs(E_centres) / dE

    rates = {}
    for name, (I_eV, zeta, n, l) in subshells.items():
        rates[name] = n_atoms_nm3 * bcf_cross_section_nm2(E_centres, I_eV, zeta, n, l) * speed_nm_fs(E_centres)

    birth_group = {}
    for name, E in birth.items():
        if E >= E_edges[0]:
            g = 0
        elif E < E_edges[-1]:
            g = n_groups
        else:
            g = int(np.searchsorted(-E_edges, -E, side='right') - 1)
        birth_group[name] = g

    return {'E_edges': E_edges, 'E_centres': E_centres, 'k_down_fs': k_down, 'rates_fs': rates,
            'birth_group': birth_group}


def ladder_yields(ladder, birth_group=0):
    """Ionisations per primary born into `birth_group` over its whole slowing-down (sum over the
    groups it passes through of rate/k_down). Used as a smoke test against Part VII section 5
    (7.09 keV primary: 2p3/2 0.109, 2p1/2 0.053, 2s 0.035, 3p 4.7, 3d 63)."""
    k = ladder['k_down_fs']
    return {name: float(np.sum(r[birth_group:] / k[birth_group:])) for name, r in ladder['rates_fs'].items()}
