"""
Electron-impact ionisation (EII) parameters for the free-electron slowing-down ladder
(docs/theory-eii-and-free-electrons.md, Part VII; docs/eii-free-electrons-implementation-plan.md).

Pure-Python formulas, evaluated once in `XLO_sim.__init__` from the `eii:` config block:

- `bcf_cross_section_nm2`: Burgess-Chidichimo EII cross section in the continuum-lowering form of van
  den Berg et al., PRL 120, 055002 (2018) (Part VII Eq. VII.1), with R0 = I'/I = 1 by default.
- `stopping_power_eV_nm`: Joy-Luo modified Bethe stopping power of solid Cu.
- `build_ladder`: log-spaced energy groups, the continuous-slowing-down transfer rate out of each
  group (Eq. VII.3), and per-group EII rates n*sigma*v at one free electron per atom (Eq. VII.2).
  With `anchor_birth_energies` every birth energy is a group edge, so each source's electrons start at
  the top of their own group.
- `build_fixed_levels`: the no-slowing-down bound: one level per birth energy, no transfer.
- `secondary_matrix`: where the secondary (delta) electron of each ionisation is born, from the
  binary-encounter spectrum dsigma/dW ~ 1/(W + I)^2 on 0 <= W <= (E - I)/2.

The model these build is explained in docs/theory-eii-electron-ladder-explained.md.
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

# Birth energies (eV) of each electron source at 8048 eV (XATOM, Part VII sec 1): 2p photoelectrons
# 7076-7096 (2s: 6949), M-shell photoelectrons 7.9-8.0 keV, KLL Auger 6.9-7.2 keV, L-MM Auger
# 750-960, Coster-Kronig 12-106. With the default ladder (E_top 7100) photo_M lands in the top group
# with photo, as before it had its own entry.
DEFAULT_BIRTH_ENERGIES_EV = {'photo': 7090.0, 'photo_M': 7950.0, 'KLL': 7100.0, 'LMM': 870.0, 'CK': 60.0}
MERGE_LOG_TOL = 0.02  # birth energies within 2% of each other share an edge / a fixed level


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


def _merged_levels(energies_eV, tol=MERGE_LOG_TOL):
    """Energies sorted high to low, dropping any within `tol` (log) of a higher one already kept."""
    kept = []
    for E in sorted(energies_eV, reverse=True):
        if not kept or np.log(kept[-1] / E) > tol:
            kept.append(float(E))
    return kept


def ladder_edges(E_top_eV, E_bottom_eV, n_groups, anchors_eV=()):
    """Group edges, high to low, log-spaced from E_top to E_bottom. Every anchor strictly inside (and not
    within MERGE_LOG_TOL of another edge) becomes an edge itself, and the n_groups groups are shared
    out between the segments between anchors so that the widest group (in log E) is as narrow as
    possible, at least one group per segment."""
    inside = [E for E in _merged_levels(anchors_eV)
              if np.log(E_top_eV / E) > MERGE_LOG_TOL and np.log(E / E_bottom_eV) > MERGE_LOG_TOL]
    points = np.array([E_top_eV] + inside + [E_bottom_eV], dtype=float)
    spans = np.log(points[:-1] / points[1:])
    if n_groups < len(spans):
        raise ValueError(f"eii.n_groups = {n_groups} is fewer than the {len(spans)} segments between "
                         f"the anchored birth energies {inside}")
    alloc = np.ones(len(spans), dtype=int)
    for _ in range(n_groups - len(spans)):
        alloc[np.argmax(spans / alloc)] += 1
    edges = [points[0]]
    for i, m in enumerate(alloc):
        edges.extend(np.geomspace(points[i], points[i + 1], m + 1)[1:])
    return np.array(edges)


def _group_rates(n_atoms_nm3, E_eV, subshells):
    return {name: n_atoms_nm3 * bcf_cross_section_nm2(E_eV, I_eV, zeta, n, l) * speed_nm_fs(E_eV)
            for name, (I_eV, zeta, n, l) in subshells.items()}


def build_ladder(n_atoms_nm3, n_groups=6, E_top_eV=7100.0, E_bottom_eV=30.0, subshells=None,
                 birth_energies_eV=None, stopping=None, anchor_birth_energies=False):
    """Discrete slowing-down ladder (Part VII section 4).

    Groups g = 0..n_groups-1 cover [E_bottom, E_top] (group 0 highest); a final bin g = n_groups
    collects electrons below E_bottom (too slow to ionise anything once E_bottom is the lowest
    threshold; no EII, no further transfer), so the sum over all n_groups+1 bins is exactly the number
    of free electrons ever produced.

    anchor_birth_energies=False: plain log spacing, and a source is born into the group that contains
    its birth energy (the original ladder). True: every birth energy is an edge (ladder_edges), and a
    source is born into the group whose top edge is nearest its birth energy, so its electrons start
    at the top of that group rather than at an energy up to one group width too high.

    Returns a dict with
      E_edges (n_groups+1,), E_centres (n_groups,), k_down_fs (n_groups,) -- transfer rate g -> g+1,
      rates_fs {subshell: (n_groups,)} -- n sigma v at one free electron per atom,
      birth_group {name: g} for each entry of birth_energies_eV, subshells, fixed_energy (False).
    """
    subshells = dict(DEFAULT_SUBSHELLS if subshells is None else subshells)
    stopping = {} if stopping is None else dict(stopping)
    birth = dict(DEFAULT_BIRTH_ENERGIES_EV)
    if birth_energies_eV:
        birth.update(birth_energies_eV)

    if anchor_birth_energies:
        E_edges = ladder_edges(E_top_eV, E_bottom_eV, n_groups, birth.values())
    else:
        E_edges = np.geomspace(E_top_eV, E_bottom_eV, n_groups + 1)
    E_centres = np.sqrt(E_edges[:-1] * E_edges[1:])
    dE = E_edges[:-1] - E_edges[1:]
    k_down = stopping_power_eV_nm(E_centres, **stopping) * speed_nm_fs(E_centres) / dE

    birth_group = {}
    for name, E in birth.items():
        if E >= E_edges[0]:
            g = 0
        elif E < E_edges[-1]:
            g = n_groups
        elif anchor_birth_energies:
            g = int(np.argmin(np.abs(np.log(E_edges[:-1] / E))))
        else:
            g = int(np.searchsorted(-E_edges, -E, side='right') - 1)
        birth_group[name] = g

    return {'E_edges': E_edges, 'E_centres': E_centres, 'k_down_fs': k_down,
            'rates_fs': _group_rates(n_atoms_nm3, E_centres, subshells), 'birth_group': birth_group,
            'subshells': subshells, 'fixed_energy': False}


def build_fixed_levels(n_atoms_nm3, E_bottom_eV=16.5, subshells=None, birth_energies_eV=None):
    """Fixed-energy electrons, the no-slowing-down bound: one level per distinct birth energy (energies
    within MERGE_LOG_TOL share a level, at the higher one), k_down = 0, so every electron keeps its
    birth energy and keeps ionising for the rest of the window. Sources born below E_bottom go
    straight into the final bin. Same dict as build_ladder; E_edges are geometric midpoints between
    the levels, for plotting only. The yield per electron then grows with the window instead of
    being fixed by the electron's energy (docs/theory-eii-electron-ladder-explained.md sec 6)."""
    subshells = dict(DEFAULT_SUBSHELLS if subshells is None else subshells)
    birth = dict(DEFAULT_BIRTH_ENERGIES_EV)
    if birth_energies_eV:
        birth.update(birth_energies_eV)
    E_centres = np.array(_merged_levels([E for E in birth.values() if E >= E_bottom_eV]))
    G = len(E_centres)
    top = E_centres[0] * (np.sqrt(E_centres[0] / E_centres[1]) if G > 1 else 1.05)
    E_edges = np.concatenate([[top], np.sqrt(E_centres[:-1] * E_centres[1:]), [E_bottom_eV]])
    birth_group = {name: (G if E < E_bottom_eV else int(np.argmin(np.abs(np.log(E_centres / E)))))
                   for name, E in birth.items()}
    return {'E_edges': E_edges, 'E_centres': E_centres, 'k_down_fs': np.zeros(G),
            'rates_fs': _group_rates(n_atoms_nm3, E_centres, subshells), 'birth_group': birth_group,
            'subshells': subshells, 'fixed_energy': True}


def secondary_fractions(E_eV, I_eV, W_edges_eV):
    """Fractions of the secondary electrons of one ionisation (primary energy E_eV, binding I_eV) born
    in each energy interval [W_edges[k+1], W_edges[k]] (edges high to low, last edge 0 for the
    sub-threshold bin). Binary-encounter (Rutherford-like) spectrum dsigma/dW ~ 1/(W + I)^2 for the
    secondary energy 0 <= W <= W_max = (E - I)/2, the slower of the two outgoing electrons by
    convention. Its median is ~I and its mean ~I (ln(W_max/I) - 1): a 7 keV primary gives 3d
    secondaries of ~70 eV on average, half of them below the 3d threshold."""
    W_max = 0.5 * (E_eV - I_eV)
    if W_max <= 0.0:
        return np.zeros(len(W_edges_eV) - 1)

    def cdf(W):
        W = np.clip(W, 0.0, W_max)
        return (1.0 / I_eV - 1.0 / (W + I_eV)) / (1.0 / I_eV - 1.0 / (W_max + I_eV))

    W_edges = np.asarray(W_edges_eV, dtype=float)
    return cdf(W_edges[:-1]) - cdf(W_edges[1:])


def secondary_matrix(ladder):
    """C[d, g] (fs^-1 per electron per atom, per target atom): rate at which one electron per atom in
    group g sets secondaries into group d (the last row is the bin below E_bottom), summed over every
    subshell it ionises -- valence and M shell included, whether or not the model tracks that
    ionisation's effect on the atom: the delta electron exists either way. Evaluated at the group
    centres. The primary's own energy loss (and so the secondary's energy) is already in the stopping
    power, so the secondaries add electrons, not energy."""
    E_c = ladder['E_centres']
    W_edges = np.concatenate([ladder['E_edges'], [0.0]])   # groups 0..G-1, then the bin [0, E_bottom]
    G = len(E_c)
    C = np.zeros((G + 1, G))
    for name, (I_eV, zeta, n, l) in ladder['subshells'].items():
        r = ladder['rates_fs'][name]
        for g in range(G):
            if r[g] > 0.0:
                C[:, g] += r[g] * secondary_fractions(E_c[g], I_eV, W_edges)
    return C


def ladder_yields(ladder, birth_group=0):
    """Ionisations per primary born into `birth_group` over its whole slowing-down (sum over the
    groups it passes through of rate/k_down). Used as a smoke test against Part VII section 5
    (7.09 keV primary: 2p3/2 0.109, 2p1/2 0.053, 2s 0.035, 3p 4.7, 3d 63)."""
    k = ladder['k_down_fs']
    return {name: float(np.sum(r[birth_group:] / k[birth_group:])) for name, r in ladder['rates_fs'].items()}
