"""
Electron-impact ionisation (EII) by the free electrons, as a discrete slowing-down ladder.

The electrons that photoionisation and Auger decay release (keV energies) slow down in the metal and
ionise more atoms on the way. `build_ladder` splits their energy range into groups, from which the
electrons move down at the continuous-slowing-down rate (Joy-Luo stopping power), and gives the EII rate
of every target subshell per electron per atom in each group. `secondary_matrix` sends the secondary
electron of each ionisation into the group of its energy.

Units: nm, fs, eV; cross sections in nm^2, rates in fs^-1.
"""

import numpy as np

BOHR_NM = 0.0529177210903
RYDBERG_EV = 13.605693122994
C_NM_FS = 299.792458
ME_C2_EV = 510998.95

# Neutral-Cu subshells as seen by a fast projectile: (binding energy eV, occupation, n, l). Binding
# energies from XATOM with relativistic corrections; occupations for metallic 3d10 4s1. 3p and 3d are
# summed over j.
SUBSHELLS = {
    '2p3/2': (951.6, 4, 2, 1),
    '2p1/2': (972.2, 2, 2, 1),
    '2s': (1098.7, 2, 2, 0),
    '3s': (128.9, 2, 3, 0),
    '3p': (86.3, 6, 3, 1),
    '3d': (16.5, 10, 3, 2),
}

# Birth energies (eV) of each electron source at 8048 eV photon energy (XATOM): 2p photoelectrons,
# M-shell photoelectrons, KLL Auger, L-MM Auger and Coster-Kronig electrons.
BIRTH_ENERGIES_EV = {'photo': 7090.0, 'photo_M': 7950.0, 'KLL': 7100.0, 'LMM': 870.0, 'CK': 60.0}
MERGE_LOG_TOL = 0.02  # birth energies within 2% of each other share a group edge

# Burgess-Chidichimo fit constants c1..c9
BCF_C = (1.0633, 0.6895, -0.4284, 1.6925, -1.7140, 2.2244, -0.5020, 0.5961, -0.1629)

# Bote, Salvat, Jablonski & Powell, At. Data Nucl. Data Tables 95, 871 (2009): fits to DWBA (overvoltage
# U = E/edge <= 16) and PWBA (U > 16) inner-shell EII cross sections. Cu row: per subshell the edge (eV),
# A[5] (U <= 16), G[4], Anlj and Be. The 3d is a valence shell in Cu and has no entry.
BOTE_SALVAT_CU = {
    '2s': (1093.70, (0.0237, 0.000123, -0.0273, 0.0163, 0.00688), (0.0494, 8.23, -0.173, 0.0823), 4.59e-7, 1.02),
    '2p1/2': (966.028, (0.0307, 0.000197, -0.037, 0.06, -0.0988), (0.0401, 7.26, -0.151, 0.0813), 8.76e-7, 0.959),
    '2p3/2': (944.886, (0.0446, 0.000283, -0.0537, 0.0872, -0.144), (0.0361, 7.25, -0.0956, 0.0668), 1.8e-6, 0.959),
    '3s': (127.925, (0.149, 0.000706, -0.181, -0.197, 1.08), (0.0219, 11.1, -0.102, 0.0657), 2.02e-6, 1.69),
    '3p1/2': (87.0366, (0.231, 0.000892, -0.352, 0.387, -0.24), (0.0245, 12.5, -0.124, 0.0793), 2.66e-6, 1.34),
    '3p3/2': (84.3061, (0.339, 0.0013, -0.526, 0.609, -0.464), (0.0235, 12.4, -0.119, 0.0771), 5.67e-6, 1.33),
}
# model subshell -> Bote-Salvat subshells summed
BOTE_SALVAT_MAP = {'2p3/2': ('2p3/2',), '2p1/2': ('2p1/2',), '2s': ('2s',), '3s': ('3s',), '3p': ('3p1/2', '3p3/2')}


def bcf_cross_section_nm2(E_eV, I_eV, zeta, n, l):
    """Burgess-Chidichimo EII cross section (nm^2) of a subshell with binding energy I_eV and occupation
    zeta, for a projectile of kinetic energy E_eV. Used for the 3d, which Bote-Salvat does not cover."""
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = BCF_C
    E = np.asarray(E_eV, dtype=float)
    x = np.clip(I_eV / E, 0.0, 1.0)
    bracket = ((c1 + (c2 + c3 * l) / n) * np.log(np.maximum(E / I_eV, 1.0))
               + (c4 + (c5 + c6 * l) / n) * (1.0 - x)
               + (c7 + (c8 + c9 * l) / n) * (1.0 - x) ** 2)
    sigma = np.pi * BOHR_NM ** 2 / (I_eV / RYDBERG_EV) ** 2 * zeta * I_eV / E * bracket
    return np.where(E > I_eV, sigma, 0.0)


def bote_salvat_cross_section_nm2(E_eV, subshell):
    """Bote-Salvat EII cross section (nm^2) of one model subshell of Cu (a key of BOTE_SALVAT_MAP)."""
    E = np.atleast_1d(np.asarray(E_eV, dtype=float))
    out = np.zeros_like(E)
    rev = 5.10998918e5
    for name in BOTE_SALVAT_MAP[subshell]:
        edge, A, G, Anlj, Be = BOTE_SALVAT_CU[name]
        U = E / edge
        lo = (U > 1.0) & (U <= 16.0)
        opu = 1.0 / (1.0 + U[lo])
        f = A[0] + A[1] * U[lo] + opu * (A[2] + opu ** 2 * (A[3] + opu ** 2 * A[4]))
        out[lo] += (U[lo] - 1.0) * (f / U[lo]) ** 2
        hi = U > 16.0
        Eh = E[hi]
        beta2 = Eh * (Eh + 2.0 * rev) / (Eh + rev) ** 2
        x = np.sqrt(Eh * (Eh + 2.0 * rev)) / rev
        fu = (2.0 * np.log(x) - beta2) * (1.0 + G[0] / x) + G[1] + G[2] * np.sqrt(rev / (Eh + rev)) + G[3] / x
        out[hi] += Anlj / beta2 * U[hi] / (U[hi] + Be) * fu
    out *= 4.0 * np.pi * BOHR_NM ** 2
    return out if np.ndim(E_eV) else float(out[0])


def cross_section_nm2(E_eV, name):
    """EII cross section of one model subshell: Bote-Salvat where it exists, Burgess-Chidichimo for the 3d."""
    if name in BOTE_SALVAT_MAP:
        return bote_salvat_cross_section_nm2(E_eV, name)
    return bcf_cross_section_nm2(E_eV, *SUBSHELLS[name])


def speed_nm_fs(E_eV):
    """Relativistic electron speed (nm/fs) at kinetic energy E_eV."""
    gamma = 1.0 + np.asarray(E_eV, dtype=float) / ME_C2_EV
    return C_NM_FS * np.sqrt(1.0 - 1.0 / gamma ** 2)


def stopping_power_eV_nm(E_eV, rho_g_cm3=8.96, Z=29, A=63.546, J_eV=322.0, k=0.83):
    """Joy-Luo modified Bethe stopping power (eV/nm) of solid Cu:
    S = 78500 rho Z / (A E) ln(1.166 (E + k J) / J) keV/cm with E in keV; the k J term keeps it finite
    below the mean excitation energy J."""
    E_keV = np.asarray(E_eV, dtype=float) / 1e3
    J_keV = J_eV / 1e3
    S_keV_cm = 78500.0 * rho_g_cm3 * Z / (A * E_keV) * np.log(1.166 * (E_keV + k * J_keV) / J_keV)
    return S_keV_cm * 1e3 / 1e7


def _merged_levels(energies_eV, tol=MERGE_LOG_TOL):
    """Energies sorted high to low, dropping any within `tol` (in log E) of a higher one already kept."""
    kept = []
    for E in sorted(energies_eV, reverse=True):
        if not kept or np.log(kept[-1] / E) > tol:
            kept.append(float(E))
    return kept


def ladder_edges(E_top_eV, E_bottom_eV, n_groups, anchors_eV=()):
    """Group edges, high to low, log-spaced from E_top to E_bottom. Every anchor inside the range becomes
    an edge, and the groups are shared out between the segments so that the widest group (in log E) is
    as narrow as possible, with at least one group per segment."""
    inside = [E for E in _merged_levels(anchors_eV)
              if np.log(E_top_eV / E) > MERGE_LOG_TOL and np.log(E / E_bottom_eV) > MERGE_LOG_TOL]
    points = np.array([E_top_eV] + inside + [E_bottom_eV], dtype=float)
    spans = np.log(points[:-1] / points[1:])
    if n_groups < len(spans):
        raise ValueError(f"eii.n_groups = {n_groups} is fewer than the {len(spans)} segments between "
                         f"the birth energies {inside}")
    alloc = np.ones(len(spans), dtype=int)
    for _ in range(n_groups - len(spans)):
        alloc[np.argmax(spans / alloc)] += 1
    edges = [points[0]]
    for i, m in enumerate(alloc):
        edges.extend(np.geomspace(points[i], points[i + 1], m + 1)[1:])
    return np.array(edges)


def build_ladder(n_atoms_nm3, n_groups=24, E_top_eV=7950.0, E_bottom_eV=16.5):
    """
    The slowing-down ladder. Groups g = 0..n_groups-1 cover [E_bottom, E_top], group 0 highest; every
    birth energy is a group edge, and each source is born into the group whose top edge it sits on. A
    final bin g = n_groups collects the electrons below E_bottom, which neither ionise nor move, so the
    sum over all n_groups + 1 bins counts every free electron ever produced.

    Returns a dict with
      E_edges (n_groups + 1,), E_centres (n_groups,)
      k_down_fs (n_groups,)             transfer rate g -> g + 1
      rates_fs {subshell: (n_groups,)}  n sigma v per target atom at one free electron per atom
      birth_group {source: g}
    """
    E_edges = ladder_edges(E_top_eV, E_bottom_eV, n_groups, BIRTH_ENERGIES_EV.values())
    E_centres = np.sqrt(E_edges[:-1] * E_edges[1:])
    dE = E_edges[:-1] - E_edges[1:]
    k_down = stopping_power_eV_nm(E_centres) * speed_nm_fs(E_centres) / dE

    birth_group = {}
    for name, E in BIRTH_ENERGIES_EV.items():
        if E >= E_edges[0]:
            birth_group[name] = 0
        elif E < E_edges[-1]:
            birth_group[name] = n_groups
        else:
            birth_group[name] = int(np.argmin(np.abs(np.log(E_edges[:-1] / E))))

    rates = {name: n_atoms_nm3 * cross_section_nm2(E_centres, name) * speed_nm_fs(E_centres) for name in SUBSHELLS}
    return {'E_edges': E_edges, 'E_centres': E_centres, 'k_down_fs': k_down, 'rates_fs': rates,
            'birth_group': birth_group}


def secondary_fractions(E_eV, I_eV, W_edges_eV):
    """Fractions of the secondary electrons of one ionisation (primary energy E_eV, binding I_eV) born in
    each interval [W_edges[k+1], W_edges[k]] (edges high to low, last edge 0). Binary-encounter spectrum
    dsigma/dW ~ 1/(W + I)^2 for the slower outgoing electron, 0 <= W <= (E - I)/2."""
    W_max = 0.5 * (E_eV - I_eV)
    if W_max <= 0.0:
        return np.zeros(len(W_edges_eV) - 1)

    def cdf(W):
        W = np.clip(W, 0.0, W_max)
        return (1.0 / I_eV - 1.0 / (W + I_eV)) / (1.0 / I_eV - 1.0 / (W_max + I_eV))

    W_edges = np.asarray(W_edges_eV, dtype=float)
    return cdf(W_edges[:-1]) - cdf(W_edges[1:])


def secondary_matrix(ladder):
    """C[d, g] (fs^-1): rate at which one electron per atom in group g sets secondaries into group d (the
    last row is the bin below E_bottom), summed over every subshell it ionises, whether or not the model
    tracks that ionisation's effect on the atom. The primary's energy loss, and so the secondary's
    energy, is already in the stopping power: secondaries add electrons, not energy."""
    E_c = ladder['E_centres']
    W_edges = np.concatenate([ladder['E_edges'], [0.0]])
    G = len(E_c)
    C = np.zeros((G + 1, G))
    for name, (I_eV, zeta, n, l) in SUBSHELLS.items():
        r = ladder['rates_fs'][name]
        for g in range(G):
            if r[g] > 0.0:
                C[:, g] += r[g] * secondary_fractions(E_c[g], I_eV, W_edges)
    return C
