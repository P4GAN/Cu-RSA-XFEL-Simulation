import numpy as np
from numba import njit
from . import tools


@njit(cache=True, fastmath=True)
def _MB_nlevel_regular_core(rho_ijxy, Omega_plus_sxy, Omega_minus_sxy, Tijs_plus, Tijs_minus,
                             Mij, Gamma_sp_Gij, S_ion_Fif, feed_diag_ixy, Delta_ij,
                             J_Omega_minus_xy, J_Omega_plus_xy,):
    nlevel = rho_ijxy.shape[0]
    s_dim = Tijs_plus.shape[2]
    nx = rho_ijxy.shape[2]
    ny = rho_ijxy.shape[3]

    Hint = np.zeros((nlevel, nlevel, nx, ny), dtype=np.complex128)
    for i in range(nlevel):
        for j in range(nlevel):
            for x in range(nx):
                for y in range(ny):
                    acc = 0.0j
                    for s in range(s_dim):
                        acc += Tijs_plus[i, j, s] * Omega_plus_sxy[s, x, y] + Tijs_minus[i, j, s] * Omega_minus_sxy[s, x, y]
                    Hint[i, j, x, y] = 1j * acc

    # Per-level spontaneous-feed (Gamma_sp_Gij, self-referential to this block's own diagonal) and
    # ionization rate, shared by every (i, j) pair below. External population feed (ground pump,
    # 2s-Auger, spectator photoionization, ...) is precomputed by the caller into feed_diag_ixy,
    # since it depends on *other* states that are frozen for this RK4 evaluation.
    diag_feed = np.zeros((nlevel, nx, ny), dtype=np.complex128)
    gamma_ion = np.zeros((nlevel, nx, ny), dtype=np.complex128)
    for i in range(nlevel):
        for x in range(nx):
            for y in range(ny):
                diag_sum = 0.0j
                for s in range(nlevel):
                    diag_sum += Gamma_sp_Gij[i, s] * rho_ijxy[s, s, x, y]

                diag_feed[i, x, y] = diag_sum + feed_diag_ixy[i, x, y]

                gamma_ion[i, x, y] = S_ion_Fif[0, i] * J_Omega_minus_xy[x, y] + \
                                     S_ion_Fif[1, i] * J_Omega_plus_xy[x, y]

    drho = np.zeros((nlevel, nlevel, nx, ny), dtype=np.complex128)
    for i in range(nlevel):
        for j in range(nlevel):
            for x in range(nx):
                for y in range(ny):
                    comm = 0.0j
                    for s in range(nlevel):
                        comm += Hint[i, s, x, y] * rho_ijxy[s, j, x, y] - rho_ijxy[i, s, x, y] * Hint[s, j, x, y]

                    val = comm - Mij[i, j] * rho_ijxy[i, j, x, y]
                    if i == j:
                        val += diag_feed[i, x, y]
                    else:
                        val += -1j * Delta_ij[i, j] * rho_ijxy[i, j, x, y]
                    val += -0.5 * (gamma_ion[i, x, y] + gamma_ion[j, x, y]) * rho_ijxy[i, j, x, y]
                    drho[i, j, x, y] = val

    return drho


def MB_nlevel_regular(t, rho_ijxy, params):
    """
    Calculate the regular part of the Maxwell-Bloch equations for the density matrix. This includes stimulated emission, pumping from the ground state with the pump and seed fields, and decay rates due to spontaneous emission and photoionization of ionic states with the pump and seed fields. If run_mode is 'consecutive', the pump field and ground state population are pre-configured, so the ionization of the seed field does not affect the ground state population, and ionization of the ionic states does not affect the pump field. If "spontaneous_only" is True, the stimulated emission contribution is neglected.

    Parameters
    ----------
    t
    rho_ijxy: np.ndarray
        Density matrix at given t,z
    params: list
        List containing the XLO_sim object, seed field Rabi frequency at given t,z, t index, z index, ground state population, pump flux, seed flux for the -1 polarization, seed flux for the +1 polarization (all at given t,z)

    Returns
    -------
    np.ndarray

    """    
    
    X, Omega_psxy, rho_ground_xy, rho_2s_xy, J_Omega_minus_xy, J_Omega_plus_xy  = params

    Omega_plus_sxy = Omega_psxy[0, :, :, :]
    Omega_minus_sxy = Omega_psxy[1, :, :, :]

    feed_diag_ixy = feed_diag_base_block(X, rho_ground_xy, rho_2s_xy, J_Omega_minus_xy, J_Omega_plus_xy)

    return _MB_nlevel_regular_core(
        rho_ijxy, Omega_plus_sxy, Omega_minus_sxy, X.Tijs_plus, X.Tijs_minus,
        X.Mij, X.Gamma_sp_Gij, X.S_ion_Fi[:, :], feed_diag_ixy, X.Delta_ij,
        J_Omega_minus_xy, J_Omega_plus_xy,
    )


def feed_diag_base_block(X, rho_ground_xy, rho_2s_xy, J_Omega_minus_xy, J_Omega_plus_xy):
    """
    External population feed into the base 6-level block's diagonal: ground-state photoionization
    (pump + seed fields, Eq. 14) plus 2s-hole Auger feeding (Eq. M2). Reproduces exactly what used
    to be computed inline inside `_MB_nlevel_regular_core` before the satellite-block refactor
    (docs/theory-and-2s-satellite-pathways.md).

    Parameters
    ----------
    X
        XLO_sim object
    rho_ground_xy: np.ndarray
        Ground state population at given t,z
    rho_2s_xy: np.ndarray
        2s hole level population at given t,z
    J_Omega_minus_xy, J_Omega_plus_xy: np.ndarray
        Seed field photon fluxes at given t,z

    Returns
    -------
    np.ndarray

    """

    S_ground_Fim = X.S_ground_Fi[0, :X.nlevel]
    S_ground_Fip = X.S_ground_Fi[1, :X.nlevel]
    auger_diag = np.diag(X.auger_feeding_matrix)

    feed = np.einsum('i,xy->ixy', S_ground_Fim, J_Omega_minus_xy * rho_ground_xy)
    feed += np.einsum('i,xy->ixy', S_ground_Fip, J_Omega_plus_xy * rho_ground_xy)
    feed += np.einsum('i,xy->ixy', auger_diag, rho_2s_xy)

    return feed


def feed_diag_satellite_block(X, chan, rho_2s_xy, rho_base_ijxy, rho_sat_ijxy, J_Omega_minus_xy, J_Omega_plus_xy):
    """
    External population feed into one 2s-hole satellite channel's local block diagonal
    (docs/theory-and-2s-satellite-pathways.md, Part II): 2s-hole Auger decay spread evenly over the
    4 lower-manifold (2p+X) msublevels (Eq. S2), plus sublevel-preserving spectator photoionization
    of the base block's 2p-hole diagonal into the same lower manifold (Eq. S3), plus
    sublevel-preserving spectator photoionization of the base block's 1s-hole diagonal into the
    2 upper-manifold (1sX) msublevels (Eq. S4), plus a direct K-hole (1s) non-radiative "KLM-type"
    Auger feed into the same lower manifold, sourced from the base block's own bare K-hole
    population rather than rho_2s_xy (Gamma_A_K_fs; a second, independent production route
    alongside Eq. S2's 2s-hole route -- see the inline comment at its call site below. Defaults to
    0/no-op, since no config currently supplies Gamma_A_K_eV).

    Double-satellite channels (docs/double-spectator-satellite-implementation-plan.md, chan.feed_from
    non-empty) instead feed exclusively from one or more *parent* satellite channels' own lower
    (L_k, local indices 0-3) manifold: a fraction of that parent's Gamma_L_eV decay -- previously
    100% generic loss -- is redirected here (the physical mechanism: the parent's spectator hole
    itself Auger-decays a second time, landing on this double-spectator configuration, while the
    parent's 2p+ core hole survives as a bystander -- see the implementation plan doc for why this
    dominates over any cross-section-driven route). Gamma_A_fs/S_feed_2p/S_feed_1s are all 0 for
    these channels (XLO_sim.py defaults), so the Eq. S2/S3/S4 terms below contribute nothing and
    this is the entire feed.

    When chan carries the 2p1/2-satellite extension (X.use_L2_satellite_pathway, chan.Mij sized
    X.satellite_nlevel = X.nlevel_base + 2), also feeds the extra 2p1/2+X msublevels (local indices
    X.nlevel_base, X.nlevel_base+1): 2s-hole Auger decay via the channel's OWN Gamma_A_L2_fs (a
    different, independently-tabulated XATOM Auger line from the Lk feed's Gamma_A_fs -- see
    xatom_tools.auger_partial_rate_L2_eV), spread evenly over the 2 msublevels, plus
    sublevel-preserving spectator photoionization of the *base block's own 2p1/2-hole* diagonal
    (rho_base_ijxy[X.nlevel_base + offset, ...], which only exists/is nonzero when
    X.use_L2_pathway is on -- guaranteed here since use_L2_satellite_pathway requires it, checked
    at construction time in XLO_sim.py).

    Parameters
    ----------
    X
        XLO_sim object
    chan
        This channel's parameter holder (XLO_sim.py::satellite_channel_params entry)
    rho_2s_xy: np.ndarray
        2s hole level population at given t,z
    rho_base_ijxy: np.ndarray
        Base block's density matrix at given t,z (sized X.nlevel, i.e. X.nlevel_base + 2 when
        X.use_L2_pathway is on -- always true here whenever chan carries the L2k extension)
    rho_sat_ijxy: list of np.ndarray
        Every satellite channel's own local block density matrix at given t,z, pre-update, in the
        same order as X.satellite_channel_params -- only read from when chan.feed_from is
        non-empty (double-satellite channels), to pull the parent channel(s)' own lower (L_k)
        manifold population. Unused (may be an empty list) for every pre-existing channel.
    J_Omega_minus_xy, J_Omega_plus_xy: np.ndarray
        Seed field photon fluxes at given t,z

    Returns
    -------
    np.ndarray

    """

    # The satellite block's own local level count is chan.Mij.shape[0] -- X.nlevel_base (6) for
    # the Kalpha1-satellite-only case, X.satellite_nlevel (8) when this channel also carries the
    # 2p1/2-satellite extension -- NOT rho_base_ijxy.shape[0], which is the *base* block's level
    # count and independently grows to 8 when use_L2_pathway extends it. Local indices
    # 0..X.nlevel_base-1 have the same L3/K meaning in both blocks (any extension is appended
    # after, at indices >= X.nlevel_base), so reading rho_base_ijxy[i, i] for i < X.nlevel_base is
    # correct regardless of whether the base block was itself extended.
    nlevel_sat = chan.Mij.shape[0]
    nlevel_base = X.nlevel_base
    nx, ny = rho_2s_xy.shape
    feed = np.zeros((nlevel_sat, nx, ny), dtype=complex)

    ei_L3_sat = X.ei_L3[:nlevel_base]
    ei_K_sat_local = X.ei_K[:nlevel_base]
    auger_weight = ei_L3_sat / np.sum(ei_L3_sat)
    auger_weight_K = ei_K_sat_local / np.sum(ei_K_sat_local)
    feed[:nlevel_base] += np.einsum('i,xy->ixy', auger_weight * chan.Gamma_A_fs, rho_2s_xy)

    # Direct K-hole (1s) non-radiative Auger feed ("KLM"-type: the 1s hole is filled by a 2p
    # electron while an M-shell electron is ejected instead), landing directly on this channel's
    # own Lk manifold with no 2s-hole intermediate needed -- a second, independent production
    # route alongside the 2s-Auger feed just above, sourced from the BASE block's own bare K-hole
    # (1s) population instead of rho_2s_xy. This population is already leaving the base block's
    # K-hole diagonal via its own total decay width (X.Mij[K,K]/Gamma_sp_Gij, unchanged by this
    # term) -- Gamma_A_K_fs only gives an explicit destination to part of that pre-existing,
    # previously-untracked non-radiative loss, exactly the same bookkeeping role Gamma_A_2s_eV
    # already plays for the 2s-hole's own total decay width. Spread evenly via the same
    # auger_weight convention as the 2s-Auger feed -- a simplification (the true angular
    # branching should mirror the Kalpha1/Gij radiative branching, since it's the same "which 2p
    # electron fills the 1s vacancy" physics with the ejected particle swapped for an electron
    # instead of a photon; revisit if this rate turns out to be non-negligible once real XATOM
    # numbers are available). Gamma_A_K_fs defaults to 0 (XLO_sim.py), so this is an exact no-op
    # unless a config explicitly supplies Gamma_A_K_eV.
    rho_K_base_xy = sum(np.real(rho_base_ijxy[i, i]) for i in range(nlevel_base) if ei_K_sat_local[i] > 0)
    feed[:nlevel_base] += np.einsum('i,xy->ixy', auger_weight * chan.Gamma_A_K_fs, rho_K_base_xy)

    # Double-satellite feed (docs/double-spectator-satellite-implementation-plan.md section 3):
    # redirected fraction of one or more parent channels' own Gamma_L_eV (manifold='lower') or
    # Gamma_K_eV (manifold='upper') decay, spread evenly over this channel's own corresponding 4
    # lower-manifold or 2 upper-manifold msublevels -- same auger_weight convention as the Eq. S2
    # term just above (no known angular dependence to do otherwise; the parent's spectator hole
    # decaying doesn't carry information about which of the parent's own already-populated
    # msublevels maps to which of this channel's). 'lower': parent's own L_k (2p+X) population
    # feeds this channel's L_k (2p+XX) manifold -- e.g. 2p+3p+'s spectator hole decaying while its
    # 2p+ core hole survives lands on 2p+3d+3d+. 'upper': parent's own U_k (1sX) population feeds
    # this channel's U_k (1sXX) manifold -- the analogous mechanism, one core-hole species over
    # (1s3p+'s spectator hole decaying while its 1s core hole survives lands on 1s3d+3d+); smaller
    # branching than 'lower' since the 1s core hole's own decay is a faster competing channel, but
    # still a real, sizable (>50%) fraction of Gamma_K_eV, confirmed via direct XATOM -decay calls
    # on 1s1_3pX,Y (see implementation plan doc). 'L2' is the same mechanism one tier deeper again
    # (docs/double-spectator-satellite-implementation-plan.md section 9): a parent channel's own
    # L2k (2p1/2+X, local indices nlevel_base:nlevel_sat) population feeds this channel's own L2k
    # (2p1/2+XX) manifold -- only present/nonzero when use_L2_satellite_pathway is on, in which
    # case both parent and child blocks share the same nlevel_sat (X.satellite_nlevel), so the L2k
    # index range is identical in both. No-op (chan.feed_from == []) for every pre-existing,
    # cross-section-fed channel.
    n_L2 = nlevel_sat - nlevel_base
    for parent_index, Gamma_feed_fs, manifold in chan.feed_from:
        parent_rho = rho_sat_ijxy[parent_index]
        if manifold == 'L2':
            parent_pop_xy = sum(np.real(parent_rho[i, i]) for i in range(nlevel_base, nlevel_base + n_L2))
            feed[nlevel_base:nlevel_base + n_L2] += np.einsum(
                'i,xy->ixy', np.ones(n_L2) / n_L2, Gamma_feed_fs * parent_pop_xy)
            continue
        src_mask, dst_weight = (ei_L3_sat, auger_weight) if manifold == 'lower' else (ei_K_sat_local, auger_weight_K)
        parent_pop_xy = sum(np.real(parent_rho[i, i]) for i in range(nlevel_base) if src_mask[i] > 0)
        feed[:nlevel_base] += np.einsum('i,xy->ixy', dst_weight, Gamma_feed_fs * parent_pop_xy)

    rate_2p_xy = chan.S_feed_2p[0] * J_Omega_minus_xy + chan.S_feed_2p[1] * J_Omega_plus_xy
    rate_1s_xy = chan.S_feed_1s[0] * J_Omega_minus_xy + chan.S_feed_1s[1] * J_Omega_plus_xy

    for i in range(nlevel_base):
        if ei_L3_sat[i] > 0:
            feed[i] += rate_2p_xy * rho_base_ijxy[i, i]
        else:
            feed[i] += rate_1s_xy * rho_base_ijxy[i, i]

    if nlevel_sat > nlevel_base:
        n_L2 = nlevel_sat - nlevel_base
        feed[nlevel_base:] += np.einsum(
            'i,xy->ixy', (chan.Gamma_A_L2_fs / n_L2) * np.ones(n_L2), rho_2s_xy)

        # L2k-manifold analogue of the K-hole KLM feed above (1s hole filled by a 2p1/2 electron
        # instead of 2p3/2): same source population (rho_K_base_xy), same even-spread convention,
        # different destination manifold and its own independent rate Gamma_A_K_to_L2_fs (defaults
        # to 0 -- no-op -- exactly like Gamma_A_L2_fs's own role for the 2s-Auger feed).
        feed[nlevel_base:] += np.einsum(
            'i,xy->ixy', (chan.Gamma_A_K_to_L2_fs / n_L2) * np.ones(n_L2), rho_K_base_xy)

        rate_2p1_xy = chan.S_feed_2p1[0] * J_Omega_minus_xy + chan.S_feed_2p1[1] * J_Omega_plus_xy
        for offset in range(n_L2):
            # Base-global index of the base block's own 2p1/2 sublevel this local L2k sublevel
            # draws from -- same offset past nlevel_base in both blocks by construction (each
            # block's own L2-like manifold is always appended immediately after its own
            # nlevel_base-sized corner), not a coincidence of nlevel_base's specific value.
            i_base = nlevel_base + offset
            feed[nlevel_base + offset] += rate_2p1_xy * rho_base_ijxy[i_base, i_base]

    return feed


def MB_satellite_block_regular(t, rho_ijxy, params):
    """
    Calculate the regular part of the Maxwell-Bloch equations for one 2s-hole satellite channel's
    local block (docs/theory-and-2s-satellite-pathways.md, Part II) -- 6 levels (2p3/2+X<->1sX), or
    8 when this channel also carries the 2p1/2-satellite extension (X.use_L2_satellite_pathway).
    Structurally identical to `MB_nlevel_regular`, reusing the base block's Tijs and, by default,
    Mij/Gamma_sp_Gij, but fed by `feed_diag_satellite_block` instead of ground-state pumping, and
    detuned from the Kalpha1 rotating frame by the channel's own per-level Delta_ij.

    Parameters
    ----------
    t
    rho_ijxy: np.ndarray
        This channel's local density matrix at given t,z
    params: list
        List containing the XLO_sim object, the channel's parameter holder, seed field Rabi
        frequency at given t,z, the base block's density matrix at given t,z, 2s hole level
        population, every satellite channel's own local block density matrix at given t,z
        (pre-update, only read from for double-satellite channels -- see
        feed_diag_satellite_block), seed flux for the -1 and +1 polarizations (all at given t,z).
        NOTE: pump flux is not included here (see XLO_sim.py's satellite_channel_params
        construction) -- only the seed/Kalpha1-field-driven part of Eq. S3/S4 is applied so far.

    Returns
    -------
    np.ndarray

    """

    X, chan, Omega_psxy, rho_base_ijxy, rho_2s_xy, rho_sat_ijxy, J_Omega_minus_xy, J_Omega_plus_xy = params

    Omega_plus_sxy = Omega_psxy[0, :, :, :]
    Omega_minus_sxy = Omega_psxy[1, :, :, :]

    feed_diag_ixy = feed_diag_satellite_block(X, chan, rho_2s_xy, rho_base_ijxy, rho_sat_ijxy, J_Omega_minus_xy, J_Omega_plus_xy)

    return _MB_nlevel_regular_core(
        rho_ijxy, Omega_plus_sxy, Omega_minus_sxy, X.Tijs_plus_satellite, X.Tijs_minus_satellite,
        chan.Mij, chan.Gamma_sp_Gij, chan.S_ion_Fi[:, :],
        feed_diag_ixy, chan.Delta_ij,
        J_Omega_minus_xy, J_Omega_plus_xy,
    )


def MB_other_regular(t, rho_other_xy, params):
    """
    Calculate the change of population of the additional ionic level due to pumping from the ground state and photoionization with the pump and seed fields.
    Parameters
    ----------
    t
    rho_other_xy: np.ndarray
        Population of the additional ionic level at given t,z
    params: list
        List containing the XLO_sim object, seed field Rabi frequency at given t,z, t index, z index, ground state population, pump flux, seed flux for the -1 polarization, seed flux for the +1 polarization (all at given t,z)

    Returns
    -------
    np.ndarray

    """
      
    X, rho_ground_xy, J_Omega_minus_xy, J_Omega_plus_xy = params
    
    drho_pump = np.einsum('f, fxy->xy', X.S_ground_Fi[:, -1], np.array([J_Omega_minus_xy, J_Omega_plus_xy])) * rho_ground_xy
    drho_ion = -np.einsum('f, fxy->xy', X.S_other_F[:], np.array([J_Omega_minus_xy, J_Omega_plus_xy])) * rho_other_xy

    return drho_pump + drho_ion


def MB_2s_regular(t, rho_2s_xy, params):
    """
    Calculate the change of population of the 2s hole level due to pumping from the ground state, photoionization with the pump and seed fields and decay.
    Parameters
    ----------
    t
    rho_2s_xy: np.ndarray
        Population of the 2s hole level at given t,z
    params: list
        List containing the XLO_sim object, seed field Rabi frequency at given t,z, t index, z index, ground state population, pump flux, seed flux for the -1 polarization, seed flux for the +1 polarization (all at given t,z)

    Returns
    -------
    np.ndarray

    """
      
    X, rho_ground_xy, J_Omega_minus_xy, J_Omega_plus_xy = params
    
    drho_pump = np.einsum('f, fxy->xy', X.S_ground_Fi[:, -2], np.array([J_Omega_minus_xy, J_Omega_plus_xy])) * rho_ground_xy
    drho_ion = -np.einsum('f, fxy->xy', X.S_2s_F[:], np.array([J_Omega_minus_xy, J_Omega_plus_xy])) * rho_2s_xy
    drho_2s_decay = -X.GammaL1fsm1N * rho_2s_xy

    return drho_pump + drho_ion + drho_2s_decay




def MB_ground_regular(t, rho_ground_xy, params):
    """
    Calculate the change of ground state population due to photoionization with the pump and seed fields.

    Parameters
    ----------
    t
    rho_ground_xy: np.ndarray
        Ground state population at given t,z
    params: list
        List containing the XLO_sim object, pump flux, seed flux for the -1 polarization, seed flux for the +1 polarization (all at given t,z), t index and z index

    Returns
    -------
    np.ndarray

    """
    
    X, J_Omega_minus_xy, J_Omega_plus_xy = params

    drho = -np.einsum('fi, fxy->xy', X.S_ground_Fi[:, :], np.array([J_Omega_minus_xy, J_Omega_plus_xy])) * rho_ground_xy

    return drho




def Omega_source_regular(X, rho_ijxy, Tijs_plus=None, Tijs_minus=None):
    """
    Calculate the classical contribution to ASE field at a given point of space and time (t,z).

    Parameters
    ----------
    rho_ijxy: np.ndarray
        Density matrix a given t,z
    Tijs_plus, Tijs_minus: np.ndarray or None
        Dipole tensors sized to match rho_ijxy's own local level count. Default (None) to X's base
        block tensors (X.Tijs_plus/X.Tijs_minus, sized X.nlevel) -- pass X.Tijs_plus_satellite/
        X.Tijs_minus_satellite (sized X.nlevel_base) when rho_ijxy is a satellite channel's local
        block instead, since that block stays nlevel_base-sized even when use_L2_pathway extends
        the base block to X.nlevel = nlevel_base + 2.
    params: list
        List containing the XLO_sim object, t index and z index

    Returns
    -------
    np.ndarray

    """
    if Tijs_plus is None:
        Tijs_plus = X.Tijs_plus
    if Tijs_minus is None:
        Tijs_minus = X.Tijs_minus

    rho_ijxy_hermitian = rho_ijxy + np.conj(np.swapaxes(rho_ijxy, 0, 1))

    Omega_plus_source = np.einsum('ijs, jixy-> sxy', Tijs_minus, rho_ijxy_hermitian)
    Omega_minus_source = np.einsum('jis, ijxy-> sxy', Tijs_plus, rho_ijxy_hermitian)

    return X.field_source_factor * np.einsum('p, psxy -> psxy', X.e_sign, np.asarray([Omega_plus_source, Omega_minus_source]))


def absorption(X, rho_ground_xyz, rho_other_xyz, rho_2s_xyz, rho_ijxyz, rho_sat_ijxyz_list=None):
    """
    Calculate the absorption coefficient of the pump or seed field due to photoionization of the ground and all ionic states, and the compound. The ground state population is not pre-configured.
    Parameters
    ----------
    rho_ground_xyz: np.ndarray
        Ground state population at given t,z
    rho_other_xyz: np.ndarray
        Additional ionic population at given t,z
    rho_2s_xyz: np.ndarray
        2s hole level population at given t,z
    rho_ijxyz: np.ndarray
        Density matrix at given t,z
    rho_sat_ijxyz_list: list of np.ndarray or None
        Density matrices of the 2s-hole satellite channels (docs/theory-and-2s-satellite-pathways.md,
        Part II) at given t,z, in the same order as `X.satellite_channel_params`. Their further-
        ionization loss cross sections (`chan.S_ion_Fi`, Eq. S8) default to zero, so this is a no-op
        until that data is supplied. Pass None (default) to skip entirely.

    Returns
    -------
    np.ndarray

    """

    kappa_Omega_sxyz = X.n * np.einsum('xyz, si->sxyz', rho_ground_xyz, X.S_ground_Fi[:, :]) + \
                        X.n * np.einsum('xyz, s->sxyz', rho_other_xyz, X.S_other_F[:]) + \
                        X.n * np.einsum('xyz, s->sxyz', rho_2s_xyz, X.S_2s_F[:]) + \
                        X.n * np.einsum('si, iixyz->sxyz', X.S_ion_Fi[:, :], rho_ijxyz) + \
                        X.n * X.sigma_compound_Ka1

    if rho_sat_ijxyz_list is not None:
        for chan, rho_sat_ijxyz in zip(X.satellite_channel_params, rho_sat_ijxyz_list):
            kappa_Omega_sxyz = kappa_Omega_sxyz + X.n * np.einsum('si, iixyz->sxyz', chan.S_ion_Fi[:, :], rho_sat_ijxyz)

    return np.array([kappa_Omega_sxyz, kappa_Omega_sxyz])
