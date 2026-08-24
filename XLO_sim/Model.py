import numpy as np
from numba import njit
from . import tools


@njit(cache=True, fastmath=True)
def _MB_nlevel_regular_core(rho_ijxy, Omega_plus_sxy, Omega_minus_sxy, Tijs_plus, Tijs_minus,
                             Mij, Gamma_sp_Gij, S_ion_Fif, feed_diag_ixy, Delta, sign_ij,
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
                        val += -1j * Delta * sign_ij[i, j] * rho_ijxy[i, j, x, y]
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
        X.Mij, X.Gamma_sp_Gij, X.S_ion_Fi[:, :], feed_diag_ixy, 0.0, X.sign_ij_block,
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


def feed_diag_satellite_block(X, chan, rho_2s_xy, rho_base_ijxy, J_Omega_minus_xy, J_Omega_plus_xy):
    """
    External population feed into one 2s-hole satellite channel's local 6-level block diagonal
    (docs/theory-and-2s-satellite-pathways.md, Part II): 2s-hole Auger decay spread evenly over the
    4 lower-manifold (2p+X) msublevels (Eq. S2), plus sublevel-preserving spectator photoionization
    of the base block's 2p-hole diagonal into the same lower manifold (Eq. S3), plus
    sublevel-preserving spectator photoionization of the base block's 1s-hole diagonal into the
    2 upper-manifold (1sX) msublevels (Eq. S4).

    Parameters
    ----------
    X
        XLO_sim object
    chan
        This channel's parameter holder (XLO_sim.py::satellite_channel_params entry)
    rho_2s_xy: np.ndarray
        2s hole level population at given t,z
    rho_base_ijxy: np.ndarray
        Base 6-level block's density matrix at given t,z
    J_Omega_minus_xy, J_Omega_plus_xy: np.ndarray
        Seed field photon fluxes at given t,z

    Returns
    -------
    np.ndarray

    """

    nlevel = rho_base_ijxy.shape[0]
    nx, ny = rho_2s_xy.shape
    feed = np.zeros((nlevel, nx, ny), dtype=complex)

    auger_weight = X.ei_L3 / np.sum(X.ei_L3)
    feed += np.einsum('i,xy->ixy', auger_weight * chan.Gamma_A_fs, rho_2s_xy)

    rate_2p_xy = chan.S_feed_2p[0] * J_Omega_minus_xy + chan.S_feed_2p[1] * J_Omega_plus_xy
    rate_1s_xy = chan.S_feed_1s[0] * J_Omega_minus_xy + chan.S_feed_1s[1] * J_Omega_plus_xy

    for i in range(nlevel):
        if X.ei_L3[i] > 0:
            feed[i] += rate_2p_xy * rho_base_ijxy[i, i]
        else:
            feed[i] += rate_1s_xy * rho_base_ijxy[i, i]

    return feed


def MB_satellite_block_regular(t, rho_ijxy, params):
    """
    Calculate the regular part of the Maxwell-Bloch equations for one 2s-hole satellite channel's
    local 6-level block (docs/theory-and-2s-satellite-pathways.md, Part II). Structurally identical
    to `MB_nlevel_regular`, reusing the base block's Tijs and, by default, Mij/Gamma_sp_Gij, but fed
    by `feed_diag_satellite_block` instead of ground-state pumping, and detuned from the Kalpha1
    rotating frame by the channel's own Delta.

    Parameters
    ----------
    t
    rho_ijxy: np.ndarray
        This channel's local density matrix at given t,z
    params: list
        List containing the XLO_sim object, the channel's parameter holder, seed field Rabi
        frequency at given t,z, the base block's density matrix at given t,z, 2s hole level
        population, seed flux for the -1 and +1 polarizations (all at given t,z). NOTE: pump flux
        is not included here (see XLO_sim.py's satellite_channel_params construction) -- only the
        seed/Kalpha1-field-driven part of Eq. S3/S4 is applied so far.

    Returns
    -------
    np.ndarray

    """

    X, chan, Omega_psxy, rho_base_ijxy, rho_2s_xy, J_Omega_minus_xy, J_Omega_plus_xy = params

    Omega_plus_sxy = Omega_psxy[0, :, :, :]
    Omega_minus_sxy = Omega_psxy[1, :, :, :]

    feed_diag_ixy = feed_diag_satellite_block(X, chan, rho_2s_xy, rho_base_ijxy, J_Omega_minus_xy, J_Omega_plus_xy)

    return _MB_nlevel_regular_core(
        rho_ijxy, Omega_plus_sxy, Omega_minus_sxy, X.Tijs_plus, X.Tijs_minus,
        chan.Mij, chan.Gamma_sp_Gij, chan.S_ion_Fi[:, :],
        feed_diag_ixy, chan.Delta_fs, X.sign_ij_block,
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




def Omega_source_regular(X, rho_ijxy):
    """
    Calculate the classical contribution to ASE field at a given point of space and time (t,z).

    Parameters
    ----------
    rho_ijxy: np.ndarray
        Density matrix a given t,z
    params: list
        List containing the XLO_sim object, t index and z index

    Returns
    -------
    np.ndarray

    """
        
    rho_ijxy_hermitian = rho_ijxy + np.conj(np.swapaxes(rho_ijxy, 0, 1))
    
    Omega_plus_source = np.einsum('ijs, jixy-> sxy', X.Tijs_minus, rho_ijxy_hermitian)
    Omega_minus_source = np.einsum('jis, ijxy-> sxy', X.Tijs_plus, rho_ijxy_hermitian)
        
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
