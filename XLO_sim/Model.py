import numpy as np
from numba import njit
from . import tools


@njit(cache=True, fastmath=True)
def _MB_nlevel_regular_core(rho_ijxy, Omega_plus_sxy, Omega_minus_sxy, Tijs_plus, Tijs_minus,
                             Mij, Gamma_sp_Gij, S_ground_Fif, S_ion_Fif,
                             auger_feeding_diag,
                             rho_ground_xy, rho_2s_xy, J_Omega_minus_xy, J_Omega_plus_xy):
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

    # Per-level pump source and ionization rate, shared by every (i, j) pair below
    pump_term = np.zeros((nlevel, nx, ny), dtype=np.complex128)
    gamma_ion = np.zeros((nlevel, nx, ny), dtype=np.complex128)
    for i in range(nlevel):
        for x in range(nx):
            for y in range(ny):
                pump_i = 0.0j
                for s in range(nlevel):
                    pump_i += Gamma_sp_Gij[i, s] * rho_ijxy[s, s, x, y]

                pump_i += S_ground_Fif[0, i] * J_Omega_minus_xy[x, y] * rho_ground_xy[x, y]
                pump_i += S_ground_Fif[1, i] * J_Omega_plus_xy[x, y] * rho_ground_xy[x, y]
                pump_i += auger_feeding_diag[i] * rho_2s_xy[x, y]
                pump_term[i, x, y] = pump_i

                gion = S_ion_Fif[0, i] * J_Omega_minus_xy[x, y] + S_ion_Fif[1, i] * J_Omega_plus_xy[x, y]
                gamma_ion[i, x, y] = gion

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
                        val += pump_term[i, x, y]
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

    return _MB_nlevel_regular_core(
        rho_ijxy, Omega_plus_sxy, Omega_minus_sxy, X.Tijs_plus, X.Tijs_minus,
        X.Mij, X.Gamma_sp_Gij, X.S_ground_Fi[:, :-1],  X.S_ion_Fi[:, :], 
        np.diag(X.auger_feeding_matrix), rho_ground_xy, rho_2s_xy, 
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


def absorption(X, rho_ground_xyz, rho_other_xyz, rho_2s_xyz, rho_ijxyz):
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

    Returns
    -------
    np.ndarray

    """
    
    kappa_Omega_sxyz = X.n * np.einsum('xyz, si->sxyz', rho_ground_xyz, X.S_ground_Fi[:, :]) + \
                        X.n * np.einsum('xyz, s->sxyz', rho_other_xyz, X.S_other_F[:]) + \
                        X.n * np.einsum('xyz, s->sxyz', rho_2s_xyz, X.S_2s_F[:]) + \
                        X.n * np.einsum('si, iixyz->sxyz', X.S_ion_Fi[:, :], rho_ijxyz) + \
                        X.n * X.sigma_compound_Ka1

    return np.array([kappa_Omega_sxyz, kappa_Omega_sxyz])
