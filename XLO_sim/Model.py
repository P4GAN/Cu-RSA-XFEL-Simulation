import numpy as np
from numba import njit
from . import tools


@njit(cache=True, fastmath=True)
def _MB_nlevel_regular_core(rho_ijxy, Omega_plus_sxy, Omega_minus_sxy, Tijs_plus, Tijs_minus,
                             Mij, Gamma_sp_Gij, S_ground_Fi0, S_ground_Fif, S_ion_Fi0, S_ion_Fif,
                             rho_ground_xy, J_Omega_minus_xy, J_Omega_plus_xy, J_P_xy):
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
                diag_sum = 0.0j
                for s in range(nlevel):
                    diag_sum += Gamma_sp_Gij[i, s] * rho_ijxy[s, s, x, y]

                pump_i = diag_sum + S_ground_Fi0[i] * J_P_xy[x, y] * rho_ground_xy[x, y]
                pump_i += S_ground_Fif[0, i] * J_Omega_minus_xy[x, y] * rho_ground_xy[x, y]
                pump_i += S_ground_Fif[1, i] * J_Omega_plus_xy[x, y] * rho_ground_xy[x, y]
                pump_term[i, x, y] = pump_i

                gion = S_ion_Fi0[i] * J_P_xy[x, y]
                gion += S_ion_Fif[0, i] * J_Omega_minus_xy[x, y]
                gion += S_ion_Fif[1, i] * J_Omega_plus_xy[x, y]
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
    
    X, Omega_psxy, it, iz, rho_ground_xy, J_P_xy, J_Omega_minus_xy, J_Omega_plus_xy  = params

    Omega_plus_sxy = Omega_psxy[0, :, :, :]
    Omega_minus_sxy = Omega_psxy[1, :, :, :]

    ######### Old unoptimized version

    # Hint = 1j * (np.einsum('ijs, sxy->ijxy', X.Tijs_plus, Omega_plus_sxy) + np.einsum('ijs, sxy->ijxy', X.Tijs_minus, Omega_minus_sxy)) 
    # drho_MB = np.einsum('isxy,sjxy->ijxy', Hint, rho_ijxy) - np.einsum('isxy,sjxy->ijxy', rho_ijxy, Hint)

    # drho_pump = np.einsum('ij, ijxy->ijxy', -X.Mij, rho_ijxy) + np.einsum('ij, ixy->ijxy', X.delta_ij, (np.einsum('is, ssxy->ixy', X.Gamma_sp_Gij, rho_ijxy) + np.einsum('i, xy->ixy', X.S_ground_Fi[0, :-1], J_P_xy * rho_ground_xy)))
    # gamma_ion_ixy = np.einsum('i, xy->ixy', X.S_ion_Fi[0, :], J_P_xy)

    # J_Omega_fxy = np.array([J_Omega_minus_xy, J_Omega_plus_xy])

    # drho_pump += np.einsum('ij, ixy->ijxy', X.delta_ij, np.einsum('fi, fxy->ixy', X.S_ground_Fi[1:, :-1], np.einsum('fxy, xy-> fxy', J_Omega_fxy, rho_ground_xy)))
    # gamma_ion_ixy += np.einsum('fi, fxy->ixy', X.S_ion_Fi[1:, :], J_Omega_fxy)
    
    # drho_ion = - 1.0 / 2.0 * (np.einsum('ixy, ijxy->ijxy', gamma_ion_ixy, rho_ijxy) + np.einsum('jxy, ijxy->ijxy', gamma_ion_ixy, rho_ijxy))
    
    # return drho_MB + drho_pump + drho_ion

    #########

    # New version without einsum, optimized with numba
    return _MB_nlevel_regular_core(
        rho_ijxy, Omega_plus_sxy, Omega_minus_sxy, X.Tijs_plus, X.Tijs_minus,
        X.Mij, X.Gamma_sp_Gij, X.S_ground_Fi[0, :-1], X.S_ground_Fi[1:, :-1],
        X.S_ion_Fi[0, :], X.S_ion_Fi[1:, :],
        rho_ground_xy, J_Omega_minus_xy, J_Omega_plus_xy, J_P_xy,
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
      
    X, Omega_psxy, it, iz, rho_ground_xy, J_P_xy, J_Omega_minus_xy, J_Omega_plus_xy = params
    
    drho_pump = np.einsum('f, fxy->xy', X.S_ground_Fi[:, -1], np.array([J_P_xy, J_Omega_minus_xy, J_Omega_plus_xy])) * rho_ground_xy
    drho_ion = -np.einsum('f, fxy->xy', X.S_other_F[:], np.array([J_P_xy, J_Omega_minus_xy, J_Omega_plus_xy])) * rho_other_xy

    return drho_pump + drho_ion



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
    
    X, J_P_xy, J_Omega_minus_xy, J_Omega_plus_xy, it, iz = params

    drho = -np.einsum('fi, fxy->xy', X.S_ground_Fi[:, :], np.array([J_P_xy, J_Omega_minus_xy, J_Omega_plus_xy])) * rho_ground_xy

    return drho




def Omega_source_regular(rho_ijxy, params):
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
    
    X, it, iz = params
    
    rho_ijxy_hermitian = rho_ijxy + np.conj(np.swapaxes(rho_ijxy, 0, 1))
    
    Omega_plus_source = np.einsum('ijs, jixy-> sxy', X.Tijs_minus, rho_ijxy_hermitian)
    Omega_minus_source = np.einsum('jis, ijxy-> sxy', X.Tijs_plus, rho_ijxy_hermitian)
        
    return X.field_source_factor * np.einsum('p, psxy -> psxy', X.e_sign, np.asarray([Omega_plus_source, Omega_minus_source]))


def pump_two_level(t, Rho, params):

    rho_0, rho_i = Rho
    X, j0, i, j = params

    f1 = -X.sigma_ground_pump * rho_0[:, :] * j0[i, :, :]
    f2 = X.sigma1_pump_1s * rho_0[:, :] * j0[i, :, :] 
    
    return np.array([f1, f2])


def absorption_Omega(rho_ijxyz, rho_other_xyz, params):
    """
    Calculate the absorption coefficient of the seed field due to photoionization of the ground and all ionic states, and the compound. The ground state population is pre-configured.

    Parameters
    ----------
    rho_ijxyz:
        Density matrix at given t,z
    rho_other_xyz:
        Additional ionic population at given t,z
    params: list
        List containing the XLO_sim object, t index and z index

    Returns
    -------
    np.ndarray

    """
    
    X, it, iz = params

    kappa_Omega_sxyz = X.n * np.einsum('xyz, si->sxyz', X.rho_0_3D[it, :, :, iz-1:iz+1], X.S_ground_Fi[1:, :]) + X.n * np.einsum('s, xyz->sxyz', X.S_other_F[1:],  rho_other_xyz) + X.n * np.einsum('si, iixyz->sxyz', X.S_ion_Fi[1:, :], rho_ijxyz) + X.n * X.sigma_compound_Ka1

    return np.array([kappa_Omega_sxyz, kappa_Omega_sxyz])


def absorption(rho_ground_xyz, rho_other_xyz, rho_ijxyz, params):
    """
    Calculate the absorption coefficient of the pump or seed field due to photoionization of the ground and all ionic states, and the compound. The ground state population is not pre-configured.
    Parameters
    ----------
    rho_ground_xyz: np.ndarray
        Ground state population at given t,z
    rho_other_xyz: np.ndarray
        Additional ionic population at given t,z
    rho_ijxyz: np.ndarray
        Density matrix at given t,z
    params: list
        List containing the XLO_sim object, t index, z index, and field name

    Returns
    -------
    np.ndarray

    """
    
    X, it, iz, field = params

    if field == 'pump':
        kappa_P_xyz = X.n * X.sigma_ground_pump * rho_ground_xyz + X.n * X.S_other_F[0] * rho_other_xyz + X.n * np.einsum('i, iixyz->xyz', X.S_ion_Fi[0, :], rho_ijxyz) + X.n * X.sigma_compound_pump

        return kappa_P_xyz

    else:
        kappa_Omega_sxyz = X.n * np.einsum('xyz, si->sxyz', rho_ground_xyz, X.S_ground_Fi[1:, :]) + X.n * np.einsum('xyz, s->sxyz', rho_other_xyz, X.S_other_F[1:]) + X.n * np.einsum('si, iixyz->sxyz', X.S_ion_Fi[1:, :], rho_ijxyz) + X.n * X.sigma_compound_Ka1

        return np.array([kappa_Omega_sxyz, kappa_Omega_sxyz])

