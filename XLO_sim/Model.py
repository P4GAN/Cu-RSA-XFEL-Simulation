import numpy as np
import tools


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

    Hint = 1j * (np.einsum('ijs, sxy->ijxy', X.Tijs_plus, Omega_plus_sxy) + np.einsum('ijs, sxy->ijxy', X.Tijs_minus, Omega_minus_sxy)) 
    drho_MB = np.einsum('isxy,sjxy->ijxy', Hint, rho_ijxy) - np.einsum('isxy,sjxy->ijxy', rho_ijxy, Hint)

    drho_pump = np.einsum('ij, ijxy->ijxy', -X.Mij, rho_ijxy) + np.einsum('ij, ixy->ijxy', X.delta_ij, (np.einsum('is, ssxy->ixy', X.Gamma_sp_fsm1N * X.Gij, rho_ijxy) + np.einsum('i, xy->ixy', X.S_ground_Fi[0, :-2], J_P_xy * rho_ground_xy)))
    gamma_ion_ixy = np.einsum('i, xy->ixy', X.S_ion_Fi[0, :], J_P_xy)

    drho_pump += np.einsum('ij, ixy->ijxy', X.delta_ij, np.einsum('fi, fxy->ixy', X.S_ground_Fi[1:, :-2], np.einsum('fxy, xy-> fxy', [J_Omega_minus_xy, J_Omega_plus_xy], rho_ground_xy)))
    gamma_ion_ixy += np.einsum('fi, fxy->ixy', X.S_ion_Fi[1:, :], np.array([J_Omega_minus_xy, J_Omega_plus_xy]))
    
    drho_ion = - 1.0 / 2.0 * (np.einsum('ixy, ijxy->ijxy', gamma_ion_ixy, rho_ijxy) + np.einsum('jxy, ijxy->ijxy', gamma_ion_ixy, rho_ijxy))
    
    return drho_MB + drho_pump + drho_ion


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
      
    X, Omega_psxy, it, iz, rho_ground_xy, J_P_xy, J_Omega_minus_xy, J_Omega_plus_xy = params
    
    drho_pump = np.einsum('f, fxy->xy', X.S_ground_Fi[:, -2], np.array([J_P_xy, J_Omega_minus_xy, J_Omega_plus_xy])) * rho_ground_xy
    drho_ion = -np.einsum('f, fxy->xy', X.S_2s_F[:], np.array([J_P_xy, J_Omega_minus_xy, J_Omega_plus_xy])) * rho_2s_xy
    drho_2s_decay = -X.GammaL1fsm1N * rho_2s_xy

    return drho_pump + drho_ion + drho_2s_decay


def MB_2p3_3d5_regular(t, rho_2p3_3d5_xy, params):
    """
    Calculate the change of population of the 2p3_3d5 hole level due to auger decay from the 2s level, photoionization with the pump and seed fields and decay.
    Parameters
    ----------
    t
    rho_2p3_3d5_xy: np.ndarray
        Population of the 2p3_3d5 hole level at given t,z
    params: list
        List containing the XLO_sim object, seed field Rabi frequency at given t,z, t index, z index, ground state population, pump flux, seed flux for the -1 polarization, seed flux for the +1 polarization (all at given t,z)

    Returns
    -------
    np.ndarray

    """
      
    X, Omega_psxy, it, iz, rho_2s_xy, J_P_xy, J_Omega_minus_xy, J_Omega_plus_xy = params
    
    drho_ion = -np.einsum('f, fxy->xy', X.S_2p3_3d5_F[:], np.array([J_P_xy, J_Omega_minus_xy, J_Omega_plus_xy])) * rho_2p3_3d5_xy
    drho_2p3_3d5_decay = -X.Gamma3M45fsm1N * rho_2p3_3d5_xy
    drho_2p3_3d5_auger_feeding = X.GammaA_L1_to_L3M45fs1N * rho_2s_xy

    return drho_ion + drho_2p3_3d5_decay + drho_2p3_3d5_auger_feeding


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


def absorption(rho_ground_xyz, rho_other_xyz, rho_2s_xyz, rho_2p3_3d5_xyz, rho_ijxyz, params):
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
    rho_2p3_3d5_xyz: np.ndarray
        2p3_3d5 hole level population at given t,z
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
        kappa_P_xyz =   X.n * X.sigma_ground_pump * rho_ground_xyz + \
                        X.n * X.S_other_F[0] * rho_other_xyz + \
                        X.n * X.S_2s_F[0] * rho_2s_xyz + \
                        X.n * X.S_2p3_3d5_F[0] * rho_2p3_3d5_xyz + \
                        X.n * np.einsum('i, iixyz->xyz', X.S_ion_Fi[0, :], rho_ijxyz) + \
                        X.n * X.sigma_compound_pump

        return kappa_P_xyz

    else:
        kappa_Omega_sxyz = X.n * np.einsum('xyz, si->sxyz', rho_ground_xyz, X.S_ground_Fi[1:, :]) + \
                            X.n * np.einsum('xyz, s->sxyz', rho_other_xyz, X.S_other_F[1:]) + \
                            X.n * np.einsum('xyz, s->sxyz', rho_2s_xyz, X.S_2s_F[1:]) + \
                            X.n * np.einsum('xyz, s->sxyz', rho_2p3_3d5_xyz, X.S_2p3_3d5_F[1:]) + \
                            X.n * np.einsum('si, iixyz->sxyz', X.S_ion_Fi[1:, :], rho_ijxyz) + \
                            X.n * X.sigma_compound_Ka1

        return np.array([kappa_Omega_sxyz, kappa_Omega_sxyz])

