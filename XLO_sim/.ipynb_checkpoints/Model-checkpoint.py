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

    if X.turn_off_coherences:
        mask_coh = generate_coherence_mask(X)

    if X.is_spontaneous_only:
        drho_MB = 0.0
    else:     
        Hint = 1j * (np.einsum('ijs, sxy->ijxy', X.Tijs_plus, Omega_plus_sxy) + np.einsum('ijs, sxy->ijxy', X.Tijs_minus, Omega_minus_sxy)) 
        drho_MB = np.einsum('isxy,sjxy->ijxy', Hint, rho_ijxy) - np.einsum('isxy,sjxy->ijxy', rho_ijxy, Hint)

        if X.turn_off_coherences:
            drho_MB = mask_coh * drho_MB

    
    if X.run_mode == 'consecutive':

        drho_pump = np.einsum('ij, ijxy->ijxy', -X.Mij, rho_ijxy) + np.einsum('ij, ixy->ijxy', X.delta_ij, (X.pump_itxyz[:, it, :, :, iz] + np.einsum('is, ssxy->ixy', X.Gamma_sp_fsm1N * X.Gij, rho_ijxy)))
        gamma_ion_ixy = np.einsum('i, xy->ixy', X.S_ion_Fi[0, :], X.j_3D[it, :, :, iz])

        if X.enable_self_absorption:
            J_Omega_minus_xy = np.real(Omega_plus_sxy[0, :, :] * Omega_minus_sxy[0, :, :]) / X.flux_factor
            J_Omega_plus_xy = np.real(Omega_plus_sxy[1, :, :] * Omega_minus_sxy[1, :, :]) / X.flux_factor

            drho_pump += np.einsum('ij, ixy->ijxy', X.delta_ij, np.einsum('si, sxy->ixy', X.S_ground_Fi[1:, :-1], np.einsum('sxy, xy-> sxy', np.array([J_Omega_minus_xy, J_Omega_plus_xy]), X.rho_0_3D[it, :, :, iz])))


    if X.run_mode == 'simultaneous':
        
        drho_pump = np.einsum('ij, ijxy->ijxy', -X.Mij, rho_ijxy) + np.einsum('ij, ixy->ijxy', X.delta_ij, (np.einsum('is, ssxy->ixy', X.Gamma_sp_fsm1N * X.Gij, rho_ijxy) + np.einsum('i, xy->ixy', X.S_ground_Fi[0, :-1], J_P_xy * rho_ground_xy)))
        gamma_ion_ixy = np.einsum('i, xy->ixy', X.S_ion_Fi[0, :], J_P_xy)

        if X.enable_self_absorption:
            drho_pump += np.einsum('ij, ixy->ijxy', X.delta_ij, np.einsum('fi, fxy->ixy', X.S_ground_Fi[1:, :-1], np.einsum('fxy, xy-> fxy', [J_Omega_minus_xy, J_Omega_plus_xy], rho_ground_xy)))
            
    if X.enable_self_absorption:
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
    
    if X.run_mode == 'consecutive':
        
        drho_pump = X.S_ground_Fi[0, -1] * X.j_3D[it, :, :, iz] * X.rho_0_3D[it, :, :, iz]
        drho_ion = - X.S_other_F[0] * X.j_3D[it, :, :, iz] * rho_other_xy
        
        if X.enable_self_absorption:
            Omega_plus_sxy = Omega_psxy[0, :, :, :]
            Omega_minus_sxy = Omega_psxy[1, :, :, :]

            J_Omega_minus_xy = np.abs(Omega_plus_sxy[0, :, :] * Omega_minus_sxy[0, :, :] / X.flux_factor)
            J_Omega_plus_xy = np.abs(Omega_plus_sxy[1, :, :] * Omega_minus_sxy[1, :, :] / X.flux_factor)

            drho_pump += np.einsum('f, fxy->xy', X.S_ground_Fi[1:, -1], np.array([J_Omega_minus_xy, J_Omega_plus_xy])) * X.rho_0_3D[it, :, :, iz]
            
        
    if X.run_mode == 'simultaneous':
        
        drho_pump = X.S_ground_Fi[0, -1] * J_P_xy * rho_ground_xy
        drho_ion = - X.S_other_F[0] * J_P_xy * rho_other_xy
        
        if X.enable_self_absorption:
            drho_pump += np.einsum('f, fxy->xy', X.S_ground_Fi[1:, -1], np.array([J_Omega_minus_xy, J_Omega_plus_xy])) * rho_ground_xy

    if X.enable_self_absorption:
        drho_ion -= np.einsum('f, fxy->xy', X.S_other_F[1:], np.array([J_Omega_minus_xy, J_Omega_plus_xy])) * rho_other_xy

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

    drho = -np.einsum('i, xy->xy', X.S_ground_Fi[0, :], J_P_xy) * rho_ground_xy

    if X.enable_self_absorption:
        drho += -np.einsum('fi, fxy->xy', X.S_ground_Fi[1:, :], np.array([J_Omega_minus_xy, J_Omega_plus_xy])) * rho_ground_xy

    return drho




def MB_nlevel_noise(rho_ijxy, E_sxy, G_sxy, params):
    """
    Calculate the noise part of the Maxwell-Bloch equations for the density matrix.

    Parameters
    ----------
    rho_ijxy: np.ndarray
        Density matrix at given t,z
    sqrtTrhoT_sxy: np.ndarray
        square root of T^(-) rho T^(+) at given t,z
    params: list
        List containing the XLO_sim object, t index and z index

    Returns
    -------
    np.ndarray

    """
    
    X, it, iz = params

    if (X.is_seeded_only==True):
        return 0.0 + 1j * 0.0

    xi_plus_sxy = X.noise_pstxyz[0, :, it, :, :, iz]
    xi_minus_sxy = X.noise_pstxyz[1, :, it, :, :, iz]
    xi_sxy = xi_plus_sxy.copy()
    xi_dag_sxy = xi_minus_sxy.copy()
       
    f_sxy       = (1.0 / X.noise_f_factor) * np.sqrt(E_sxy / (E_sxy - G_sxy)) * xi_sxy / X.dV
    f_dag_sxy   = (1.0 / X.noise_f_factor) * np.sqrt(E_sxy / (E_sxy - G_sxy)) * xi_dag_sxy / X.dV
    f_c_sxy     = X.noise_f_factor * np.sqrt((E_sxy - G_sxy) / E_sxy) * np.conj(xi_sxy) / X.dt
    f_dag_c_sxy = X.noise_f_factor * np.sqrt((E_sxy - G_sxy) / E_sxy) * np.conj(xi_dag_sxy) / X.dt
    
    if (X.enable_noise_mask==True):
        
        noise_mask = generate_noise_mask(E_sxy - G_sxy, X.Esxy_noise_cutoff)
        f_sxy       *= noise_mask
        f_dag_sxy *= noise_mask
        f_c_sxy *= noise_mask
        f_dag_c_sxy *= noise_mask
        
    drho_noise_f_c = np.einsum('ikxy, kjxy->ijxy', np.einsum('iks, sxy ->ikxy', X.Tijs_minus, f_c_sxy), rho_ijxy) + np.einsum('ikxy, kjxy->ijxy', rho_ijxy, np.einsum('kjs, sxy ->kjxy', X.Tijs_plus, f_dag_c_sxy))
    Omega_plus_self_noise__sxy  = 1j * (3.0/(8.0 * np.pi)) * X.lambdaKalpha1N**2 * X.Gamma_sp_fsm1N * f_sxy * X.dz / 2.0
    Omega_minus_self_noise__sxy = -1j * (3.0/(8.0 * np.pi)) * X.lambdaKalpha1N**2 * X.Gamma_sp_fsm1N * f_dag_sxy * X.dz / 2.0
    Hint_self_noise_ijxy = 1j * (np.einsum('ijs, sxy->ijxy', X.Tijs_plus, Omega_plus_self_noise__sxy) + np.einsum('ijs, sxy->ijxy', X.Tijs_minus, Omega_minus_self_noise__sxy)) 
    drho_field_self_noise_MB = np.einsum('isxy,sjxy->ijxy', Hint_self_noise_ijxy, rho_ijxy) - np.einsum('isxy,sjxy->ijxy', rho_ijxy, Hint_self_noise_ijxy)
    

    return X.dt * (drho_noise_f_c + drho_field_self_noise_MB)


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


def Omega_source_noise(E_sxy, G_sxy, params):
    """
    Calculate the noise contribution to ASE field at a given point of space and time (t,z).

    Parameters
    ----------
    sqrtTrhoT_sxy: np.ndarray
        square root of T^(-) rho T^(+) at given t,z
    params: list
        List containing the XLO_sim object, t index and z index

    Returns
    -------
    np.ndarray

    """
    
    X, it, iz = params
    
    if (X.is_seeded_only==True):
        return 0.0 + 1j * 0.0
    
    xi_plus_sxy = X.noise_pstxyz[0, :, it, :, :, iz]
    xi_minus_sxy = X.noise_pstxyz[1, :, it, :, :, iz]
    xi_sxy = xi_plus_sxy.copy()
    xi_dag_sxy = xi_minus_sxy.copy()
    
    dOmega_noise_psxy = np.zeros((2, 2, X.xgrid, X.ygrid), dtype=complex)
        
    f_sxy       = (1.0 / X.noise_f_factor) * np.sqrt(E_sxy / (E_sxy - G_sxy)) * xi_sxy / X.dV
    f_dag_sxy   = (1.0 / X.noise_f_factor) * np.sqrt(E_sxy / (E_sxy - G_sxy)) * xi_dag_sxy / X.dV
    
    if (X.enable_noise_mask==True):
        noise_mask = generate_noise_mask(E_sxy - G_sxy, X.Esxy_noise_cutoff)
        f_sxy       *= noise_mask
        f_dag_sxy   *= noise_mask

    
    dOmega_noise_psxy[0, :, :, :] += 1j * (3.0 / (8.0 * np.pi)) * X.lambdaKalpha1N**2 * X.Gamma_sp_fsm1N * X.dz * f_sxy
    dOmega_noise_psxy[1, :, :, :] += -1j * (3.0 / (8.0 * np.pi)) * X.lambdaKalpha1N**2 * X.Gamma_sp_fsm1N * X.dz * f_dag_sxy
       
    return dOmega_noise_psxy


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

        if (X.enable_pump_absorption==False):
            kappa_P_xyz *= 0.0

        return kappa_P_xyz

    else:
        kappa_Omega_sxyz = X.n * np.einsum('xyz, si->sxyz', rho_ground_xyz, X.S_ground_Fi[1:, :]) + X.n * np.einsum('xyz, s->sxyz', rho_other_xyz, X.S_other_F[1:]) + X.n * np.einsum('si, iixyz->sxyz', X.S_ion_Fi[1:, :], rho_ijxyz) + X.n * X.sigma_compound_Ka1

        return np.array([kappa_Omega_sxyz, kappa_Omega_sxyz])

    

def generate_noise_mask(E_sxy, val):
        
    noise_mask = np.ones(np.shape(E_sxy))
    noise_mask[np.where(np.real(E_sxy) < val)] = 0.0
    
    return noise_mask


def generate_coherence_mask(X):
    
    mask_coh=np.ones((X.nlevel, X.nlevel, X.xgrid, X.ygrid))
    
    if(X.nlevel==6):
        
        for i in range(X.nlevel):
            for j in range(X.nlevel) :
                if(i==j):
                     mask_coh[i, j, :, :] = 0.0
                elif(i<4):
                    if(j<4):
                        mask_coh[i, j, :, :] = 0.0
    
        mask_coh[5, 4, :, :] = 0
        mask_coh[4, 5, :, :] = 0

    return mask_coh