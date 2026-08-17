import numpy as np
import scipy.constants as sp_const
from scipy.interpolate import RegularGridInterpolator
import scipy.constants as sp_const
import scipy.special as sp_func
import scipy.signal as sp_sign

from ocelot.optics.new_wave import *
import logging
logging.getLogger('ocelot').setLevel(logging.CRITICAL + 1)

def gaussian_pulse(X):
    """
    Generate a temporal Gaussian intensity profile.

    Parameters
    ----------
    X
        XLO_sim object

    Returns
    -------
    np.ndarray

    """

    return  1.0 / (np.sqrt(2.0 * np.pi) * X.sigma_t) * np.exp(-(X.t - X.t0)**2 / 2.0 / X.sigma_t**2)


def Gaussian_pulse_3D(X):
    """
    Generate a three-dimensional spatio-temporal Gaussian field profile.
    Parameters
    ----------
    X
        XLO_sim object

    Returns
    -------
    np.ndarray

    """
    
    return  np.sqrt(X.N_pump_photons / (2.0 * np.pi * X.sigma_r**2) / (np.sqrt(2.0 * np.pi) * X.sigma_t) * np.exp(-1.0 * (X.x_mesh**2 + X.y_mesh**2) / 2.0 / X.sigma_r**2) * np.exp(-(X.t_mesh - X.t0)**2 / 2.0 / X.sigma_t**2))





def Gaussian_pulse_3D_t_shifted(X):
    """
    Generate a three-dimensional spatio-temporal Gaussian field profile with peak at tmax (it is convenient for attosecond poulses where the evolution is after the pump)
    Parameters
    ----------
    X
        XLO_sim object

    Returns
    -------
    np.ndarray

    """
    
    return  np.sqrt(X.N_pump_photons / (2.0 * np.pi * X.sigma_r**2) / (np.sqrt(2.0 * np.pi) * X.sigma_t) * np.exp(-1.0 * (X.x_mesh**2 + X.y_mesh**2) / 2.0 / X.sigma_r**2) * np.exp(-(X.t_mesh - X.t_pump_max)**2 / 2.0 / X.sigma_t**2))


def Gaussian_pulse_3D_t_shifted_chirped(X):
    """
    Generate a three-dimensional spatio-temporal Gaussian field profile with peak at tmax (it is convenient for attosecond poulses where the evolution is after the pump) with chirp
    Parameters
    ----------
    X
        XLO_sim object

    Returns
    -------
    np.ndarray

    """
    
    return  np.exp(1j * X.chirp_rad_fs2 * (X.t_mesh - X.t_pump_max)**2) * np.sqrt(X.N_pump_photons / (2.0 * np.pi * X.sigma_r**2) / (np.sqrt(2.0 * np.pi) * X.sigma_t) * np.exp(-1.0 * (X.x_mesh**2 + X.y_mesh**2) / 2.0 / X.sigma_r**2) * np.exp(-(X.t_mesh - X.t_pump_max)**2 / 2.0 / X.sigma_t**2))


def Gaussian_pulse_3D_t_shifted_splitted(X):
    """
    Generate a three-dimensional spatio-temporal Gaussian field profile with peak at tmax (it is convenient for attosecond poulses where the evolution is after the pump) and splitted into two parts delayed by a splitted_delay_t
    Parameters
    ----------
    X
        XLO_sim object

    Returns
    -------
    np.ndarray

    """
    p1 = np.sqrt(
        X.first_peak_ratio * X.N_pump_photons 
        / (2.0 * np.pi * X.sigma_r**2) 
        / (np.sqrt(2.0 * np.pi) * X.sigma_t) 
        * np.exp(-1.0 * (X.x_mesh**2 + X.y_mesh**2) / 2.0 / X.sigma_r**2) 
        * np.exp(-(X.t_mesh - X.t_pump_max)**2 / 2.0 / X.sigma_t**2)
    )
    
    p2 = np.sqrt(
        (1 - X.first_peak_ratio) * X.N_pump_photons 
        / (2.0 * np.pi * X.sigma_r**2) 
        / (np.sqrt(2.0 * np.pi) * X.sigma_t) 
        * np.exp(-1.0 * (X.x_mesh**2 + X.y_mesh**2) / 2.0 / X.sigma_r**2) 
        * np.exp(-(X.t_mesh - X.t_pump_max - X.splitted_delay_t)**2 / 2.0 / X.sigma_t**2)
    )
    
    return  p1 + p2

def shift_P_txy(array, t_peak, tmax_fs):

    tmax = array.shape[0]
    
    max_time_index = tmax // 2

    t_peak_index = int(t_peak / tmax_fs * tmax)

    shift_amount = t_peak_index - max_time_index
    
    
    shifted_array = np.zeros_like(array)
    if shift_amount > 0:
        padded_array = np.pad(array, [(abs(shift_amount), 0), (0, 0), (0, 0)], mode='constant')
        shifted_array += padded_array[0:tmax, :, :] 
    else:
        padded_array = np.pad(array, [(0, abs(shift_amount)), (0, 0), (0, 0)], mode='constant')
        shifted_array += padded_array[abs(shift_amount)-1:-1, :, :]
  
    return shifted_array


def Ocelot_SASE_pulse_pump_txy(X):
    SASE = RadiationField()  # initialize RadiationField object
    
    # The transverse domain is considered to be space [m], since I hace input
    # parameters in time [fs], the correct conversion is needed
    
    sigma_rx = X.pump_width_FWHM_x / (2 * np.sqrt(2*np.log(2))) 
    sigma_ry = X.pump_width_FWHM_y / (2 * np.sqrt(2*np.log(2)))
    sigma_t  = X.pump_duration_FWHM_t / (2 * np.sqrt(2*np.log(2)))
    
    N_pump_photons = X.E_pump_uJ * 1e-6 / (X.hwKalpha1N * sp_const.e)
    print('number of pump photons = ' + f"{N_pump_photons:.1e}")
    
    kwargs={'xlamds':1e-9*X.lambdaPump,                     # [m] - central wavelength
            'seed': X.random_seed,
            'shape':(X.xgrid, X.ygrid, X.tgrid),            # size of field matrix (x,y,z=ct) (number of points)
            'dgrid':(2e-9*X.xmax, 2e-9*X.ymax, 1e-15*X.tmax*sp_const.c),                # size of field grid (max value) 
            'power_rms':(1e-9*sigma_rx, 1e-9*sigma_ry, 1e-15*sigma_t*sp_const.c),   # rms size of radiation distribution
            'power_center':(0,0,None),                      # (x,y,z) [m] - position of the radiation distribution
            'power_angle':(0,0),                            # (x,y) [rad] - angle of further radiation propagation
            'power_waistpos':(0,0),                         # (Z_x,Z_y) [m] downstrean location of the waist of the beam
            'wavelength':None,                              # central frequency of the radiation, if different from xlamds
            'zsep':None,                                    # distance between slices in z as zsep*xlamds
            'freq_chirp':0,                                 # dw/dt=[1/fs**2] - requency chirp of the beam around power_center[2]
            'en_pulse':N_pump_photons*X.hwPump*sp_const.e,        # total energy or max power of the pulse, use only one
            'power':None,
            'rho':X.FEL_bandwidth/2
            }
    
    SASE = imitate_sase_dfl(**kwargs);
    
    field_txy = SASE.fld
    
    # Normalization: it has to be compatible with the units used in the rest of
    # the code. imitate_sase_dfl uses SI units [m, s, J], need to normalize with
    # respect to [nm, fs, eV]
    
    dx = 1e9 * (SASE.Lx() / SASE.Nx())
    dy = 1e9 * (SASE.Ly() / SASE.Ny())
    dt = 1e15 * (SASE.Lz() / SASE.Nz()) / sp_const.c
    
    norm = np.sqrt(np.sum(np.abs(field_txy)**2 * dx * dy * dt))
    
    field_txy = field_txy / norm
    
    return np.sqrt(N_pump_photons) * field_txy


def Gaussian_pulse_aniso_pump(X):
          
    sigma_rx = X.pump_width_FWHM_x / (2 * np.sqrt(2*np.log(2)))
    sigma_ry = X.pump_width_FWHM_y / (2 * np.sqrt(2*np.log(2)))
    sigma_t  = X.pump_duration_FWHM_t / (2 * np.sqrt(2*np.log(2)))
    
    field_txy = np.sqrt( 
        1 / (2.0 * np.pi * sigma_rx * sigma_ry) 
        / (np.sqrt(2.0 * np.pi) * sigma_t) 
        * np.exp(-1.0 * X.x_mesh**2 / 2.0 / sigma_rx**2) 
        * np.exp(-1.0 * X.y_mesh**2 / 2.0 / sigma_ry**2) 
        * np.exp(-(X.t_mesh - X.t0)**2 / 2.0 / sigma_t**2)
    )
    
    norm = np.sqrt(np.sum(np.abs(field_txy)**2 * X.dx * X.dy * X.dt))
    
    print('norm = ',norm)
    
    field_txy = field_txy / norm
    
    N_pump_photons = X.E_pump_uJ * 1e-6 / (X.hwKalpha1N * sp_const.e)
    print('number of pump photons = ' + f"{N_pump_photons:.1e}")
    
        
    return np.sqrt(N_pump_photons) * field_txy



def Gaussian_pulse_aniso_seed(X):
          
    seed_sigma_rx = X.seed_width_FWHM_x / (2 * np.sqrt(2*np.log(2)))
    seed_sigma_ry = X.seed_width_FWHM_y / (2 * np.sqrt(2*np.log(2)))
    seed_sigma_t  = X.seed_duration_FWHM_t / (2 * np.sqrt(2*np.log(2)))
    
    field_txy = np.sqrt( 
        1 / (2.0 * np.pi * seed_sigma_rx * seed_sigma_ry) 
        / (np.sqrt(2.0 * np.pi) * seed_sigma_t) 
        * np.exp(-1.0 * X.x_mesh**2 / 2.0 / seed_sigma_rx**2) 
        * np.exp(-1.0 * X.y_mesh**2 / 2.0 / seed_sigma_ry**2) 
        * np.exp(-(X.t_mesh - X.t0)**2 / 2.0 / seed_sigma_t**2)
    )
    
    norm = np.sqrt(np.sum(np.abs(field_txy)**2 * X.dx * X.dy * X.dt))
    
    print('norm = ',norm)
    
    field_txy = field_txy / norm
    
    N_seed_photons = X.E_seed_uJ * 1e-6 / (X.hwKalpha1N * sp_const.e)
    print('number of seed photons = ' + f"{N_seed_photons:.1e}")
    
    seed = np.sqrt(N_seed_photons) * field_txy
    
    Omega_seed_pstxy = np.zeros((2, 2, X.tgrid, X.xgrid, X.ygrid), dtype=complex)
    fluxfield2Rabi = np.sqrt((3.0 * X.lambdaKalpha1N**2 * X.Gamma_sp_fsm1N / 8.0 / np.pi))
    Omega_seed_pstxy[0,1,:,:,:] = fluxfield2Rabi * seed # in linear polarization basis, along y-axis
    Omega_seed_pstxy[1,:,:,:,:] = np.conj(Omega_seed_pstxy[0,:,:,:,:])
    
    return Omega_seed_pstxy 


def roll_zeropad(a, shift, axis=None):
    """
    Roll array elements along a given axis.

    Elements off the end of the array are treated as zeros.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int
        The number of places by which elements are shifted.
    axis : int, optional
        The axis along which elements are shifted.  By default, the array
        is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as `a`.

    See Also
    --------
    roll     : Elements that roll off one end come back on the other.
    rollaxis : Roll the specified axis backwards, until it lies in a
               given position.

    Examples
    --------
    >>> x = np.arange(10)
    >>> roll_zeropad(x, 2)
    array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    >>> roll_zeropad(x, -2)
    array([2, 3, 4, 5, 6, 7, 8, 9, 0, 0])

    >>> x2 = np.reshape(x, (2,5))
    >>> x2
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> roll_zeropad(x2, 1)
    array([[0, 0, 1, 2, 3],
           [4, 5, 6, 7, 8]])
    >>> roll_zeropad(x2, -2)
    array([[2, 3, 4, 5, 6],
           [7, 8, 9, 0, 0]])
    >>> roll_zeropad(x2, 1, axis=0)
    array([[0, 0, 0, 0, 0],
           [0, 1, 2, 3, 4]])
    >>> roll_zeropad(x2, -1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, 1, axis=1)
    array([[0, 0, 1, 2, 3],
           [0, 5, 6, 7, 8]])
    >>> roll_zeropad(x2, -2, axis=1)
    array([[2, 3, 4, 0, 0],
           [7, 8, 9, 0, 0]])

    >>> roll_zeropad(x2, 50)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, -50)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, 0)
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])

    """
    a = np.asanyarray(a)
    if shift == 0: return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift,n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift,n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res

def bragg_angle_for_energy(E_eV, d_hkl_A):
    """
    Bragg angle needed for a crystal reflection to select a given photon energy.

    Parameters
    ----------
    E_eV : float
        Target photon energy [eV].
    d_hkl_A : float
        Lattice plane spacing [Angstrom].

    Returns
    -------
    float
        Bragg angle [rad].
    """
    hc_eVA = 12398.42  # eV*Angstrom
    return np.arcsin(hc_eVA / (2 * d_hkl_A * E_eV))


def Roh(X, target_energy_eV):
    """
    Response function for X-ray Bragg diffraction for 111 silicon
    http://dx.doi.org/10.1103/PhysRevSTAB.15.100702
    
    """
    d_hkl_A=5.4310/np.sqrt(3)

    chi_h = -0.79955e-05 + 1j*0.24361e-06
    chi_mh = chi_h
    chi_0 = -0.15127e-04 + 1j*0.34955e-06

    theta_B = bragg_angle_for_energy(target_energy_eV, d_hkl_A)

    omega_Bragg = target_energy_eV / X.hbar

    # Detuning of the selected energy from the Ka1 line
    domega = (target_energy_eV - X.hwKalpha1N) / X.hbar

    t = X.t - (X.tmax/2)

    Tg = (2 * np.sin(theta_B)**2) / (omega_Bragg * np.sqrt(chi_h*chi_mh))
    exparg = - (omega_Bragg * np.imag(chi_0) * t) / (2 * np.sin(theta_B)**2)

    return np.heaviside(t, 1) * (sp_func.jv(1, t/Tg) / (1j*t)) * np.exp(exparg) * np.exp(1j * domega * t)

def Ocelot_SASE_seed_111_dcm_pstxy(X):
    SASE = RadiationField()  # initialize RadiationField object

    target_energy_eV = X.monochromator_target_energy_eV
    
    seed_sigma_rx = X.seed_width_FWHM_x / (2 * np.sqrt(2*np.log(2)))
    seed_sigma_ry = X.seed_width_FWHM_y / (2 * np.sqrt(2*np.log(2)))
    seed_sigma_t  = X.seed_duration_FWHM_t / (2 * np.sqrt(2*np.log(2)))
    
    # The transverse domain is considered to be space [m], since I hace input
    # parameters in time [fs], the correct conversion is needed
    
    kwargs={'xlamds':1e-9*X.lambdaKalpha1N,                     # [m] - central wavelength
            'seed': X.random_seed,
            'shape':(X.xgrid, X.ygrid, X.tgrid),            # size of field matrix (x,y,z=ct) (number of points)
            'dgrid':(2e-9*X.xmax, 2e-9*X.ymax, 1e-15*X.tmax*sp_const.c),                # size of field grid (max value) 
            'power_rms':(1e-9*seed_sigma_rx, 1e-9*seed_sigma_ry, 1e-15*seed_sigma_t*sp_const.c),   # rms size of radiation distribution
            'power_center':(0,0,None),                      # (x,y,z) [m] - position of the radiation distribution
            'power_angle':(0,0),                            # (x,y) [rad] - angle of further radiation propagation
            'power_waistpos':(0,0),                         # (Z_x,Z_y) [m] downstrean location of the waist of the beam
            'wavelength':None,                              # central frequency of the radiation, if different from xlamds
            'zsep':None,                                    # distance between slices in z as zsep*xlamds
            'freq_chirp':0,                                 # dw/dt=[1/fs**2] - requency chirp of the beam around power_center[2]
            'en_pulse':X.E_seed_uJ*1e-6,        # total energy or max power of the pulse, use only one
            'power':None,
            'rho':X.seed_FEL_bandwidth/2
            }
    
    SASE = imitate_sase_dfl(**kwargs);
    
    field_txy = SASE.fld  
    
    # Effect of monochromator: convolution between field_t and Roh
    field_temp_t0 = np.einsum('txy -> t', field_txy) # extract t domain
    field_temp_t0 = roll_zeropad(field_temp_t0, int(X.seed_delay/X.dt))

    field_temp_xy = np.einsum('txy -> xy', field_txy)

    field_temp_t1 = sp_sign.convolve(field_temp_t0, Roh(X, target_energy_eV), mode='same')
    field_temp_t2 = sp_sign.convolve(field_temp_t1, Roh(X, target_energy_eV), mode='same')
    
    field_txy = np.einsum('t, xy -> txy', field_temp_t2, field_temp_xy)
    
    norm = np.sqrt(np.sum(np.abs(field_txy)**2 * X.dx * X.dy * X.dt))
    
    field_txy = field_txy / norm
    
    N_seed_photons = X.E_seed_uJ * 1e-6 / (X.hwKalpha1N * sp_const.e)
    # print('number of seed photons = ' + f"{N_seed_photons:.1e}")
    
    SASE_for_seed = np.sqrt(N_seed_photons) * field_txy
    
    Omega_seed_pstxy = np.zeros((2, 2, X.tgrid, X.xgrid, X.ygrid), dtype=complex)
    fluxfield2Rabi = np.sqrt((3.0 * X.lambdaKalpha1N**2 * X.Gamma_sp_fsm1N / 8.0 / np.pi))
    Omega_seed_pstxy[0,1,:,:,:] = fluxfield2Rabi * SASE_for_seed # in linear polarization basis, along y-axis
    Omega_seed_pstxy[1,:,:,:,:] = np.conj(Omega_seed_pstxy[0,:,:,:,:])
        
    return Omega_seed_pstxy



def Ocelot_SASE_seed_pstxy(X):
    SASE = RadiationField()  # initialize RadiationField object
    
    seed_sigma_rx = X.seed_width_FWHM_x / (2 * np.sqrt(2*np.log(2)))
    seed_sigma_ry = X.seed_width_FWHM_y / (2 * np.sqrt(2*np.log(2)))
    seed_sigma_t  = X.seed_duration_FWHM_t / (2 * np.sqrt(2*np.log(2)))
    
    # The transverse domain is considered to be space [m], since I hace input
    # parameters in time [fs], the correct conversion is needed
    
    kwargs={'xlamds':1e-9*X.lambdaKalpha1N,                     # [m] - central wavelength
            'seed': X.random_seed,
            'shape':(X.xgrid, X.ygrid, X.tgrid),            # size of field matrix (x,y,z=ct) (number of points)
            'dgrid':(2e-9*X.xmax, 2e-9*X.ymax, 1e-15*X.tmax*sp_const.c),                # size of field grid (max value) 
            'power_rms':(1e-9*seed_sigma_rx, 1e-9*seed_sigma_ry, 1e-15*seed_sigma_t*sp_const.c),   # rms size of radiation distribution
            'power_center':(0,0,None),                      # (x,y,z) [m] - position of the radiation distribution
            'power_angle':(0,0),                            # (x,y) [rad] - angle of further radiation propagation
            'power_waistpos':(0,0),                         # (Z_x,Z_y) [m] downstrean location of the waist of the beam
            'wavelength':None,                              # central frequency of the radiation, if different from xlamds
            'zsep':None,                                    # distance between slices in z as zsep*xlamds
            'freq_chirp':0,                                 # dw/dt=[1/fs**2] - requency chirp of the beam around power_center[2]
            'en_pulse':X.E_seed_uJ*1e-6,        # total energy or max power of the pulse, use only one
            'power':None,
            'rho':X.seed_FEL_bandwidth/2
            }
    
    SASE = imitate_sase_dfl(**kwargs);
    
    field_txy = SASE.fld
    
    norm = np.sqrt(np.sum(np.abs(field_txy)**2 * X.dx * X.dy * X.dt))
    
    field_txy = field_txy / norm
    
    N_seed_photons = X.E_seed_uJ * 1e-6 / (X.hwKalpha1N * sp_const.e)
    # print('number of seed photons = ' + f"{N_seed_photons:.1e}")
    
    SASE_for_seed = np.sqrt(N_seed_photons) * field_txy
    
    Omega_seed_pstxy = np.zeros((2, 2, X.tgrid, X.xgrid, X.ygrid), dtype=complex)
    fluxfield2Rabi = np.sqrt((3.0 * X.lambdaKalpha1N**2 * X.Gamma_sp_fsm1N / 8.0 / np.pi))
    Omega_seed_pstxy[0,1,:,:,:] = fluxfield2Rabi * SASE_for_seed # in linear polarization basis, along y-axis
    Omega_seed_pstxy[1,:,:,:,:] = np.conj(Omega_seed_pstxy[0,:,:,:,:])
        
    return Omega_seed_pstxy


def Gaussian_pulse_3D_with_q(X, k=None, N_photons=None):
    """
    Generate a complex three-dimensional spatio-temporal Gaussian profile of field Rabi frequency expressed in terms of the q parameter.

    Parameters
    ----------
    X
        XLO_sim object
    k
        Radiation wavenumber
    N_photons
        Integrated number of photons

    Returns
    -------
    np.ndarray

    """

    if (k is None):
        k = X.kp
        
    if (N_photons is None):
        N_photons = X.N_pump_photons
        
    qx = 1j * X.zR
    qy = 1j * X.zR

    ux = 1.0 / np.sqrt(qx) * np.exp(-1j * X.kp * X.x_mesh**2 / 2.0 / qx)
    uy = 1.0 / np.sqrt(qy) * np.exp(-1j * X.kp * X.y_mesh**2 / 2.0 / qy)
    ut = 1.0 / (np.sqrt(2.0 * np.pi) * X.sigma_t) * np.exp(-(X.t_mesh - X.t0)**2 / 2.0 / X.sigma_t**2)

    eta = 2.0 * k * X.zR * X.sigma_t / np.sqrt(np.pi)
    
    return np.sqrt(eta) * np.sqrt(N_photons) * ux * uy * ut


def Gaussian_pulse_3D_with_q_chirp(X, bt, v0, k=None):
    """
    Generate a complex three-dimensional spatio-temporal Gaussian profile of field Rabi frequency expressed in terms of the q parameter.

    Parameters
    ----------
    X
        XLO_sim object
    bt
        phase terms
    v0
        phase terms
    k
        Radiation wavenumber

    Returns
    -------
    np.ndarray

    """

    if (k is None):
        k = X.kp
    
    qx = 1j * X.zR
    qy = 1j * X.zR
    ux = 1.0 / np.sqrt(qx) * np.exp(-1j * X.kp * X.x_mesh**2 / 2.0 / qx)
    uy = 1.0 / np.sqrt(qy) * np.exp(-1j * X.kp * X.y_mesh**2 / 2.0 / qy)
    exponent =- (X.t_mesh - X.t0)**2 / 2.0 / X.sigma_t**2
    exponent = exponent + 1j * bt * (X.t_mesh-X.t0)**2 + 2 * np.pi * 1j * v0 * (X.t_mesh-X.t0)
    ut =  1.0 / (np.sqrt(2.0 * np.pi) * X.sigma_t) * np.exp(exponent)
    eta = 2.0 * k * X.zR * X.sigma_t / np.sqrt(np.pi)

    return np.sqrt(eta) * np.sqrt(X.N_pump_photons) * ux * uy * ut
    

def Gaussian_from_mesh(X, mesh, moments):
    
    omega_nn, th_nxx, th_nyy = mesh
    sigma_w, sigma_th_x, sigma_th_y = moments
    
    return -1j * np.exp(-1.0 * (th_nxx**2 / 2.0 / sigma_th_x**2 + th_nyy**2 / 2.0 / sigma_th_y**2) ) * np.exp(-(omega_nn - 0.0)**2 / 2.0 / sigma_w**2)
    

def uniform_field_txy(X, Nphotons):

    eta = X.dt * X.dx * X.dy * X.xgrid * X.ygrid * X.tgrid / (3.0 * X.lambdaKalpha1N**2 * X.Gamma_sp_fsm1N / 8.0 / np.pi) 
    
    return 0.0 * X.x_mesh + np.sqrt(Nphotons / eta)


def uniform_seed_field_txy(X, Nphotons_spol, Nphotons_ppol, Dphi=0):

    uniform_seed_field = np.zeros_like(X.noise_pstxyz)[:, :, :, :, :, 0]

    field_spol = uniform_field_txy(X, Nphotons_spol)
    field_ppol = np.exp(1j * Dphi) * uniform_field_txy(X, Nphotons_ppol)

    uniform_seed_field[0, 0, :, :, :] = field_spol
    uniform_seed_field[1, 0, :, :, :] = np.conj(field_spol)
    uniform_seed_field[0, 1, :, :, :] = field_ppol 
    uniform_seed_field[1, 1, :, :, :] = np.conj(field_ppol)

    return uniform_seed_field
    

def Gaussian_seed_field_txy(X, Nphotons_spol, Nphotons_ppol, Dphi=0):

    #what's wrong with this function?

    Gaussian_seed_field = np.zeros_like(X.noise_pstxyz)[:, :, :, :, :, 0]

    field_spol = Gaussian_pulse_3D_with_q(X, X.k0, Nphotons_spol)
    field_ppol = np.exp(1j * Dphi) * Gaussian_pulse_3D_with_q(X, X.k0, Nphotons_ppol)

    Gaussian_seed_field[0, 0, :, :, :] = field_spol
    Gaussian_seed_field[1, 0, :, :, :] = np.conj(field_spol)
    Gaussian_seed_field[0, 1, :, :, :] = field_ppol
    Gaussian_seed_field[1, 1, :, :, :] = np.conj(field_ppol)

    return Gaussian_seed_field



def SASE_pulse_3D_with_q(X, k=None):
    """
    Generate a complex three-dimensional spatio-temporal SASE profile of field Rabi frequency expressed in terms of the q parameter.

    Parameters
    ----------
    X
        XLO_sim object
    k
        Radiation wavenumber

    Returns
    -------
    np.ndarray

    """
    
    np.random.seed(seed)
    

    if (k is None):
        k = X.kp

    qx = 1j * X.zR
    qy = 1j * X.zR

    ux = np.exp(-1j * X.kp * X.x_mesh**2 / 2.0 / qx)
    uy = np.exp(-1j * X.kp * X.y_mesh**2 / 2.0 / qy)

    t0_array = np.random.normal(0.0, X.sigma_t, X.N_modes)
    phi_mean = -1.0j * X.kp * X.c * np.mean(t0_array)

    ut = np.einsum('ntxy, n->txy', np.exp(- (X.t_mesh[np.newaxis, :, :, :] - t0_array[:, np.newaxis, np.newaxis, np.newaxis] - X.t0)**2 / 4.0 / X.sigma_coh**2), np.exp(1.0j * X.kp * X.c * t0_array)) * np.exp(phi_mean)

    scaling = X.N_modes
    for i in range(X.N_modes):
        for j in range(i+1, X.N_modes):
            scaling += 2.0 * np.cos(X.kp * X.c * (t0_array[j] - t0_array[i])) * np.exp(-(t0_array[i] - t0_array[j])**2 / 8.0 / X.sigma_coh**2)
    scaling *= np.sqrt(2.0 * np.pi) * X.sigma_coh

    eta = X.kp / (np.pi * X.zR * scaling)
    
    return np.sqrt(eta) * np.sqrt(X.N_pump_photons) * ux * uy * ut


def linear_to_circular(X, field_linear):

    field_circular = np.zeros_like(field_linear)
    field_circular[0, :, :, :, :] = np.einsum('stxy,qs->qtxy', field_linear[0, :, :, :, :], np.linalg.inv(X.transform_matrix))
    field_circular[1, :, :, :, :] = np.einsum('stxy,qs->qtxy', field_linear[1, :, :, :, :], np.conj(np.linalg.inv(X.transform_matrix)))
    
    return field_circular
    

def circular_to_linear(X, field_circular):

    field_linear = np.zeros_like(field_circular)
    field_linear[0, :, :, :, :] = np.einsum('stxy,qs->qtxy', field_circular[0, :, :, :, :], X.transform_matrix)
    field_linear[1, :, :, :, :] = np.einsum('stxy,qs->qtxy', field_circular[1, :, :, :, :], np.conj(X.transform_matrix))

    return field_linear


def nphoton_sz(X):
    """
    Calculate the number of seed photons for different field polarizations as function of the target position z.

    Parameters
    ----------
    X
        XLO_sim object

    Returns
    -------
    np.ndarray

    """
    
    return X.dt * X.dx * X.dy / (3.0 * X.lambdaKalpha1N**2 * X.Gamma_sp_fsm1N / 8.0 / np.pi) * np.sum(X.Omega_pstxyz[0,:,:,:,:,:] * X.Omega_pstxyz[1,:,:,:,:,:], axis=(1, 2, 3))


def nphoton_reg_sz(X):
    """
    Calculate the regularized number of seed photons for different field polarizations as function of the target position z.

    Parameters
    ----------
    X
        XLO_sim object

    Returns
    -------
    np.ndarray

    """
    
    return X.dt * X.dx * X.dy / (3.0 * X.lambdaKalpha1N**2 * X.Gamma_sp_fsm1N / 8.0 / np.pi) * 1.0 / 2.0 * (np.sum(X.Omega_pstxyz[0,:,1:,:,:,:] * X.Omega_pstxyz[1,:,:-1,:,:,:], axis=(1, 2, 3)) + np.sum(X.Omega_pstxyz[0,:,:-1,:,:,:] * X.Omega_pstxyz[1,:,1:,:,:,:], axis=(1, 2, 3)))


def nphoton_pump_z(X):
    """
    Calculate the number of pump photons as function of the target position z.

    Parameters
    ----------
    X
        XLO_sim object

    Returns
    -------
    np.ndarray

    """
    
    return X.dt * X.dx * X.dy * np.sum(X.J_P_txyz, axis=(0, 1, 2))


def random_vector_normal(size, seed=None):
    """
    Generate a sample from the standard normal distribution.

    Parameters
    ----------
    size: tuple
        Dimensions of the returned array
    seed: int
        Seed for the random number generator

    Returns
    -------
    np.ndarray

    """
    
    np.random.seed(seed)
  
    return np.random.randn(*size)


def random_vector_binary(size, seed=None):
    """
    Generate a random sample from elements [-1, 1].

    Parameters
    ----------
    size: tuple
        Dimensions of the returned array
    seed: int
        Seed for the random number generator

    Returns
    -------
    np.ndarray

    """
    
    np.random.seed(seed)

    return np.random.choice([-1, 1], size)


def random_gaussian_noise_nlevel_pstxyz(X):
    """
    Generate the array of noise factors for the seed field and density matrix, drawn from a standard normal distribution.

    Parameters
    ----------
    X
        XLO_sim object

    Returns
    -------
    np.ndarray

    """

    return (random_vector_normal((2, 2, X.tgrid, X.xgrid, X.ygrid, X.zgrid), X.random_seed) + 1j * random_vector_normal((2, 2, X.tgrid, X.xgrid, X.ygrid, X.zgrid), X.random_seed)) / np.sqrt(2.0)


def random_binary_noise_nlevel_pstxyz(X):
    """
    Generate the array of noise factors for the seed field and density matrix, drawn from array [-1, 1].

    Parameters
    ----------
    X
        XLO_sim object

    Returns
    -------
    np.ndarray

    """

    return (random_vector_binary((2, 2, X.tgrid, X.xgrid, X.ygrid, X.zgrid), X.random_seed) + 1j * random_vector_binary((2, 2, X.tgrid, X.xgrid, X.ygrid, X.zgrid), X.random_seed)) / np.sqrt(2.0)



def RK45_step(f, y, x0, dx, params):
    """
    Calculate one step of time propagation with the explicit Runge-Kutta method of order 4(5).

    Parameters
    ----------
    f: callable
        Right-hand side of the system
    y: array
        Initial state
    x0: float
        Initial time
    dx: float
        Time step size
    params: list

    Returns
    -------
    np.ndarray

    """

    k1 = f(x0, y, params) 
    k2 = f(x0 + 0.5 * dx, y + 0.5 * k1 * dx, params) 
    k3 = f(x0 + 0.5 * dx, y + 0.5 * k2 * dx, params) 
    k4 = f(x0 + dx, y + k3 * dx, params) 

    return (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dx / 6.0


def find_fwhm(x, y):
    max_value = np.max(y)
    max_index = np.argmax(y)
    half_max = max_value / 2
    idx_left = 0
    idx_right = len(y) - 1
    for i in range(0, max_index, 1):
        if y[i] > half_max:
            idx_left = i
            break
    for i in range(len(y)-1, max_index, -1):
        if y[i] > half_max:
            idx_right = i
            break
    return x[idx_right] - x[idx_left]

   
def transmittance_to_absorbance(T, base=np.e):
    """
    Convert transmittance to absorbance.

    Parameters
    ----------
    T : array_like
        Transmittance, I_out / I_in.
    base : float, optional
        Logarithm base. Use `np.e` (default) for the natural-log optical
        depth (A_e = mu * L, directly comparable to an absorption
        coefficient), or 10 for the decadic optical density (A = -log10(T)).

    Returns
    -------
    np.ndarray
        Absorbance.

    """

    return -np.log(T) / np.log(base)



def fft_field_t_y_to_w_thy(X, field_pstxy, ypad, tpad):
    """
    Fourier transform a field from the (time, y) domain to the (omega, theta_y) domain.

    Zero-pads the t and y axes, then FFTs along them and applies a linear
    phase correction for the fact that the simulation window is not centered
    at t=0/y=0 (it is centered at X.t_pump_max/X.ymax), so the resulting
    energy/angle map has the correct phase reference.

    Parameters
    ----------
    X
        XLO_sim object
    field_pstxy : np.ndarray
        Field array with axes (p, s, t, x, y), e.g. a z-slice of X.Omega_pstxyz.
    ypad : int
        Zero-padding added on each side of the y axis before the FFT (sets theta_y resolution).
    tpad : int
        Zero-padding added on each side of the t axis before the FFT (sets omega resolution).

    Returns
    -------
    extent_w_thy : list of float
        [omega_min, omega_max, theta_y_min, theta_y_max] (omega in eV, theta_y in mrad),
        the plot extent of the transformed field.
    field_psxwThy : np.ndarray
        The FFT'd, phase-corrected field with axes (p, s, x, omega, theta_y).

    """

    coeffs = (2.0 * np.pi * X.hbar, 2.0 * np.pi / X.k0)
    shifts = (X.t_pump_max + tpad * X.dt, X.ymax + ypad * X.dy)
    field_psxty = np.einsum('pstxy->psxty',field_pstxy)
    pad_shape = [(0, 0), (0, 0), (0, 0), (tpad, tpad), (ypad, ypad)]
    domains_sim, meshes_sim, step_sizes_sim = X.optics.nd_kspace(coeffs, (X.tgrid, X.ygrid), (tpad, ypad), (X.dt, X.dy))
    phasors = [np.exp(1j * 2.0 * np.pi * mesh * shift / coeff) for coeff, mesh, shift in zip(coeffs, meshes_sim, shifts)]

    field_psxty_pad = X.optics.my_pad(field_psxty, pad_shape)
    field_fft_pad_phased = X.optics.my_fft_phased(field_psxty_pad, (3, 4), phasors)

    extent_w_thy = [
        np.min(domains_sim[0][tpad:X.tgrid + tpad]), np.max(domains_sim[0][tpad:X.tgrid + tpad]),
        1e3*np.min(domains_sim[1]), 1e3*np.max(domains_sim[1])]

    return extent_w_thy, field_fft_pad_phased 



def SF_spectrum_w(X, zint, ypad, tpad):
    """
    Compute the spectral intensity of the in-sample field at a given depth z.

    X.Omega_pstxyz is the coherent Rabi-frequency field propagating through
    the sample: it is seeded with the injected probe field at z=0 and picks
    up the medium's stimulated/spontaneous response as it propagates (see
    Sample.py). Following this codebase's convention (e.g. I_SF_w_thy,
    convert_SF_phnm2fs_Wcm2) it is referred to as the "SF" field throughout,
    though depending on `zint` it may still be seed-dominated rather than
    purely spontaneously emitted.

    Takes that field at z-grid index `zint`, transforms it to the (omega,
    theta_y) domain via `fft_field_t_y_to_w_thy`, and forms the intensity from
    the product of the p=0 (Omega) and p=1 (Omega*) components. Returns the
    spectrum both integrated over theta_y and evaluated on-axis (theta_y = 0).

    Parameters
    ----------
    X
        XLO_sim object
    zint : int
        Grid index along z (propagation depth into the sample) at which to evaluate the spectrum.
    ypad : int
        Zero-padding added to the y axis before the FFT (see `fft_field_t_y_to_w_thy`).
    tpad : int
        Zero-padding added to the t axis before the FFT (see `fft_field_t_y_to_w_thy`).

    Returns
    -------
    womega_ar : np.ndarray
        Photon energy grid, in eV.
    I_int_thy_w : np.ndarray
        SF spectral intensity integrated over theta_y, as a function of omega.
    I_thy0_w : np.ndarray
        On-axis (theta_y = 0) SF spectral intensity, as a function of omega.

    """

    OmegaSF_z_pstxy = X.Omega_pstxyz[:, :, :, :, :, zint]
    extent_w_thy, OmegaSF_z_psxwThy = fft_field_t_y_to_w_thy(X, OmegaSF_z_pstxy, ypad, tpad)
    I_SF_w_thy = np.real(
        np.einsum('sxwy,sxwy->wy', OmegaSF_z_psxwThy[0,:,:,:,:], OmegaSF_z_psxwThy[1,:,:,::-1,::-1])
    )  

    ygrid_pad = X.ygrid + 2*ypad
    tgrid_pad = X.tgrid + 2*tpad
    womega_ar  = np.linspace(extent_w_thy[0], extent_w_thy[1], tgrid_pad)
    theta_y_ar = np.linspace(extent_w_thy[2], extent_w_thy[3], ygrid_pad)

    dtheta_y = theta_y_ar[1] - theta_y_ar[0] 
    I_int_thy_w = np.real(dtheta_y * np.einsum('wa -> w', I_SF_w_thy))
    I_thy0_w = np.real(I_SF_w_thy[:,int(ygrid_pad/2)])
    
        
    return womega_ar, I_int_thy_w, I_thy0_w 