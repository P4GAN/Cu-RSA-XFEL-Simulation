import numpy as np
import scipy.constants as sp_const
from scipy.interpolate import RegularGridInterpolator

from mini_ocelot.new_wave import *


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
    
    sigma_rx = X.config['pump_width_FWHM_x'] / (2 * np.sqrt(2*np.log(2))) 
    sigma_ry = X.config['pump_width_FWHM_y'] / (2 * np.sqrt(2*np.log(2)))
    sigma_t  = X.config['pump_duration_FWHM_t'] / (2 * np.sqrt(2*np.log(2)))
    
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
    
    # plt.figure()
    # plt.imshow(np.abs(field_txy[0,:,:])**2)
    # plt.figure()
    # #plt.plot(np.linspace(-SASE.Lz()/sp_const.c*1.e15, SASE.Lz()*1.e15/sp_const.c, X.tgrid), np.abs(field_txy[:,1,1])**2)
    # plt.plot(np.linspace(0, SASE.Lz()*1.e15/sp_const.c, X.tgrid), np.abs(field_txy[:,1,1])**2)
    # plt.xlabel('t (fs)')
    # plt.show()
        
    return np.sqrt(N_pump_photons) * field_txy#shift_P_txy(np.sqrt(X.N_pump_photons) * field_txy, X.t_pump_max, X.tmax)


def Gaussian_pulse_aniso_pump(X):
          
    sigma_rx = X.config['pump_width_FWHM_x'] / (2 * np.sqrt(2*np.log(2)))
    sigma_ry = X.config['pump_width_FWHM_y'] / (2 * np.sqrt(2*np.log(2)))
    sigma_t  = X.config['pump_duration_FWHM_t'] / (2 * np.sqrt(2*np.log(2)))
    
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
          
    seed_sigma_rx = X.config['seed_width_FWHM_x'] / (2 * np.sqrt(2*np.log(2)))
    seed_sigma_ry = X.config['seed_width_FWHM_y'] / (2 * np.sqrt(2*np.log(2)))
    seed_sigma_t  = X.config['seed_duration_FWHM_t'] / (2 * np.sqrt(2*np.log(2)))
    
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

def Roh(X):
    chi_h = -0.79955e-05 + 1j*0.24361e-06
    chi_mh = chi_h
    chi_0 = -0.15127e-04 + 1j*0.34955e-06
    theta_B = (np.pi/180) * 14.221
    
    t = X.t - (X.tmax/2)
    
    Tg = (2 * np.sin(theta_B)**2) / ( (X.hwKalpha1N/X.hbar) * np.sqrt(chi_h*chi_mh))
    exparg = - ( (X.hwKalpha1N/X.hbar) * np.imag(chi_0) * t) / (2 * np.sin(theta_B)**2)
    
    # plt.plot(np.real((sp_func.jv(1, t/Tg) / (1j*t)) * np.exp(exparg)))
    # plt.plot(np.imag((sp_func.jv(1, t/Tg) / (1j*t)) * np.exp(exparg)))
    # plt.show()
    
    return np.heaviside(t, 1) * (sp_func.jv(1, t/Tg) / (1j*t)) * np.exp(exparg)

def Ocelot_SASE_seed_220_dbm_pstxy(X):
    SASE = RadiationField()  # initialize RadiationField object
    
    seed_sigma_rx = X.config['seed_width_FWHM_x'] / (2 * np.sqrt(2*np.log(2)))
    seed_sigma_ry = X.config['seed_width_FWHM_y'] / (2 * np.sqrt(2*np.log(2)))
    seed_sigma_t  = X.config['seed_duration_FWHM_t'] / (2 * np.sqrt(2*np.log(2)))
    
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
        
    # plt.figure()
    # plt.imshow(np.abs(field_txy[0,:,:])**2)
    # plt.figure()
    # plt.plot(np.linspace(-SASE.Lz()/sp_const.c, SASE.Lz()/sp_const.c, X.tgrid), np.abs(field_txy[:,1,1])**2)
    # plt.show()
    
    # Effect of mono: convolution between field_t and Roh
    field_temp_t0 = np.einsum('txy -> t', field_txy) # extract t domain
    field_temp_t0 = roll_zeropad(field_temp_t0, int(X.seed_delay/X.dt))
    
    field_temp_xy = np.einsum('txy -> xy', field_txy)
    
    field_temp_t1 = sp_sign.convolve(field_temp_t0, Roh(X), mode='same')
    field_temp_t2 = sp_sign.convolve(field_temp_t1, Roh(X), mode='same')
    
    field_txy = np.einsum('t, xy -> txy', field_temp_t2, field_temp_xy)
    
    # plt.plot(np.abs(1e12*Roh(X))**2)
    # plt.show()
    # plt.plot(np.abs(field_temp_t0)**2)
    # plt.show()
    # plt.plot(np.abs(0.1*field_temp_t1)**2)
    # plt.show()
    # plt.plot(np.abs(0.01*field_temp_t2)**2)
    # plt.show()
    # plt.plot(np.abs(field_txy[:,0,0])**2)
    # plt.show()  
       
    
    # Normalization: it has to be compatible with the units used in the rest of
    # the code. imitate_sase_dfl uses SI units [m, s, J], need to normalize with
    # respect to [nm, fs, eV]
    
    # dx = 1e9 * (SASE.Lx() / (SASE.Nx()-1))
    # dy = 1e9 * (SASE.Ly() / (SASE.Ny()-1))
    # dt = 1e15 * (SASE.Lz() / (SASE.Nz()-1)) / sp_const.c
    
    norm = np.sqrt(np.sum(np.abs(field_txy)**2 * X.dx * X.dy * X.dt))
    
    field_txy = field_txy / norm
    
    # plt.figure()
    # plt.imshow(np.abs(field_txy[0,:,:])**2)
    # plt.figure()
    # #plt.plot(np.linspace(-SASE.Lz()/sp_const.c*1.e15, SASE.Lz()*1.e15/sp_const.c, X.tgrid), np.abs(field_txy[:,1,1])**2)
    # plt.plot(np.linspace(0, SASE.Lz()*1.e15/sp_const.c, X.tgrid), np.abs(field_txy[:,1,1])**2)
    # plt.xlabel('t (fs)')
    # plt.show()
    
    N_seed_photons = X.E_seed_uJ * 1e-6 / (X.hwKalpha1N * sp_const.e)
    print('number of seed photons = ' + f"{N_seed_photons:.1e}")
    
    SASE_for_seed = np.sqrt(N_seed_photons) * field_txy
    
    Omega_seed_pstxy = np.zeros((2, 2, X.tgrid, X.xgrid, X.ygrid), dtype=complex)
    fluxfield2Rabi = np.sqrt((3.0 * X.lambdaKalpha1N**2 * X.Gamma_sp_fsm1N / 8.0 / np.pi))
    Omega_seed_pstxy[0,1,:,:,:] = fluxfield2Rabi * SASE_for_seed # in linear polarization basis, along y-axis
    Omega_seed_pstxy[1,:,:,:,:] = np.conj(Omega_seed_pstxy[0,:,:,:,:])
        
    return Omega_seed_pstxy





def Ocelot_SASE_seed_pstxy(X):
    SASE = RadiationField()  # initialize RadiationField object
    
    seed_sigma_rx = X.config['seed_width_FWHM_x'] / (2 * np.sqrt(2*np.log(2)))
    seed_sigma_ry = X.config['seed_width_FWHM_y'] / (2 * np.sqrt(2*np.log(2)))
    seed_sigma_t  = X.config['seed_duration_FWHM_t'] / (2 * np.sqrt(2*np.log(2)))
    
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
    
    # Normalization: it has to be compatible with the units used in the rest of
    # the code. imitate_sase_dfl uses SI units [m, s, J], need to normalize with
    # respect to [nm, fs, eV]
    
    # dx = 1e9 * (SASE.Lx() / (SASE.Nx()-1))
    # dy = 1e9 * (SASE.Ly() / (SASE.Ny()-1))
    # dt = 1e15 * (SASE.Lz() / (SASE.Nz()-1)) / sp_const.c

    
    norm = np.sqrt(np.sum(np.abs(field_txy)**2 * X.dx * X.dy * X.dt))
    
    field_txy = field_txy / norm
    
    # plt.figure()
    # plt.imshow(np.abs(field_txy[0,:,:])**2)
    # plt.figure()
    # #plt.plot(np.linspace(-SASE.Lz()/sp_const.c*1.e15, SASE.Lz()*1.e15/sp_const.c, X.tgrid), np.abs(field_txy[:,1,1])**2)
    # plt.plot(np.linspace(0, SASE.Lz()*1.e15/sp_const.c, X.tgrid), np.abs(field_txy[:,1,1])**2)
    # plt.xlabel('t (fs)')
    # plt.show()
    
    N_seed_photons = X.E_seed_uJ * 1e-6 / (X.hwKalpha1N * sp_const.e)
    print('number of seed photons = ' + f"{N_seed_photons:.1e}")
    
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
    
    return X.dt * X.dx * X.dy * np.sum(X.j_3D, axis=(0, 1, 2))


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


