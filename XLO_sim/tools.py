import os
import resource
import sys
import time
import traceback
import multiprocessing as mp
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml
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
    
    kwargs={'xlamds':1e-9*X.lambdaCenter,                     # [m] - central wavelength
            'seed': X.random_seed,
            'shape':(X.xgrid, X.ygrid, X.tgrid),            # size of field matrix (x,y,z=ct) (number of points)
            'dgrid':(2e-9*X.xmax, 2e-9*X.ymax, 1e-15*X.tmax*sp_const.c),                # size of field grid (max value) 
            'power_rms':(1e-9*seed_sigma_rx, 1e-9*seed_sigma_ry, 1e-15*seed_sigma_t*sp_const.c),   # rms size of radiation distribution
            'power_center':(0,0,1e-15*X.t_peak*sp_const.c),  # (x,y,z) [m] - position of the radiation distribution
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
    
    kwargs={'xlamds':1e-9*X.lambdaCenter,                     # [m] - central wavelength
            'seed': X.random_seed,
            'shape':(X.xgrid, X.ygrid, X.tgrid),            # size of field matrix (x,y,z=ct) (number of points)
            'dgrid':(2e-9*X.xmax, 2e-9*X.ymax, 1e-15*X.tmax*sp_const.c),                # size of field grid (max value)
            'power_rms':(1e-9*seed_sigma_rx, 1e-9*seed_sigma_ry, 1e-15*seed_sigma_t*sp_const.c),   # rms size of radiation distribution
            'power_center':(0,0,1e-15*X.t_peak*sp_const.c),  # (x,y,z) [m] - position of the radiation distribution
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


def Gaussian_from_mesh(X, mesh, moments):
    
    omega_nn, th_nxx, th_nyy = mesh
    sigma_w, sigma_th_x, sigma_th_y = moments
    
    return -1j * np.exp(-1.0 * (th_nxx**2 / 2.0 / sigma_th_x**2 + th_nyy**2 / 2.0 / sigma_th_y**2) ) * np.exp(-(omega_nn - 0.0)**2 / 2.0 / sigma_w**2)
    

def uniform_field_txy(X, Nphotons):

    eta = X.dt * X.dx * X.dy * X.xgrid * X.ygrid * X.tgrid / (3.0 * X.lambdaKalpha1N**2 * X.Gamma_sp_fsm1N / 8.0 / np.pi) 
    
    return 0.0 * X.x_mesh + np.sqrt(Nphotons / eta)


def uniform_seed_field_txy(X, Nphotons_spol, Nphotons_ppol, Dphi=0):

    uniform_seed_field = np.zeros((2, 2, X.tgrid, X.xgrid, X.ygrid), dtype=complex)

    field_spol = uniform_field_txy(X, Nphotons_spol)
    field_ppol = np.exp(1j * Dphi) * uniform_field_txy(X, Nphotons_ppol)

    uniform_seed_field[0, 0, :, :, :] = field_spol
    uniform_seed_field[1, 0, :, :, :] = np.conj(field_spol)
    uniform_seed_field[0, 1, :, :, :] = field_ppol 
    uniform_seed_field[1, 1, :, :, :] = np.conj(field_ppol)

    return uniform_seed_field
    

def Gaussian_seed_field_txy(X, Nphotons_spol, Nphotons_ppol, Dphi=0):

    #what's wrong with this function?

    Gaussian_seed_field = np.zeros((2, 2, X.tgrid, X.xgrid, X.ygrid), dtype=complex)

    field_spol = Gaussian_pulse_3D_with_q(X, X.k0, Nphotons_spol)
    field_ppol = np.exp(1j * Dphi) * Gaussian_pulse_3D_with_q(X, X.k0, Nphotons_ppol)

    Gaussian_seed_field[0, 0, :, :, :] = field_spol
    Gaussian_seed_field[1, 0, :, :, :] = np.conj(field_spol)
    Gaussian_seed_field[0, 1, :, :, :] = field_ppol
    Gaussian_seed_field[1, 1, :, :, :] = np.conj(field_ppol)

    return Gaussian_seed_field


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



def fft_field_t_y_to_w_thy(X, field_pstxy, ypad, tpad, window_alpha=0.25):
    """
    Fourier transform a field from the (time, y) domain to the (omega, theta_y) domain.

    Applies a Tukey window to the t and y axes, zero-pads them, then FFTs
    along them and applies a linear phase correction for the fact that the
    simulation window is not centered at t=0/y=0 (it is centered at
    X.t_peak/X.ymax), so the resulting energy/angle map has the correct
    phase reference.

    The window is needed because zero-padding a field that has not decayed
    to (numerical) zero by the edge of the simulation grid creates a step
    discontinuity there; that discontinuity's Fourier transform decays only
    algebraically (Gibbs-type leakage), contaminating the spectrum/angular
    map at large detuning/angle with a numerical artifact rather than the
    physical (typically much faster decaying) tail. Tapering the field to
    zero before padding removes that discontinuity.

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
    window_alpha : float
        Shape parameter of the Tukey window applied to the t and y axes
        before padding (0 = no tapering/rectangular window, 1 = Hann window).

    Returns
    -------
    extent_w_thy : list of float
        [omega_min, omega_max, theta_y_min, theta_y_max] (omega in eV, theta_y in mrad),
        the plot extent of the transformed field.
    field_psxwThy : np.ndarray
        The FFT'd, phase-corrected field with axes (p, s, x, omega, theta_y).

    """

    coeffs = (2.0 * np.pi * X.hbar, 2.0 * np.pi / X.k0)
    shifts = (X.t_peak + tpad * X.dt, X.ymax + ypad * X.dy)
    field_psxty = np.einsum('pstxy->psxty',field_pstxy)

    window_t = sp_sign.windows.tukey(X.tgrid, alpha=window_alpha)
    window_y = sp_sign.windows.tukey(X.ygrid, alpha=window_alpha)
    field_psxty = field_psxty * window_t[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis] \
                               * window_y[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

    pad_shape = [(0, 0), (0, 0), (0, 0), (tpad, tpad), (ypad, ypad)]
    domains_sim, meshes_sim, step_sizes_sim = X.optics.nd_kspace(coeffs, (X.tgrid, X.ygrid), (tpad, ypad), (X.dt, X.dy))
    phasors = [np.exp(1j * 2.0 * np.pi * mesh * shift / coeff) for coeff, mesh, shift in zip(coeffs, meshes_sim, shifts)]

    field_psxty_pad = X.optics.my_pad(field_psxty, pad_shape)
    field_fft_pad_phased = X.optics.my_fft_phased(field_psxty_pad, (3, 4), phasors)

    extent_w_thy = [
        np.min(domains_sim[0][tpad:X.tgrid + tpad]), np.max(domains_sim[0][tpad:X.tgrid + tpad]),
        1e3*np.min(domains_sim[1]), 1e3*np.max(domains_sim[1])]

    return extent_w_thy, field_fft_pad_phased 



def SF_spectrum_w(X, zint, ypad, tpad, window_alpha=0.25):
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
    theta_y) domain via `fft_field_t_y_to_w_thy`, and forms the intensity as
    |FFT(Omega)|^2 from the p=0 (Omega) component and its conjugate (rather
    than using the stored p=1 (Omega*) component, whose transform would need
    reflecting about zero frequency/angle to line up -- exact only for the
    interior bins of an even-length FFT grid, not the Nyquist edge bins).
    Returns the spectrum both integrated over theta_y and evaluated on-axis
    (theta_y = 0).

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
    window_alpha : float
        Tukey window shape parameter applied before padding/FFT (see `fft_field_t_y_to_w_thy`).

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
    extent_w_thy, OmegaSF_z_psxwThy = fft_field_t_y_to_w_thy(X, OmegaSF_z_pstxy, ypad, tpad, window_alpha)
    I_SF_w_thy = np.real(
        np.einsum('sxwy,sxwy->wy', OmegaSF_z_psxwThy[0,:,:,:,:], np.conj(OmegaSF_z_psxwThy[0,:,:,:,:]))
    )

    ygrid_pad = X.ygrid + 2*ypad
    tgrid_pad = X.tgrid + 2*tpad
    womega_ar  = np.linspace(extent_w_thy[0], extent_w_thy[1], tgrid_pad)
    theta_y_ar = np.linspace(extent_w_thy[2], extent_w_thy[3], ygrid_pad)

    dtheta_y = theta_y_ar[1] - theta_y_ar[0] 
    I_int_thy_w = np.real(dtheta_y * np.einsum('wa -> w', I_SF_w_thy))
    I_thy0_w = np.real(I_SF_w_thy[:,int(ygrid_pad/2)])


    return womega_ar, I_int_thy_w, I_thy0_w


def compute_run_outputs(X, tpad, ypad):
    """Post-process one finished XLO_sim repetition into the arrays the sweep
    scripts save. Shared by run_mono_sweep.py, run_intensity_sweep.py,
    run_duration_sweep.py and run_convergence_sweep.py, which differ only in
    how they seed/configure X before calling X.run_3D().

    rho_eg_t_last is contracted directly from a single (x, y, z) point of
    X.rho_ijtxyz rather than via the full-grid P_pstxyz = einsum(...) used
    historically -- same result (P_pstxyz's only consumer was that one
    point), but avoids materializing a (p, s, t, x, y, z) array per worker
    just to throw away everything but one (s, t) slice.
    """
    womega_ar, I_int_thy_w_0, I_thy0_w_0 = SF_spectrum_w(X, 0, ypad, tpad)
    womega_ar, I_int_thy_w_last, I_thy0_w_last = SF_spectrum_w(X, -1, ypad, tpad)

    # Lean (keep_z_history=False) runs only store a length-1 (x,y) footprint at the center pixel;
    # clamping resolves to the true center in full mode and to 0 (where lean mode wrote it) otherwise.
    cx = min(int(X.xgrid / 2), X.rho_ijtxyz.shape[3] - 1)
    cy = min(int(X.ygrid / 2), X.rho_ijtxyz.shape[4] - 1)
    rho_ijt_center = X.rho_ijtxyz[:, :, :, cx, cy, -1]

    I_t_0 = np.einsum('stxy,stxy->t', X.Omega_pstxyz[0, :, :, :, :, 0], X.Omega_pstxyz[1, :, :, :, :, 0])
    I_t_last = np.einsum('stxy,stxy->t', X.Omega_pstxyz[0, :, :, :, :, -1], X.Omega_pstxyz[1, :, :, :, :, -1])
    rho_ee_t_last = np.einsum('ijs, jkt, kis-> t', X.Tijs_minus, rho_ijt_center, X.Tijs_plus, optimize=True)
    rho_gg_t_last = np.einsum('ijs, jkt, kis-> t', X.Tijs_plus, rho_ijt_center, X.Tijs_minus, optimize=True)
    rho_eg_t_last = np.einsum('ijs,jit->st', X.Tijs_minus, rho_ijt_center, optimize=True)[0]
    rho_ground_t_last = X.rho_ground_txyz[:, cx, cy, -1]
    rho_other_t_last = X.rho_other_txyz[:, cx, cy, -1]
    rho_2s_t_last = X.rho_2s_txyz[:, cx, cy, -1]
    t_axis = X.t

    # rho_gg_t_last/rho_eg_t_last sum the L3 (Kalpha1) and L2 (Kalpha2) contributions together when
    # use_L2_pathway is on. The keys below isolate each manifold/coherence individually instead, by
    # masking Tijs_plus/Tijs_minus with ei_L3/ei_L2 (zero, hence harmless, when L2 is off).
    Tijs_plus_L3 = X.Tijs_plus * X.ei_L3[None, :, None]
    Tijs_minus_L3 = X.Tijs_minus * X.ei_L3[:, None, None]
    Tijs_plus_L2 = X.Tijs_plus * X.ei_L2[None, :, None]
    Tijs_minus_L2 = X.Tijs_minus * X.ei_L2[:, None, None]
    rho_l3_t_last = np.einsum('ijs, jkt, kis-> t', Tijs_plus_L3, rho_ijt_center, Tijs_minus_L3, optimize=True)
    rho_l2_t_last = np.einsum('ijs, jkt, kis-> t', Tijs_plus_L2, rho_ijt_center, Tijs_minus_L2, optimize=True)
    rho_eg_l3_t_last = np.einsum('ijs,jit->st', Tijs_minus_L3, rho_ijt_center, optimize=True).sum(axis=0)
    rho_eg_l2_t_last = np.einsum('ijs,jit->st', Tijs_minus_L2, rho_ijt_center, optimize=True).sum(axis=0)

    # Stacked into (n_sat, t) arrays, rather than a name-keyed dict, so accumulate_run_outputs'
    # np.stack/np.isnan reduction applies to these like any other per-repetition key;
    # satellite_channel_names carries the per-row labels as a run-level (non-accumulated) axis.
    n_sat = len(X.satellite_channel_params)
    satellite_channel_names = tuple(chan.name for chan in X.satellite_channel_params)
    rho_ee_t_last_sat = np.zeros((n_sat, X.tgrid), dtype=complex)
    rho_gg_t_last_sat = np.zeros((n_sat, X.tgrid), dtype=complex)
    rho_eg_t_last_sat = np.zeros((n_sat, X.tgrid), dtype=complex)

    # Per-channel L3-only/L2-only counterparts of rho_l3_t_last/rho_l2_t_last above, built once
    # since every channel shares the same local template.
    Tijs_plus_L3_sat = X.Tijs_plus_satellite * X.ei_L3_satellite[None, :, None]
    Tijs_minus_L3_sat = X.Tijs_minus_satellite * X.ei_L3_satellite[:, None, None]
    Tijs_plus_L2_sat = X.Tijs_plus_satellite * X.ei_L2_satellite[None, :, None]
    Tijs_minus_L2_sat = X.Tijs_minus_satellite * X.ei_L2_satellite[:, None, None]
    rho_l3_t_last_sat = np.zeros((n_sat, X.tgrid), dtype=complex)
    rho_l2_t_last_sat = np.zeros((n_sat, X.tgrid), dtype=complex)
    rho_eg_l3_t_last_sat = np.zeros((n_sat, X.tgrid), dtype=complex)
    rho_eg_l2_t_last_sat = np.zeros((n_sat, X.tgrid), dtype=complex)

    for k, rho_sat_ijtxyz in enumerate(X.rho_sat_ijtxyz):
        rho_sat_ijt_center = rho_sat_ijtxyz[:, :, :, cx, cy, -1]
        rho_ee_t_last_sat[k] = np.einsum(
            'ijs, jkt, kis-> t', X.Tijs_minus_satellite, rho_sat_ijt_center, X.Tijs_plus_satellite, optimize=True)
        rho_gg_t_last_sat[k] = np.einsum(
            'ijs, jkt, kis-> t', X.Tijs_plus_satellite, rho_sat_ijt_center, X.Tijs_minus_satellite, optimize=True)
        rho_eg_t_last_sat[k] = np.einsum(
            'ijs,jit->st', X.Tijs_minus_satellite, rho_sat_ijt_center, optimize=True)[0]
        rho_l3_t_last_sat[k] = np.einsum(
            'ijs, jkt, kis-> t', Tijs_plus_L3_sat, rho_sat_ijt_center, Tijs_minus_L3_sat, optimize=True)
        rho_l2_t_last_sat[k] = np.einsum(
            'ijs, jkt, kis-> t', Tijs_plus_L2_sat, rho_sat_ijt_center, Tijs_minus_L2_sat, optimize=True)
        rho_eg_l3_t_last_sat[k] = np.einsum(
            'ijs,jit->st', Tijs_minus_L3_sat, rho_sat_ijt_center, optimize=True).sum(axis=0)
        rho_eg_l2_t_last_sat[k] = np.einsum(
            'ijs,jit->st', Tijs_minus_L2_sat, rho_sat_ijt_center, optimize=True).sum(axis=0)

    return {
        "womega_ar": womega_ar,
        "I_int_thy_w_0": I_int_thy_w_0,
        "I_int_thy_w_last": I_int_thy_w_last,
        "I_thy0_w_0": I_thy0_w_0,
        "I_thy0_w_last": I_thy0_w_last,
        "I_t_0": I_t_0,
        "I_t_last": I_t_last,
        "rho_ee_t_last": rho_ee_t_last,
        "rho_gg_t_last": rho_gg_t_last,
        "rho_eg_t_last": rho_eg_t_last,
        "rho_l3_t_last": rho_l3_t_last,
        "rho_l2_t_last": rho_l2_t_last,
        "rho_eg_l3_t_last": rho_eg_l3_t_last,
        "rho_eg_l2_t_last": rho_eg_l2_t_last,
        "rho_ground_t_last": rho_ground_t_last,
        "rho_other_t_last": rho_other_t_last,
        "rho_2s_t_last": rho_2s_t_last,
        "rho_ee_t_last_sat": rho_ee_t_last_sat,
        "rho_gg_t_last_sat": rho_gg_t_last_sat,
        "rho_eg_t_last_sat": rho_eg_t_last_sat,
        "rho_l3_t_last_sat": rho_l3_t_last_sat,
        "rho_l2_t_last_sat": rho_l2_t_last_sat,
        "rho_eg_l3_t_last_sat": rho_eg_l3_t_last_sat,
        "rho_eg_l2_t_last_sat": rho_eg_l2_t_last_sat,
        "satellite_channel_names": satellite_channel_names,
        "t_axis": t_axis,
    }


# Arrays that are deterministic functions of the grid config, not the random
# SASE draw -- identical across every repetition of a config, so they are
# saved as-is rather than accumulated into a sum/sumsq pair.
RUN_OUTPUT_AXIS_KEYS = ("womega_ar", "t_axis", "satellite_channel_names")


def accumulate_run_outputs(results):
    """Fold a list of per-repetition output dicts (from compute_run_outputs)
    into one dict of summary statistics, saved once per SLURM array task
    instead of once per repetition.

    Every non-axis array is reduced to '<key>_sum' (sum over repetitions,
    NaN entries excluded), '<key>_sumsq' (sum of |value|**2 over the same
    non-NaN entries -- real-valued even for complex inputs, since what
    convergence/error-bar checks need here is the spread of each
    quantity's magnitude, not its complex-valued "variance"), and
    '<key>_count' (element-wise count of repetitions that contributed a
    non-NaN value -- an array, not a scalar, since a numerically unstable
    repetition can produce NaN at only some (t/w) indices of a key while
    leaving the rest finite). 'n_reps' is the plain repetition count
    (attempted, not just valid). Because sums/counts are additive, multiple
    chunks' saved accumulators (e.g. from different array tasks writing
    into the same runs_.../ folder) can be combined losslessly later by
    summing again and dividing sum by count -- see data_from_folder().
    """
    acc = {"n_reps": len(results)}
    for key in results[0]:
        if key in RUN_OUTPUT_AXIS_KEYS:
            acc[key] = results[0][key]
            continue
        stacked = np.stack([r[key] for r in results])
        valid = ~np.isnan(stacked)
        n_bad = int((~valid).sum())
        if n_bad:
            print(f"Warning: {key} has {n_bad} NaN value(s) across {len(results)} repetitions "
                  f"-- excluded from the accumulated sum", flush=True)
        finite = np.where(valid, stacked, 0)
        acc[f"{key}_sum"] = finite.sum(axis=0)
        acc[f"{key}_sumsq"] = (np.abs(finite) ** 2).sum(axis=0)
        acc[f"{key}_count"] = valid.sum(axis=0)
    return acc


def format_duration(seconds):
    """Format a duration in seconds as e.g. '1h 02m 05.3s', '2m 05.3s', or '45.3s'."""
    hours, rem = divmod(float(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{int(hours)}h {int(minutes):02d}m {secs:04.1f}s"
    if minutes:
        return f"{int(minutes)}m {secs:04.1f}s"
    return f"{secs:.1f}s"


def peak_memory_gb(who=resource.RUSAGE_SELF):
    """Peak RSS so far, in GiB, for this process (default) or its terminated
    children (who=resource.RUSAGE_CHILDREN -- only reflects a child once it has
    exited and been reaped, e.g. after ProcessPoolExecutor.shutdown(wait=True)).
    resource.getrusage's ru_maxrss is KiB on Linux (the SLURM/Maxwell case this
    matters for) but bytes on macOS/BSD -- normalized here either way."""
    ru_maxrss = resource.getrusage(who).ru_maxrss
    kb = ru_maxrss / 1024 if sys.platform == "darwin" else ru_maxrss
    return kb / (1024 ** 2)


def run_sweep_chunk(run_simulation, yaml_path, reps, run_path, output_stem, nproc,
                     checkpoint_every=None, ctx_method="fork"):
    """Run `reps` repetitions of `run_simulation(yaml_path, rep)` across a
    process pool and save the accumulated (sum/sumsq/count) result. Shared
    by all run_*_sweep.py scripts.

    Written to survive a failure mode seen on the SLURM cluster these run
    on: a job whose per-repetition logs show every repetition has started
    (or even finished), yet the job itself hangs past its --time limit
    with nothing written to disk. Two things about the old
    multiprocessing.Pool.starmap-based loop caused that:

    - If a worker dies unexpectedly (most plausibly OOM-killed by the
      kernel -- these sweeps run dozens of multi-GB worker processes per
      node), Pool.starmap blocks forever waiting for a result that will
      never arrive: nothing is logged, no exception is raised, and the
      job just sits until SLURM's wall-clock kill fires. This uses
      concurrent.futures.ProcessPoolExecutor instead, which detects the
      dead worker and raises BrokenProcessPool on the next result fetch,
      so the failure is visible in the job's .err log instead of being
      indistinguishable from an ordinary timeout.
    - The accumulated sum/sumsq/count of whatever repetitions have
      completed so far is checkpointed to '<output_stem>.partial.npz'
      every `checkpoint_every` completions (default: nproc, i.e. roughly
      once per parallel wave). If the job is killed anyway (e.g. a
      repetition that's genuinely stuck rather than dead, which a broken
      pool won't catch), the repetitions completed up to the last
      checkpoint are still on disk -- data_from_folder() already sums
      every .npz chunk file it finds in a runs_.../ folder, so a leftover
      partial file is picked up automatically instead of losing the whole
      chunk's work. The partial file is removed once the chunk finishes
      successfully and its final file is written, so a later successful
      retry of the same chunk doesn't double-count it.

    Parameters
    ----------
    run_simulation : callable(yaml_path, rep) -> dict
        Per-repetition worker function (e.g. run_mono_sweep.run_simulation).
        Must be a module-level function so it can be pickled to workers.
    yaml_path : str
    reps : sequence of int
    run_path : str
        Folder to write '<output_stem>_<timestamp>.npz' /
        '<output_stem>.partial.npz' into.
    output_stem : str
        Filename stem, e.g. 'run_at_seed_30.0_uJ__reps_0-20'.
    nproc : int
        Worker processes.
    checkpoint_every : int, optional
        Completions between checkpoint writes (default: nproc).
    ctx_method : str
        multiprocessing start method (default 'fork', matching the
        notebooks this is a batch-job counterpart of).

    Returns
    -------
    str
        Path to the final saved .npz file.
    """
    checkpoint_every = checkpoint_every or max(1, nproc)
    partial_path = os.path.join(run_path, f"{output_stem}.partial.npz")

    chunk_t0 = time.perf_counter()
    ctx = mp.get_context(ctx_method)
    results = []
    executor = ProcessPoolExecutor(max_workers=nproc, mp_context=ctx)
    futures = {executor.submit(run_simulation, yaml_path, rep): rep for rep in reps}
    try:
        for i, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if i % checkpoint_every == 0 and i < len(reps):
                elapsed = time.perf_counter() - chunk_t0
                np.savez_compressed(partial_path, **accumulate_run_outputs(results))
                print(f"checkpoint: {len(results)}/{len(reps)} repetitions done "
                      f"({format_duration(elapsed)} elapsed, {format_duration(elapsed / len(results))}/rep avg) "
                      f"-> {partial_path}", flush=True)
    except Exception as e:
        elapsed = time.perf_counter() - chunk_t0
        print(f"{type(e).__name__} after {len(results)}/{len(reps)} completed repetitions "
              f"({format_duration(elapsed)} elapsed) -- a worker likely died (e.g. OOM-killed) or raised; "
              f"traceback follows:", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        if results:
            np.savez_compressed(partial_path, **accumulate_run_outputs(results))
            print(f"Saved {len(results)} completed repetitions to {partial_path} before re-raising", flush=True)
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    elapsed = time.perf_counter() - chunk_t0
    # RUSAGE_CHILDREN is only accurate once every worker has been reaped, which
    # shutdown(wait=True) above just guaranteed -- this is the peak RSS any
    # single worker process hit, maxed over every repetition this chunk ran.
    print(f"chunk finished: {len(results)} repetitions in {format_duration(elapsed)} "
          f"({format_duration(elapsed / max(1, len(results)))}/rep avg, {nproc} processes, "
          f"peak worker mem {peak_memory_gb(resource.RUSAGE_CHILDREN):.2f} GB)", flush=True)

    date_string = np.datetime_as_string(np.datetime64('now'))
    final_path = os.path.join(run_path, f"{output_stem}_{date_string}.npz")
    np.savez_compressed(final_path, **accumulate_run_outputs(results))
    if os.path.exists(partial_path):
        os.remove(partial_path)
    return final_path


def data_from_folder(folder_path, group_keys, aux_keys=(), reverse=True):
    """Aggregate a sweep's runs_.../ subfolders of accumulate_run_outputs()
    chunk files into per-group mean (and std, where defined) arrays.

    Each runs_.../ subfolder may contain several chunk .npz files (one per
    SLURM array task that contributed repetitions to that group); their
    sum/sumsq/count are combined by summing and dividing sum (and sumsq) by
    the combined element-wise count, so a group's mean is correct even if
    its chunks covered different repetition counts or had NaN entries
    excluded at different (t/w) indices.

    Chunk files predate '<key>_count' (see accumulate_run_outputs) fall
    back to that chunk's plain n_reps as the divisor for every element of
    that key; if such an old chunk's sum/sumsq themselves contain NaN (from
    before the run-level NaN fix), that chunk's contribution is dropped at
    just the affected elements rather than poisoning the whole group's
    mean.

    Parameters
    ----------
    folder_path : str
        Sweep's top-level output directory (contains runs_.../ subfolders).
    group_keys : str or tuple of str
        YAML key(s), read from the config copied into each runs_.../
        folder, used to group/nest results -- e.g. 'E_seed_uJ', or
        ('E_seed_uJ', 'monochromator_target_energy_eV') for a nested
        (E_seed, target energy) grouping. A tuple nests one dict level per
        key, in order.
    aux_keys : sequence of str
        YAML keys to return as a flat aux_data list, read from whichever
        runs_.../ folder is processed last (matching the sweep convention
        that these are constant across the whole sweep).
    reverse : bool
        Sort direction for the outermost group level (nested levels, if
        any, always sort ascending). Default True (descending) matches the
        existing E_seed/duration-descending plotting convention.

    Returns
    -------
    sorted_accumulated_arrays : OrderedDict
        Nested (len(group_keys) levels deep) dict of {array_name: mean
        array} (plus '<array_name>_std' for every accumulated, non-axis
        array), sorted by group_keys.
    aux_data : list
        Values of aux_keys from the last-processed folder.
    total_n_reps : int
        Combined repetition count of the last-processed group (all groups
        share the same count for a well-formed sweep).
    """
    if isinstance(group_keys, str):
        group_keys = (group_keys,)

    accumulated = {}
    aux_data = None
    total_n_reps = None

    for runs_folder in os.listdir(folder_path):
        runs_path = os.path.join(folder_path, runs_folder)
        if not os.path.isdir(runs_path):
            continue

        yaml_file = next((f for f in os.listdir(runs_path) if f.endswith('.yaml')), None)
        if yaml_file is None:
            print(f"No YAML file found in {runs_path}")
            continue
        with open(os.path.join(runs_path, yaml_file), 'r') as f:
            yaml_data = yaml.safe_load(f)

        group_values = tuple(yaml_data[k] for k in group_keys)
        aux_data = [yaml_data[k] for k in aux_keys]

        sums, sumsqs, counts, axes = {}, {}, {}, {}
        n_reps = 0
        for file_name in os.listdir(runs_path):
            if not file_name.endswith('.npz'):
                continue
            data = np.load(os.path.join(runs_path, file_name))
            file_n_reps = int(data['n_reps'])
            n_reps += file_n_reps

            sum_keys = {n[:-len('_sum')] for n in data.files if n.endswith('_sum')}
            for key in sum_keys:
                chunk_sum = data[f"{key}_sum"]
                chunk_count = (data[f"{key}_count"] if f"{key}_count" in data.files
                               else np.full(chunk_sum.shape, file_n_reps, dtype=float))
                bad = np.isnan(chunk_sum)
                if np.any(bad):
                    chunk_sum = np.where(bad, 0, chunk_sum)
                    chunk_count = np.where(bad, 0, chunk_count)
                sums[key] = sums.get(key, 0) + chunk_sum
                counts[key] = counts.get(key, 0) + chunk_count
                if f"{key}_sumsq" in data.files:
                    chunk_sumsq = np.where(bad, 0, data[f"{key}_sumsq"])
                    sumsqs[key] = sumsqs.get(key, 0) + chunk_sumsq

            for array_name in data.files:
                if array_name in ('n_reps',) or array_name.endswith(('_sum', '_sumsq', '_count')):
                    continue
                axes[array_name] = data[array_name]

        if n_reps == 0:
            print(f"No .npz chunk files found in {runs_path}")
            continue

        group_arrays = dict(axes)
        for key, total_sum in sums.items():
            count = counts[key]
            n_starved = int((count == 0).sum())
            if n_starved:
                print(f"Warning: {key} in {runs_path} has {n_starved} element(s) with no valid "
                      f"(non-NaN) repetitions -- left as NaN")
            count_safe = np.where(count > 0, count, 1)
            mean = np.where(count > 0, total_sum / count_safe, np.nan)
            group_arrays[key] = mean
            if key in sumsqs:
                variance = np.where(count > 0, sumsqs[key] / count_safe - np.abs(mean) ** 2, np.nan)
                group_arrays[f"{key}_std"] = np.sqrt(np.clip(variance, 0, None))

        node = accumulated
        for gv in group_values[:-1]:
            node = node.setdefault(gv, {})
        node[group_values[-1]] = group_arrays
        total_n_reps = n_reps

    def sort_level(level_dict, depth):
        items = sorted(level_dict.items(), key=lambda kv: kv[0], reverse=(reverse if depth == 0 else False))
        if depth + 1 < len(group_keys):
            return OrderedDict((k, sort_level(v, depth + 1)) for k, v in items)
        return OrderedDict(items)

    sorted_accumulated_arrays = sort_level(accumulated, 0)
    return sorted_accumulated_arrays, aux_data, total_n_reps 