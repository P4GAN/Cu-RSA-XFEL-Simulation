"""
Incident X-ray pulses. Each generator returns the field at the sample entrance as a Rabi frequency
Omega[p, s, t, x, y] (fs^-1): p = 0 is the field and p = 1 its complex conjugate, s = 0, 1 the x and y
linear polarisations. Every pulse is polarised along y and normalised to E_seed_uJ.

The config key `pulse` picks the generator (make_pulse): gaussian, sase, or sase_dcm (a SASE pulse
through the Si(111) double-crystal monochromator of the self-seeded experiment).
"""

import logging

import numpy as np
import scipy.constants as sp_const
import scipy.signal as sp_sign
import scipy.special as sp_func
from ocelot.optics.new_wave import imitate_sase_dfl

logging.getLogger('ocelot').setLevel(logging.CRITICAL + 1)

FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))


def make_pulse(X):
    """The pulse named by the config key `pulse`."""
    generators = {'gaussian': gaussian_pulse, 'sase': sase_pulse, 'sase_dcm': sase_dcm_pulse}
    if X.pulse not in generators:
        raise ValueError(f"pulse must be one of {sorted(generators)}, got {X.pulse!r}")
    return generators[X.pulse](X)


def gaussian_pulse(X):
    """Transform-limited Gaussian pulse at the Kalpha1 line, centred in the time window."""
    sigma_x = X.seed_width_FWHM_x / FWHM_PER_SIGMA
    sigma_y = X.seed_width_FWHM_y / FWHM_PER_SIGMA
    sigma_t = X.seed_duration_FWHM_t / FWHM_PER_SIGMA
    field_txy = np.sqrt(
        1 / (2.0 * np.pi * sigma_x * sigma_y)
        / (np.sqrt(2.0 * np.pi) * sigma_t)
        * np.exp(-1.0 * X.x_mesh**2 / 2.0 / sigma_x**2)
        * np.exp(-1.0 * X.y_mesh**2 / 2.0 / sigma_y**2)
        * np.exp(-(X.t_mesh - X.t0)**2 / 2.0 / sigma_t**2)
    )
    return _rabi_field(X, field_txy)


def sase_pulse(X):
    """One SASE shot from OCELOT (random spikes in time, random seed X.random_seed), centred at
    seed_center_E with relative bandwidth seed_FEL_bandwidth and peak at t_peak."""
    return _rabi_field(X, _ocelot_sase(X))


def sase_dcm_pulse(X):
    """One SASE shot after the Si(111) double-crystal monochromator tuned to monochromator_target_energy_eV:
    the on-axis time profile is convolved twice with the crystal's reflection response."""
    field_txy = _ocelot_sase(X)
    field_t = _delay(np.einsum('txy -> t', field_txy), int(X.seed_delay / X.dt))
    field_xy = np.einsum('txy -> xy', field_txy)
    response = dcm_response(X, X.monochromator_target_energy_eV)
    field_t = sp_sign.convolve(field_t, response, mode='same')
    field_t = sp_sign.convolve(field_t, response, mode='same')
    return _rabi_field(X, np.einsum('t, xy -> txy', field_t, field_xy))


def _ocelot_sase(X):
    """SASE field (t, x, y) from OCELOT's imitate_sase_dfl, in its own units; lengths in m, time as c t."""
    lambda_centre_nm = 2.0 * np.pi * X.c * X.hbar / X.seed_center_E
    sase = imitate_sase_dfl(
        xlamds=1e-9 * lambda_centre_nm,
        seed=X.random_seed,
        shape=(X.xgrid, X.ygrid, X.tgrid),
        dgrid=(2e-9 * X.xmax, 2e-9 * X.ymax, 1e-15 * X.tmax * sp_const.c),
        power_rms=(1e-9 * (X.seed_width_FWHM_x / FWHM_PER_SIGMA), 1e-9 * (X.seed_width_FWHM_y / FWHM_PER_SIGMA),
                   1e-15 * (X.seed_duration_FWHM_t / FWHM_PER_SIGMA) * sp_const.c),
        power_center=(0, 0, 1e-15 * X.t_peak * sp_const.c),
        power_angle=(0, 0),
        power_waistpos=(0, 0),
        wavelength=None,
        zsep=None,
        freq_chirp=0,
        en_pulse=X.E_seed_uJ * 1e-6,
        power=None,
        rho=X.seed_FEL_bandwidth / 2,
    )
    return sase.fld


def _rabi_field(X, field_txy):
    """Scale a field (t, x, y) of any normalisation to E_seed_uJ worth of Kalpha1 photons and convert it to
    the Rabi-frequency field Omega[p, s, t, x, y], polarised along y."""
    norm = np.sqrt(np.sum(np.abs(field_txy)**2 * X.dx * X.dy * X.dt))
    field_txy = field_txy / norm
    N_photons = X.E_seed_uJ * 1e-6 / (X.hwKalpha1N * sp_const.e)
    Omega_pstxy = np.zeros((2, 2, X.tgrid, X.xgrid, X.ygrid), dtype=complex)
    Omega_pstxy[0, 1, :, :, :] = np.sqrt(X.flux_factor) * (np.sqrt(N_photons) * field_txy)
    Omega_pstxy[1, :, :, :, :] = np.conj(Omega_pstxy[0, :, :, :, :])
    return Omega_pstxy


def dcm_response(X, target_energy_eV):
    """Time-domain amplitude response of one symmetric Si(111) Bragg reflection set to target_energy_eV, in
    the frame rotating at the Kalpha1 frequency (Lindberg & Shvyd'ko, PRSTAB 15, 100702 (2012))."""
    d_hkl_A = 5.4310 / np.sqrt(3)
    chi_h = -0.79955e-05 + 1j * 0.24361e-06
    chi_mh = chi_h
    chi_0 = -0.15127e-04 + 1j * 0.34955e-06

    hc_eVA = 12398.42
    theta_B = np.arcsin(hc_eVA / (2 * d_hkl_A * target_energy_eV))
    omega_Bragg = target_energy_eV / X.hbar
    domega = (target_energy_eV - X.hwKalpha1N) / X.hbar

    t = X.t - (X.tmax / 2)
    Tg = (2 * np.sin(theta_B)**2) / (omega_Bragg * np.sqrt(chi_h * chi_mh))
    exparg = -(omega_Bragg * np.imag(chi_0) * t) / (2 * np.sin(theta_B)**2)
    return np.heaviside(t, 1) * (sp_func.jv(1, t / Tg) / (1j * t)) * np.exp(exparg) * np.exp(1j * domega * t)


def _delay(field_t, shift):
    """field_t shifted by `shift` samples, zero-filled."""
    if shift == 0:
        return field_t
    out = np.zeros_like(field_t)
    if abs(shift) >= len(field_t):
        return out
    if shift > 0:
        out[shift:] = field_t[:-shift]
    else:
        out[:shift] = field_t[-shift:]
    return out


def linear_to_circular(X, field_linear):
    """Field (p, s, t, x, y) from the linear (x, y) to the circular polarisation basis the atoms use."""
    field_circular = np.zeros_like(field_linear)
    field_circular[0] = np.einsum('stxy,qs->qtxy', field_linear[0], np.linalg.inv(X.transform_matrix))
    field_circular[1] = np.einsum('stxy,qs->qtxy', field_linear[1], np.conj(np.linalg.inv(X.transform_matrix)))
    return field_circular


def circular_to_linear(X, field_circular):
    field_linear = np.zeros_like(field_circular)
    field_linear[0] = np.einsum('stxy,qs->qtxy', field_circular[0], X.transform_matrix)
    field_linear[1] = np.einsum('stxy,qs->qtxy', field_circular[1], np.conj(X.transform_matrix))
    return field_linear
