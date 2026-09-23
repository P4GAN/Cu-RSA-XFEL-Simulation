"""
Plane-wave Maxwell-Bloch solver for the two-level Cu Kalpha1 RSA model with a ground and an
auxiliary state: the numerical counterpart of the analytic model in ../RSA-derivation
(RSA_technical_spec.md sec. 3, RSA_derivation_attempt1.md Eqs. (1)-(4)).

Independent of the XLO_sim class: no magnetic sublevels (so no dark state), no 2s/satellite/
middlemen pathways and no transverse diffraction. Without diffraction every point of the focus
is its own 1D (t, z) problem. The beam average uses the attempt-1 substitution
v = local/peak fluence, under which the energy-weighted focal average is a uniform average over
v in [0, 1] (derivation Eq. 32); the v integral is done with Gauss-Legendre nodes, plus one
on-axis node (v = 1, weight 0).

Per atom, in the frame rotating at the carrier omega_c (delta = omega_c - omega_ul):

    dg/dt = -sig_g J g
    du/dt = -a u + q                                  a = Gamma_u + sig_u J
    dl/dt = sig_L3 J g + Gamma_sp u - b l - q         b = Gamma_l + sig_l J
    dc/dt = -(gamma - i delta) c + (i/2) Om (l - u)   gamma = (a + b)/2 + gamma_phi
    x     = 1 - g - l - u

    dOm/dz = (i/2) n_a sigma0 Gamma_sp c - (1/2) kappa_nr Om,    J = |Om|^2 / (sigma0 Gamma_sp)
    kappa_nr = n_a (sig_g g + sig_l l + sig_u u + sig_x x)

with c = rho_ul and q = Im(Om* c) (mode "mb", full Maxwell-Bloch). Mode "re" is the rate-
equation limit: the coherence is slaved to the instantaneous populations, c = (i/2)(l - u) K with
dK/dt = Om - (gamma - i delta) K (the field filtered by the line's Lorentzian), and again
q = Im(Om* c) = (l - u) Re(Om* K) / 2, so the atoms absorb exactly what the field loses. For a
stationary field q = W (l - u) with W = derivation Eq. (8). (Before 2026-09-22 "re" used
q = gamma |K|^2 (l - u) / 2, XLO_sim's use_rate_equations form. That has the same time integral
only while l - u is constant: the field lost up to 7 % (mono, 100 uJ) and 65 % (SASE, 200 uJ) more
resonant photons than the atoms absorbed. Runs saved with that form lack the key re_closure.)
"re" relaxes l - u at up to Om^2/gamma, which exceeds Om once Om > gamma, so its time step is
also bounded by that rate (choose_dt).

x collects every atom that has left {g, l, u}: other-shell photoionisation, the non-Kalpha1 decay
of the 1s hole, decay and photoionisation of both holes. It absorbs with sig_x, as in the
attempt-1 script (the spec's eq. 3.3 has sig_x = 0).

Numerics: RK4 in t on a uniform grid, with the field at the half steps from 4-point cubic
interpolation. Heun (predictor-corrector) in z, so each z step costs two passes over the time
grid. The incident field is the exact local drive at z = 0 (no J(z=0) lag), and every recorded
plane is read out after exactly round(z/dz) steps.

Units: time fs, rates fs^-1, lengths cm, cross sections cm^2, n_a cm^-3, flux J cm^-2 fs^-1:
the analytic script's units, so its parameter dictionaries carry over unchanged. Om is in
rad/fs.
"""

import copy
import hashlib
import os
import subprocess

import numpy as np
import yaml
from numba import njit, prange

HBAR = 0.6582119569      # eV fs
HC = 12398.419843        # eV Angstrom
EV = 1.602176634e-19     # J
FWHM_TO_STD = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
MODES = ("mb", "re")
BEAMS = ("mono", "sase")
INT0_KEYS = ("Phi", "N_res", "N_nr", "N_L3", "N_fluo", "N_q")
SASE_GEN_DT = 0.01       # fs; SASE shots are drawn on this grid whatever the time step, so runs
                         # with different grid.dt_fs see identical pulses (paired convergence tests)


# ----------------------------------------------------------------------------------------------
# Configuration and parameters
# ----------------------------------------------------------------------------------------------
def load_config(path, overrides=()):
    """Read a YAML config and apply 'section.key=value' overrides (value parsed as YAML)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, overrides)
    # PyYAML (YAML 1.1) reads 8.25e22 without an exponent sign as a string; fail loudly.
    for section in ("atom", "mono", "sase", "grid"):
        bad = [k for k, v in cfg[section].items() if isinstance(v, str)]
        if bad:
            raise ValueError(f"{path}: {section}.{bad} parsed as strings; write exponents with a sign (8.25e+22)")
    return cfg


def apply_overrides(cfg, overrides):
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} is not of the form section.key=value")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node:
                raise KeyError(f"override {item!r}: no section {part!r}")
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"override {item!r}: no key {parts[-1]!r} (typo?)")
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def rates(atom):
    """Atomic constants in internal units. Same transformation as rsa_transmittance.rates(),
    including the bright-fraction rescaling sig_L3 -> f sig_L3, sigma0 -> sigma0 / f."""
    r = dict(atom)
    r["Gu"] = atom["Gamma_u_eV"] / HBAR
    r["Gl"] = atom["Gamma_l_eV"] / HBAR
    r["Gsp"] = atom["Gamma_sp_eV"] / HBAR
    r["gphi"] = atom["gamma_phi_eV"] / HBAR
    lam = HC / atom["E_ul"] * 1e-8
    f = atom.get("bright_fraction", 1.0)
    r["sigma0"] = atom["sigma0_prefactor"] * lam**2 / f
    r["sig_L3"] = atom["sig_L3"] * f
    r["s0G"] = r["sigma0"] * r["Gsp"]
    return r


def weak_field_constants(r):
    """J -> 0 values of the quantities that set the regimes (derivation Eqs. 7, 11, 35)."""
    Gu, Gl, Gsp = r["Gu"], r["Gl"], r["Gsp"]
    gam = 0.5 * (Gu + Gl) + r["gphi"]
    Wsat = Gu * Gl / (Gu + Gl - Gsp)
    return dict(gamma=gam, Wsat=Wsat, sig_pk=r["s0G"] / (2.0 * gam),
                c0=r["sig_L3"] / r["sig_g"] * Gu / (Gu + Gl - Gsp),
                k0=r["n_a"] * r["sig_g"], eta_sat=Gu / (Gu + Gl - Gsp))


def pvec(r):
    return np.array([r["Gu"], r["Gl"], r["Gsp"], r["gphi"], r["sig_g"], r["sig_L3"], r["sig_u"],
                     r["sig_l"], r["sig_x"], r["n_a"], r["s0G"]], dtype=np.float64)


def beam_peak(beam, E_uJ, E_ph):
    """On-axis peak fluence [cm^-2] and flux [cm^-2 fs^-1] of a Gaussian focus (fluence FWHM
    wx, wy) and Gaussian envelope (intensity FWHM tau). Same as rsa_transmittance.beam_peak."""
    N = np.asarray(E_uJ) * 1e-6 * beam["T_beamline"] / (np.asarray(E_ph) * EV)
    A_eff = np.pi * beam["wx_nm"] * beam["wy_nm"] * 1e-14 / (4.0 * np.log(2.0))
    tau_eff = beam["tau_fs"] * np.sqrt(np.pi / (4.0 * np.log(2.0)))
    F_pk = N / A_eff
    return F_pk, F_pk / tau_eff


def tl_bandwidth_eV(tau_fs):
    """Spectral intensity FWHM of a transform-limited Gaussian pulse of intensity FWHM tau."""
    return 4.0 * np.log(2.0) * HBAR / tau_fs


def v_nodes(n_v):
    """Gauss-Legendre nodes/weights on v in [0, 1], plus an on-axis node v = 1 with weight 0."""
    x, w = np.polynomial.legendre.leggauss(n_v)
    return np.append(0.5 * (x + 1.0), 1.0), np.append(0.5 * w, 0.0)


def mono_detunings(segments):
    """Concatenate [start, stop, step] segments (stop inclusive) into one sorted grid [eV]."""
    pts = []
    for start, stop, step in segments:
        n = int(round((stop - start) / step))
        pts.append(start + step * np.arange(n + 1))
    return np.unique(np.round(np.concatenate(pts), 6))


def z_plan(grid):
    """Number of z steps, dz [cm] and the step index of every recorded thickness."""
    dz_um = grid["dz_um"]
    n_z = int(round(grid["zmax_um"] / dz_um))
    rec = np.array([int(round(z / dz_um)) for z in grid["z_record_um"]], dtype=np.int64)
    if np.any(rec < 1) or np.any(rec > n_z) or np.any(np.diff(rec) <= 0):
        raise ValueError(f"z_record_um {grid['z_record_um']} must be increasing, >= dz and <= zmax")
    return n_z, dz_um * 1e-4, rec, rec * dz_um


def time_grid(half_window_fs, dt):
    n = int(round(2.0 * half_window_fs / dt))
    n += n % 2
    return -half_window_fs + dt * np.arange(n)


def tl_shape(t, tau_fs):
    """Real TL Gaussian amplitude, normalised so that sum |f|^2 dt = tau_eff, i.e. J = J_pk f^2."""
    f = np.exp(-2.0 * np.log(2.0) * t**2 / tau_fs**2)
    tau_eff = tau_fs * np.sqrt(np.pi / (4.0 * np.log(2.0)))
    return f * np.sqrt(tau_eff / (np.sum(f**2) * (t[1] - t[0])))


def sase_shapes(n_shots, t, tau_fs, bw_eV, rng):
    """Chaotic pulses: complex Gaussian noise with a Gaussian stationary power spectrum, times the
    Gaussian envelope. The stationary width is chosen so that the ensemble-averaged spectrum
    (stationary spectrum convolved with the envelope's) has FWHM bw_eV. Each shot is normalised
    to sum |f|^2 dt = tau_eff, i.e. every shot carries exactly the nominal pulse energy."""
    n = t.size
    dt = t[1] - t[0]
    nu = 2.0 * np.pi * np.fft.fftfreq(n, dt)
    bw_stat = np.sqrt(max(bw_eV**2 - tl_bandwidth_eV(tau_fs)**2, (0.1 * bw_eV)**2))
    sig = bw_stat / HBAR * FWHM_TO_STD                 # std of the stationary power spectrum
    filt = np.exp(-nu**2 / (4.0 * sig**2))
    z = (rng.standard_normal((n_shots, n)) + 1j * rng.standard_normal((n_shots, n))) / np.sqrt(2.0)
    xi = np.fft.ifft(np.fft.fft(z, axis=1) * filt, axis=1)
    f = xi * np.exp(-2.0 * np.log(2.0) * t**2 / tau_fs**2)
    tau_eff = tau_fs * np.sqrt(np.pi / (4.0 * np.log(2.0)))
    return f * np.sqrt(tau_eff / (np.sum(np.abs(f)**2, axis=1, keepdims=True) * dt))


def upsample(f, m):
    """Band-limited (FFT zero-padding) interpolation onto a grid m times finer."""
    if m == 1:
        return f
    n = f.shape[-1]
    F = np.fft.fft(f, axis=-1)
    Fp = np.zeros(f.shape[:-1] + (n * m,), dtype=complex)
    Fp[..., : n // 2] = F[..., : n // 2]
    Fp[..., -(n // 2):] = F[..., -(n // 2):]
    return np.fft.ifft(Fp, axis=-1) * m


def spectrum_axis(n_t, dt, E_frame):
    """Photon energy of every FFT bin for spectra computed with spectral_amplitude()."""
    return E_frame + HBAR * 2.0 * np.pi * np.fft.fftfreq(n_t, dt)


def spectral_amplitude(f):
    """sum_t f(t) exp(+i nu t): a field component Om ~ exp(-i Delta t) in the carrier frame
    (photon energy E_frame + hbar Delta) peaks at nu = +Delta. Constant factors cancel in every
    ratio taken from it."""
    return np.fft.ifft(f, axis=-1)


# ----------------------------------------------------------------------------------------------
# Numba kernels
# ----------------------------------------------------------------------------------------------
@njit(inline="always")
def _rhs(yg, yl, yu, yc, O, dlt, cpl, Gu, Gl, Gsp, gphi, sg, sL3, su, sl, inv_s0G, re):
    J = (O.real * O.real + O.imag * O.imag) * inv_s0G
    a = Gu + su * J
    b = Gl + sl * J
    gam = 0.5 * (a + b) + gphi
    w = yl - yu
    Oc = cpl * O
    if re:
        q = 0.5 * w * (Oc.real * yc.real + Oc.imag * yc.imag)    # Im(conj(Om) c), c = (i/2) w K
        dc = Oc - (gam - 1j * dlt) * yc
    else:
        q = Oc.real * yc.imag - Oc.imag * yc.real          # Im(conj(Om) c)
        dc = -(gam - 1j * dlt) * yc + 0.5j * Oc * w
    return -sg * J * yg, sL3 * J * yg + Gsp * yu - b * yl - q, -a * yu + q, dc


@njit
def _plane(Om, dlt, cpl, dt, p, re, g, l, u, c, q):
    """RK4 over the time grid at one z plane, driven by Om (field at the grid points). Fills the
    populations, the physical coherence c (re: (i/2)(l-u)K) and the l->u exchange rate q."""
    Gu = p[0]; Gl = p[1]; Gsp = p[2]; gphi = p[3]; sg = p[4]; sL3 = p[5]; su = p[6]; sl = p[7]
    inv_s0G = 1.0 / p[10]
    n = Om.shape[0]
    h = dt
    h2 = 0.5 * dt
    yg = 1.0
    yl = 0.0
    yu = 0.0
    yc = 0.0 + 0.0j
    g[0] = 1.0; l[0] = 0.0; u[0] = 0.0; c[0] = 0.0 + 0.0j; q[0] = 0.0
    for i in range(n - 1):
        O0 = Om[i]
        O1 = Om[i + 1]
        if i >= 1 and i <= n - 3:
            Oh = (9.0 * (O0 + O1) - Om[i - 1] - Om[i + 2]) * 0.0625
        elif i == 0:
            Oh = (3.0 * O0 + 6.0 * O1 - Om[2]) * 0.125
        else:
            Oh = (6.0 * O0 + 3.0 * O1 - Om[n - 3]) * 0.125
        k1g, k1l, k1u, k1c = _rhs(yg, yl, yu, yc, O0, dlt, cpl, Gu, Gl, Gsp, gphi, sg, sL3, su, sl, inv_s0G, re)
        k2g, k2l, k2u, k2c = _rhs(yg + h2 * k1g, yl + h2 * k1l, yu + h2 * k1u, yc + h2 * k1c, Oh,
                                  dlt, cpl, Gu, Gl, Gsp, gphi, sg, sL3, su, sl, inv_s0G, re)
        k3g, k3l, k3u, k3c = _rhs(yg + h2 * k2g, yl + h2 * k2l, yu + h2 * k2u, yc + h2 * k2c, Oh,
                                  dlt, cpl, Gu, Gl, Gsp, gphi, sg, sL3, su, sl, inv_s0G, re)
        k4g, k4l, k4u, k4c = _rhs(yg + h * k3g, yl + h * k3l, yu + h * k3u, yc + h * k3c, O1,
                                  dlt, cpl, Gu, Gl, Gsp, gphi, sg, sL3, su, sl, inv_s0G, re)
        yg += h / 6.0 * (k1g + 2.0 * k2g + 2.0 * k3g + k4g)
        yl += h / 6.0 * (k1l + 2.0 * k2l + 2.0 * k3l + k4l)
        yu += h / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        yc += h / 6.0 * (k1c + 2.0 * k2c + 2.0 * k3c + k4c)
        g[i + 1] = yg
        l[i + 1] = yl
        u[i + 1] = yu
        if re:
            c[i + 1] = 0.5j * (yl - yu) * yc
            q[i + 1] = 0.5 * (yl - yu) * cpl * (O1.real * yc.real + O1.imag * yc.imag)
        else:
            c[i + 1] = yc
            q[i + 1] = cpl * (O1.real * yc.imag - O1.imag * yc.real)


@njit(parallel=True)
def _march(Om_in, dlt, cpl, dt, dz, n_z, rec, p, re, keep_fields, src_slot, ts_slot, ts_stride,
           out_flu, out_pop, out_int0, out_fields, out_src0, out_ts):
    """Propagate every trajectory (row of Om_in) through n_z Heun steps of size dz. Trajectories
    are independent (no diffraction), so they run in parallel."""
    n_traj, n_t = Om_in.shape
    n_rec = rec.shape[0]
    Gsp = p[2]; sg = p[4]; sL3 = p[5]; su = p[6]; sl = p[7]; sx = p[8]; na = p[9]; s0G = p[10]
    res_fac = 0.5j * na * s0G
    inv_s0G = 1.0 / s0G
    n_ts_t = out_ts.shape[2]
    for k in prange(n_traj):
        Om = Om_in[k].copy()
        Omp = np.empty(n_t, np.complex128)
        src = np.empty(n_t, np.complex128)
        c = np.empty(n_t, np.complex128)
        g = np.empty(n_t)
        l = np.empty(n_t)
        u = np.empty(n_t)
        q = np.empty(n_t)
        ir = 0
        for iz in range(n_z + 1):
            _plane(Om, dlt[k], cpl[k], dt, p, re, g, l, u, c, q)
            for i in range(n_t):
                knr = na * (sg * g[i] + sl * l[i] + su * u[i] + sx * (1.0 - g[i] - l[i] - u[i]))
                src[i] = res_fac * c[i] - 0.5 * knr * Om[i]

            if iz == 0:
                Phi = 0.0; Nres = 0.0; Nnr = 0.0; NL3 = 0.0; Nfl = 0.0; Nq = 0.0
                for i in range(n_t):
                    O = Om[i]
                    J = (O.real * O.real + O.imag * O.imag) * inv_s0G
                    Phi += J
                    Nres += O.real * c[i].imag - O.imag * c[i].real
                    Nnr += (sg * g[i] + sl * l[i] + su * u[i] + sx * (1.0 - g[i] - l[i] - u[i])) * J
                    NL3 += sL3 * J * g[i]
                    Nfl += Gsp * u[i]
                    Nq += q[i]
                out_int0[k, 0] = Phi * dt
                out_int0[k, 1] = Nres * dt
                out_int0[k, 2] = Nnr * dt
                out_int0[k, 3] = NL3 * dt
                out_int0[k, 4] = Nfl * dt
                out_int0[k, 5] = Nq * dt
                s = src_slot[k]
                if s >= 0:
                    for i in range(n_t):
                        rr = res_fac * c[i]
                        out_src0[s, 0, i] = rr
                        out_src0[s, 1, i] = src[i] - rr
                s = ts_slot[k]
                if s >= 0:
                    for j in range(n_ts_t):
                        i = j * ts_stride
                        O = Om[i]
                        out_ts[s, 0, j] = g[i]
                        out_ts[s, 1, j] = l[i]
                        out_ts[s, 2, j] = u[i]
                        out_ts[s, 3, j] = c[i].real
                        out_ts[s, 4, j] = c[i].imag
                        out_ts[s, 5, j] = (O.real * O.real + O.imag * O.imag) * inv_s0G

            if ir < n_rec and rec[ir] == iz:
                F = 0.0
                for i in range(n_t):
                    F += Om[i].real * Om[i].real + Om[i].imag * Om[i].imag
                out_flu[k, ir] = F * inv_s0G * dt
                out_pop[k, ir, 0] = g[n_t - 1]
                out_pop[k, ir, 1] = 1.0 - g[n_t - 1] - l[n_t - 1] - u[n_t - 1]
                if keep_fields:
                    for i in range(n_t):
                        out_fields[k, ir, i] = Om[i]
                ir += 1

            if iz == n_z:
                break
            for i in range(n_t):
                Omp[i] = Om[i] + dz * src[i]
            _plane(Omp, dlt[k], cpl[k], dt, p, re, g, l, u, c, q)
            for i in range(n_t):
                knr = na * (sg * g[i] + sl * l[i] + su * u[i] + sx * (1.0 - g[i] - l[i] - u[i]))
                Om[i] = Om[i] + 0.5 * dz * (src[i] + res_fac * c[i] - 0.5 * knr * Omp[i])


def march(Om_in, dlt, cpl, dt, dz_cm, n_z, rec, r, mode, keep_fields=False, src_slot=None,
          ts_slot=None, ts_stride=1):
    """Allocate the outputs and run the kernel. Returns
      flu    [traj, rec]      fluence (int J dt) at every recorded plane
      pop    [traj, rec, 2]   g and x at the end of the window at every recorded plane
      int0   [traj, 6]        entrance-plane time integrals, INT0_KEYS: fluence, resonant photons
                              absorbed per atom (field side, int Im(Om* c) dt), non-resonant photons
                              per atom, 2p3/2 holes made per atom, Kalpha1 photons emitted per
                              atom (int Gamma_sp u dt), resonant exchange (atom side, int q dt)
      fields [traj, rec, t]   complex64 fields at the recorded planes (keep_fields)
      src0   [slot, 2, t]     complex64 dOm/dz at z = 0, split into resonant / non-resonant parts
      ts     [slot, 6, t_s]   float32 g, l, u, Re c, Im c, J at z = 0, every ts_stride steps"""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    Om_in = np.ascontiguousarray(Om_in, dtype=np.complex128)
    n_traj, n_t = Om_in.shape
    n_rec = rec.size
    src_slot = np.full(n_traj, -1, np.int64) if src_slot is None else np.asarray(src_slot, np.int64)
    ts_slot = np.full(n_traj, -1, np.int64) if ts_slot is None else np.asarray(ts_slot, np.int64)
    n_src = max(1, int(src_slot.max()) + 1)
    n_ts = max(1, int(ts_slot.max()) + 1)
    n_ts_t = (n_t - 1) // ts_stride + 1
    out = dict(
        flu=np.zeros((n_traj, n_rec)),
        pop=np.zeros((n_traj, n_rec, 2)),
        int0=np.zeros((n_traj, 6)),
        fields=np.zeros((n_traj, n_rec, n_t) if keep_fields else (1, 1, 1), np.complex64),
        src0=np.zeros((n_src, 2, n_t) if src_slot.max() >= 0 else (1, 2, 1), np.complex64),
        ts=np.zeros((n_ts, 6, n_ts_t) if ts_slot.max() >= 0 else (1, 6, 1), np.float32),
    )
    _march(Om_in, np.ascontiguousarray(dlt, np.float64), np.ascontiguousarray(cpl, np.float64),
           float(dt), float(dz_cm), int(n_z), np.asarray(rec, np.int64), pvec(r), mode == "re",
           bool(keep_fields), src_slot, ts_slot, int(ts_stride),
           out["flu"], out["pop"], out["int0"], out["fields"], out["src0"], out["ts"])
    return out


# ----------------------------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------------------------
RE_STEP = 1.0            # mode "re": relax_ref dt <= RE_STEP. RK4 is stable to 2.79 on a decaying
                         # mode and within 2 % of exp(-x) at x = 1; the old Rabi-only step reached
                         # 3.1 at 200 uJ (mono) and diverged


def stiff_rate(r, Om_ref, mode):
    """Fastest decay rate the RK4 step must resolve besides the Rabi frequency: in "re" the
    populations relax toward the slaved coherence at 2W <= Om^2/gamma (resonant bound, J -> 0
    gamma). Zero for "mb", where the stiffest scale is Om itself."""
    return Om_ref**2 / weak_field_constants(r)["gamma"] if mode == "re" else 0.0


def choose_dt(grid, Om_ref, relax_ref=0.0):
    """Largest step <= dt_fs that is an integer fraction of dt_fs with Om_ref dt <= rabi_step and
    relax_ref dt <= RE_STEP. Returns (dt, m) with dt = dt_fs / m."""
    m = max(1, int(np.ceil(grid["dt_fs"] * Om_ref / grid["rabi_step"])),
            int(np.ceil(grid["dt_fs"] * relax_ref / RE_STEP)))
    return grid["dt_fs"] / m, m


def run_mono(cfg, E_uJ, mode, log=print):
    """Photon-energy scan with a transform-limited Gaussian pulse. Every (detuning, v) pair is a
    trajectory; one extra trajectory per v node has the Kalpha1 coupling off (non-resonant
    baseline). Returns per-node arrays; beam/axis averages are left to the analysis."""
    r = rates(cfg["atom"])
    beam, gr = cfg["mono"], cfg["grid"]
    dE = mono_detunings(beam["scan_eV"])
    E0 = r["E_ul"] + dE
    v, wv = v_nodes(gr["n_v"])
    n_v, n_dE = v.size, dE.size
    F_pk, J_pk = beam_peak(beam, E_uJ, E0)
    F_ref, J_ref = beam_peak(beam, E_uJ, r["E_ul"])
    Om_pk = np.sqrt(r["s0G"] * J_pk.max())
    dt, _ = choose_dt(gr, Om_pk, stiff_rate(r, Om_pk, mode))
    t = time_grid(gr["t_half_window_fs"], dt)
    shape = tl_shape(t, beam["tau_fs"])
    n_z, dz_cm, rec, z_rec_um = z_plan(gr)

    amp = np.sqrt(r["s0G"] * J_pk[:, None] * v[None, :]).reshape(-1)
    amp_ref = np.sqrt(r["s0G"] * J_ref * v)
    Om_in = np.concatenate([amp[:, None] * shape[None, :], amp_ref[:, None] * shape[None, :]])
    dlt = np.concatenate([np.repeat(dE / HBAR, n_v), np.zeros(n_v)])
    cpl = np.concatenate([np.ones(n_dE * n_v), np.zeros(n_v)])

    # on-axis entrance-plane time series at the scan points nearest the requested detunings
    bloch_idx = np.unique([int(np.argmin(np.abs(dE - d))) for d in beam["bloch_detunings_eV"]])
    ts_slot = np.full(Om_in.shape[0], -1, np.int64)
    for s, j in enumerate(bloch_idx):
        ts_slot[j * n_v + (n_v - 1)] = s
    ts_stride = max(1, int(round(gr["ts_stride_fs"] / dt)))

    log(f"mono {mode} {E_uJ:g} uJ: {Om_in.shape[0]} trajectories x {t.size} steps (dt {dt:.4g} fs) "
        f"x {n_z} z steps; Omega_pk/gamma0 = {np.sqrt(r['s0G'] * J_pk.max()) / weak_field_constants(r)['gamma']:.3g}")
    out = march(Om_in, dlt, cpl, dt, dz_cm, n_z, rec, r, mode, ts_slot=ts_slot, ts_stride=ts_stride)

    nm = n_dE * n_v
    Phi_in = out["int0"][:, 0]
    T = out["flu"] / Phi_in[:, None]
    return dict(
        beam="mono", mode=mode, re_closure="field", E_uJ=float(E_uJ), dt=dt, n_t=t.size, dz_um=gr["dz_um"],
        z_rec_um=z_rec_um, v=v, wv=wv, dE=dE, E0=E0, F_pk=F_pk, J_pk=J_pk,
        T=T[:nm].reshape(n_dE, n_v, -1), T_ref=T[nm:],
        int0=out["int0"][:nm].reshape(n_dE, n_v, 6), int0_ref=out["int0"][nm:],
        pop_end=out["pop"][:nm].reshape(n_dE, n_v, -1, 2), pop_end_ref=out["pop"][nm:],
        ts=out["ts"], ts_t=t[::ts_stride][: out["ts"].shape[2]], ts_dE=dE[bloch_idx],
    )


def run_sase(cfg, E_uJ, mode, log=print):
    """Spectrally resolved SASE transmission. Shots are processed in chunks; spectra are reduced
    per statistical block. T(E) at a recorded thickness is num_out_view[view, block] / den[block]
    (view 0 = on axis, 1 = beam average), T_nonres = num_ref_view / den_ref. The entrance-plane
    (L -> 0) absorption coefficient at node m is kres_num[block, m] / den[block] (resonant) and
    knr_num / den (non-resonant), exactly -d ln S(E)/dz at z = 0; nodes combine with wv."""
    r = rates(cfg["atom"])
    beam, gr = cfg["sase"], cfg["grid"]
    v, wv = v_nodes(gr["n_v"])
    n_v = v.size
    F_pk, J_pk = beam_peak(beam, E_uJ, beam["E_c"])
    n_z, dz_cm, rec, z_rec_um = z_plan(gr)
    n_rec = rec.size
    n_shots, n_blocks = beam["n_shots"], beam["n_blocks"]
    if n_shots % n_blocks:
        raise ValueError("sase.n_shots must be a multiple of sase.n_blocks")
    block = n_shots // n_blocks

    rng = np.random.default_rng(beam["seed"])
    t0 = time_grid(gr["t_half_window_fs"], SASE_GEN_DT)
    shapes0 = sase_shapes(n_shots, t0, beam["tau_fs"], beam["bw_eV"], rng)
    I_rel = np.abs(shapes0) ** 2
    env = np.exp(-4.0 * np.log(2.0) * t0**2 / beam["tau_fs"]**2)
    Om_ref = np.sqrt(r["s0G"] * J_pk * np.percentile(I_rel[:, env > 0.5], 99.9))
    m_up = max(int(np.ceil(SASE_GEN_DT / gr["dt_fs"] - 1e-9)),
               int(np.ceil(SASE_GEN_DT * Om_ref / gr["rabi_step"])),
               int(np.ceil(SASE_GEN_DT * stiff_rate(r, Om_ref, mode) / RE_STEP)))
    dt = SASE_GEN_DT / m_up
    n_t = t0.size * m_up
    t = -gr["t_half_window_fs"] + dt * np.arange(n_t)
    dlt_c = (beam["E_c"] - r["E_ul"]) / HBAR

    E_full = spectrum_axis(n_t, dt, beam["E_c"])
    keep = np.abs(E_full - r["E_ul"]) <= 30.0
    order = np.argsort(E_full[keep])
    E_axis = E_full[keep][order]
    n_w = E_axis.size

    # chunk = largest divisor of the block size whose stored fields fit max_field_elements
    per_shot = n_v * n_rec * n_t
    chunk = max(1, min(block, int(gr["max_field_elements"] // per_shot)))
    while block % chunk:
        chunk -= 1
    ts_stride = max(1, int(round(gr["ts_stride_fs"] / dt)))
    n_bloch = min(beam["n_bloch_shots"], n_shots)

    den = np.zeros((n_blocks, n_w))
    num_out = np.zeros((n_blocks, n_v, n_rec, n_w))
    kres_num = np.zeros((n_blocks, n_v, n_w))
    knr_num = np.zeros((n_blocks, n_v, n_w))
    int0 = np.zeros((n_blocks, n_v, 6))
    pop_end = np.zeros((n_v, n_rec, 2))
    ts, ts_t = None, t[::ts_stride]
    amp = np.sqrt(r["s0G"] * J_pk * v)

    log(f"sase {mode} {E_uJ:g} uJ: {n_shots} shots x {n_v} nodes, {n_t} steps (dt {dt:.4g} fs, "
        f"x{m_up}) x {n_z} z steps, chunks of {chunk} shots; Omega_ref/gamma0 = "
        f"{Om_ref / weak_field_constants(r)['gamma']:.3g}")

    norm = r["s0G"] * J_pk * v                   # |incident amplitude|^2 per unit |shape|^2, per node

    def reduce_chunk(shp, out, with_src):
        """Spectral sums over the shots of one chunk (shp [n_c, n_t] incident shapes). Every node's
        output is divided by its own incident scale, so num / den is that node's T(E)."""
        n_c = shp.shape[0]
        A_sh = spectral_amplitude(shp)[:, keep][:, order]                              # [n_c, n_w]
        A_out = spectral_amplitude(out["fields"])[..., keep][..., order]                # [n_c*n_v, n_rec, n_w]
        S_out = (np.abs(A_out.reshape(n_c, n_v, n_rec, n_w)) ** 2).sum(axis=0) / norm[:, None, None]
        if not with_src:
            return (np.abs(A_sh) ** 2).sum(axis=0), S_out, None, None
        src = spectral_amplitude(out["src0"])[..., keep][..., order].reshape(n_c, n_v, 2, n_w)
        A_in = amp[None, :, None] * A_sh[:, None, :]                                   # [n_c, n_v, n_w]
        kr = (-2.0 * np.real(np.conj(A_in) * src[:, :, 0])).sum(axis=0) / norm[:, None]
        kn = (-2.0 * np.real(np.conj(A_in) * src[:, :, 1])).sum(axis=0) / norm[:, None]
        return (np.abs(A_sh) ** 2).sum(axis=0), S_out, kr, kn

    for start in range(0, n_shots, chunk):
        shots = np.arange(start, start + chunk)
        shp = upsample(shapes0[shots], m_up)
        Om_in = (amp[None, :, None] * shp[:, None, :]).reshape(-1, n_t)
        src_slot = np.arange(Om_in.shape[0], dtype=np.int64)
        ts_slot = np.full(Om_in.shape[0], -1, np.int64)
        for s in shots[shots < n_bloch]:
            ts_slot[(s - start) * n_v + (n_v - 1)] = s
        out = march(Om_in, np.full(Om_in.shape[0], dlt_c), np.ones(Om_in.shape[0]), dt, dz_cm,
                    n_z, rec, r, mode, keep_fields=True, src_slot=src_slot, ts_slot=ts_slot,
                    ts_stride=ts_stride)
        b = start // block
        d, so, kr, kn = reduce_chunk(shp, out, True)
        den[b] += d
        num_out[b] += so
        kres_num[b] += kr
        knr_num[b] += kn
        int0[b] += out["int0"].reshape(chunk, n_v, 6).sum(axis=0)
        pop_end += out["pop"].reshape(chunk, n_v, n_rec, 2).sum(axis=0)
        if ts_slot.max() >= 0:
            if ts is None:
                ts = np.zeros((n_bloch, 6, out["ts"].shape[2]), np.float32)
            sel = shots[shots < n_bloch]
            ts[sel] = out["ts"][sel]
        log(f"  shots {start}-{start + chunk - 1} done")

    # non-resonant baseline: the first n_ref shots again with the Kalpha1 coupling off
    n_ref = min(beam["n_ref_shots"], n_shots)
    den_ref = np.zeros(n_w)
    num_ref = np.zeros((n_v, n_rec, n_w))
    for start in range(0, n_ref, chunk):
        shots = np.arange(start, min(start + chunk, n_ref))
        shp = upsample(shapes0[shots], m_up)
        Om_in = (amp[None, :, None] * shp[:, None, :]).reshape(-1, n_t)
        out = march(Om_in, np.full(Om_in.shape[0], dlt_c), np.zeros(Om_in.shape[0]), dt, dz_cm,
                    n_z, rec, r, mode, keep_fields=True)
        d, so, _, _ = reduce_chunk(shp, out, False)
        den_ref += d
        num_ref += so

    # finite-thickness spectra are only ever read in the two views, so store those (float64:
    # -ln(T/T_nonres)/L at thin L subtracts two numbers close to 1)
    views = np.stack([np.eye(n_v)[n_v - 1], wv])                          # [axis, beam]
    return dict(
        beam="sase", mode=mode, re_closure="field", E_uJ=float(E_uJ), dt=dt, n_t=n_t, dz_um=gr["dz_um"],
        z_rec_um=z_rec_um, v=v, wv=wv, E_axis=E_axis, F_pk=F_pk, J_pk=J_pk,
        n_shots=n_shots, n_blocks=n_blocks, n_ref_shots=n_ref,
        den=den, num_out_view=np.einsum("bmrw,km->kbrw", num_out, views),
        kres_num=kres_num.astype(np.float32), knr_num=knr_num.astype(np.float32),
        den_ref=den_ref, num_ref_view=np.einsum("mrw,km->krw", num_ref, views),
        int0=int0, pop_end=pop_end / n_shots,
        ts=ts if ts is not None else np.zeros((0, 6, 0), np.float32), ts_t=ts_t,
    )


# ----------------------------------------------------------------------------------------------
# Provenance and self-checks
# ----------------------------------------------------------------------------------------------
def provenance(repo_root):
    """git HEAD, dirty flag and the sha1 of this file, saved with every output (cluster runs have
    silently used stale code before)."""
    info = {"code_sha1": hashlib.sha1(open(__file__, "rb").read()).hexdigest()}
    try:
        info["git_head"] = subprocess.run(["git", "-C", repo_root, "rev-parse", "HEAD"], capture_output=True,
                                          text=True, check=True).stdout.strip()
        info["git_dirty"] = bool(subprocess.run(["git", "-C", repo_root, "status", "--porcelain"],
                                                capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        info["git_head"], info["git_dirty"] = "unknown", True
    info["module_path"] = os.path.abspath(__file__)
    return info


def self_check(cfg, log=print):
    """Fast checks of the kernel against closed forms (a few trajectories, a few seconds after JIT
    compilation). Raises AssertionError on failure.
      1. Flat-top drive, weak and strong, on and off resonance, MB and RE: at the plateau the
         populations equal the rate-equation steady state (derivation Eq. 12) and the coherence
         equals (i/2) Om w / (gamma - i delta) (the exact Bloch steady state).
      2. One z step: the fluence lost equals the photons absorbed at z = 0 (field bookkeeping).
      3. Spectral convention: a pulse at carrier offset +5 eV peaks at E_frame + 5 eV.
      4. Both modes, smooth and chaotic pulse: the resonant photons the field loses equal those the
         atoms absorb, and "re" stays bounded at the Rabi frequency of 500 uJ (mono, on axis) with
         the step from choose_dt."""
    atom = dict(cfg["atom"])
    atom["sig_g"] = atom["sig_L3"] = 1e-25        # no depletion, so the plateau is a true steady state
    r = rates(atom)
    wf = weak_field_constants(r)
    dt = 0.01
    t = time_grid(40.0, dt)
    flat = np.exp(-0.5 * (t / 22.0) ** 16)       # super-Gaussian, flat over |t| < ~18 fs
    i0 = t.size // 2
    for s_target in (0.01, 30.0):
        J = s_target * wf["Wsat"] / wf["sig_pk"]
        for d_eV in (0.0, 1.5):
            dlt = d_eV / HBAR
            for mode in MODES:
                Om = np.sqrt(r["s0G"] * J) * flat
                out = march(Om[None, :], np.array([dlt]), np.array([1.0]), dt, 1e-9, 0,
                            np.array([0]), r, mode, ts_slot=np.array([0]), ts_stride=1)
                g, l, u, cr, ci, Jt = out["ts"][0][:, i0].astype(float)
                a = r["Gu"] + r["sig_u"] * J
                b = r["Gl"] + r["sig_l"] * J
                gam = 0.5 * (a + b) + r["gphi"]
                W = r["s0G"] * J * gam / (2.0 * (gam**2 + dlt**2))
                S = r["sig_L3"] * J * g
                D = a * b + W * (a + b - r["Gsp"])
                l_ss, u_ss = S * (a + W) / D, S * W / D
                c_ss = 0.5j * np.sqrt(r["s0G"] * J) * (l - u) / (gam - 1j * dlt)
                err = max(abs(l / l_ss - 1), abs(u / u_ss - 1), abs((cr + 1j * ci) / c_ss - 1))
                log(f"  steady state s={s_target:g} delta={d_eV:g} eV {mode}: max rel. error {err:.2e}")
                assert err < 2e-3, "plateau does not match the steady state"

    r = rates(cfg["atom"])
    tau = 6.0
    t = time_grid(25.0, dt)
    for J_pk in (1e16, 3e19):
        Om = np.sqrt(r["s0G"] * J_pk) * tl_shape(t, tau)
        dz = 1e-7
        out = march(Om[None, :], np.array([0.0]), np.array([1.0]), dt, dz, 1, np.array([1]), r, "mb")
        Phi0, Nres, Nnr = out["int0"][0, :3]
        lost = (Phi0 - out["flu"][0, 0]) / dz
        expect = r["n_a"] * (Nres + Nnr)
        log(f"  photon bookkeeping J_pk={J_pk:.0e}: lost/absorbed - 1 = {lost / expect - 1:.2e}")
        assert abs(lost / expect - 1) < 1e-3, "field loss does not match the absorbed photons"

    f = tl_shape(t, tau) * np.exp(-1j * (5.0 / HBAR) * t)
    E = spectrum_axis(t.size, dt, 8000.0)
    peak = E[np.argmax(np.abs(spectral_amplitude(f)))]
    log(f"  spectral convention: +5 eV carrier offset peaks at {peak - 8000.0:+.3f} eV")
    assert abs(peak - 8005.0) < 0.1, "spectral axis sign is wrong"

    dt_s = 0.002
    t = time_grid(25.0, dt_s)
    chaotic = sase_shapes(1, t, tau, 15.0, np.random.default_rng(1))[0]
    for label, shape in (("TL", tl_shape(t, tau)), ("chaotic", chaotic)):
        Om = np.sqrt(r["s0G"] * 1e19) * shape
        for mode in MODES:
            out = march(Om[None, :], np.array([0.0]), np.array([1.0]), dt_s, 1e-9, 0, np.array([0]), r, mode)
            Nres, Nq = out["int0"][0, 1], out["int0"][0, 5]
            log(f"  {label} pulse, {mode}: field-side / atom-side resonant photons - 1 = {Nres / Nq - 1:.1e}")
            assert abs(Nres / Nq - 1) < 1e-9, "resonant photons lost by the field != absorbed by the atoms"

    J_pk = beam_peak(cfg["mono"], 500.0, r["E_ul"])[1]
    Om_pk = np.sqrt(r["s0G"] * J_pk)
    dt_s, m = choose_dt(cfg["grid"], Om_pk, stiff_rate(r, Om_pk, "re"))
    t = time_grid(25.0, dt_s)
    out = march((Om_pk * tl_shape(t, tau))[None, :], np.array([0.0]), np.array([1.0]), dt_s, 1e-9, 0,
                np.array([0]), r, "re", ts_slot=np.array([0]), ts_stride=1)
    pops = out["ts"][0, :3].astype(float)
    ok = np.all(np.isfinite(pops)) and pops.min() > -1e-6 and pops.max() < 1 + 1e-6
    log(f"  re at 500 uJ (Omega_pk/gamma0 = {Om_pk / weak_field_constants(r)['gamma']:.0f}, dt = {dt_s:.2e} fs): "
        f"populations in [{pops.min():.2e}, {pops.max():.3f}]")
    assert ok, "rate-equation mode is unstable at high field"
    return True


def pipeline_check(cfg, log=print):
    """Tiny end-to-end runs of run_mono / run_sase (seconds) that check their normalisation:
      1. mono, very weak pulse, on axis at the line centre: the entrance-plane kappa_res equals
         the exact linear response of the same pulse (2p3/2 holes made by J(t) and decaying at
         Gamma_l, probed through the Lorentzian coherence), computed here by direct quadrature.
      2. sase: -ln T(E) / dz of a one-step slab equals the entrance-plane kappa_0(E)
         (resonant + non-resonant) to O(kappa dz)."""
    small = apply_overrides(cfg, [
        "grid.zmax_um=0.1", "grid.dz_um=0.1", "grid.z_record_um=[0.1]", "grid.n_v=2",
        "mono.scan_eV=[[-1.0, 1.0, 1.0]]", "mono.bloch_detunings_eV=[0.0]",
        "sase.n_shots=4", "sase.n_blocks=2", "sase.n_ref_shots=2", "sase.n_bloch_shots=1"])
    quiet = lambda *a, **k: None
    r = rates(small["atom"])

    E_uJ = 1e-5
    res = run_mono(small, E_uJ, "mb", log=quiet)
    j, ax = int(np.argmin(np.abs(res["dE"]))), res["v"].size - 1
    Phi, Nres = res["int0"][j, ax, :2]
    t = time_grid(small["grid"]["t_half_window_fs"], res["dt"])
    dt = res["dt"]
    Om = np.sqrt(r["s0G"] * res["J_pk"][j]) * tl_shape(t, small["mono"]["tau_fs"])
    J = np.abs(Om) ** 2 / r["s0G"]

    def lin_filter(src, lam):
        """y' = -lam y + src, trapezoidal exponential integrator."""
        y = np.zeros_like(src)
        e = np.exp(-lam * dt)
        for i in range(src.size - 1):
            y[i + 1] = y[i] * e + 0.5 * dt * (src[i] * e + src[i + 1])
        return y

    l_lin = lin_filter(r["sig_L3"] * J + 0j, r["Gl"]).real
    c_lin = lin_filter(0.5j * Om * l_lin, 0.5 * (r["Gu"] + r["Gl"]) + r["gphi"])
    N_lin = np.sum(np.imag(np.conj(Om) * c_lin)) * dt
    err = Nres / N_lin - 1
    log(f"  mono {E_uJ:g} uJ on axis: kappa_res(line) = {r['n_a'] * Nres / Phi:.4e} /cm, "
        f"exact linear response {r['n_a'] * N_lin / Phi:.4e} /cm ({err:+.1e})")
    assert abs(err) < 1e-3, "mono entrance-plane absorption does not match the linear response"

    res = run_sase(small, 1.0, "mb", log=quiet)
    ax = res["v"].size - 1
    den = res["den"].sum(axis=0)
    T_slab = res["num_out_view"][0].sum(axis=0)[0] / den
    k_slab = -np.log(T_slab) / (res["z_rec_um"][0] * 1e-4)
    k0 = (res["kres_num"] + res["knr_num"]).sum(axis=0)[ax] / den
    band = np.abs(res["E_axis"] - small["sase"]["E_c"]) < 10.0
    err = np.max(np.abs(k_slab[band] / k0[band] - 1))
    log(f"  sase 1 uJ on axis: one-step slab vs entrance-plane kappa, max rel. difference {err:.1e} "
        f"(kappa dz ~ {np.max(k0[band]) * res['z_rec_um'][0] * 1e-4:.1e})")
    assert err < 1e-2, "sase slab transmission does not match the entrance-plane absorption"
    return True
