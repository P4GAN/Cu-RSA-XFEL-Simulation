import numpy as np

from . import model
from .optics import Optics
from .pulses import linear_to_circular


def rk4_step(f, y, t0, dt, params):
    """Increment of y over one classical fourth-order Runge-Kutta step of dy/dt = f(t, y, params)."""
    k1 = f(t0, y, params)
    k2 = f(t0 + 0.5 * dt, y + 0.5 * k1 * dt, params)
    k3 = f(t0 + 0.5 * dt, y + 0.5 * k2 * dt, params)
    k4 = f(t0 + dt, y + k3 * dt, params)
    return (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0


class Sample:
    """
    The z-marching loop. The sample is cut into zgrid planes. At each plane, for every time step, the
    atoms are integrated forward one RK4 step, driven by the local field; the field is then propagated
    to the next plane (Fresnel diffraction plus photoionisation absorption), and the field radiated by
    the atomic coherences is added. The field that leaves plane iz drives plane iz + 1.

    Two implementations of the same numerics: _run_full keeps every z plane of every array (for
    plot.py), _run_lean keeps a rolling two-plane window plus the entrance and exit fields, and the full
    density matrices at the centre pixel of the last plane (for sweeps).
    """

    def __init__(self, X, seed_field):
        self.optics = Optics(X)
        self.seed_field = linear_to_circular(X, seed_field)

    @staticmethod
    def _photon_flux(X, Omega_ps):
        """Photon fluxes (J_minus, J_plus) of the two circular polarisations, for a field with leading
        axes (p, s)."""
        J_minus = np.real(Omega_ps[0, 0] * Omega_ps[1, 0] / X.flux_factor)
        J_plus = np.real(Omega_ps[0, 1] * Omega_ps[1, 1] / X.flux_factor)
        return J_minus, J_plus

    @staticmethod
    def _step_increments(X, it, Omega_it, rho_ijxy, rho_sat_ijxy, rho_ground_xy, rho_other_xy, rho_2s_xy,
                         rho_mid_xy, rho_e_gxy, J_minus_xy, J_plus_xy):
        """RK4 increments of every population over time step it, all driven by start-of-step values."""
        t0 = it * X.dt
        mid_xy = rho_mid_xy if X.use_middlemen else None
        eii_R_xy = model.eii_rates_xy(X, rho_e_gxy) if X.use_eii else None

        d_rho = rk4_step(model.base_block_rhs, rho_ijxy, t0, X.dt,
                                [X, Omega_it, rho_ground_xy, rho_2s_xy, J_minus_xy, J_plus_xy, eii_R_xy])
        d_other = rk4_step(model.other_rhs, rho_other_xy, t0, X.dt,
                                  [X, rho_ground_xy, J_minus_xy, J_plus_xy, eii_R_xy])
        d_2s = rk4_step(model.twos_rhs, rho_2s_xy, t0, X.dt,
                               [X, rho_ground_xy, J_minus_xy, J_plus_xy, eii_R_xy])
        eii_ground_loss_xy = np.sum(eii_R_xy, axis=0) * rho_ground_xy if X.use_eii else None
        d_ground = rk4_step(model.ground_rhs, rho_ground_xy, t0, X.dt,
                                   [X, J_minus_xy, J_plus_xy, eii_ground_loss_xy])
        d_sat = [
            rk4_step(model.satellite_block_rhs, rho_sat_ijxy[k], t0, X.dt,
                            [X, chan, Omega_it, rho_ijxy, rho_2s_xy, rho_sat_ijxy, J_minus_xy, J_plus_xy,
                             mid_xy, eii_R_xy])
            for k, chan in enumerate(X.satellite_channel_params)
        ]

        d_mid = d_e = None
        if X.use_middlemen:
            gain_xy, loss_rate_xy = model.middleman_gain_loss(X, rho_ground_xy, rho_other_xy, rho_2s_xy, rho_ijxy,
                                                              rho_sat_ijxy, J_minus_xy, J_plus_xy, eii_R_xy)
            d_mid = rk4_step(model.middleman_rhs, rho_mid_xy, t0, X.dt, [gain_xy, loss_rate_xy])
        if X.use_eii:
            prod_gxy = model.electron_production_gxy(X, rho_ground_xy, rho_other_xy, rho_2s_xy, mid_xy, rho_ijxy,
                                                     rho_sat_ijxy, J_minus_xy, J_plus_xy, rho_e_gxy)
            d_e = rk4_step(model.electron_rhs, rho_e_gxy, t0, X.dt, [X, prod_gxy])
        return d_rho, d_other, d_2s, d_ground, d_sat, d_mid, d_e

    def _propagate(self, X, Omega_psxy, kappa_Omega_psxyz, rho_ijxy, rho_sat_ijxy):
        """Field leaving this plane: diffraction and absorption over the step from the previous plane
        (none at the entrance face, where kappa is None), plus the field the coherences radiate."""
        if kappa_Omega_psxyz is not None:
            Omega_psxy = self.optics.propagate(X, Omega_psxy, kappa_Omega_psxyz)
        Omega_psxy += X.dz * model.field_source(X, rho_ijxy)
        for rho_sat in rho_sat_ijxy:
            Omega_psxy += X.dz * model.field_source(X, rho_sat)
        return Omega_psxy

    def run(self, X):
        if X.keep_z_history:
            if getattr(X, "recorder", None) is not None:
                raise ValueError("a recorder only runs on the lean path: set X.keep_z_history = False")
            self._run_full(X)
        else:
            self._run_lean(X)

    def _run_full(self, X):
        nt, nx, ny, nz = X.tgrid, X.xgrid, X.ygrid, X.zgrid
        n_sat = len(X.satellite_channel_params)

        rho_ijtxyz = np.zeros((X.nlevel, X.nlevel, nt, nx, ny, nz), dtype=complex)
        rho_sat_ijtxyz = [np.zeros((X.nlevel, X.nlevel, nt, nx, ny, nz), dtype=complex) for _ in range(n_sat)]
        rho_ground_txyz = np.ones((nt, nx, ny, nz), dtype=complex)
        rho_other_txyz = np.zeros((nt, nx, ny, nz), dtype=complex)
        rho_2s_txyz = np.zeros((nt, nx, ny, nz), dtype=complex)
        rho_mid_txyz = np.zeros((nt, nx, ny, nz)) if X.use_middlemen else None
        rho_e_gtxyz = np.zeros((X.eii_G, nt, nx, ny, nz)) if X.use_eii else None

        # Omega_pstxyz[..., iz] is the field that drives plane iz; the last slot ends up holding the
        # field that leaves the sample.
        Omega_pstxyz = np.zeros((2, 2, nt, nx, ny, nz), dtype=complex)
        Omega_pstxyz[:, :, :, :, :, 0] = self.seed_field
        Omega_pstxy = Omega_pstxyz[:, :, :, :, :, 0].copy()
        J_Omega_minus_txy, J_Omega_plus_txy = np.zeros((2, nt, nx, ny))
        J_Omega_minus_txy[:], J_Omega_plus_txy[:] = self._photon_flux(X, Omega_pstxy)

        for iz in range(nz):
            rho_ijxy = rho_ijtxyz[:, :, 0, :, :, iz].copy()
            rho_sat_ijxy = [rho_sat_ijtxyz[k][:, :, 0, :, :, iz].copy() for k in range(n_sat)]
            rho_ground_xy = rho_ground_txyz[0, :, :, iz].copy()
            rho_other_xy = rho_other_txyz[0, :, :, iz].copy()
            rho_2s_xy = rho_2s_txyz[0, :, :, iz].copy()
            rho_mid_xy = np.zeros((nx, ny)) if X.use_middlemen else None
            rho_e_gxy = np.zeros((X.eii_G, nx, ny)) if X.use_eii else None

            for it in range(nt):
                d_rho, d_other, d_2s, d_ground, d_sat, d_mid, d_e = self._step_increments(
                    X, it, Omega_pstxy[:, :, it, :, :], rho_ijxy, rho_sat_ijxy, rho_ground_xy, rho_other_xy,
                    rho_2s_xy, rho_mid_xy, rho_e_gxy, J_Omega_minus_txy[it], J_Omega_plus_txy[it])

                rho_ijxy += d_rho
                rho_ground_xy += d_ground
                rho_other_xy += d_other
                rho_2s_xy += d_2s
                if X.use_middlemen:
                    rho_mid_xy = rho_mid_xy + d_mid
                    rho_mid_txyz[it, :, :, iz] = rho_mid_xy
                if X.use_eii:
                    rho_e_gxy = rho_e_gxy + d_e
                    rho_e_gtxyz[:, it, :, :, iz] = rho_e_gxy
                for k in range(n_sat):
                    rho_sat_ijxy[k] = rho_sat_ijxy[k] + d_sat[k]
                    rho_sat_ijtxyz[k][:, :, it, :, :, iz] = rho_sat_ijxy[k]
                rho_ground_txyz[it, :, :, iz] = rho_ground_xy
                rho_other_txyz[it, :, :, iz] = rho_other_xy
                rho_2s_txyz[it, :, :, iz] = rho_2s_xy
                rho_ijtxyz[:, :, it, :, :, iz] = rho_ijxy

                # absorption over the step from plane iz - 1 to plane iz (plane 0 is the entrance face)
                kappa_Omega_psxyz = None
                if iz != 0:
                    window = slice(iz - 1, iz + 1)
                    kappa_Omega_psxyz = model.absorption(
                        X, rho_ground_txyz[it, :, :, window], rho_other_txyz[it, :, :, window],
                        rho_2s_txyz[it, :, :, window], rho_ijtxyz[:, :, it, :, :, window],
                        [rho_sat_ijtxyz[k][:, :, it, :, :, window] for k in range(n_sat)],
                        rho_mid_txyz[it, :, :, window] if X.use_middlemen else None)
                Omega_pstxy[:, :, it, :, :] = self._propagate(
                    X, Omega_pstxy[:, :, it, :, :], kappa_Omega_psxyz, rho_ijtxyz[:, :, it, :, :, iz],
                    [rho_sat_ijtxyz[k][:, :, it, :, :, iz] for k in range(n_sat)])
                J_Omega_minus_txy[it], J_Omega_plus_txy[it] = self._photon_flux(X, Omega_pstxy[:, :, it, :, :])

            Omega_pstxyz[:, :, :, :, :, min(iz + 1, nz - 1)] = Omega_pstxy

        self.rho_ijtxyz = rho_ijtxyz
        self.rho_sat_ijtxyz = rho_sat_ijtxyz
        self.Omega_pstxyz = Omega_pstxyz
        self.rho_ground_txyz = np.real(rho_ground_txyz)
        self.rho_2s_txyz = np.real(rho_2s_txyz)
        self.rho_other_txyz = np.real(rho_other_txyz)
        self.rho_mid_txyz = rho_mid_txyz
        self.rho_e_gtxyz = rho_e_gtxyz

    def _run_lean(self, X):
        """
        The same numerics as _run_full, keeping only what the sweep outputs read: the
        entrance and exit fields, and every population at the centre pixel of the last plane. The
        absorption only needs the populations of the previous plane, so those are kept in a rolling
        window (level diagonals only, in single precision). An optional X.recorder is handed every
        step's state; it only reads it.
        """
        nlevel, nt, nx, ny, nz = X.nlevel, X.tgrid, X.xgrid, X.ygrid, X.zgrid
        n_sat = len(X.satellite_channel_params)
        recorder = getattr(X, "recorder", None)
        cx, cy = int(X.xgrid / 2), int(X.ygrid / 2)
        diag = np.arange(nlevel)

        Omega_pstxy = self.seed_field.copy()
        Omega_pstxy_entrance = Omega_pstxy.copy()
        J_Omega_minus_txy, J_Omega_plus_txy = np.zeros((2, nt, nx, ny))
        J_Omega_minus_txy[:], J_Omega_plus_txy[:] = self._photon_flux(X, Omega_pstxy)

        prev = None
        for iz in range(nz):
            rho_ijxy = np.zeros((nlevel, nlevel, nx, ny), dtype=complex)
            rho_sat_ijxy = [np.zeros((nlevel, nlevel, nx, ny), dtype=complex) for _ in range(n_sat)]
            rho_ground_xy = np.ones((nx, ny), dtype=complex)
            rho_other_xy = np.zeros((nx, ny), dtype=complex)
            rho_2s_xy = np.zeros((nx, ny), dtype=complex)
            rho_mid_xy = np.zeros((nx, ny)) if X.use_middlemen else None
            rho_e_gxy = np.zeros((X.eii_G, nx, ny)) if X.use_eii else None

            # this plane's history, read back as the previous plane by the next one
            curr = {
                'ground': np.empty((nt, nx, ny), dtype=np.float32),
                'other': np.empty((nt, nx, ny), dtype=np.float32),
                '2s': np.empty((nt, nx, ny), dtype=np.float32),
                'diag': np.empty((nlevel, nt, nx, ny), dtype=np.complex64),
                'sat_diag': [np.empty((nlevel, nt, nx, ny), dtype=np.complex64) for _ in range(n_sat)],
                'mid': np.empty((nt, nx, ny), dtype=np.float32) if X.use_middlemen else None,
            }

            is_last_plane = iz == nz - 1
            if is_last_plane:
                rho_ijt_centre = np.empty((nlevel, nlevel, nt), dtype=complex)
                rho_sat_ijt_centre = [np.empty((nlevel, nlevel, nt), dtype=complex) for _ in range(n_sat)]
                rho_e_gt_centre = np.empty((X.eii_G, nt)) if X.use_eii else None

            for it in range(nt):
                d_rho, d_other, d_2s, d_ground, d_sat, d_mid, d_e = self._step_increments(
                    X, it, Omega_pstxy[:, :, it, :, :], rho_ijxy, rho_sat_ijxy, rho_ground_xy, rho_other_xy,
                    rho_2s_xy, rho_mid_xy, rho_e_gxy, J_Omega_minus_txy[it], J_Omega_plus_txy[it])

                rho_ijxy += d_rho
                rho_ground_xy += d_ground
                rho_other_xy += d_other
                rho_2s_xy += d_2s
                if X.use_middlemen:
                    rho_mid_xy = rho_mid_xy + d_mid
                    curr['mid'][it] = rho_mid_xy
                if X.use_eii:
                    rho_e_gxy = rho_e_gxy + d_e
                    if is_last_plane:
                        rho_e_gt_centre[:, it] = rho_e_gxy[:, cx, cy]
                for k in range(n_sat):
                    rho_sat_ijxy[k] = rho_sat_ijxy[k] + d_sat[k]
                    curr['sat_diag'][k][:, it] = rho_sat_ijxy[k][diag, diag]
                    if is_last_plane:
                        rho_sat_ijt_centre[k][:, :, it] = rho_sat_ijxy[k][:, :, cx, cy]
                curr['ground'][it] = np.real(rho_ground_xy)
                curr['other'][it] = np.real(rho_other_xy)
                curr['2s'][it] = np.real(rho_2s_xy)
                curr['diag'][:, it] = rho_ijxy[diag, diag]
                if is_last_plane:
                    rho_ijt_centre[:, :, it] = rho_ijxy[:, :, cx, cy]
                if recorder is not None:
                    # the field here is still the one that drove this step
                    recorder.record_step(it, Omega_pstxy[:, :, it, :, :], rho_ijxy, rho_sat_ijxy,
                                         rho_ground_xy, rho_other_xy, rho_2s_xy, rho_mid_xy, rho_e_gxy)

                kappa_Omega_psxyz = None
                if iz != 0:
                    kappa_Omega_psxyz = model.absorption(
                        X, _window(prev['ground'][it], curr['ground'][it]),
                        _window(prev['other'][it], curr['other'][it]),
                        _window(prev['2s'][it], curr['2s'][it]),
                        _diag_window(prev['diag'][:, it], curr['diag'][:, it]),
                        [_diag_window(prev['sat_diag'][k][:, it], curr['sat_diag'][k][:, it]) for k in range(n_sat)],
                        _window(prev['mid'][it], curr['mid'][it]) if X.use_middlemen else None)
                Omega_pstxy[:, :, it, :, :] = self._propagate(X, Omega_pstxy[:, :, it, :, :], kappa_Omega_psxyz,
                                                              rho_ijxy, rho_sat_ijxy)
                J_Omega_minus_txy[it], J_Omega_plus_txy[it] = self._photon_flux(X, Omega_pstxy[:, :, it, :, :])

            if recorder is not None:
                recorder.end_plane(iz, Omega_pstxy)
            prev = curr

        # Everything is stored with singleton (x, y, z) axes, so the output has the same layout as the
        # full path's, at one pixel and one plane.
        self.rho_ijtxyz = rho_ijt_centre[:, :, :, None, None, None]
        self.rho_sat_ijtxyz = [rho[:, :, :, None, None, None] for rho in rho_sat_ijt_centre]
        self.Omega_pstxyz = np.stack([Omega_pstxy_entrance, Omega_pstxy], axis=-1)
        self.rho_ground_txyz = curr['ground'][:, cx, cy][:, None, None, None]
        self.rho_2s_txyz = curr['2s'][:, cx, cy][:, None, None, None]
        self.rho_other_txyz = curr['other'][:, cx, cy][:, None, None, None]
        self.rho_mid_txyz = curr['mid'][:, cx, cy][:, None, None, None] if X.use_middlemen else None
        self.rho_e_gtxyz = rho_e_gt_centre[:, :, None, None, None] if X.use_eii else None


def _window(prev_xy, curr_xy):
    """Populations of the previous and current plane, stacked along a last axis of length 2."""
    return np.stack([prev_xy, curr_xy], axis=-1).astype(complex)


def _diag_window(prev_ixy, curr_ixy):
    """A (level, level, x, y, 2) window holding only the level diagonals, which is all model.absorption reads."""
    nlevel = prev_ixy.shape[0]
    diag = np.arange(nlevel)
    window = np.zeros((nlevel, nlevel) + prev_ixy.shape[1:] + (2,), dtype=complex)
    window[diag, diag, ..., 0] = prev_ixy
    window[diag, diag, ..., 1] = curr_ixy
    return window
