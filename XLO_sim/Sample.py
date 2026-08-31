import numpy as np
import matplotlib.pyplot as plt
from . import tools
from . import Model
from . import Optics as XLO_optics


class XLO_sample:

    
    def __init__(self, X, seed_field=None):
        self.optics = XLO_optics.XLO_optics(X)
        self.optics.Greens_function_numerical_3D(X)
        X.Gxyz = self.optics.Gxyz
        
        if (seed_field is None):
            self.is_seeded = False  
            # Simulation startup: noise
        else:
            self.is_seeded = True
            # Simulation startup: seeded
            # Seed field will be converted to RH and LH polarization
            self.seed_field = tools.linear_to_circular(X, seed_field)
    
            
    def init_n_level_3D(self, X):
        """
        Create arrays for storing the density matrix elements and emitted field amplitudes for propagation with and without field absorption. Check if the seed field is present and set up the initial conditions.

        Parameters
        ----------
        X
            XLO_sim object

        Returns
        -------
        np.ndarray
            Density matrix.
        np.ndarray
            ASE field amplitude.

        """

        rho_ijtxyz = np.zeros((X.nlevel, X.nlevel, X.tgrid, X.xgrid, X.ygrid, X.zgrid), dtype=complex)
        Omega_pstxyz = (0.0 + 0 * 1j) * np.ones((2, 2, X.tgrid, X.xgrid, X.ygrid, X.zgrid), dtype=complex)

        if (self.is_seeded == True):
            Omega_pstxyz[:,:,:,:,:,0] = self.seed_field
            
        rho_ground_txyz = np.ones((X.tgrid, X.xgrid, X.ygrid, X.zgrid), dtype=complex)
        rho_other_txyz = np.zeros((X.tgrid, X.xgrid, X.ygrid, X.zgrid), dtype=complex)
        rho_2s_txyz = np.zeros((X.tgrid, X.xgrid, X.ygrid, X.zgrid), dtype=complex)

        # One local X.satellite_nlevel-level block per channel (not X.nlevel, which grows
        # independently when use_L2_pathway extends the base block). Empty list if none configured.
        rho_sat_ijtxyz = [np.zeros((X.satellite_nlevel, X.satellite_nlevel, X.tgrid, X.xgrid, X.ygrid, X.zgrid), dtype=complex)
                          for _ in X.satellite_channel_params]

        J_Omega_minus_txy = np.zeros((X.tgrid, X.xgrid, X.ygrid))
        J_Omega_plus_txy = np.zeros((X.tgrid, X.xgrid, X.ygrid))

        return rho_ground_txyz, rho_other_txyz, rho_2s_txyz, rho_ijtxyz, rho_sat_ijtxyz, Omega_pstxyz, J_Omega_minus_txy, J_Omega_plus_txy


  
    def evaluate_n_level_3D(self, X):
        """
        Perform calculation of the ASE field generation and ionic density matrices evolution, with and without field absorption.

        Parameters
        ----------
        X
            XLO_sim object

        Returns
        -------

        """
        if getattr(X, "keep_z_history", True):
            self._evaluate_n_level_3D_full(X)
        else:
            self._evaluate_n_level_3D_lean(X)

    def _evaluate_n_level_3D_full(self, X):
        """
        Original full-z-history implementation of evaluate_n_level_3D. Stores every z plane of
        every array, which Plot.py/notebooks need but costs tens of GB per repetition at
        production grid sizes -- batch run_*_sweep.py scripts set X.keep_z_history=False to use
        _evaluate_n_level_3D_lean instead (same physics, only keeps what's read back).
        """

        rho_ground_txyz, rho_other_txyz, rho_2s_txyz, rho_ijtxyz, rho_sat_ijtxyz, Omega_pstxyz, J_Omega_minus_txy, J_Omega_plus_txy = self.init_n_level_3D(X)

        Omega_pstxy = Omega_pstxyz[:, :, :, :, :, 0].copy()
        
        n_sat = len(X.satellite_channel_params)

        # Main loop begins. Iterate over longitudinal coordinate
        for iz in range(0, X.zgrid):

            rho_ijxy = rho_ijtxyz[:, :, 0, :, :, iz].copy()
            rho_sat_ijxy = [rho_sat_ijtxyz[k][:, :, 0, :, :, iz].copy() for k in range(n_sat)]

            rho_ground_xy = rho_ground_txyz[0, :, :, iz].copy()
            rho_other_xy = rho_other_txyz[0, :, :, iz].copy()
            rho_2s_xy = rho_2s_txyz[0, :, :, iz].copy()

            ######################
            # Loop over simulation time window begins
            for it in range(0, X.tgrid):
                d_rho_it_reg = tools.RK45_step(Model.MB_nlevel_regular, rho_ijxy, it * X.dt, X.dt, [X, Omega_pstxy[:, :, it, :, :], rho_ground_xy, rho_2s_xy, J_Omega_minus_txy[it, :, :], J_Omega_plus_txy[it, :, :]])
                d_rho_other_it = tools.RK45_step(Model.MB_other_regular, rho_other_xy, it * X.dt, X.dt, [X, rho_ground_xy, J_Omega_minus_txy[it, :, :], J_Omega_plus_txy[it, :, :]])
                d_rho_2s_it = tools.RK45_step(Model.MB_2s_regular, rho_2s_xy, it * X.dt, X.dt, [X, rho_ground_xy, J_Omega_minus_txy[it, :, :], J_Omega_plus_txy[it, :, :]])
                d_rho_ground_it = tools.RK45_step(Model.MB_ground_regular, rho_ground_xy, it * X.dt, X.dt, [X, J_Omega_minus_txy[it, :, :], J_Omega_plus_txy[it, :, :]])

                # One block per satellite channel, fed from the base block's/rho_2s_xy's pre-update
                # (start-of-step) values, same convention as the other blocks above.
                d_rho_sat_it = [
                    tools.RK45_step(Model.MB_satellite_block_regular, rho_sat_ijxy[k], it * X.dt, X.dt,
                                     [X, chan, Omega_pstxy[:, :, it, :, :], rho_ijxy, rho_2s_xy, rho_sat_ijxy,
                                      J_Omega_minus_txy[it, :, :], J_Omega_plus_txy[it, :, :]])
                    for k, chan in enumerate(X.satellite_channel_params)
                ]

                rho_ijxy += d_rho_it_reg
                rho_ground_xy += d_rho_ground_it
                rho_other_xy += d_rho_other_it
                rho_2s_xy += d_rho_2s_it
                for k in range(n_sat):
                    rho_sat_ijxy[k] = rho_sat_ijxy[k] + d_rho_sat_it[k]
                    rho_sat_ijtxyz[k][:, :, it, :, :, iz] = rho_sat_ijxy[k]

                rho_ground_txyz[it, :, :, iz] = rho_ground_xy
                rho_other_txyz[it, :, :, iz] = rho_other_xy
                rho_2s_txyz[it, :, :, iz] = rho_2s_xy
                rho_ijtxyz[:, :, it, :, :, iz] = rho_ijxy

                rho_sat_ijxyz_list = [rho_sat_ijtxyz[k][:, :, it, :, :, iz-1:iz+1] for k in range(n_sat)]

                if (iz != 0):
                    kappa_Omega_psxyz = Model.absorption(X, rho_ground_txyz[it, :, :, iz-1:iz+1], rho_other_txyz[it, :, :, iz-1:iz+1],
                                                         rho_2s_txyz[it, :, :, iz-1:iz+1],
                                                         rho_ijtxyz[:, :, it, :, :, iz-1:iz+1],
                                                         rho_sat_ijxyz_list)
                    Omega_pstxy[:, :, it, :, :] = self.optics.Fresnel_propagator_with_absorption(X, Omega_pstxy[:, :, it, :, :], X.dz, iz * X.dz, kappa_Omega_psxyz, X.lambdaKalpha1N)

                Omega_pstxy[:, :, it, :, :] +=  1.0 * X.dz * Model.Omega_source_regular(X, rho_ijtxyz[:, :, it, :, :, iz])
                for k in range(n_sat):
                    Omega_pstxy[:, :, it, :, :] += 1.0 * X.dz * Model.Omega_source_regular(
                        X, rho_sat_ijtxyz[k][:, :, it, :, :, iz], X.Tijs_plus_satellite, X.Tijs_minus_satellite)

                J_Omega_minus_txy[it, :, :] = np.real(Omega_pstxy[0, 0, it, :, :] * Omega_pstxy[1, 0, it, :, :] / X.flux_factor)
                J_Omega_plus_txy[it, :, :] = np.real(Omega_pstxy[0, 1, it, :, :] * Omega_pstxy[1, 1, it, :, :] / X.flux_factor)

            # ######################
            # Loop over simulation time window ends

            if (iz != X.zgrid-1):
                Omega_pstxyz[:, :, :, :, :, iz + 1] = Omega_pstxy

        ######################
        # Main loop ends

        self.rho_ijtxyz = rho_ijtxyz
        self.rho_sat_ijtxyz = rho_sat_ijtxyz
        self.Omega_pstxyz = Omega_pstxyz

        self.rho_ground_txyz = np.real(rho_ground_txyz)

        self.rho_2s_txyz = np.real(rho_2s_txyz)
        self.rho_other_txyz = np.real(rho_other_txyz)

    def _evaluate_n_level_3D_lean(self, X):
        """
        Memory-lean counterpart of _evaluate_n_level_3D_full for batch/statistics jobs
        (X.keep_z_history=False): identical z-marching physics, but keeps only what
        tools.compute_run_outputs reads back instead of every z plane -- 2-slot rolling prev/curr
        buffers for rho_ground/other/2s_txyz and for the diagonal of rho_ijtxyz/rho_sat_ijtxyz
        (Model.absorption only reads the diagonal of the latter two), plus a center-pixel-only,
        final-z-only full matrix for rho_ijtxyz/rho_sat_ijtxyz. rho_ground/other/2s_txyz are also
        narrowed to float32/complex64 (their imaginary part is always exactly 0).

        Deliberately reproduces _evaluate_n_level_3D_full's off-by-one for Omega_pstxyz (index -1
        there is the state after the *second-to-last* iz, not the true final one) so lean and full
        mode agree exactly -- do not "fix" this without also changing the full-history path.

        Not used for interactive/notebook work -- Plot.py needs the full z/x/y profile.
        """

        nlevel, tgrid, xgrid, ygrid, zgrid = X.nlevel, X.tgrid, X.xgrid, X.ygrid, X.zgrid
        cx, cy = int(X.xgrid / 2), int(X.ygrid / 2)  # matches tools.compute_run_outputs exactly

        # t=0 initial-condition template, identical for every z -- re-copied into the rolling "xy"
        # working array at the start of every iz iteration rather than stored per-z.
        rho_ij_ic_xy = np.zeros((nlevel, nlevel, xgrid, ygrid), dtype=complex)

        n_sat = len(X.satellite_channel_params)
        satellite_nlevel = X.satellite_nlevel
        rho_sat_ij_ic_xy = [np.zeros((satellite_nlevel, satellite_nlevel, X.xgrid, X.ygrid), dtype=complex)
                            for k in range(n_sat)]

        rho_ground_ic_xy = np.ones((xgrid, ygrid), dtype=complex)
        rho_other_ic_xy = np.zeros((xgrid, ygrid), dtype=complex)
        rho_2s_ic_xy = np.zeros((xgrid, ygrid), dtype=complex)

        if self.is_seeded == True:
            Omega_pstxy = self.seed_field.copy()
        else:
            Omega_pstxy = np.zeros((2, 2, tgrid, xgrid, ygrid), dtype=complex)

        Omega_pstxyz_z0 = Omega_pstxy.copy()
        Omega_pstxyz_zlast = Omega_pstxyz_z0

        J_Omega_minus_txy = np.zeros((tgrid, xgrid, ygrid))
        J_Omega_plus_txy = np.zeros((tgrid, xgrid, ygrid))

        diag_idx = np.arange(nlevel)
        sat_diag_idx = np.arange(satellite_nlevel)

        prev_rho_ground_txy = prev_rho_other_txy = prev_rho_2s_txy = prev_rho_diag_txy = None
        prev_rho_sat_diag_txy = [None for k in range(n_sat)]

        # Populated only during the FINAL iz (see docstring) -- the full (off-diagonal included)
        # but center-pixel-only history that becomes self.rho_ijtxyz/rho_sat_ijtxyz.
        final_rho_ijt_center = None
        final_rho_sat_ijt_center = [None for k in range(n_sat)]

        for iz in range(0, zgrid):

            rho_ijxy = rho_ij_ic_xy.copy()
            rho_sat_ijxy = [rho_sat_ij_ic_xy[k].copy() for k in range(n_sat)]
            rho_ground_xy = rho_ground_ic_xy.copy()
            rho_other_xy = rho_other_ic_xy.copy()
            rho_2s_xy = rho_2s_ic_xy.copy()

            # float32/complex64: these buffers only ever get read back through Model.absorption()'s
            # lookback window (upcast to complex there), never fed into the RK4 state itself, so
            # this only narrows the stored history trace -- the single biggest memory lever on
            # run_mono_sweep.py's OOM'ing workers.
            curr_rho_ground_txy = np.empty((tgrid, xgrid, ygrid), dtype=np.float32)
            curr_rho_other_txy = np.empty((tgrid, xgrid, ygrid), dtype=np.float32)
            curr_rho_2s_txy = np.empty((tgrid, xgrid, ygrid), dtype=np.float32)
            curr_rho_diag_txy = np.empty((nlevel, tgrid, xgrid, ygrid), dtype=np.complex64)
            curr_rho_sat_diag_txy = [np.empty((satellite_nlevel, tgrid, xgrid, ygrid), dtype=np.complex64)
                                     for k in range(n_sat)]

            is_final_iz = (iz == zgrid - 1)
            if is_final_iz:
                final_rho_ijt_center = np.empty((nlevel, nlevel, tgrid), dtype=complex)
                final_rho_sat_ijt_center = [np.empty((satellite_nlevel, satellite_nlevel, tgrid), dtype=complex)
                                            for k in range(n_sat)]

            ######################
            # Loop over simulation time window begins
            for it in range(0, tgrid):
                d_rho_it_reg = tools.RK45_step(Model.MB_nlevel_regular, rho_ijxy, it * X.dt, X.dt, [X, Omega_pstxy[:, :, it, :, :], rho_ground_xy, rho_2s_xy, J_Omega_minus_txy[it, :, :], J_Omega_plus_txy[it, :, :]])
                d_rho_other_it = tools.RK45_step(Model.MB_other_regular, rho_other_xy, it * X.dt, X.dt, [X, rho_ground_xy, J_Omega_minus_txy[it, :, :], J_Omega_plus_txy[it, :, :]])
                d_rho_2s_it = tools.RK45_step(Model.MB_2s_regular, rho_2s_xy, it * X.dt, X.dt, [X, rho_ground_xy, J_Omega_minus_txy[it, :, :], J_Omega_plus_txy[it, :, :]])
                d_rho_ground_it = tools.RK45_step(Model.MB_ground_regular, rho_ground_xy, it * X.dt, X.dt, [X, J_Omega_minus_txy[it, :, :], J_Omega_plus_txy[it, :, :]])

                # One block per satellite channel, fed from the base block's/rho_2s_xy's pre-update
                # (start-of-step) values, same convention as the other blocks above.
                d_rho_sat_it = [
                    tools.RK45_step(Model.MB_satellite_block_regular, rho_sat_ijxy[k], it * X.dt, X.dt,
                                     [X, chan, Omega_pstxy[:, :, it, :, :], rho_ijxy, rho_2s_xy, rho_sat_ijxy,
                                      J_Omega_minus_txy[it, :, :], J_Omega_plus_txy[it, :, :]])
                    for k, chan in enumerate(X.satellite_channel_params)
                ]

                rho_ijxy += d_rho_it_reg
                rho_ground_xy += d_rho_ground_it
                rho_other_xy += d_rho_other_it
                rho_2s_xy += d_rho_2s_it

                for k in range(n_sat):
                    rho_sat_ijxy[k] = rho_sat_ijxy[k] + d_rho_sat_it[k]
                    curr_rho_sat_diag_txy[k][:, it, :, :] = rho_sat_ijxy[k][sat_diag_idx, sat_diag_idx, :, :]
                    if is_final_iz:
                        final_rho_sat_ijt_center[k][:, :, it] = rho_sat_ijxy[k][:, :, cx, cy]

                curr_rho_ground_txy[it, :, :] = np.real(rho_ground_xy)
                curr_rho_other_txy[it, :, :] = np.real(rho_other_xy)
                curr_rho_2s_txy[it, :, :] = np.real(rho_2s_xy)
                curr_rho_diag_txy[:, it, :, :] = rho_ijxy[diag_idx, diag_idx, :, :]
                if is_final_iz:
                    final_rho_ijt_center[:, :, it] = rho_ijxy[:, :, cx, cy]

                if iz != 0:
                    # Same [iz-1, iz] window Model.absorption() reads in the full-history path.
                    window_ground = np.stack([prev_rho_ground_txy[it, :, :],
                                              curr_rho_ground_txy[it, :, :]], axis=-1).astype(complex)
                    window_other = np.stack([prev_rho_other_txy[it, :, :],
                                             curr_rho_other_txy[it, :, :]], axis=-1).astype(complex)
                    window_2s = np.stack([prev_rho_2s_txy[it, :, :],
                                          curr_rho_2s_txy[it, :, :]], axis=-1).astype(complex)
                    # Only the diagonal is populated -- Model.absorption never reads off-diagonal
                    # entries of this argument (see docstring).
                    window_ij = np.zeros((nlevel, nlevel, xgrid, ygrid, 2), dtype=complex)
                    window_ij[diag_idx, diag_idx, :, :, 0] = prev_rho_diag_txy[:, it, :, :]
                    window_ij[diag_idx, diag_idx, :, :, 1] = curr_rho_diag_txy[:, it, :, :]
                    window_sat_ij = []
                    for k in range(n_sat):
                        w = np.zeros((satellite_nlevel, satellite_nlevel, xgrid, ygrid, 2), dtype=complex)
                        w[sat_diag_idx, sat_diag_idx, :, :, 0] = prev_rho_sat_diag_txy[k][:, it, :, :]
                        w[sat_diag_idx, sat_diag_idx, :, :, 1] = curr_rho_sat_diag_txy[k][:, it, :, :]
                        window_sat_ij.append(w)

                    kappa_Omega_psxyz = Model.absorption(X, window_ground, window_other,
                                                         window_2s, window_ij, window_sat_ij)
                    Omega_pstxy[:, :, it, :, :] = self.optics.Fresnel_propagator_with_absorption(X, Omega_pstxy[:, :, it, :, :], X.dz, iz * X.dz, kappa_Omega_psxyz, X.lambdaKalpha1N)

                Omega_pstxy[:, :, it, :, :] += 1.0 * X.dz * Model.Omega_source_regular(X, rho_ijxy)
                for k in range(n_sat):
                    Omega_pstxy[:, :, it, :, :] += 1.0 * X.dz * Model.Omega_source_regular(
                        X, rho_sat_ijxy[k], X.Tijs_plus_satellite, X.Tijs_minus_satellite)

                J_Omega_minus_txy[it, :, :] = np.real(Omega_pstxy[0, 0, it, :, :] * Omega_pstxy[1, 0, it, :, :] / X.flux_factor)
                J_Omega_plus_txy[it, :, :] = np.real(Omega_pstxy[0, 1, it, :, :] * Omega_pstxy[1, 1, it, :, :] / X.flux_factor)

            # ######################
            # Loop over simulation time window ends

            # Reproduces the full-history path's off-by-one (see docstring).
            if iz == zgrid - 2:
                Omega_pstxyz_zlast = Omega_pstxy.copy()

            prev_rho_ground_txy = curr_rho_ground_txy
            prev_rho_other_txy = curr_rho_other_txy
            prev_rho_2s_txy = curr_rho_2s_txy
            prev_rho_diag_txy = curr_rho_diag_txy
            prev_rho_sat_diag_txy = curr_rho_sat_diag_txy

        ######################
        # Main loop ends

        self.rho_ijtxyz = final_rho_ijt_center[:, :, :, np.newaxis, np.newaxis, np.newaxis]
        self.rho_sat_ijtxyz = []
        for k in range(n_sat):
            self.rho_sat_ijtxyz.append(final_rho_sat_ijt_center[k][:, :, :, np.newaxis, np.newaxis, np.newaxis])
        self.Omega_pstxyz = np.stack([Omega_pstxyz_z0, Omega_pstxyz_zlast], axis=-1)

        self.rho_ground_txyz = curr_rho_ground_txy[:, :, :, np.newaxis]

        self.rho_2s_txyz = curr_rho_2s_txy[:, :, :, np.newaxis]
        self.rho_other_txyz = curr_rho_other_txy[:, :, :, np.newaxis]
