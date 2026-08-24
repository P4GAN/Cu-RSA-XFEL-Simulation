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

        # 2s-hole satellite channels (docs/theory-and-2s-satellite-pathways.md, Part II): one full
        # 6-level block per channel, same shape as rho_ijtxyz. Empty list if none configured.
        rho_sat_ijtxyz = [np.zeros((X.nlevel, X.nlevel, X.tgrid, X.xgrid, X.ygrid, X.zgrid), dtype=complex)
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
        Original full-z-history implementation of evaluate_n_level_3D (see
        that docstring). Stores every z plane of every field/density-matrix
        array for the whole run, which is what Plot.py and the analysis
        notebooks need to show propagation through the sample depth -- at
        production grid sizes (e.g. tgrid=3200, x=y=z=25) that's tens of GB
        per repetition (rho_ijtxyz alone is ~29 GB), so this path is only
        suitable for interactive/notebook use, not the batch run_*_sweep.py
        scripts. Those pass X.keep_z_history=False to use
        _evaluate_n_level_3D_lean instead, which reproduces the exact same
        z-marching physics but only keeps the z=0/z=-1 planes those scripts'
        outputs actually read (see tools.compute_run_outputs).
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

                # 2s-hole satellite channels (docs/theory-and-2s-satellite-pathways.md, Part II):
                # one full 6-level block per channel, fed from the base block's and rho_2s_xy's
                # pre-update (start-of-step) values, same convention as the other blocks above.
                d_rho_sat_it = [
                    tools.RK45_step(Model.MB_satellite_block_regular, rho_sat_ijxy[k], it * X.dt, X.dt,
                                     [X, chan, Omega_pstxy[:, :, it, :, :], rho_ijxy, rho_2s_xy,
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
                    Omega_pstxy[:, :, it, :, :] += 1.0 * X.dz * Model.Omega_source_regular(X, rho_sat_ijtxyz[k][:, :, it, :, :, iz])

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
        Memory-lean counterpart of _evaluate_n_level_3D_full for batch/statistics
        jobs (X.keep_z_history=False): same z-marching physics, same RK45 calls
        in the same order on the same inputs, but storage is cut from O(zgrid)
        to O(1) per array by keeping only what's ever read back afterwards.

        Two categories of array, treated differently:

        - rho_ground/other/2s/ijtxyz: Model.absorption() reads a 2-plane
          [iz-1, iz] window while marching (see _evaluate_n_level_3D_full), so
          these need a 2-slot rolling buffer (prev/curr z) during the loop.
          Nothing downstream reads any z but the last one (tools.compute_run_outputs
          only ever indexes X.rho_ijtxyz/rho_ground_txyz/etc at [...,-1]), so
          after the loop only the last z's buffer is exposed, as a length-1 z
          axis (so existing [...,-1]/[...,0]-style indexing elsewhere keeps
          working unchanged).
        - Omega_pstxyz/Pfield_txyz/J_Omega_(minus|plus)_txyz: never
          read back from storage during the marching -- only the live "xy"
          rolling variables (Omega_pstxy etc.) are, so no rolling buffer is
          needed at all, just a snapshot at z=0 and z=-1, taken at exactly the
          same point in the loop _evaluate_n_level_3D_full would have written
          them into the big array. That includes replicating
          _evaluate_n_level_3D_full's existing off-by-one for Omega_pstxyz:
          `if (iz != X.zgrid-1): Omega_pstxyz[...,iz+1] = Omega_pstxy` means
          Omega_pstxyz[...,-1] there is actually the state after the
          *second-to-last* iz's update, not the true final one -- reproduced
          here rather than "fixed", so lean and full mode agree exactly.

        Only rho_ground/other/2s_txyz get the extra real-dtype narrowing (fix
        for the complex128-storing-real-values waste in init_n_level_3D):
        verified empirically (see PR discussion) that their imaginary part is
        exactly 0 throughout the run, since Model.MB_ground/other/2s_regular
        are pure population rate equations with no phase terms -- rho_ijtxyz
        stays complex since its off-diagonal coherences are genuinely complex.

        Not used for interactive/notebook work -- Plot.py and the analysis
        notebooks need the full z profile, which this deliberately discards.
        """

        nlevel, tgrid, xgrid, ygrid, zgrid = X.nlevel, X.tgrid, X.xgrid, X.ygrid, X.zgrid

        # t=0 initial-condition templates. Identical for every z in
        # init_n_level_3D (each is set via a uniform [...,:,:,:] assignment
        # over all x,y,z), so unlike the full-history path there is no need to
        # store/re-read per-z copies of these -- just re-copy the same
        # template into the rolling "xy" working array at the start of every
        # iz iteration, exactly reproducing what
        # `rho_ijtxyz[:,:,0,:,:,iz].copy()` etc. would have read.
        rho_ij_ic_xy = np.zeros((nlevel, nlevel, xgrid, ygrid), dtype=complex)

        n_sat = len(X.satellite_channel_params)
        rho_sat_ij_ic_xy = [np.zeros((X.nlevel, X.nlevel, X.xgrid, X.ygrid), dtype=complex)
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

        prev_rho_ground_txy = prev_rho_other_txy = prev_rho_2s_txy = prev_rho_ijtxy = None
        prev_rho_sat_ijtxy = [None for k in range(n_sat)]

        for iz in range(0, zgrid):

            rho_ijxy = rho_ij_ic_xy.copy()
            rho_sat_ijxy = [rho_sat_ij_ic_xy[k].copy() for k in range(n_sat)]
            rho_ground_xy = rho_ground_ic_xy.copy()
            rho_other_xy = rho_other_ic_xy.copy()
            rho_2s_xy = rho_2s_ic_xy.copy()

            curr_rho_ground_txy = np.empty((tgrid, xgrid, ygrid))
            curr_rho_other_txy = np.empty((tgrid, xgrid, ygrid))
            curr_rho_2s_txy = np.empty((tgrid, xgrid, ygrid))
            curr_rho_ijtxy = np.empty((nlevel, nlevel, tgrid, xgrid, ygrid), dtype=complex)
            curr_rho_sat_ijtxy = [np.empty((nlevel, nlevel, tgrid, xgrid, ygrid), dtype=complex)
                                  for k in range(n_sat)]

            ######################
            # Loop over simulation time window begins
            for it in range(0, tgrid):
                d_rho_it_reg = tools.RK45_step(Model.MB_nlevel_regular, rho_ijxy, it * X.dt, X.dt, [X, Omega_pstxy[:, :, it, :, :], rho_ground_xy, rho_2s_xy, J_Omega_minus_txy[it, :, :], J_Omega_plus_txy[it, :, :]])
                d_rho_other_it = tools.RK45_step(Model.MB_other_regular, rho_other_xy, it * X.dt, X.dt, [X, rho_ground_xy, J_Omega_minus_txy[it, :, :], J_Omega_plus_txy[it, :, :]])
                d_rho_2s_it = tools.RK45_step(Model.MB_2s_regular, rho_2s_xy, it * X.dt, X.dt, [X, rho_ground_xy, J_Omega_minus_txy[it, :, :], J_Omega_plus_txy[it, :, :]])
                d_rho_ground_it = tools.RK45_step(Model.MB_ground_regular, rho_ground_xy, it * X.dt, X.dt, [X, J_Omega_minus_txy[it, :, :], J_Omega_plus_txy[it, :, :]])

                # 2s-hole satellite channels (docs/theory-and-2s-satellite-pathways.md, Part II):
                # one full 6-level block per channel, fed from the base block's and rho_2s_xy's
                # pre-update (start-of-step) values, same convention as the other blocks above.
                d_rho_sat_it = [
                    tools.RK45_step(Model.MB_satellite_block_regular, rho_sat_ijxy[k], it * X.dt, X.dt,
                                     [X, chan, Omega_pstxy[:, :, it, :, :], rho_ijxy, rho_2s_xy,
                                      J_Omega_minus_txy[it, :, :], J_Omega_plus_txy[it, :, :]])
                    for k, chan in enumerate(X.satellite_channel_params)
                ]

                rho_ijxy += d_rho_it_reg
                rho_ground_xy += d_rho_ground_it
                rho_other_xy += d_rho_other_it
                rho_2s_xy += d_rho_2s_it

                for k in range(n_sat):
                    rho_sat_ijxy[k] = rho_sat_ijxy[k] + d_rho_sat_it[k]
                    curr_rho_sat_ijtxy[k][:, :, it, :, :] = rho_sat_ijxy[k]

                curr_rho_ground_txy[it, :, :] = np.real(rho_ground_xy)
                curr_rho_other_txy[it, :, :] = np.real(rho_other_xy)
                curr_rho_2s_txy[it, :, :] = np.real(rho_2s_xy)
                curr_rho_ijtxy[:, :, it, :, :] = rho_ijxy

                if iz != 0:
                    # Same [iz-1, iz] window Model.absorption() would have gotten
                    # from rho_ground_txyz[it,:,:,iz-1:iz+1] etc. in the
                    # full-history path -- cast back to complex to match its
                    # input dtype exactly (these hold real-only values either way).
                    window_ground = np.stack([prev_rho_ground_txy[it, :, :], 
                                              curr_rho_ground_txy[it, :, :]], axis=-1).astype(complex)
                    window_other = np.stack([prev_rho_other_txy[it, :, :], 
                                             curr_rho_other_txy[it, :, :]], axis=-1).astype(complex)
                    window_2s = np.stack([prev_rho_2s_txy[it, :, :], 
                                          curr_rho_2s_txy[it, :, :]], axis=-1).astype(complex)
                    window_ij = np.stack([prev_rho_ijtxy[:, :, it, :, :], 
                                          curr_rho_ijtxy[:, :, it, :, :]], axis=-1)
                    window_sat_ij = [np.stack([prev_rho_sat_ijtxy[k][:, :, it, :, :], 
                                               curr_rho_sat_ijtxy[k][:, :, it, :, :]], axis=-1)
                                     for k in range(n_sat)]

                    kappa_Omega_psxyz = Model.absorption(X, window_ground, window_other, 
                                                         window_2s, window_ij, window_sat_ij)
                    Omega_pstxy[:, :, it, :, :] = self.optics.Fresnel_propagator_with_absorption(X, Omega_pstxy[:, :, it, :, :], X.dz, iz * X.dz, kappa_Omega_psxyz, X.lambdaKalpha1N)

                Omega_pstxy[:, :, it, :, :] += 1.0 * X.dz * Model.Omega_source_regular(X, curr_rho_ijtxy[:, :, it, :, :])
                for k in range(n_sat):
                    Omega_pstxy[:, :, it, :, :] += 1.0 * X.dz * Model.Omega_source_regular(X, curr_rho_sat_ijtxy[k][:, :, it, :, :])

                J_Omega_minus_txy[it, :, :] = np.real(Omega_pstxy[0, 0, it, :, :] * Omega_pstxy[1, 0, it, :, :] / X.flux_factor)
                J_Omega_plus_txy[it, :, :] = np.real(Omega_pstxy[0, 1, it, :, :] * Omega_pstxy[1, 1, it, :, :] / X.flux_factor)

            # ######################
            # Loop over simulation time window ends

            # Matches `if (iz != X.zgrid-1): Omega_pstxyz[...,iz+1] = Omega_pstxy`
            # in the full-history path, evaluated only at the one iz where
            # iz+1 == zgrid-1 (see docstring for the off-by-one this reproduces).
            if iz == zgrid - 2:
                Omega_pstxyz_zlast = Omega_pstxy.copy()

            prev_rho_ground_txy = curr_rho_ground_txy
            prev_rho_other_txy = curr_rho_other_txy
            prev_rho_2s_txy = curr_rho_2s_txy
            prev_rho_ijtxy = curr_rho_ijtxy
            prev_rho_sat_ijtxy = curr_rho_sat_ijtxy

        ######################
        # Main loop ends

        self.rho_ijtxyz = curr_rho_ijtxy[:, :, :, :, :, np.newaxis]
        self.rho_sat_ijtxyz = []
        for k in range(n_sat):
            self.rho_sat_ijtxyz.append(curr_rho_sat_ijtxy[k][:, :, :, :, :, np.newaxis])
        self.Omega_pstxyz = np.stack([Omega_pstxyz_z0, Omega_pstxyz_zlast], axis=-1)

        self.rho_ground_txyz = curr_rho_ground_txy[:, :, :, np.newaxis]

        self.rho_2s_txyz = curr_rho_2s_txy[:, :, :, np.newaxis]
        self.rho_other_txyz = curr_rho_other_txy[:, :, :, np.newaxis]
