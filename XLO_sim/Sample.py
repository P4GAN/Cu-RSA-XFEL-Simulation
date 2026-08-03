import numpy as np
import matplotlib.pyplot as plt
import tools
import Model
import Optics as XLO_optics


class XLO_sample:

    
    def __init__(self, X, seed_field=None):
        
        X.noise_pstxyz = tools.random_gaussian_noise_nlevel_pstxyz(X)
        self.optics = XLO_optics.XLO_optics(X)
        self.optics.Greens_function_numerical_3D(X)
        X.Gxyz = self.optics.Gxyz
        
        if (seed_field is None):
            self.is_seeded = 0  
            print('Simulation startup: noise')
        else:
            self.is_seeded = 1
            print('Simulation startup: seeded')
            print('Seed field will be converted to RH and LH polarization')
            self.seed_field = tools.linear_to_circular(X, seed_field)
    
        self.pump_field = X.pump_pulse_3D.copy()

            
        
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

        if X.initial_population == 'upper':
            if X.nlevel == 6:
                for i in range(4, 6):
                    rho_ijtxyz[i, i, 0, :, :, :] = 1.0/2.0 + 1j * 0.0
            elif X.nlevel == 2:
                rho_ijtxyz[1, 1, 0, :, :, :] = 1.0 + 1j * 0.0
        elif X.initial_population == 'lower':
            if X.nlevel == 6:
                for i in range(4):
                    rho_ijtxyz[i, i, 0, :, :, :] = 1.0/4.0 + 1j * 0.0
            elif X.nlevel == 2:
                rho_ijtxyz[0, 0, 0, :, :, :] = 1.0 + 1j * 0.0
        
        if (self.is_seeded == 1):
            Omega_pstxyz[:,:,:,:,:,0] = self.seed_field
            
        if X.initial_population == 'ground':
            rho_ground_txyz = np.ones((X.tgrid, X.xgrid, X.ygrid, X.zgrid), dtype=complex)
        else:
            rho_ground_txyz = np.zeros((X.tgrid, X.xgrid, X.ygrid, X.zgrid), dtype=complex)
            
        rho_other_txyz = np.zeros((X.tgrid, X.xgrid, X.ygrid, X.zgrid), dtype=complex)
        rho_2s_txyz = np.zeros((X.tgrid, X.xgrid, X.ygrid, X.zgrid), dtype=complex)
        Pfield_txyz = np.zeros((X.tgrid, X.xgrid, X.ygrid, X.zgrid), dtype=complex)
        J_P_txyz = np.zeros((X.tgrid, X.xgrid, X.ygrid, X.zgrid))
        Pfield_txyz[:, :, :, 0] = self.pump_field.copy()
        J_P_txyz[:, :, :, 0] = np.real(Pfield_txyz[:, :, :, 0] * np.conj(Pfield_txyz[:, :, :, 0]))
        J_Omega_minus_txyz = np.zeros((X.tgrid, X.xgrid, X.ygrid, X.zgrid))
        J_Omega_plus_txyz = np.zeros((X.tgrid, X.xgrid, X.ygrid, X.zgrid))
        
        return rho_ground_txyz, rho_other_txyz, rho_2s_txyz, Pfield_txyz, J_P_txyz, rho_ijtxyz, Omega_pstxyz, J_Omega_minus_txyz, J_Omega_plus_txyz

    
    
    def configure(self, X):
        """
        Perform the pre-calculation of the pump field flux and ground state population with and without diffraction. Calculate the pumping rates for the ionic state populations.

        Parameters
        ----------
        X
            XLO_sim object

        Returns
        -------

        """
        
        rho_0_txyz = np.zeros((X.tgrid, X.xgrid, X.ygrid, X.zgrid))
        rho_i_txyz = np.zeros((X.tgrid, X.xgrid, X.ygrid, X.zgrid))
        j_txyz = np.zeros((X.tgrid, X.xgrid, X.ygrid, X.zgrid))
        
        if X.initial_population == 'ground':
            rho_0_txy = np.ones((X.tgrid, X.xgrid, X.ygrid))
        else:
            rho_0_txy = np.zeros((X.tgrid, X.xgrid, X.ygrid))
            
        if X.initial_population == 'upper':
            rho_i_txy = np.ones((X.tgrid, X.xgrid, X.ygrid))
        else:
            rho_i_txy = np.zeros((X.tgrid, X.xgrid, X.ygrid))

        j0_txy = np.real(self.pump_field * np.conj(self.pump_field))
        j_txyz[:, :, :, 0] = j0_txy

        if X.enable_pump_diffraction == True:
            pfield_txyz = np.zeros((X.tgrid, X.xgrid, X.ygrid, X.zgrid), dtype=complex)
            pfield0_txy = self.pump_field.copy()
            pfield_txyz[:, :, :, 0] = pfield0_txy

        for j in range(0, X.zgrid):
            
            rho_0_val = rho_0_txy[0, :, :].copy()
            rho_i_val = rho_i_txy[0, :, :].copy()
                     
            for i in range(0, X.tgrid):
                    
                delta_Rho_3D = tools.RK45_step(Model.pump_two_level, [rho_0_val, rho_i_val], i*X.dt, X.dt, [X, j0_txy, i, j])
                rho_0_val += delta_Rho_3D[0]
                rho_i_val += delta_Rho_3D[1]

                rho_0_txyz[i, :, :, j] = rho_0_val
                rho_i_txyz[i, :, :, j] = rho_i_val
                
                if (X.enable_pump_diffraction == True) and (j != 0):
                    kappa_p_xyz = (X.n * X.sigma_ground_pump * rho_0_txyz[i, :, :, j - 1: j + 1] + X.n * X.sigma_compound_pump) * (1.0 + 0.0 * 1j)
                    
                    pfield0_txy[i, :, :] = self.optics.Fresnel_propagator_with_absorption(X, pfield0_txy[i, :, :], X.dz, j * X.dz, kappa_p_xyz, X.lambdaPump, 'pump')
                    j0_txy[i, :, :] = np.real(pfield0_txy[i, :, :] * np.conj(pfield0_txy[i, :, :]))           
            
            
            if (X.enable_pump_diffraction == False) and (j != 0):
                j0_txy -= 1.0 * X.dz * j0_txy * X.n * (X.sigma_ground_pump * rho_0_txyz[:, :, :, j] + X.sigma_compound_pump)
                
            j_txyz[:, :, :, j] = j0_txy

        self.rho_0_3D = rho_0_txyz
        self.rho_i_3D = rho_i_txyz
        self.j_3D = j_txyz 
    
        if X.enable_pump_diffraction: 
            print('Pump field diffraction: enabled')
        else:
            print('Pump field diffraction: disabled')

            

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
             
        rho_ground_txyz, rho_other_txyz, rho_2s_txyz, Pfield_txyz, J_P_txyz, rho_ijtxyz, Omega_pstxyz, J_Omega_minus_txyz, J_Omega_plus_txyz = self.init_n_level_3D(X)
        
        Pfield_txy = Pfield_txyz[:, :, :, 0].copy()
        J_P_txy = J_P_txyz[:, :, :, 0].copy()
        J_Omega_minus_txy = J_Omega_minus_txyz[:, :, :, 0].copy()
        J_Omega_plus_txy = J_Omega_plus_txyz[:, :, :, 0].copy()

       
        Omega_pstxy = Omega_pstxyz[:, :, :, :, :, 0].copy()
        
        
        # Main loop begins. Iterate over longitudinal coordinate        
        for iz in range(0, X.zgrid):

            rho_ijxy = rho_ijtxyz[:, :, 0, :, :, iz].copy()
            
            rho_ground_xy = rho_ground_txyz[0, :, :, iz].copy()
            rho_other_xy = rho_other_txyz[0, :, :, iz].copy()
            rho_2s_xy = rho_2s_txyz[0, :, :, iz].copy()

            print ("Slice: ", iz)
                        
            ######################
            # Loop over simulation time window begins      
            for it in range(0, X.tgrid):
                d_rho_it_reg = tools.RK45_step(Model.MB_nlevel_regular, rho_ijxy, it * X.dt, X.dt, [X, Omega_pstxy[:, :, it, :, :], it, iz, rho_ground_xy, rho_2s_xy, J_P_txy[it, :, :], J_Omega_minus_txy[it, :, :], J_Omega_plus_txy[it, :, :]])
                d_rho_other_it = tools.RK45_step(Model.MB_other_regular, rho_other_xy, it * X.dt, X.dt, [X, Omega_pstxy[:, :, it, :, :], it, iz, rho_ground_xy, J_P_txy[it, :, :], J_Omega_minus_txy[it, :, :], J_Omega_plus_txy[it, :, :]])
                d_rho_2s_it = tools.RK45_step(Model.MB_2s_regular, rho_2s_xy, it * X.dt, X.dt, [X, Omega_pstxy[:, :, it, :, :], it, iz, rho_ground_xy, J_P_txy[it, :, :], J_Omega_minus_txy[it, :, :], J_Omega_plus_txy[it, :, :]])
                d_rho_ground_it = tools.RK45_step(Model.MB_ground_regular, rho_ground_xy, it * X.dt, X.dt, [X, J_P_txy[it, :, :], J_Omega_minus_txy[it, :, :], J_Omega_plus_txy[it, :, :], it, iz])

                rho_ijxy += d_rho_it_reg #+ d_rho_it_noise 
                rho_ground_xy += d_rho_ground_it
                rho_other_xy += d_rho_other_it
                rho_2s_xy += d_rho_2s_it

                rho_ground_txyz[it, :, :, iz] = rho_ground_xy
                rho_other_txyz[it, :, :, iz] = rho_other_xy
                rho_2s_txyz[it, :, :, iz] = rho_2s_xy
                rho_ijtxyz[:, :, it, :, :, iz] = rho_ijxy
                
                kappa_P_xyz = Model.absorption(rho_ground_txyz[it, :, :, iz-1:iz+1], rho_other_txyz[it, :, :, iz-1:iz+1], 
                                               rho_2s_txyz[it, :, :, iz-1:iz+1], 
                                               rho_ijtxyz[:, :, it, :, :, iz-1:iz+1], [X, it, iz, 'pump'])
                
                if (X.enable_pump_diffraction == True) and (iz != 0):
                    Pfield_txy[it, :, :] = self.optics.Fresnel_propagator_with_absorption(X, Pfield_txy[it, :, :], X.dz, iz * X.dz, kappa_P_xyz, X.lambdaPump, 'pump')
                    J_P_txy[it, :, :] = np.real(Pfield_txy[it, :, :] * np.conj(Pfield_txy[it, :, :]))
                    
                if (X.enable_pump_diffraction == False) and (iz != 0):
                    J_P_txy[it, :, :] -= X.dz * 0.5 * np.real(kappa_P_xyz[:, :, 0] + kappa_P_xyz[:, :, 1]) * J_P_txy[it, :, :]
                
                if (X.enable_self_absorption == True) and (iz != 0):
                    kappa_Omega_psxyz = Model.absorption(rho_ground_txyz[it, :, :, iz-1:iz+1], rho_other_txyz[it, :, :, iz-1:iz+1], 
                                                         rho_2s_txyz[it, :, :, iz-1:iz+1], 
                                                         rho_ijtxyz[:, :, it, :, :, iz-1:iz+1], [X, it, iz, 'field'])
                    Omega_pstxy[:, :, it, :, :] = self.optics.Fresnel_propagator_with_absorption(X, Omega_pstxy[:, :, it, :, :], X.dz, iz * X.dz, kappa_Omega_psxyz, X.lambdaKalpha1N, 'ASE') 
                    
                if (X.enable_self_absorption == False) and (iz != 0):
                    Omega_pstxy[:, :, it, :, :] = self.optics.Fresnel_propagator_no_absorption(X, Omega_pstxy[:, :, it, :, :], X.dz, iz * X.dz)

                Omega_pstxy[:, :, it, :, :] +=  1.0 * X.dz * Model.Omega_source_regular(rho_ijtxyz[:, :, it, :, :, iz], [X, it, iz]) 
                    
                J_Omega_minus_txy[it, :, :] = np.real(Omega_pstxy[0, 0, it, :, :] * Omega_pstxy[1, 0, it, :, :] / X.flux_factor)
                J_Omega_plus_txy[it, :, :] = np.real(Omega_pstxy[0, 1, it, :, :] * Omega_pstxy[1, 1, it, :, :] / X.flux_factor)

            # ######################    
            # Loop over simulation time window ends    
            
            if (iz != X.zgrid-1):
                Omega_pstxyz[:, :, :, :, :, iz + 1] = Omega_pstxy

            Pfield_txyz[:, :, :, iz] = Pfield_txy
            J_P_txyz[:, :, :, iz] = J_P_txy
            J_Omega_minus_txyz[:, :, :, iz] = J_Omega_minus_txy
            J_Omega_plus_txyz[:, :, :, iz] = J_Omega_plus_txy
                
        ######################            
        # Main loop ends     

        self.rho_ijtxyz = rho_ijtxyz
        self.Omega_pstxyz = Omega_pstxyz
        
        self.rho_0_3D = np.real(rho_ground_txyz)
        self.j_3D = J_P_txyz
        self.Pfield_txyz = Pfield_txyz

        self.rho_2s_txyz = np.real(rho_2s_txyz)
        self.rho_other_txyz = np.real(rho_other_txyz)
        