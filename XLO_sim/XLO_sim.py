import yaml
import numpy as np
import scipy.constants as sp_const
import matplotlib.pyplot as plt
import tools
import Model
import Sample as XLO_sample
import Optics as XLO_optics
import h5py


class XLO_sim:

    def __init__(self, YAML):

        self.hbar = sp_const.hbar / sp_const.e * 1.0e15
        self.c = sp_const.c / 1.0e6

        self.config = yaml.load(open(YAML), Loader=yaml.FullLoader)
        
        for key, value in self.config.items():
            setattr(self, key, value)

        if 'initial_population' not in self.config:
            self.initial_population = 'ground'

        self.sigma_ground_pump = self.sigma1_pump_1s + self.sigma1_pump_2p3 + self.sigma1_pump_other
        self.sigma_compound_pump = np.sum(element['N_atoms'] * element['sigma_compound_pump'] for element in self.compound.values())
        self.sigma_compound_Ka1 = np.sum(element['N_atoms'] * element['sigma_compound_Ka1'] for element in self.compound.values())
            
        self.is_kfilter = self.config['enable_kfilter']

        self.lambdaPump = 2.0 * np.pi * self.c * self.hbar / self.hwPump
        self.kp = 2.0 * np.pi / self.lambdaPump
        
        self.lambdaKalpha1N = 2.0 * np.pi * self.c * self.hbar / self.hwKalpha1N
        self.k0 = 2.0 * np.pi / self.lambdaKalpha1N

        self.Hij = np.tril(np.ones((self.nlevel, self.nlevel)), -1)
        self.delta_ij = np.eye(self.nlevel, self.nlevel)              
        
        self.dz = self.zmax / (self.zgrid - 1.0)
        self.z = np.linspace(0, self.zmax, self.zgrid)
        
        # self.sigma_r = self.config['pump_width_FWHM'] / (2 * np.sqrt(2*np.log(2)))
        # self.sigma_rx = self.sigma_r
        # self.sigma_ry = self.sigma_r 
        # self.sigma_t = self.config['pulse_duration_FWHM_t'] / (2 * np.sqrt(2*np.log(2)))
        
        # self.zR = 4 * np.pi * self.sigma_r**2 / self.lambdaPump 
        self.t0 = self.tmax / 2.0
        
        # self.theta_geom = np.arctan(4.0 * self.sigma_r / self.zmax)
        # self.theta_pump = self.lambdaPump / np.pi / 2.0 / self.sigma_r * self.M2
        # self.theta_max = np.max((self.theta_geom, self.theta_pump))
        
        
        self.optics = XLO_optics.XLO_optics(self)

        if self.auto_grid == True:
            print('Autogrid is enabled. Analyzing geometry...')
            print('theta_geom (mrad): ', self.theta_geom * 1e3, '; theta_pump (mrad): ', self.theta_pump * 1e3, '; theta_max (mrad): ', self.theta_max * 1e3)
            self.xmax = 2.0 * self.sigma_r + self.zmax * self.theta_max + self.lambdaPump/ 2.0 / self.sigma_r * self.zmax
            self.ymax = 2.0 * self.sigma_r + self.zmax * self.theta_max + self.lambdaPump/ 2.0 / self.sigma_r * self.zmax
            print('Resized xmax, ymax to: ', self.xmax, self.ymax)
            
            self.xgrid = int(2.0 * self.xmax / (self.lambdaKalpha1N / 5.0 / self.theta_max))
            self.ygrid = int(2.0 * self.ymax / (self.lambdaKalpha1N / 5.0 / self.theta_max))
            print('Resized xgrid, ygrid to: ', self.xgrid, self.ygrid)

            
        domains_txy, meshes_txy, step_sizes_txy = self.optics.nd_space((0, -self.xmax, -self.ymax), (self.tmax, self.xmax, self.ymax), (self.tgrid, self.xgrid, self.ygrid))
        
        self.t_mesh, self.x_mesh, self.y_mesh = meshes_txy
        self.t = domains_txy[0]
        self.dt, self.dx, self.dy = step_sizes_txy

        
        if self.pump_pulse_format == 'Gaussian':            
            self.pump_pulse_3D = tools.Gaussian_pulse_3D(self)                
            print('Using Gaussian pump profile')
            print("FWHM beam size: ", 2.355 * self.sigma_r, "nm")
            
        if self.pump_pulse_format == 'Gaussian_pulse_aniso_pump':            
            self.pump_pulse_3D = tools.Gaussian_pulse_aniso_pump(self)                
            print('Using Gaussian_pulse_aniso_pump')
            self.t_pump_max = self.tmax / 2
            
        if self.pump_pulse_format == 'Gaussian_shifted':
            self.pump_pulse_3D = tools.Gaussian_pulse_3D_t_shifted(self)                
            print('Using Gaussian shifted pump profile')
            print("FWHM beam size: ", 2.355 * self.sigma_r, "nm")
            
        if self.pump_pulse_format == 'Gaussian_shifted_splitted':
            self.pump_pulse_3D = tools.Gaussian_pulse_3D_t_shifted_splitted(self)                
            print('Using Gaussian shifted pump profile')
            print("FWHM beam size: ", 2.355 * self.sigma_r, "nm")
            
        if self.pump_pulse_format == 'Gaussian_pulse_3D_t_shifted_chirped':
            self.pump_pulse_3D = tools.Gaussian_pulse_3D_t_shifted_chirped(self)                
            print('Using Gaussian shifted chirped pump profile')
            print("FWHM beam size: ", 2.355 * self.sigma_r, "nm")
            
        if self.pump_pulse_format == 'Ocelot_SASE_pulse_pump_txy':
            self.pump_pulse_3D = tools.Ocelot_SASE_pulse_pump_txy(self)                
            print('Ocelot_SASE_pulse_pump_txy')
            self.t_pump_max = self.tmax / 2
            
        if self.pump_pulse_format == 'SASE':
            self.N_modes = self.config['N_modes']
            self.sigma_coh = self.config['sigma_coh']
            self.pump_pulse_3D = tools.SASE_pulse_3D_with_q(self)
            print('Using SASE pump profile')

        if self.pump_pulse_format == 'loadh5':
            print('loading from h5')
            h5f = h5py.File(self.pump_h5file_name, 'r')
            self.pump_pulse_3D= h5f['pulse'][:]
            h5f.close()
            pulse_shape=self.pump_pulse_3D.shape
            
            if(pulse_shape[0]!=self.tgrid) :
                print('pulse t grid ' , pulse_shape[0], ' tgrid ', self.tgrid)
                raise Exception('loaded pulse time grid must equal configured grid size')
            if(pulse_shape[1]!=self.xgrid) :
                print('pulse x grid ' , pulse_shape[1], ' xgrid ', self.xgrid)
                raise Exception('loaded pulse x grid must equal configured grid size')
            if(pulse_shape[2]!=self.ygrid) :
                print('pulse y grid ' , pulse_shape[2], ' ygrid ', self.ygrid)
                raise Exception('loaded pulse y grid must equal configured grid size')
            print('Using Loaded pump profile from h5 file')
            
        if self.pump_pulse_format == 'GenesisV2':
            self.fname_GenV2 = self.config['fname_GenV2']
            self.ncar_GenV2 = self.config['ncar_GenV2']
            mag_factor = self.config['mag_factor_GenV2']
            crop_t = self.config['crop_t']
            crop_x = self.config['crop_x']
            self.dgrid_GenV2 = self.config['dgrid_GenV2'] / mag_factor
            self.zsep_GenV2 = self.config['zsep_GenV2']
            self.pump_pulse_3D = tools.pump_from_file_genesis2_DFL(self, crop_t, crop_x)
            print('Loaded genesis2 DFL file')
            
        if self.pump_pulse_format == 'GenesisV4':
            print('Imported pump')

        self.GammaKfsm1N = 1.0 / (self.hbar / self.config['GammaKeVN'])
        self.GammaL2fsm1N = 1.0 / (self.hbar / self.config['GammaL2eVN'])
        self.GammaL3fsm1N = 1.0 / (self.hbar / self.config['GammaL3eVN'])
        self.GammarKalpha1fsm1N = 1.0 / (self.hbar / self.config['GammarKalpha1eVN'])
        self.GammarKalpha2fsm1N = 1.0 / (self.hbar / self.config['GammarKalpha2eVN'])
        self.Gamma_sp_fsm1N = self.GammarKalpha1fsm1N + self.GammarKalpha2fsm1N

        self.f05Kalpha12A = self.config['GammarKalpha2eVN'] / self.config['GammarKalpha1eVN']

        self.DeltaomegaL2mL3A = (self.hwKalpha1N - self.hwKalpha2N) / self.hbar

        if (self.config['random_seed'] == -1):
            self.random_seed = None
        else:
            self.random_seed = self.config['random_seed']
            
        self.flux_factor = 3.0 * self.lambdaKalpha1N ** 2 * self.Gamma_sp_fsm1N / 8.0 / np.pi
        self.field_source_factor = 1j * 3.0 * self.lambdaKalpha1N**2 * self.Gamma_sp_fsm1N * self.n / 16.0 / np.pi 
        self.Gamma_ij = 0.5 * (self.GammaKfsm1N + self.GammaL3fsm1N)
        self.convert_pump_phnm2fs_Wcm2 = (self.hwPump     * 1.602e-19 / (1e-9)**2 / 1e-15) / (1 / (1e-2)**2)
        self.convert_SF_phnm2fs_Wcm2   = (self.hwKalpha1N * 1.602e-19 / (1e-9)**2 / 1e-15) / (1 / (1e-2)**2)
        
        self.noise_f_factor = np.sqrt((3.0/(8.0 * np.pi)) * (self.lambdaKalpha1N**2 / (2.0 * self.dx * self.dy)) * self.Gamma_sp_fsm1N * self.dt)
        self.dV = self.dx * self.dy * self.dz
        
        self.e_sign = np.asarray([1, -1])
        self.e_pol = np.asarray([1, 1])
        
        Tijs = np.zeros((self.nlevel, self.nlevel, 2), dtype=complex)
        Gij = np.zeros((self.nlevel, self.nlevel), dtype=complex)
        S_ground_Fi = np.zeros((3, self.nlevel+1))
        S_ion_Fi = np.zeros((3, self.nlevel))
        S_other_F = np.zeros(3)
        
        if (self.nlevel)==6:
                        
            Tijs[0,4,0] = 1.0 / np.sqrt(3)
            Tijs[1,5,0] = 1.0 / 3.0
            Tijs[2,4,1] = 1.0 / 3.0
            Tijs[3,5,1] = 1.0 / np.sqrt(3)
            Tijs[4,0,0] = 1.0 / np.sqrt(3)
            Tijs[4,2,1] = 1.0 / 3.0
            Tijs[5,1,0] = 1.0 / 3.0
            Tijs[5,3,1] = 1.0 / np.sqrt(3)
            
            Gij[0,4] = 1.0 / 3.0
            Gij[1,4] = 2.0 / 9.0
            Gij[1,5] = 1.0 / 9.0
            Gij[2,4] = 1.0 / 9.0
            Gij[2,5] = 2.0 / 9.0
            Gij[3,5] = 1.0 / 3.0

            S_ground_Fi[0, 0] = self.sigma1_pump_2p3 * 0.27
            S_ground_Fi[0, 1] = self.sigma1_pump_2p3 * 0.23
            S_ground_Fi[0, 2] = self.sigma1_pump_2p3 * 0.23
            S_ground_Fi[0, 3] = self.sigma1_pump_2p3 * 0.27
            S_ground_Fi[0, 4] = self.sigma1_pump_1s * 0.5
            S_ground_Fi[0, 5] = self.sigma1_pump_1s * 0.5
            S_ground_Fi[0, 6] = self.sigma1_pump_other

            S_ground_Fi[1, 0] = self.sigma1_Ka1_2p3 * 0.12
            S_ground_Fi[1, 1] = self.sigma1_Ka1_2p3 * 0.18
            S_ground_Fi[1, 2] = self.sigma1_Ka1_2p3 * 0.28
            S_ground_Fi[1, 3] = self.sigma1_Ka1_2p3 * 0.42
            S_ground_Fi[1, 6] = self.sigma1_Ka1_other

            S_ground_Fi[2, 0] = self.sigma1_Ka1_2p3 * 0.42
            S_ground_Fi[2, 1] = self.sigma1_Ka1_2p3 * 0.28
            S_ground_Fi[2, 2] = self.sigma1_Ka1_2p3 * 0.18
            S_ground_Fi[2, 3] = self.sigma1_Ka1_2p3 * 0.12
            S_ground_Fi[2, 6] = self.sigma1_Ka1_other

            S_ion_Fi[0, 0] = self.sigma2_pump_2p3 * 1.05
            S_ion_Fi[0, 1] = self.sigma2_pump_2p3 * 0.95
            S_ion_Fi[0, 2] = self.sigma2_pump_2p3 * 0.95
            S_ion_Fi[0, 3] = self.sigma2_pump_2p3 * 1.05
            S_ion_Fi[0, 4] = self.sigma2_pump_1s * 1.0
            S_ion_Fi[0, 5] = self.sigma2_pump_1s * 1.0

            S_ion_Fi[1, 0] = self.sigma2_Ka1_2p3 * 0.70
            S_ion_Fi[1, 1] = self.sigma2_Ka1_2p3 * 0.83
            S_ion_Fi[1, 2] = self.sigma2_Ka1_2p3 * 1.06
            S_ion_Fi[1, 3] = self.sigma2_Ka1_2p3 * 1.41
            S_ion_Fi[1, 4] = self.sigma2_Ka1_1s * 0.75
            S_ion_Fi[1, 5] = self.sigma2_Ka1_1s * 1.25

            S_ion_Fi[2, 0] = self.sigma2_Ka1_2p3 * 1.41
            S_ion_Fi[2, 1] = self.sigma2_Ka1_2p3 * 1.06
            S_ion_Fi[2, 2] = self.sigma2_Ka1_2p3 * 0.83
            S_ion_Fi[2, 3] = self.sigma2_Ka1_2p3 * 0.70
            S_ion_Fi[2, 4] = self.sigma2_Ka1_1s * 1.25
            S_ion_Fi[2, 5] = self.sigma2_Ka1_1s * 0.75

            S_other_F[0] = self.sigma2_pump_other
            S_other_F[1] = self.sigma2_Ka1_other
            S_other_F[2] = self.sigma2_Ka1_other

            self.ei_L3 = np.asarray([1, 1, 1, 1, 0, 0])
            self.ei_K = np.asarray([0, 0, 0, 0, 1, 1])
            
        if (self.nlevel)==2:
            
            Tijs[0,1,0] = 1.0
            Tijs[1,0,0] = 1.0

            Gij[0,1] = 1.0

            S_ground_Fi[0, 1] = self.sigma_1s
            S_ground_Fi[0, 2] = self.sigma_other

            self.ei_L3 = np.asarray([1, 0])
            self.ei_K = np.asarray([0, 1])
            
        self.Tijs = Tijs
        self.Gij = Gij
        self.S_ground_Fi = S_ground_Fi
        self.S_other_F = S_other_F
        self.S_ion_Fi = S_ion_Fi
        self.Tijs_plus = np.einsum('ijs, ij->ijs', self.Tijs, self.Hij)
        self.Tijs_minus = np.einsum('ijs, ji->ijs', self.Tijs, self.Hij)
        self.Mij = self.GammaL3fsm1N * np.outer(self.ei_L3, self.ei_L3) + self.GammaKfsm1N * np.outer(self.ei_K, self.ei_K) + (self.GammaKfsm1N + self.GammaL3fsm1N) * (np.outer(self.ei_L3, self.ei_K) + np.outer(self.ei_K, self.ei_L3)) / 2.0
        self.transform_matrix = np.asarray([[1, 1], [1j, -1j]]) / np.sqrt(2.0) # transformation matrix from circular polarization vectors to Cartesian

        
            
    def configure(self, seed_field=None):
        """
        Check if the seeding field is present, and pass it to the Sample object. Pre-compute the pump field dynamics with or without diffracion.

        Parameters
        ----------
        seed_field : ndarray or None
            Seed field amplitude at the beginning of gain medium. Default is starting from noise (None). Seed field is assumed to have no detuning with the Kalpha1 transition.

        Returns
        -------

        """

        sample = XLO_sample.XLO_sample(self, seed_field)
        self.sample = sample
        print("Configured XLO simulation in simultaneous mode")



    def run_3D(self):
        """
        Calculate ASE field propagation and density matrix evolution with or without absorption of the ASE field.

        Returns
        -------

        """

        self.sample.evaluate_n_level_3D(self)
        self.rho_0_3D = self.sample.rho_0_3D
        self.j_3D = self.sample.j_3D
        self.nphoton_pump_z = tools.nphoton_pump_z(self)

        self.rho_ijtxyz = self.sample.rho_ijtxyz
        self.Omega_pstxyz = self.sample.Omega_pstxyz
        self.nphoton_sz = tools.nphoton_sz(self)
        self.nphoton_reg_sz = tools.nphoton_reg_sz(self)
        self.nphoton_pump_z = tools.nphoton_pump_z(self)
        if self.nphoton_pump_z[0] != 0.0:
            self.pump_transmission = self.nphoton_pump_z[-1] / self.nphoton_pump_z[0]
        
        self.E_sstxyz = np.einsum('ijp,jktxyz,kir->prtxyz', self.Tijs_minus, self.rho_ijtxyz, self.Tijs_plus)
        self.G_sstxyz = np.einsum('ijr,jktxyz,kip->prtxyz', self.Tijs_plus, self.rho_ijtxyz, self.Tijs_minus)

        self.P_pstxyz = np.array([ np.einsum('ijs,jitxyz->stxyz', self.Tijs_minus, self.rho_ijtxyz) , np.einsum('ijtxyz,jis->stxyz', self.rho_ijtxyz, self.Tijs_plus) ])
        
        if self.is_Cartesian_pol:
            Omega_pqtxy = tools.circular_to_linear(self, self.Omega_pstxyz[:, :, :, :, :, -1])
            self.Omega_qtxy = (Omega_pqtxy[0, :, :, :, :] + np.conj(Omega_pqtxy[1, :, :, :, :])) / 2.0
            
        self.Pfield_txyz = self.sample.Pfield_txyz

        
    def pre_run_3D(self):
        """
        Calculate ASE field propagation and density matrix evolution with or without absorption of the ASE field.

        Returns
        -------

        """

        self.sample.evaluate_rho_ij_with_pump_3D(self)

        self.pre_rho_ijtxyz = self.sample.rho_ijtxyz
        self.Omega_pstxyz = self.sample.Omega_pstxyz

        self.P_pstxyz = np.array([ np.einsum('ijs,jitxyz->stxyz', self.Tijs_minus, self.pre_rho_ijtxyz) , np.einsum('ijtxyz,jis->stxyz', self.pre_rho_ijtxyz, self.Tijs_plus) ])