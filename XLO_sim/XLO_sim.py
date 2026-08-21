import yaml
import numpy as np
import scipy.constants as sp_const
import matplotlib.pyplot as plt
from . import tools
from . import Model
from . import Sample as XLO_sample
from . import Optics as XLO_optics
import h5py


class XLO_sim:

    def __init__(self, YAML):

        self.hbar = sp_const.hbar / sp_const.e * 1.0e15
        self.c = sp_const.c / 1.0e6

        self.config = yaml.load(open(YAML), Loader=yaml.FullLoader)
        
        for key, value in self.config.items():
            setattr(self, key, value)

        if 'keep_z_history' not in self.config:
            # True preserves the original behavior: every z plane of every
            # field/density-matrix array is stored, which is what Plot.py and
            # the analysis notebooks need to show propagation through the
            # sample depth. At production grid sizes that's tens of GB per
            # repetition (see Sample._evaluate_n_level_3D_full), so the batch
            # run_*_sweep.py scripts set this to False on X before configure()/
            # run_3D() -- they only ever read the z=0/z=-1 planes anyway (see
            # tools.compute_run_outputs), and Sample._evaluate_n_level_3D_lean
            # reproduces the same z-marching physics storing only those.
            self.keep_z_history = True

        self.sigma_compound_Ka1 = sum(element['N_atoms'] * element['sigma_compound_Ka1'] for element in self.compound.values())
            
        self.lambdaKalpha1N = 2.0 * np.pi * self.c * self.hbar / self.hwKalpha1N
        self.k0 = 2.0 * np.pi / self.lambdaKalpha1N

        self.Hij = np.tril(np.ones((self.nlevel, self.nlevel)), -1)
        self.delta_ij = np.eye(self.nlevel, self.nlevel)              
        
        self.dz = self.zmax / (self.zgrid - 1.0)
        self.z = np.linspace(0, self.zmax, self.zgrid)
        
        self.t0 = self.tmax / 2.0
               
        self.optics = XLO_optics.XLO_optics(self)
            
        domains_txy, meshes_txy, step_sizes_txy = self.optics.nd_space((0, -self.xmax, -self.ymax), (self.tmax, self.xmax, self.ymax), (self.tgrid, self.xgrid, self.ygrid))
        
        self.t_mesh, self.x_mesh, self.y_mesh = meshes_txy
        self.t = domains_txy[0]
        self.dt, self.dx, self.dy = step_sizes_txy

        self.GammaKfsm1N = self.config['GammaKeVN'] / self.hbar
        self.GammaL3fsm1N = self.config['GammaL3eVN'] / self.hbar
        self.GammarKalpha1fsm1N = self.config['GammarKalpha1eVN'] / self.hbar
        self.GammarKalpha2fsm1N = self.config['GammarKalpha2eVN'] / self.hbar
        self.GammaL1fsm1N = self.config['GammaL1eVN'] / self.hbar
        self.GammaA_L1_to_L3M45fs1N = self.config['GammaA_L1_to_L3M45eVN'] / self.hbar
        self.Gamma_sp_fsm1N = self.GammarKalpha1fsm1N + self.GammarKalpha2fsm1N

        self.f05Kalpha12A = self.config['GammarKalpha2eVN'] / self.config['GammarKalpha1eVN']

        self.DeltaomegaL2mL3A = (self.hwKalpha1N - self.hwKalpha2N) / self.hbar

        if (self.config['random_seed'] == -1):
            self.random_seed = None
        else:
            self.random_seed = self.config['random_seed']
            
        self.flux_factor = 3.0 * self.lambdaKalpha1N ** 2 * self.Gamma_sp_fsm1N / 8.0 / np.pi
        self.field_source_factor = 1j * 3.0 * self.lambdaKalpha1N**2 * self.Gamma_sp_fsm1N * self.n / 16.0 / np.pi 
        self.Gamma_ij = 0.5 * (self.GammaKfsm1N + self.GammaL3fsm1N) + self.additional_dephasing
        self.convert_SF_phnm2fs_Wcm2   = (self.hwKalpha1N * 1.602e-19 / (1e-9)**2 / 1e-15) / (1 / (1e-2)**2)
                
        self.e_sign = np.asarray([1, -1])
        self.e_pol = np.asarray([1, 1])
        
        Tijs = np.zeros((self.nlevel, self.nlevel, 2), dtype=complex)
        Gij = np.zeros((self.nlevel, self.nlevel), dtype=complex)
        S_ground_Fi = np.zeros((2, self.nlevel+2)) # Includes transition to 2s hole and other
        S_ion_Fi = np.zeros((2, self.nlevel))
        S_other_F = np.zeros(2)
        S_2s_F = np.zeros(2)

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

            S_ground_Fi[0, 0] = self.sigma1_Ka1_2p3 * 0.12
            S_ground_Fi[0, 1] = self.sigma1_Ka1_2p3 * 0.18
            S_ground_Fi[0, 2] = self.sigma1_Ka1_2p3 * 0.28
            S_ground_Fi[0, 3] = self.sigma1_Ka1_2p3 * 0.42
            S_ground_Fi[0, 6] = self.sigma1_Ka1_2s
            S_ground_Fi[0, 7] = self.sigma1_Ka1_other

            S_ground_Fi[1, 0] = self.sigma1_Ka1_2p3 * 0.42
            S_ground_Fi[1, 1] = self.sigma1_Ka1_2p3 * 0.28
            S_ground_Fi[1, 2] = self.sigma1_Ka1_2p3 * 0.18
            S_ground_Fi[1, 3] = self.sigma1_Ka1_2p3 * 0.12
            S_ground_Fi[1, 6] = self.sigma1_Ka1_2s
            S_ground_Fi[1, 7] = self.sigma1_Ka1_other

            S_ion_Fi[0, 0] = self.sigma2_Ka1_2p3 * 0.70
            S_ion_Fi[0, 1] = self.sigma2_Ka1_2p3 * 0.83
            S_ion_Fi[0, 2] = self.sigma2_Ka1_2p3 * 1.06
            S_ion_Fi[0, 3] = self.sigma2_Ka1_2p3 * 1.41
            S_ion_Fi[0, 4] = self.sigma2_Ka1_1s * 0.75
            S_ion_Fi[0, 5] = self.sigma2_Ka1_1s * 1.25

            S_ion_Fi[1, 0] = self.sigma2_Ka1_2p3 * 1.41
            S_ion_Fi[1, 1] = self.sigma2_Ka1_2p3 * 1.06
            S_ion_Fi[1, 2] = self.sigma2_Ka1_2p3 * 0.83
            S_ion_Fi[1, 3] = self.sigma2_Ka1_2p3 * 0.70
            S_ion_Fi[1, 4] = self.sigma2_Ka1_1s * 1.25
            S_ion_Fi[1, 5] = self.sigma2_Ka1_1s * 0.75

            self.ei_L3 = np.asarray([1, 1, 1, 1, 0, 0])
            self.ei_K = np.asarray([0, 0, 0, 0, 1, 1])
            
        if (self.nlevel)==2:
            Tijs[0,1,0] = np.sqrt(2.0/3.0)
            Tijs[1,0,0] = np.sqrt(2.0/3.0)
            Tijs[0,1,1] = np.sqrt(2.0/3.0)   
            Tijs[1,0,1] = np.sqrt(2.0/3.0) 

            Gij[0,1] = 2.0 / 3.0

            S_ground_Fi[0, 0] = self.sigma1_Ka1_2p3 
            S_ground_Fi[0, 2] = self.sigma1_Ka1_2s
            S_ground_Fi[0, 3] = self.sigma1_Ka1_other

            S_ground_Fi[1, 0] = self.sigma1_Ka1_2p3 
            S_ground_Fi[1, 2] = self.sigma1_Ka1_2s
            S_ground_Fi[1, 3] = self.sigma1_Ka1_other

            S_ion_Fi[0, 0] = self.sigma2_Ka1_2p3 
            S_ion_Fi[0, 1] = self.sigma2_Ka1_1s 

            S_ion_Fi[1, 0] = self.sigma2_Ka1_2p3 
            S_ion_Fi[1, 1] = self.sigma2_Ka1_1s 

            self.ei_L3 = np.asarray([1, 0])
            self.ei_K = np.asarray([0, 1])

        S_other_F[0] = self.sigma2_Ka1_other
        S_other_F[1] = self.sigma2_Ka1_other

        S_2s_F[0] = self.sigma2_Ka1_2s
        S_2s_F[1] = self.sigma2_Ka1_2s
            
        self.Tijs = Tijs
        self.Gij = Gij
        self.Gamma_sp_Gij = self.Gamma_sp_fsm1N * Gij
        self.S_ground_Fi = S_ground_Fi
        self.S_other_F = S_other_F
        self.S_ion_Fi = S_ion_Fi
        self.Tijs_plus = np.einsum('ijs, ij->ijs', self.Tijs, self.Hij)
        self.Tijs_minus = np.einsum('ijs, ji->ijs', self.Tijs, self.Hij)
        self.Mij = (self.GammaL3fsm1N * np.outer(self.ei_L3, self.ei_L3) + 
                    self.GammaKfsm1N * np.outer(self.ei_K, self.ei_K) + 
                    self.Gamma_ij * (np.outer(self.ei_L3, self.ei_K) + np.outer(self.ei_K, self.ei_L3)))
        # Incoherent 2s -> 2p_3/2 3d_5/2 Auger feeding
        if self.use_2s_pathway == True:
            self.auger_feeding_matrix = np.diag(self.ei_L3 / np.sum(self.ei_L3)) * self.GammaA_L1_to_L3M45fs1N
            self.S_2s_F = S_2s_F
        else:
            self.auger_feeding_matrix = np.zeros((self.nlevel, self.nlevel))
            self.S_2s_F = np.zeros(3)
            self.S_ground_Fi[:, -2] = 0.0
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



    def run_3D(self):
        """
        Calculate ASE field propagation and density matrix evolution with or without absorption of the ASE field.

        Returns
        -------

        """

        self.sample.evaluate_n_level_3D(self)
        self.rho_ground_txyz = self.sample.rho_ground_txyz
        self.rho_other_txyz = self.sample.rho_other_txyz
        self.rho_2s_txyz = self.sample.rho_2s_txyz

        self.rho_ijtxyz = self.sample.rho_ijtxyz
        self.Omega_pstxyz = self.sample.Omega_pstxyz

        if self.is_Cartesian_pol:
            Omega_pqtxy = tools.circular_to_linear(self, self.Omega_pstxyz[:, :, :, :, :, -1])
            self.Omega_qtxy = (Omega_pqtxy[0, :, :, :, :] + np.conj(Omega_pqtxy[1, :, :, :, :])) / 2.0
            
