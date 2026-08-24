import types
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

        if 'satellite_channels' not in self.config:
            self.satellite_channels = []

        # 2p1/2 (L2, Kalpha2) pathway (docs/theory-and-2s-satellite-pathways.md, Part III):
        # shares the SAME 1s (K) population as the base 2p3/2 (L3) block -- unlike the satellite
        # channels, this is not a separate configuration, so it can't be a separate block; it
        # extends the base block itself by 2 levels (local indices nlevel_base, nlevel_base+1).
        self.use_L2_pathway = self.config.get('use_L2_pathway', False)
        nlevel_base = self.nlevel  # as read from YAML: the base block's own level count (6 or 2)
        if self.use_L2_pathway and nlevel_base != 6:
            raise ValueError('use_L2_pathway requires nlevel: 6 (2p1/2 extends the full sublevel-resolved base block)')
        if self.use_L2_pathway:
            self.nlevel = nlevel_base + 2

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
        # GammaL2eVN (2p1/2 hole width) is only needed/read when use_L2_pathway is on; no
        # base-config default exists for it (see theory doc Part III, xatom_tools.py can compute
        # it via state_total_decay_width_eV, calibrated against GammaL3eVN's own XATOM/literature
        # ratio the same way the satellite channels' Gamma_L_eV/Gamma_K_eV are).
        if self.use_L2_pathway:
            self.GammaL2fsm1N = self.config['GammaL2eVN'] / self.hbar
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

        if nlevel_base==6:

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

            self.ei_L3 = np.zeros(self.nlevel); self.ei_L3[0:4] = 1
            self.ei_K = np.zeros(self.nlevel); self.ei_K[4:6] = 1

        if nlevel_base==2:
            Tijs[0,1,0] = 1 # np.sqrt(2.0/3.0)
            Tijs[1,0,0] = 1 # np.sqrt(2.0/3.0)

            Gij[0,1] = 1 # 2.0 / 3.0

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

        if self.use_L2_pathway:
            # 2p1/2 (L2, Kalpha2), local indices nlevel_base (m=-1/2), nlevel_base+1 (m=+1/2).
            # Tijs/Gij derived via Clebsch-Gordan coefficients for the 2p1/2<->1s1/2 dipole
            # transition (docs/theory-and-2s-satellite-pathways.md, Part III derivation): the
            # only two paraxial (sigma=+-1) channels are 1s(m=-1/2)->2p1/2(m=+1/2) [sigma-index 1,
            # matching this code's existing sigma-index<->Delta-m convention] and
            # 1s(m=+1/2)->2p1/2(m=-1/2) [sigma-index 0], each with T=sqrt(2)/3 (T^2=2/9, matching
            # the existing 2p3/2 block's T^2=G self-consistency pattern). Gij additionally carries
            # the isotropic-only (sigma=0/pi, not part of Tijs) 1/9 branch to the *same-m* 1s
            # sublevel, using exactly the same 1/9,2/9 building-block fractions the base 2p3/2
            # block already uses.
            i_m, i_p = nlevel_base, nlevel_base + 1  # local 2p1/2 m=-1/2, m=+1/2

            Tijs[i_p, 4, 1] = np.sqrt(2.0) / 3.0
            Tijs[4, i_p, 1] = np.sqrt(2.0) / 3.0
            Tijs[i_m, 5, 0] = np.sqrt(2.0) / 3.0
            Tijs[5, i_m, 0] = np.sqrt(2.0) / 3.0

            Gij[i_p, 4] = 2.0 / 9.0
            Gij[i_m, 4] = 1.0 / 9.0
            Gij[i_m, 5] = 2.0 / 9.0
            Gij[i_p, 5] = 1.0 / 9.0

            self.ei_L2 = np.zeros(self.nlevel); self.ei_L2[i_m:i_p+1] = 1

            # Ground-state photoionization directly into 2p1/2 (theory doc Part III): total cross
            # section sigma1_Ka1_2p1 is a new required config key (get from XATOM's ground-state
            # -pcs '2p-' row, e.g. ~0.51x sigma1_Ka1_2p3 at 8047.91 eV per xatom_tools.py).
            # m-resolved branching is NOT independently derived (unlike Tijs/Gij, photoionization
            # angular branching involves the continuum electron's partial waves, not pure bound-
            # state Clebsch-Gordan algebra) -- an even 50/50 split across the 2 msublevels is used
            # as a placeholder pending a proper m-resolved calculation; flagged explicitly in the
            # theory doc.
            S_ground_Fi[0, i_m] = self.sigma1_Ka1_2p1 * 0.5
            S_ground_Fi[0, i_p] = self.sigma1_Ka1_2p1 * 0.5
            S_ground_Fi[1, i_m] = self.sigma1_Ka1_2p1 * 0.5
            S_ground_Fi[1, i_p] = self.sigma1_Ka1_2p1 * 0.5

            # Further ionization of an already-2p1/2-holed ion (mirrors the base 2p3/2 S_ion_Fi
            # rows; sigma2_Ka1_2p1 is a new config key, e.g. from XATOM's total -pcs cross section
            # of the "2p1,0" hole configuration).
            S_ion_Fi[0, i_m] = self.sigma2_Ka1_2p1
            S_ion_Fi[0, i_p] = self.sigma2_Ka1_2p1
            S_ion_Fi[1, i_m] = self.sigma2_Ka1_2p1
            S_ion_Fi[1, i_p] = self.sigma2_Ka1_2p1
        else:
            self.ei_L2 = np.zeros(self.nlevel)

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
        if self.use_L2_pathway:
            Gamma_ij_L2K = 0.5 * (self.GammaL2fsm1N + self.GammaKfsm1N) + self.additional_dephasing
            Gamma_ij_L2L3 = 0.5 * (self.GammaL2fsm1N + self.GammaL3fsm1N) + self.additional_dephasing
            self.Mij = (self.Mij + self.GammaL2fsm1N * np.outer(self.ei_L2, self.ei_L2) +
                        Gamma_ij_L2K * (np.outer(self.ei_L2, self.ei_K) + np.outer(self.ei_K, self.ei_L2)) +
                        Gamma_ij_L2L3 * (np.outer(self.ei_L2, self.ei_L3) + np.outer(self.ei_L3, self.ei_L2)))

        # +1 for (i in K, j in L3), -1 for (i in L3, j in K), 0 otherwise -- including i,j both in
        # the same manifold (degenerate msublevels, e.g. i,j both in L3), which a naive index-order
        # sign(i-j) would get wrong. Kept for the satellite channels below (each a single-Delta_k
        # block, sign_ij_block * Delta_k).
        self.sign_ij_block = np.outer(self.ei_K, self.ei_L3) - np.outer(self.ei_L3, self.ei_K)

        # General per-pair detuning matrix (docs/theory-and-2s-satellite-pathways.md, Part III):
        # Delta_ij[i,j] = f[i]-f[j], where f is a per-LEVEL "intrinsic detuning from the shared
        # Kalpha1 rotating frame" (f=0 for K/L3, f=DeltaomegaL2mL3A for L2). This generalizes (and,
        # for the 2-manifold case, is algebraically identical to) the satellite blocks' scalar
        # Delta_k * sign_ij_block -- verified: for f = Delta_k*ei_K, f[i]-f[j] equals
        # Delta_k*sign_ij_block[i,j] for every (i,j) pair. Needed here because with 3 manifolds
        # sharing one block (K, L3, L2), a single scalar+sign_ij can no longer express both the
        # K<->L2 and L3<->L2 detunings (which differ from K<->L3's zero) at once.
        f_detuning = self.DeltaomegaL2mL3A * self.ei_L2 if self.use_L2_pathway else np.zeros(self.nlevel)
        self.Delta_ij = f_detuning[:, None] - f_detuning[None, :]

        # Incoherent 2s -> 2p_3/2 3d_5/2 Auger feeding
        if self.use_2s_pathway == True:
            self.auger_feeding_matrix = np.diag(self.ei_L3 / np.sum(self.ei_L3)) * self.GammaA_L1_to_L3M45fs1N
            self.S_2s_F = S_2s_F
        else:
            self.auger_feeding_matrix = np.zeros((self.nlevel, self.nlevel))
            self.S_2s_F = np.zeros(3)
            self.S_ground_Fi[:, -2] = 0.0
        self.transform_matrix = np.asarray([[1, 1], [1j, -1j]]) / np.sqrt(2.0) # transformation matrix from circular polarization vectors to Cartesian

        # 2s-hole satellite pathways (docs/theory-and-2s-satellite-pathways.md, Part II): each
        # channel is its own detuned 6-level block, reusing Tijs/Gij verbatim and, by default,
        # the base Mij (spectator approximation) unless it overrides Gamma_L_eV/Gamma_K_eV.
        if self.satellite_channels and nlevel_base != 6:
            raise ValueError('satellite_channels requires nlevel == 6 (they reuse the full sublevel-resolved base block structure)')
        if self.satellite_channels and self.use_L2_pathway:
            raise ValueError('satellite_channels and use_L2_pathway are not yet compatible -- the satellite '
                              'blocks only reuse the base 2p3/2/1s Tijs/Gij (6 local levels); 2p1/2-spectator '
                              'satellite channels are not modeled (see theory doc Part III scope note)')

        # NOTE: pump-driven spectator photoionization (theory doc Eq. S3/S4's sigma_P * J_P term)
        # is not applied -- there is currently no per-(t,z) pump photon flux available anywhere in
        # Model.py's regular RK4 functions to multiply such a cross section by (J_Omega_minus/
        # plus_xy are seed/Kalpha1-field-only throughout this codebase, confirmed via the pre-
        # existing, unchanged MB_ground_regular). xatom/xatom_tools.py accordingly no longer
        # computes a pump cross section at all; only the seed-field-driven term is applied.
        self.satellite_channel_params = []
        for channel in self.satellite_channels:
            Delta_fs = channel['detuning_eV'] / self.hbar
            Gamma_A_fs = channel['Gamma_A_2s_eV'] / self.hbar
            S_feed_2p = np.asarray([channel['sigma_Ka1_from_2p'], channel['sigma_Ka1_from_2p']])
            S_feed_1s = np.asarray([channel['sigma_Ka1_from_1s'], channel['sigma_Ka1_from_1s']])

            if ('Gamma_L_eV' in channel) or ('Gamma_K_eV' in channel):
                GammaL_fs = channel.get('Gamma_L_eV', self.GammaL3eVN) / self.hbar
                GammaK_fs = channel.get('Gamma_K_eV', self.GammaKeVN) / self.hbar
                Gamma_coh_fs = 0.5 * (GammaL_fs + GammaK_fs) + self.additional_dephasing
                Mij = (GammaL_fs * np.outer(self.ei_L3, self.ei_L3) +
                       GammaK_fs * np.outer(self.ei_K, self.ei_K) +
                       Gamma_coh_fs * (np.outer(self.ei_L3, self.ei_K) + np.outer(self.ei_K, self.ei_L3)))
            else:
                Mij = self.Mij

            # Further-ionization loss (theory doc section 12.4, Eq. S8): sigma_ion_from_2p/1s are
            # each a single scalar (the double-hole configuration's *own* total photoionization
            # cross section, e.g. from xatom_tools.total_photoionization_cross_section_nm2) applied
            # uniformly across every msublevel of that manifold -- same spectator-approximation
            # convention as Gamma_L_eV/Gamma_K_eV above (one width/rate per manifold, not per
            # msublevel). Optional, default 0 (no loss) if not supplied, matching the doc's
            # "default to 0 unless supplied" for this deferred term.
            S_ion_Fi_chan = np.zeros((2, self.nlevel))
            S_ion_Fi_chan[:, self.ei_L3.astype(bool)] = channel.get('sigma_ion_from_2p', 0.0)
            S_ion_Fi_chan[:, self.ei_K.astype(bool)] = channel.get('sigma_ion_from_1s', 0.0)

            self.satellite_channel_params.append(types.SimpleNamespace(
                name=channel['name'],
                Delta_ij=Delta_fs * self.sign_ij_block,  # see Delta_ij derivation above
                Gamma_A_fs=Gamma_A_fs,
                S_feed_2p=S_feed_2p,
                S_feed_1s=S_feed_1s,
                Mij=Mij,
                Gamma_sp_Gij=self.Gamma_sp_Gij,
                S_ion_Fi=S_ion_Fi_chan,  # further-ionization loss (theory doc §12.4)
            ))



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
        self.rho_sat_ijtxyz = self.sample.rho_sat_ijtxyz
        self.Omega_pstxyz = self.sample.Omega_pstxyz

        if self.is_Cartesian_pol:
            Omega_pqtxy = tools.circular_to_linear(self, self.Omega_pstxyz[:, :, :, :, :, -1])
            self.Omega_qtxy = (Omega_pqtxy[0, :, :, :, :] + np.conj(Omega_pqtxy[1, :, :, :, :])) / 2.0
            
