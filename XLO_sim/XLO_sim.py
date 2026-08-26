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


def _add_L2_manifold(Tijs, Gij, ei_K, nlevel_local, i_lower_start):
    """
    Mutate Tijs/Gij in place to add a 2p1/2 (Kalpha2-analogue) manifold at local indices
    [i_lower_start, i_lower_start+1], dipole-coupled to the SAME local 1s-like manifold already
    present (the two indices where ei_K==1). Pure angular-momentum algebra (Clebsch-Gordan-derived
    1s<->2p1/2 dipole/branching values, docs/theory-and-2s-satellite-pathways.md Part III sec 18;
    sign correction in docs/2p1_2-implementation-plan.md sec 4.1) -- independent of which physical
    configuration (bare Cu ion or a 2s-hole satellite's spectator-Xk configuration) the local
    1s-like manifold represents, so this is reused verbatim for both the base block and each
    satellite channel's own local block when it also gets the 2p1/2-satellite extension.

    The two local K-like indices must follow this codebase's existing convention (established by
    the base block, PDF Eq. 18): sorted ascending, the LOWER of the two pairs with the upper
    (m=+1/2) new sublevel at sigma-index 1, the HIGHER pairs with the lower (m=-1/2) new sublevel
    at sigma-index 0 -- verified to reproduce the base block's own pre-refactor Tijs/Gij entries
    bit-for-bit when called with i_lower_start=6 on the 6-level base block (K at local 4,5).

    Returns
    -------
    np.ndarray
        ei_L2, the new manifold's indicator array (length nlevel_local).
    """
    k_indices = np.flatnonzero(ei_K)
    if k_indices.size != 2:
        raise ValueError('L2 manifold extension requires exactly 2 local 1s-like sublevels')
    k_lo, k_hi = int(k_indices[0]), int(k_indices[1])
    i_m, i_p = i_lower_start, i_lower_start + 1

    Tijs[i_p, k_lo, 1] = -np.sqrt(2.0) / 3.0
    Tijs[k_lo, i_p, 1] = -np.sqrt(2.0) / 3.0
    Tijs[i_m, k_hi, 0] = np.sqrt(2.0) / 3.0
    Tijs[k_hi, i_m, 0] = np.sqrt(2.0) / 3.0

    Gij[i_p, k_lo] = 2.0 / 9.0
    Gij[i_m, k_lo] = 1.0 / 9.0
    Gij[i_m, k_hi] = 2.0 / 9.0
    Gij[i_p, k_hi] = 1.0 / 9.0

    ei_L2 = np.zeros(nlevel_local)
    ei_L2[i_m:i_p + 1] = 1
    return ei_L2


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

        # Double-M-shell-spectator ("double-satellite") channels
        # (docs/double-spectator-satellite-implementation-plan.md): a third generation of channels,
        # fed not by a cross-section x field-flux term (like every entry above) but by redirecting
        # part of an *existing* satellite_channels entry's own Gamma_L_eV decay (its `feed_from`
        # key, resolved to parent indices below). Structurally identical local 6/8-level blocks
        # otherwise, so they're appended onto the SAME self.satellite_channels list rather than
        # tracked separately -- every downstream consumer (satellite_channel_params construction,
        # Sample.py's marching loop, Model.absorption, Omega_source_regular) already iterates that
        # one flat list generically. list(...)+list(...) rather than .extend() so this doesn't
        # mutate self.config['satellite_channels'] in place (setattr above aliased it directly).
        if 'double_satellite_channels' in self.config and self.config['double_satellite_channels']:
            self.satellite_channels = list(self.satellite_channels) + list(self.config['double_satellite_channels'])

        # 2p1/2 (L2, Kalpha2) pathway (docs/theory-and-2s-satellite-pathways.md, Part III):
        # shares the SAME 1s (K) population as the base 2p3/2 (L3) block -- unlike the satellite
        # channels, this is not a separate configuration, so it can't be a separate block; it
        # extends the base block itself by 2 levels (local indices nlevel_base, nlevel_base+1).
        self.use_L2_pathway = self.config.get('use_L2_pathway', False)
        nlevel_base = self.nlevel  # as read from YAML: the base block's own level count (6 or 2)
        self.nlevel_base = nlevel_base
        if self.use_L2_pathway and nlevel_base != 6:
            raise ValueError('use_L2_pathway requires nlevel: 6 (2p1/2 extends the full sublevel-resolved base block)')
        if self.use_L2_pathway:
            self.nlevel = nlevel_base + 2

        # 2p1/2-satellite ("Kalpha2-satellite") extension of each satellite channel: combines Part
        # II's satellite blocks with Part III's L2 pathway one level deeper -- each channel's own
        # local block gains a 2p1/2+X_k manifold, exactly as use_L2_pathway extends the base block.
        # Auto-enabled whenever both ingredients are present, no separate YAML flag: the new
        # spectator-photoionization feed into 2p1/2+X_k needs the base block's own 2p1/2
        # population, which only exists when use_L2_pathway is on (see xatom_tools.py's
        # l2_satellite_channel_parameters and the satellite_channel_params construction below).
        self.use_L2_satellite_pathway = self.use_L2_pathway and bool(self.satellite_channels)
        self.satellite_nlevel = nlevel_base + (2 if self.use_L2_satellite_pathway else 0)

        self.sigma_compound_Ka1 = sum(element['N_atoms'] * element['sigma_compound_Ka1'] for element in self.compound.values())

        self.lambdaKalpha1N = 2.0 * np.pi * self.c * self.hbar / self.hwKalpha1N
        self.lambdaCenter = 2.0 * np.pi * self.c * self.hbar / self.seed_center_E
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
            S_ground_Fi[0, self.nlevel] = self.sigma1_Ka1_2s
            S_ground_Fi[0, self.nlevel + 1] = self.sigma1_Ka1_other

            S_ground_Fi[1, 0] = self.sigma1_Ka1_2p3 * 0.42
            S_ground_Fi[1, 1] = self.sigma1_Ka1_2p3 * 0.28
            S_ground_Fi[1, 2] = self.sigma1_Ka1_2p3 * 0.18
            S_ground_Fi[1, 3] = self.sigma1_Ka1_2p3 * 0.12
            S_ground_Fi[1, self.nlevel] = self.sigma1_Ka1_2s
            S_ground_Fi[1, self.nlevel + 1] = self.sigma1_Ka1_other

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
            # 1s(m=+1/2)->2p1/2(m=-1/2) [sigma-index 0], each with |T|=sqrt(2)/3 (T^2=2/9, matching
            # the existing 2p3/2 block's T^2=G self-consistency pattern). Gij additionally carries
            # the isotropic-only (sigma=0/pi, not part of Tijs) 1/9 branch to the *same-m* 1s
            # sublevel, using exactly the same 1/9,2/9 building-block fractions the base 2p3/2
            # block already uses.
            #
            # Sign: unlike the 2p3/2 (j=l+1/2) block, whose two sigma branches carry the SAME sign
            # (verified by reproducing all 8 of its Tijs entries via the Wigner-Eckart/6j formula
            # below), the 2p1/2 (j=l-1/2) reduced matrix element picks up a relative minus sign
            # between its two m-sublevel branches -- a genuine angular-momentum-algebra asymmetry
            # between j=l+1/2 and j=l-1/2 manifolds, not a free convention choice. Confirmed via
            # sympy.physics.wigner (wigner_3j/wigner_6j) against the same formula that reproduces
            # the base block's signs exactly: <2p1/2,m=+1/2|T_{q=+1}|1s,m=-1/2> = -sqrt(2)/3, while
            # <2p1/2,m=-1/2|T_{q=-1}|1s,m=+1/2> = +sqrt(2)/3. Getting this wrong makes the two
            # m-branches interfere with the wrong relative phase in the coherent polarization sum,
            # which can flip a net-absorptive spectral feature into a net-emissive one.
            i_m, i_p = nlevel_base, nlevel_base + 1  # local 2p1/2 m=-1/2, m=+1/2

            # Tijs/Gij values (both magnitude and the j=l-1/2 relative sign) computed by the
            # shared _add_L2_manifold helper -- see its docstring; verified to reproduce this
            # block's own original inline values bit-for-bit (K at local indices 4,5, sorted
            # ascending matches k_lo=4/k_hi=5 exactly as used here before the refactor).
            self.ei_L2 = _add_L2_manifold(Tijs, Gij, self.ei_K, self.nlevel, i_lower_start=nlevel_base)

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
        # Tijs_plus[i,j]/Tijs_minus[i,j] must hold Tijs[i,j] iff i is the PHYSICALLY upper (K/1s-
        # hole) state and j is a PHYSICALLY lower (2p-hole) state, or vice versa -- this is what
        # makes Hint[i,j] (Model.py) couple to the correct field (Omega_plus for i=upper, j=lower;
        # Omega_minus for i=lower, j=upper), matching the standard RWA dipole-Hamiltonian result
        # H_int[e,g] ~ -d_eg*epsilon(t), H_int[g,e] ~ -d_ge*epsilon*(t) for a two-level atom (see
        # docs/2p1_2-implementation-plan.md, sections 2-3). self.Hij (i>j via raw array index) is
        # only a correct proxy for this when every "upper" state has a numerically larger index
        # than every "lower" state it couples to -- true for the base K(4,5)>L3(0-3) block by
        # construction, but false for L2 (6,7), which was appended AFTER K. Built from physical
        # role (ei_K/ei_L3/ei_L2) instead of raw index, this reduces to exactly self.Hij whenever
        # ei_L2 is all-zero (use_L2_pathway=False, incl. every satellite-channel config, which
        # reuses these same Tijs_plus/Tijs_minus) -- zero behavior change for any existing config.
        ei_upper = self.ei_K
        ei_lower = self.ei_L3 + self.ei_L2
        role_mask_upper_lower = np.outer(ei_upper, ei_lower)
        self.Tijs_plus = np.einsum('ijs, ij->ijs', self.Tijs, role_mask_upper_lower)
        self.Tijs_minus = np.einsum('ijs, ij->ijs', self.Tijs, role_mask_upper_lower.T)
        if self.use_L2_satellite_pathway:
            # Independent local template -- NOT sliced from the base block's own Tijs/Tijs_plus/
            # Tijs_minus, unlike the L2-off case below: the base block's global indices
            # nlevel_base,nlevel_base+1 hold the BARE 2p1/2 hole (no spectator), a different
            # physical state from a satellite channel's own local 2p1/2+X_k manifold. Reuses the
            # exact same universal Clebsch-Gordan values via _add_L2_manifold (spectator-
            # independent, per Part II's core reuse argument for the 0..5 corner), just placed in a
            # fresh self.satellite_nlevel x self.satellite_nlevel array (0..nlevel_base-1 copied
            # from the base 2p3/2<->1s corner, nlevel_base.. zero until _add_L2_manifold fills it)
            # so the satellite template's local L2 indices don't alias the base block's own global
            # L2 slot.
            Tijs_sat = np.zeros((self.satellite_nlevel, self.satellite_nlevel, 2), dtype=complex)
            Gij_sat = np.zeros((self.satellite_nlevel, self.satellite_nlevel), dtype=complex)
            Tijs_sat[:nlevel_base, :nlevel_base, :] = Tijs[:nlevel_base, :nlevel_base, :]
            Gij_sat[:nlevel_base, :nlevel_base] = Gij[:nlevel_base, :nlevel_base]
            ei_L3_sat = np.zeros(self.satellite_nlevel); ei_L3_sat[:nlevel_base] = self.ei_L3[:nlevel_base]
            ei_K_sat = np.zeros(self.satellite_nlevel); ei_K_sat[:nlevel_base] = self.ei_K[:nlevel_base]
            ei_L2_sat = _add_L2_manifold(Tijs_sat, Gij_sat, ei_K_sat, self.satellite_nlevel, i_lower_start=nlevel_base)
            role_mask_sat = np.outer(ei_K_sat, ei_L3_sat + ei_L2_sat)
            self.Tijs_plus_satellite = np.einsum('ijs, ij->ijs', Tijs_sat, role_mask_sat)
            self.Tijs_minus_satellite = np.einsum('ijs, ij->ijs', Tijs_sat, role_mask_sat.T)
        else:
            # Satellite channels (below) are always local nlevel_base (6) blocks -- 2p3/2<->1s only,
            # theory doc Part II -- regardless of whether use_L2_pathway extended the base block to 8
            # levels. L2 is appended AFTER the base 6 (local indices nlevel_base, nlevel_base+1), so it
            # only ever writes into rows/cols >= nlevel_base of Tijs/Tijs_plus/Tijs_minus above; the
            # top-left nlevel_base x nlevel_base corner is therefore byte-for-byte identical to what a
            # standalone nlevel=6 (no L2) config would have produced. Slicing recovers exactly the
            # tensors the satellite blocks need, with zero special-casing and zero behaviour change when
            # use_L2_pathway is False (slice is then the whole array).
            self.Tijs_plus_satellite = self.Tijs_plus[:nlevel_base, :nlevel_base, :]
            self.Tijs_minus_satellite = self.Tijs_minus[:nlevel_base, :nlevel_base, :]
            Gij_sat = Gij[:nlevel_base, :nlevel_base]
            ei_L3_sat = self.ei_L3[:nlevel_base]
            ei_K_sat = self.ei_K[:nlevel_base]
            ei_L2_sat = np.zeros(nlevel_base)
        # Persisted (rather than left as __init__-local) so post-run analysis code (tools.py's
        # compute_run_outputs) can isolate the L3-only/L2-only/K-only contributions to a satellite
        # channel's Tijs_plus_satellite/Tijs_minus_satellite, exactly as self.ei_L3/ei_K/ei_L2 already
        # let it do for the base block.
        self.ei_L3_satellite = ei_L3_sat
        self.ei_K_satellite = ei_K_sat
        self.ei_L2_satellite = ei_L2_sat
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
        # sign(i-j) would get wrong. Not consumed elsewhere (the satellite channels below use the
        # equivalent, L2k-extensible f_i-f_j formulation instead, theory doc Eq. K4) -- kept as a
        # documented reference for the base K<->L3 sign convention.
        self.sign_ij_block = np.outer(self.ei_K, self.ei_L3) - np.outer(self.ei_L3, self.ei_K)

        # General per-pair detuning matrix (docs/theory-and-2s-satellite-pathways.md, Part III;
        # sign re-derived in docs/2p1_2-implementation-plan.md section 7): Delta_ij[i,j] = f[i]-f[j],
        # where f is a per-LEVEL "intrinsic detuning from the shared Kalpha1 rotating frame" (f=0
        # for K/L3, f=-DeltaomegaL2mL3A for L2).
        #
        # Sign: rho_ij(t) ~ exp(-i*Delta_ij[i,j]*t) from the -1j*Delta_ij[i,j]*rho_ij term in
        # Model.py's off-diagonal equation. Omega_source_regular sources Omega_plus from
        # Tijs_minus[i,j]*rho_hermitian[j,i], and (post role-mask fix) Tijs_minus survives for
        # i=lower(L2),j=upper(K) -- so Omega_plus ~ rho_hermitian[K,L2] ~ exp(-i*Delta_ij[K,L2]*t),
        # which the FFT (numpy convention, exp(-i*omega*t)) places at bin omega=-Delta_ij[K,L2].
        # For the Kalpha2 feature to appear at NEGATIVE detuning (Kalpha2 < Kalpha1 in photon
        # energy), we need omega=-DeltaomegaL2mL3A, i.e. Delta_ij[K,L2]=+DeltaomegaL2mL3A -- which
        # requires f[L2]=-DeltaomegaL2mL3A (f[i]-f[j] with f[K]=0 gives Delta_ij[K,L2]=-f[L2]).
        # Verified empirically: with the role-mask fix alone (this sign unchanged), the spectral
        # feature appeared as a genuine dip but at +20 eV instead of -20 eV -- exactly the single
        # sign flip this correction makes.
        f_detuning = -self.DeltaomegaL2mL3A * self.ei_L2 if self.use_L2_pathway else np.zeros(self.nlevel)
        self.Delta_ij = f_detuning[:, None] - f_detuning[None, :]

        # Incoherent 2s -> 2p_3/2 3d_5/2 Auger feeding
        if self.use_2s_pathway == True:
            self.auger_feeding_matrix = np.diag(self.ei_L3 / np.sum(self.ei_L3)) * self.GammaA_L1_to_L3M45fs1N
            self.S_2s_F = S_2s_F
        else:
            self.auger_feeding_matrix = np.zeros((self.nlevel, self.nlevel))
            self.S_2s_F = np.zeros(2)
            self.S_ground_Fi[:, -2] = 0.0
        self.transform_matrix = np.asarray([[1, 1], [1j, -1j]]) / np.sqrt(2.0) # transformation matrix from circular polarization vectors to Cartesian

        # 2s-hole satellite pathways (docs/theory-and-2s-satellite-pathways.md, Part II): each
        # channel is its own detuned 6-level block, reusing Tijs/Gij verbatim and, by default,
        # the base Mij (spectator approximation) unless it overrides Gamma_L_eV/Gamma_K_eV. When
        # use_L2_satellite_pathway is also on, each channel's block grows to 8 local levels (its
        # own 2p1/2+X_k manifold, local indices nlevel_base,nlevel_base+1) -- the Kalpha2-satellite
        # analogue of Part III's base-block L2 extension, one level deeper: the channel's 1sX_k
        # upper manifold (local 4,5) is shared between the Kalpha1-satellite (Lk) and Kalpha2-
        # satellite (L2k) branches, exactly as the base block's single 1s-hole population is shared
        # between Kalpha1 (L3) and Kalpha2 (L2) -- so this must extend the channel's existing block,
        # not add a second one.
        if self.satellite_channels and nlevel_base != 6:
            raise ValueError('satellite_channels requires nlevel == 6 (they reuse the full sublevel-resolved base block structure)')

        # NOTE: pump-driven spectator photoionization (theory doc Eq. S3/S4's sigma_P * J_P term)
        # is not applied -- there is currently no per-(t,z) pump photon flux available anywhere in
        # Model.py's regular RK4 functions to multiply such a cross section by (J_Omega_minus/
        # plus_xy are seed/Kalpha1-field-only throughout this codebase, confirmed via the pre-
        # existing, unchanged MB_ground_regular). xatom/xatom_tools.py accordingly no longer
        # computes a pump cross section at all; only the seed-field-driven term is applied.
        # ei_L3_sat/ei_K_sat/ei_L2_sat/Gij_sat come from the Tijs_plus_satellite/Tijs_minus_satellite
        # construction above (already sized self.satellite_nlevel, with ei_L2_sat all-zero when
        # use_L2_satellite_pathway is off).
        Gamma_sp_Gij_sat = self.Gamma_sp_fsm1N * Gij_sat

        # Name -> index lookup for double-satellite channels' `feed_from` references (below) --
        # built once, before the main loop, so a double-satellite entry can reference a parent
        # regardless of list order (Sample.py always reads the PRE-update rho_sat_ijxy list in
        # full before updating any entry, so there is no ordering requirement -- see
        # docs/double-spectator-satellite-implementation-plan.md section 3).
        _satellite_name_to_index = {}
        for idx, channel in enumerate(self.satellite_channels):
            _satellite_name_to_index.setdefault(channel['name'], []).append(idx)

        self.satellite_channel_params = []
        for channel in self.satellite_channels:
            Delta_fs = channel['detuning_eV'] / self.hbar
            # Double-satellite channels (feed_from key present) have no Gamma_A_2s_eV/
            # sigma_Ka1_from_2p/1s -- their entire feed comes from feed_from instead (resolved
            # below), so these default to 0 rather than being required.
            Gamma_A_fs = channel.get('Gamma_A_2s_eV', 0.0) / self.hbar
            # Direct K-hole (1s) non-radiative "KLM-type" Auger feed into this channel's own Lk
            # manifold (Model.py's feed_diag_satellite_block) -- a second, independent production
            # route alongside the 2s-Auger feed above, redirecting part of the K-hole's own
            # ALREADY-occurring non-radiative decay (see the budget check after this loop).
            # Optional, defaults to 0 for any channel/config that doesn't set it (e.g. every
            # double_satellite_channels entry -- no 3-body "1s hole -> 2p-hole + 2 M-holes" KLM
            # analogue has been derived for those). The four base satellite_channels entries
            # (3d+/3d-/3p+/3p-) in config/base/*.yaml carry real values, read 2026-08-27 off a
            # live `xatom -hole 1s1 -decay` run via xatom_tools.auger_partial_rate_eV(spectator,
            # parent_hole_config='1s1', initial_label='1s0') -- e.g. 3p+'s 0.034 eV came from the
            # "1s0 - 2p+ 3p+" row. These turned out to be ~1-2 orders of magnitude smaller than the
            # existing Gamma_A_2s_eV, but the base block's own bare K-hole population is itself
            # ~1e3-1e4x larger than rho_2s across the pulse-energy range checked (SASE ensemble
            # data, data/sweep_double_satellite_and_L2_update_2) -- net effect: this new feed
            # dominates the existing 2s-driven feed by 2-3 orders of magnitude, not a small
            # correction. See the session's chat log for the full back-of-envelope; this has NOT
            # yet been validated by an actual run_3D() (only by post-hoc analysis of saved
            # populations from runs that predate this feed term).
            Gamma_A_K_fs = channel.get('Gamma_A_K_eV', 0.0) / self.hbar
            S_feed_2p = np.asarray([channel.get('sigma_Ka1_from_2p', 0.0), channel.get('sigma_Ka1_from_2p', 0.0)])
            S_feed_1s = np.asarray([channel.get('sigma_Ka1_from_1s', 0.0), channel.get('sigma_Ka1_from_1s', 0.0)])

            # feed_from (docs/double-spectator-satellite-implementation-plan.md section 3): list of
            # (parent_index_into_self.satellite_channels, Gamma_feed_fs) pairs, resolved by parent
            # name now so Model.py's feed_diag_satellite_block never has to do string lookups
            # per-timestep. Absent/empty for every pre-existing (cross-section-fed) channel.
            feed_from_resolved = []
            for feed in channel.get('feed_from', []):
                parent_name = feed['channel']
                matches = _satellite_name_to_index.get(parent_name, [])
                if len(matches) != 1:
                    raise ValueError(
                        f"double-satellite channel {channel.get('name', '?')!r}'s feed_from "
                        f"references channel {parent_name!r}, which matches {len(matches)} "
                        f"satellite_channels entries by name (must match exactly 1)"
                    )
                manifold = feed.get('manifold', 'lower')
                if manifold not in ('lower', 'upper', 'L2'):
                    raise ValueError(f"feed_from manifold must be 'lower', 'upper', or 'L2', got {manifold!r}")
                if manifold == 'L2' and not self.use_L2_satellite_pathway:
                    # A manifold='L2' entry reads/writes the L2k (2p1/2+X) slice of the local
                    # block, which only exists (nlevel_sat > nlevel_base) when
                    # use_L2_satellite_pathway is on -- without it this entry would silently do
                    # nothing (docs/double-spectator-satellite-implementation-plan.md section 9),
                    # so fail loudly instead.
                    raise ValueError(
                        f"double-satellite channel {channel.get('name', '?')!r}'s feed_from has a "
                        f"manifold='L2' entry, which requires use_L2_pathway: True (and at least "
                        f"one satellite_channels entry) to be meaningful -- unset it or remove "
                        f"the manifold='L2' entries"
                    )
                feed_from_resolved.append((matches[0], feed['Gamma_feed_eV'] / self.hbar, manifold))

            if ('Gamma_L_eV' in channel) or ('Gamma_K_eV' in channel):
                GammaL_fs = channel.get('Gamma_L_eV', self.GammaL3eVN) / self.hbar
                GammaK_fs = channel.get('Gamma_K_eV', self.GammaKeVN) / self.hbar
            else:
                GammaL_fs = self.GammaL3fsm1N
                GammaK_fs = self.GammaKfsm1N
            Gamma_coh_fs = 0.5 * (GammaL_fs + GammaK_fs) + self.additional_dephasing
            Mij = (GammaL_fs * np.outer(ei_L3_sat, ei_L3_sat) +
                   GammaK_fs * np.outer(ei_K_sat, ei_K_sat) +
                   Gamma_coh_fs * (np.outer(ei_L3_sat, ei_K_sat) + np.outer(ei_K_sat, ei_L3_sat)))

            # Per-level "intrinsic detuning from the shared Kalpha1 rotating frame" (same f-vector
            # construction as the base block's Delta_ij, theory doc Eq. K4): f=0 for Lk, f=Delta_fs
            # for Uk -- Delta_fs plays the role the base block's f[K] plays there, since this
            # channel's own Kalpha1-satellite line is itself detuned by Delta_fs from the shared
            # frame. Reduces to exactly Delta_fs*sign_ij_block for every (i,j) pair (verified: f_i-f_j
            # with f=0/Delta_fs is +Delta_fs for Uk->Lk, -Delta_fs for Lk->Uk, 0 within a manifold --
            # the same three cases sign_ij_block covers), so this is a zero-behavior-change
            # generalization when L2k is absent.
            f_local = Delta_fs * ei_K_sat

            # Further-ionization loss (theory doc section 12.4, Eq. S8): sigma_ion_from_2p/1s are
            # each a single scalar (the double-hole configuration's *own* total photoionization
            # cross section, e.g. from xatom_tools.total_photoionization_cross_section_nm2) applied
            # uniformly across every msublevel of that manifold -- same spectator-approximation
            # convention as Gamma_L_eV/Gamma_K_eV above (one width/rate per manifold, not per
            # msublevel). Optional, default 0 (no loss) if not supplied, matching the doc's
            # "default to 0 unless supplied" for this deferred term.
            S_ion_Fi_chan = np.zeros((2, self.satellite_nlevel))
            S_ion_Fi_chan[:, ei_L3_sat.astype(bool)] = channel.get('sigma_ion_from_2p', 0.0)
            S_ion_Fi_chan[:, ei_K_sat.astype(bool)] = channel.get('sigma_ion_from_1s', 0.0)

            Gamma_A_L2_fs = None
            Gamma_A_K_to_L2_fs = None
            S_feed_2p1 = None
            if self.use_L2_satellite_pathway:
                is_double_satellite = bool(channel.get('feed_from'))
                # New keys required from xatom_tools.l2_satellite_channel_parameters (theory doc:
                # this section's "2p1/2-satellite" extension, combining Part II + Part III) -- fail
                # loudly rather than silently running with an incomplete/inconsistent channel, since
                # a missing key here would otherwise show up only as a subtly wrong spectrum.
                # Double-satellite channels (docs/double-spectator-satellite-implementation-plan.md
                # section 9) don't need Gamma_A_2s_to_L2_eV/sigma_Ka1_from_2p1 -- those describe the
                # OLD (2s-Auger / cross-section-driven) feed mechanisms, which don't apply here; a
                # double-satellite channel's L2k manifold is instead fed by manifold='L2' entries
                # in its own feed_from (resolved below, same as its Lk/Uk manifolds already are).
                required_keys = ('detuning_eV_L2_split', 'Gamma_L2_eV') if is_double_satellite else (
                    'detuning_eV_L2_split', 'Gamma_A_2s_to_L2_eV', 'Gamma_L2_eV', 'sigma_Ka1_from_2p1')
                missing = [k for k in required_keys if k not in channel]
                if missing:
                    raise ValueError(
                        f"satellite_channels entry {channel.get('name', '?')!r} is missing {missing} "
                        f"(required whenever use_L2_pathway is combined with satellite_channels -- "
                        f"run xatom_tools.l2_satellite_channel_parameters()/double_spectator_L2_parameters() "
                        f"and merge its output into this channel's YAML entry, or unset "
                        f"use_L2_pathway/satellite_channels to skip the 2p1/2-satellite extension)"
                    )

                Delta_L2_split_fs = channel['detuning_eV_L2_split'] / self.hbar
                Gamma_A_L2_fs = channel.get('Gamma_A_2s_to_L2_eV', 0.0) / self.hbar
                # L2k-manifold analogue of Gamma_A_K_fs above (1s hole filled by a 2p1/2 electron
                # instead of 2p3/2) -- same optional/defaults-to-0 convention.
                Gamma_A_K_to_L2_fs = channel.get('Gamma_A_K_to_L2_eV', 0.0) / self.hbar
                GammaL2_fs = channel['Gamma_L2_eV'] / self.hbar
                S_feed_2p1 = np.asarray([channel.get('sigma_Ka1_from_2p1', 0.0), channel.get('sigma_Ka1_from_2p1', 0.0)])

                # f[L2k] = f[Uk] - Delta_L2_split, the exact per-channel analogue of the base
                # block's f[L2]=f[K]-DeltaomegaL2mL3A (theory doc Eq. K4), but using this channel's
                # OWN Kalpha1-satellite/Kalpha2-satellite splitting (xatom_tools.py's
                # l2_satellite_splitting_eV) rather than assuming the bare-ion splitting carries
                # over unchanged.
                f_local = f_local + ei_L2_sat * (Delta_fs - Delta_L2_split_fs)

                Gamma_coh_L2K_fs = 0.5 * (GammaL2_fs + GammaK_fs) + self.additional_dephasing
                Gamma_coh_L2L3_fs = 0.5 * (GammaL2_fs + GammaL_fs) + self.additional_dephasing
                Mij = (Mij + GammaL2_fs * np.outer(ei_L2_sat, ei_L2_sat) +
                       Gamma_coh_L2K_fs * (np.outer(ei_L2_sat, ei_K_sat) + np.outer(ei_K_sat, ei_L2_sat)) +
                       Gamma_coh_L2L3_fs * (np.outer(ei_L2_sat, ei_L3_sat) + np.outer(ei_L3_sat, ei_L2_sat)))

                S_ion_Fi_chan[:, ei_L2_sat.astype(bool)] = channel.get('sigma_ion_from_2p1', 0.0)

            Delta_ij_chan = f_local[:, None] - f_local[None, :]

            self.satellite_channel_params.append(types.SimpleNamespace(
                name=channel['name'],
                Delta_ij=Delta_ij_chan,
                Gamma_A_fs=Gamma_A_fs,
                Gamma_A_K_fs=Gamma_A_K_fs,
                Gamma_A_L2_fs=Gamma_A_L2_fs,  # None unless use_L2_satellite_pathway
                Gamma_A_K_to_L2_fs=Gamma_A_K_to_L2_fs,  # None unless use_L2_satellite_pathway
                feed_from=feed_from_resolved,  # [] unless this is a double-satellite channel
                S_feed_2p=S_feed_2p,
                S_feed_1s=S_feed_1s,
                S_feed_2p1=S_feed_2p1,  # None unless use_L2_satellite_pathway
                Mij=Mij,
                Gamma_sp_Gij=Gamma_sp_Gij_sat,
                S_ion_Fi=S_ion_Fi_chan,  # further-ionization loss (theory doc §12.4)
            ))

        # Budget check (mirrors the established GammaL1eVN/Gamma_A_2s_eV pattern, theory doc
        # §12.7/§27): Gamma_A_K_eV/Gamma_A_K_to_L2_eV redirect part of the K-hole's own ALREADY-
        # occurring non-radiative decay (GammaKeVN minus the radiative Kalpha1+Kalpha2 branch)
        # into the satellite ladder -- summed over every channel, this must not exceed that
        # non-radiative budget, or the feed would remove population faster than the K-hole
        # actually decays non-radiatively (Mij[K,K] = GammaKfsm1N itself is untouched by this
        # feature). Checked loudly, not silently, per this codebase's own established convention;
        # 0 for every config that doesn't set these keys, so this is a no-op check otherwise.
        Gamma_K_nonradiative_eV = (
            self.config['GammaKeVN'] - self.config['GammarKalpha1eVN'] - self.config['GammarKalpha2eVN'])
        Gamma_A_K_total_eV = sum(
            channel.get('Gamma_A_K_eV', 0.0) + channel.get('Gamma_A_K_to_L2_eV', 0.0)
            for channel in self.satellite_channels
        )
        if Gamma_A_K_total_eV > Gamma_K_nonradiative_eV:
            raise ValueError(
                f"satellite_channels' Gamma_A_K_eV/Gamma_A_K_to_L2_eV sum to "
                f"{Gamma_A_K_total_eV:.4f} eV, exceeding the K-hole's own non-radiative decay "
                f"budget (GammaKeVN - GammarKalpha1eVN - GammarKalpha2eVN = "
                f"{Gamma_K_nonradiative_eV:.4f} eV) -- this would feed the satellite ladder "
                f"faster than the K-hole actually decays non-radiatively. Reduce these values or "
                f"recheck the underlying XATOM branching."
            )

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
            
