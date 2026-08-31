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
    Mutate Tijs/Gij in place to add a 2p1/2 manifold at local indices [i_lower_start, +1],
    dipole-coupled to the local 1s-like manifold (the two indices where ei_K==1). Clebsch-Gordan
    values shared by the base block and every L2-satellite channel (docs/theory-and-2s-satellite-pathways.md
    Part III sec 18; sign per docs/2p1_2-implementation-plan.md sec 4.1). K indices sorted ascending:
    lower index pairs with the m=-1/2 new sublevel (sigma 0), higher with m=+1/2 (sigma 1).

    Returns ei_L2, the new manifold's indicator array (length nlevel_local).
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
            self.keep_z_history = True

        if 'satellite_channels' not in self.config:
            self.satellite_channels = []

        if 'double_satellite_channels' in self.config and self.config['double_satellite_channels']:
            self.satellite_channels = list(self.satellite_channels) + list(self.config['double_satellite_channels'])

        # 2p1/2 (L2, Kalpha2) pathway: shares the base block's 1s (K) population, so it extends
        # the base block by 2 levels (local indices nlevel_base, nlevel_base+1) rather than being
        # a separate block (docs/theory-and-2s-satellite-pathways.md, Part III).
        self.use_L2_pathway = self.config.get('use_L2_pathway', False)
        nlevel_base = self.nlevel
        self.nlevel_base = nlevel_base
        if self.use_L2_pathway:
            self.nlevel = nlevel_base + 2

        # Auto-enabled 2p1/2-satellite extension: when both L2 and satellite channels are on, each
        # channel's local block also gains its own 2p1/2+X_k manifold (2 extra local levels).
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
        # GammaL2eVN (2p1/2 hole width) is only required when use_L2_pathway is on.
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
            # 2p1/2 (L2), local indices nlevel_base (m=-1/2), nlevel_base+1 (m=+1/2). Tijs/Gij from
            # Clebsch-Gordan algebra (docs/theory-and-2s-satellite-pathways.md Part III). The j=l-1/2
            # relative sign between the two m-branches is physical, not a convention choice -- get it
            # wrong and a net-absorptive feature can flip to net-emissive (verified via sympy.physics.wigner).
            i_m, i_p = nlevel_base, nlevel_base + 1
            self.ei_L2 = _add_L2_manifold(Tijs, Gij, self.ei_K, self.nlevel, i_lower_start=nlevel_base)

            # Ground-state photoionization directly into 2p1/2. m-resolved branching isn't derived
            # (unlike Tijs/Gij) -- even 50/50 split across msublevels is a placeholder.
            S_ground_Fi[0, i_m] = self.sigma1_Ka1_2p1 * 0.5
            S_ground_Fi[0, i_p] = self.sigma1_Ka1_2p1 * 0.5
            S_ground_Fi[1, i_m] = self.sigma1_Ka1_2p1 * 0.5
            S_ground_Fi[1, i_p] = self.sigma1_Ka1_2p1 * 0.5

            # Further ionization of an already-2p1/2-holed ion, mirrors the base 2p3/2 S_ion_Fi rows.
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
        # Tijs_plus/Tijs_minus[i,j] must hold Tijs[i,j] iff i is the physically upper (K) state and
        # j the lower (2p-hole) state, or vice versa, so Hint (Model.py) couples to the right field.
        # self.Hij (i>j by raw index) is only a valid proxy for that while every upper state has a
        # larger index than every lower state -- true for base K(4,5)>L3(0-3), but not once L2(6,7)
        # is appended after K. Built from physical role (ei_K/ei_L3/ei_L2) instead, which reduces to
        # exactly self.Hij whenever ei_L2 is all-zero (use_L2_pathway off).
        ei_upper = self.ei_K
        ei_lower = self.ei_L3 + self.ei_L2
        role_mask_upper_lower = np.outer(ei_upper, ei_lower)
        self.Tijs_plus = np.einsum('ijs, ij->ijs', self.Tijs, role_mask_upper_lower)
        self.Tijs_minus = np.einsum('ijs, ij->ijs', self.Tijs, role_mask_upper_lower.T)
        if self.use_L2_satellite_pathway:
            # Independent local template, not sliced from the base block: the base block's global
            # nlevel_base/+1 indices hold the bare 2p1/2 hole, a different physical state from a
            # satellite channel's own local 2p1/2+X_k manifold. Same Clebsch-Gordan values via
            # _add_L2_manifold, placed in a fresh satellite_nlevel-sized array.
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
            # Satellite channels are always local nlevel_base (6) blocks (2p3/2<->1s only), even
            # when use_L2_pathway extended the base block to 8 -- L2 is appended after the base 6,
            # so the top-left nlevel_base x nlevel_base corner is unaffected and slicing recovers
            # exactly what the satellite blocks need.
            self.Tijs_plus_satellite = self.Tijs_plus[:nlevel_base, :nlevel_base, :]
            self.Tijs_minus_satellite = self.Tijs_minus[:nlevel_base, :nlevel_base, :]
            Gij_sat = Gij[:nlevel_base, :nlevel_base]
            ei_L3_sat = self.ei_L3[:nlevel_base]
            ei_K_sat = self.ei_K[:nlevel_base]
            ei_L2_sat = np.zeros(nlevel_base)
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

        # Sign convention reference for the base K<->L3 pair (+1 for i in K,j in L3; -1 reversed;
        # 0 within a manifold) -- not consumed elsewhere, the satellite channels use the equivalent
        # f_i-f_j formulation below instead (theory doc Eq. K4).
        self.sign_ij_block = np.outer(self.ei_K, self.ei_L3) - np.outer(self.ei_L3, self.ei_K)

        # Per-pair detuning from the shared Kalpha1 rotating frame: Delta_ij[i,j] = f[i]-f[j], with
        # f=0 for K/L3 and f=-DeltaomegaL2mL3A for L2 (docs/theory-and-2s-satellite-pathways.md Part
        # III; sign re-derived in docs/2p1_2-implementation-plan.md sec 7). This sign places the
        # Kalpha2 feature at negative detuning as required by the FFT's exp(-i*omega*t) convention --
        # verified empirically (the wrong sign put the feature at +20 eV instead of -20 eV).
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
        # channel is its own detuned 6-level block, reusing Tijs/Gij and, by default, the base Mij
        # unless it overrides Gamma_L_eV/Gamma_K_eV. When use_L2_satellite_pathway is also on, each
        # channel's block grows to 8 local levels (its own 2p1/2+X_k manifold) -- the Kalpha2-satellite
        # analogue of Part III's base-block L2 extension, extending the channel's existing block
        # rather than adding a second one.

        # Pump-driven spectator photoionization (Eq. S3/S4's sigma_P*J_P term) isn't applied -- no
        # per-(t,z) pump flux exists in the RK4 path; only the seed-field-driven term is used.
        Gamma_sp_Gij_sat = self.Gamma_sp_fsm1N * Gij_sat

        # Name -> index lookup for double-satellite channels' feed_from references below.
        _satellite_name_to_index = {}
        for idx, channel in enumerate(self.satellite_channels):
            _satellite_name_to_index.setdefault(channel['name'], []).append(idx)

        self.satellite_channel_params = []
        for channel in self.satellite_channels:
            Delta_fs = channel['detuning_eV'] / self.hbar
            # Double-satellite channels feed exclusively via feed_from (below), so these default to 0.
            Gamma_A_fs = channel.get('Gamma_A_2s_eV', 0.0) / self.hbar
            # Direct K-hole (1s) non-radiative "KLM-type" Auger feed into this channel's Lk manifold,
            # independent of the 2s-Auger route above; redirects part of the K-hole's own already-
            # occurring non-radiative decay (budget-checked after this loop). Optional, defaults to 0.
            # CAUTION: not yet validated by an actual run_3D(), only by post-hoc analysis of saved
            # populations (docs/double-spectator-satellite-implementation-plan.md).
            Gamma_A_K_fs = channel.get('Gamma_A_K_eV', 0.0) / self.hbar
            S_feed_2p = np.asarray([channel.get('sigma_Ka1_from_2p', 0.0), channel.get('sigma_Ka1_from_2p', 0.0)])
            S_feed_1s = np.asarray([channel.get('sigma_Ka1_from_1s', 0.0), channel.get('sigma_Ka1_from_1s', 0.0)])

            # feed_from (docs/double-spectator-satellite-implementation-plan.md sec 3): parent
            # channels (by name, resolved to index here) this channel draws population from,
            # instead of a cross section. Empty for every pre-existing, cross-section-fed channel.
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
                    # L2k slice only exists when use_L2_satellite_pathway is on -- fail loudly
                    # instead of silently doing nothing.
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

            # Per-level detuning from the shared Kalpha1 frame (f=0 for Lk, f=Delta_fs for Uk),
            # same f-vector construction as the base block's Delta_ij (theory doc Eq. K4) -- reduces
            # to Delta_fs*sign_ij_block, so this is a no-op generalization when L2k is absent.
            f_local = Delta_fs * ei_K_sat

            # Further-ionization loss (theory doc sec 12.4, Eq. S8): one cross section per manifold,
            # applied uniformly across its msublevels. Optional, defaults to 0 (no loss).
            S_ion_Fi_chan = np.zeros((2, self.satellite_nlevel))
            S_ion_Fi_chan[:, ei_L3_sat.astype(bool)] = channel.get('sigma_ion_from_2p', 0.0)
            S_ion_Fi_chan[:, ei_K_sat.astype(bool)] = channel.get('sigma_ion_from_1s', 0.0)

            Gamma_A_L2_fs = None
            Gamma_A_K_to_L2_fs = None
            S_feed_2p1 = None
            if self.use_L2_satellite_pathway:
                is_double_satellite = bool(channel.get('feed_from'))
                # New keys from xatom_tools.l2_satellite_channel_parameters; fail loudly on a missing
                # key rather than silently producing a subtly wrong spectrum. Double-satellite channels
                # skip Gamma_A_2s_to_L2_eV/sigma_Ka1_from_2p1 -- their L2k manifold is fed via
                # manifold='L2' feed_from entries instead (resolved above).
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
                # L2k analogue of Gamma_A_K_fs above (1s hole filled by 2p1/2 instead of 2p3/2).
                Gamma_A_K_to_L2_fs = channel.get('Gamma_A_K_to_L2_eV', 0.0) / self.hbar
                GammaL2_fs = channel['Gamma_L2_eV'] / self.hbar
                S_feed_2p1 = np.asarray([channel.get('sigma_Ka1_from_2p1', 0.0), channel.get('sigma_Ka1_from_2p1', 0.0)])

                # f[L2k] = f[Uk] - Delta_L2_split: per-channel analogue of the base block's
                # f[L2]=f[K]-DeltaomegaL2mL3A, using this channel's own satellite splitting.
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

        # Budget check: Gamma_A_K_eV/Gamma_A_K_to_L2_eV redirect part of the K-hole's own already-
        # occurring non-radiative decay into the satellite ladder, so summed over every channel they
        # must not exceed that budget (theory doc sec 12.7/27). No-op when unset (defaults to 0).
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
            
