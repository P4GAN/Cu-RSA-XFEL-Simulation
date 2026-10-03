import types

import numpy as np
import scipy.constants as sp_const
import yaml

from . import pulses
from .optics import Optics
from .sample import Sample

# Config keys (docs/usage.md describes each). Unknown keys are rejected, so a typo or a key from an older
# version of the code can't be silently ignored.
REQUIRED_KEYS = {
    'n', 'zmax', 'hwKalpha1N', 'hwKalpha2N',
    'GammaKeVN', 'GammaL3eVN', 'GammaL2eVN', 'GammaL1eVN', 'GammarKalpha1eVN', 'GammarKalpha2eVN',
    'GammaA_L1_to_L3M45eVN',
    'sigma1_Ka1_2p3', 'sigma1_Ka1_2p1', 'sigma1_Ka1_2s', 'sigma1_Ka1_other',
    'sigma2_Ka1_1s', 'sigma2_Ka1_2p3', 'sigma2_Ka1_2p1', 'sigma2_Ka1_2s', 'sigma2_Ka1_other',
    'use_2s_pathway',
    'pulse', 'E_seed_uJ', 'seed_duration_FWHM_t', 'seed_width_FWHM_x', 'seed_width_FWHM_y', 't_peak',
    'tgrid', 'tmax', 'xgrid', 'xmax', 'ygrid', 'ymax', 'zgrid', 'xpad', 'ypad', 'tpad',
}
OPTIONAL_KEYS = {
    'GammaA_L1_to_L2eVN', 'satellite_channels', 'double_satellite_channels', 'use_middlemen', 'middlemen',
    'L2_CK_feed', 'use_eii', 'eii',
    'seed_center_E', 'seed_FEL_bandwidth', 'monochromator_target_energy_eV', 'seed_delay', 'random_seed',
    'keep_z_history',
}


class Simulation:
    """
    One simulation: the config, the grids, and every coupling tensor of the atomic model.

        sim = Simulation("config/example.yaml")
        sim.configure()   # the incident pulse named by the config's `pulse` key
        sim.run()

    Every key of the YAML config becomes an attribute (X.tgrid, X.hwKalpha1N, ...); the code calls a
    Simulation X throughout. Units are fs, nm and eV; rates are in fs^-1 and cross sections in nm^2.

    Level structure. The base block is one density matrix over the core-hole sublevels that the
    Kalpha field couples coherently:
        0-3  2p3/2 hole (L3), its four magnetic sublevels
        4-5  1s hole (K), two sublevels
        6-7  2p1/2 hole (L2), two sublevels
    The neutral ground state, the 2s hole, and "other" (every hole the model does not resolve) are
    incoherent scalar populations. Each satellite channel (an ion with spectator holes, e.g. 3d-1)
    is its own 8-level block with the same layout, shifted by the channel's detuning.
    """

    def __init__(self, config_path):
        self.hbar = sp_const.hbar / sp_const.e * 1.0e15   # eV fs
        self.c = sp_const.c / 1.0e6                       # nm / fs

        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        missing = sorted(REQUIRED_KEYS - set(self.config))
        unknown = sorted(set(self.config) - REQUIRED_KEYS - OPTIONAL_KEYS)
        if missing or unknown:
            raise ValueError(f"{config_path}: missing keys {missing}, unknown keys {unknown}")
        for key, value in self.config.items():
            setattr(self, key, value)

        self.keep_z_history = self.config.get('keep_z_history', True)
        self.use_middlemen = bool(self.config.get('use_middlemen', False))
        self.use_eii = bool(self.config.get('use_eii', False))
        self.satellite_channels = (list(self.config.get('satellite_channels') or [])
                                   + list(self.config.get('double_satellite_channels') or []))
        self.nlevel = 8

        self.lambdaKalpha1N = 2.0 * np.pi * self.c * self.hbar / self.hwKalpha1N
        self.k0 = 2.0 * np.pi / self.lambdaKalpha1N

        self.dz = self.zmax / (self.zgrid - 1.0)
        self.z = np.linspace(0, self.zmax, self.zgrid)
        self.t0 = self.tmax / 2.0

        self.optics = Optics(self)
        domains_txy, meshes_txy, step_sizes_txy = self.optics.nd_space(
            (0, -self.xmax, -self.ymax), (self.tmax, self.xmax, self.ymax), (self.tgrid, self.xgrid, self.ygrid))
        self.t_mesh, self.x_mesh, self.y_mesh = meshes_txy
        self.t = domains_txy[0]
        self.dt, self.dx, self.dy = step_sizes_txy

        self.GammaKfsm1N = self.GammaKeVN / self.hbar
        self.GammaL3fsm1N = self.GammaL3eVN / self.hbar
        self.GammaL2fsm1N = self.GammaL2eVN / self.hbar
        self.GammaL1fsm1N = self.GammaL1eVN / self.hbar
        self.GammarKalpha1fsm1N = self.GammarKalpha1eVN / self.hbar
        self.GammarKalpha2fsm1N = self.GammarKalpha2eVN / self.hbar
        self.GammaA_L1_to_L3M45fs1N = self.GammaA_L1_to_L3M45eVN / self.hbar
        self.Gamma_sp_fsm1N = self.GammarKalpha1fsm1N + self.GammarKalpha2fsm1N
        self.DeltaomegaL2mL3A = (self.hwKalpha1N - self.hwKalpha2N) / self.hbar

        seed = self.config.get('random_seed', -1)
        self.random_seed = None if seed == -1 else seed

        # |Omega|^2 = flux_factor * photon flux (photons nm^-2 fs^-1)
        self.flux_factor = 3.0 * self.lambdaKalpha1N ** 2 * self.Gamma_sp_fsm1N / 8.0 / np.pi
        self.field_source_factor = 1j * 3.0 * self.lambdaKalpha1N**2 * self.Gamma_sp_fsm1N * self.n / 16.0 / np.pi
        self.convert_SF_phnm2fs_Wcm2 = (self.hwKalpha1N * 1.602e-19 / (1e-9)**2 / 1e-15) / (1 / (1e-2)**2)

        # Field components: p = 0 is Omega, p = 1 its conjugate; s = 0, 1 the two circular polarisations.
        self.e_sign = np.asarray([1, -1])
        self.e_pol = np.asarray([1, 1])
        self.transform_matrix = np.asarray([[1, 1], [1j, -1j]]) / np.sqrt(2.0)  # circular -> Cartesian

        self._build_base_block()
        self._build_satellite_blocks()
        self._build_untracked_outflow()
        self._build_middlemen()
        if self.use_eii:
            self._build_eii()

    def _build_base_block(self):
        n = self.nlevel
        self.ei_L3 = np.zeros(n); self.ei_L3[0:4] = 1
        self.ei_K = np.zeros(n); self.ei_K[4:6] = 1
        self.ei_L2 = np.zeros(n); self.ei_L2[6:8] = 1

        # Dipole couplings Tijs[i, j, s] for polarisation s (Clebsch-Gordan coefficients). The relative
        # sign of the two 2p1/2 branches is physical: flipping it turns Kalpha2 absorption into emission.
        Tijs = np.zeros((n, n, 2), dtype=complex)
        Tijs[0, 4, 0] = Tijs[4, 0, 0] = 1.0 / np.sqrt(3)
        Tijs[1, 5, 0] = Tijs[5, 1, 0] = 1.0 / 3.0
        Tijs[2, 4, 1] = Tijs[4, 2, 1] = 1.0 / 3.0
        Tijs[3, 5, 1] = Tijs[5, 3, 1] = 1.0 / np.sqrt(3)
        Tijs[7, 4, 1] = Tijs[4, 7, 1] = -np.sqrt(2.0) / 3.0
        Tijs[6, 5, 0] = Tijs[5, 6, 0] = np.sqrt(2.0) / 3.0

        # Branching of spontaneous Kalpha decay from each 1s sublevel j into each 2p sublevel i.
        Gij = np.zeros((n, n), dtype=complex)
        Gij[0, 4] = 1.0 / 3.0
        Gij[1, 4] = 2.0 / 9.0
        Gij[1, 5] = 1.0 / 9.0
        Gij[2, 4] = 1.0 / 9.0
        Gij[2, 5] = 2.0 / 9.0
        Gij[3, 5] = 1.0 / 3.0
        Gij[7, 4] = 2.0 / 9.0
        Gij[6, 4] = 1.0 / 9.0
        Gij[6, 5] = 2.0 / 9.0
        Gij[7, 5] = 1.0 / 9.0

        # Photoionisation of the neutral atom, S_ground_Fi[f, i] for photon polarisation f: into each
        # base-block sublevel, then into the 2s hole (column n) and "other" (column n + 1).
        S_ground_Fi = np.zeros((2, n + 2))
        S_ground_Fi[0, 0:4] = self.sigma1_Ka1_2p3 * np.array([0.12, 0.18, 0.28, 0.42])
        S_ground_Fi[1, 0:4] = self.sigma1_Ka1_2p3 * np.array([0.42, 0.28, 0.18, 0.12])
        S_ground_Fi[:, 6:8] = self.sigma1_Ka1_2p1 * 0.5
        S_ground_Fi[:, n] = self.sigma1_Ka1_2s if self.use_2s_pathway else 0.0
        S_ground_Fi[:, n + 1] = self.sigma1_Ka1_other

        # Further photoionisation of an ion in each sublevel (a loss out of the block).
        S_ion_Fi = np.zeros((2, n))
        S_ion_Fi[0, 0:4] = self.sigma2_Ka1_2p3 * np.array([0.70, 0.83, 1.06, 1.41])
        S_ion_Fi[1, 0:4] = self.sigma2_Ka1_2p3 * np.array([1.41, 1.06, 0.83, 0.70])
        S_ion_Fi[0, 4:6] = self.sigma2_Ka1_1s * np.array([0.75, 1.25])
        S_ion_Fi[1, 4:6] = self.sigma2_Ka1_1s * np.array([1.25, 0.75])
        S_ion_Fi[:, 6:8] = self.sigma2_Ka1_2p1

        self.Tijs = Tijs
        self.Gij = Gij
        self.Gamma_sp_Gij = self.Gamma_sp_fsm1N * Gij
        self.S_ground_Fi = S_ground_Fi
        self.S_ion_Fi = S_ion_Fi
        self.S_other_F = np.full(2, self.sigma2_Ka1_other)
        self.S_2s_F = np.full(2, self.sigma2_Ka1_2s) if self.use_2s_pathway else np.zeros(2)

        # Tijs_plus couples a 1s (upper) row to a 2p (lower) column, Tijs_minus the reverse, so each
        # half of the interaction picks up the right field component (Model.MB_nlevel_regular).
        upper_lower = np.outer(self.ei_K, self.ei_L3 + self.ei_L2)
        self.Tijs_plus = np.einsum('ijs, ij->ijs', Tijs, upper_lower)
        self.Tijs_minus = np.einsum('ijs, ij->ijs', Tijs, upper_lower.T)

        self.Mij = self.decay_matrix(self.GammaL3fsm1N, self.GammaKfsm1N, self.GammaL2fsm1N)
        self.Delta_ij = self.detuning_matrix(0.0, self.DeltaomegaL2mL3A)

        # 2s-hole Auger decay into the 2p3/2 (L1-L3M45 Coster-Kronig) and, in the metal, the 2p1/2
        # manifold, spread evenly over the sublevels.
        if self.use_2s_pathway:
            GammaA_L1_to_L2_fs = self.config.get('GammaA_L1_to_L2eVN', 0.0) / self.hbar
            self.auger_feeding_matrix = (np.diag(self.ei_L3 / np.sum(self.ei_L3)) * self.GammaA_L1_to_L3M45fs1N
                                         + np.diag(self.ei_L2 / np.sum(self.ei_L2)) * GammaA_L1_to_L2_fs)
        else:
            self.auger_feeding_matrix = np.zeros((n, n))

    def decay_matrix(self, Gamma_L3_fs, Gamma_K_fs, Gamma_L2_fs):
        """Mij: each level's width on the diagonal, and on every coherence the mean of its two widths."""
        width = Gamma_L3_fs * self.ei_L3 + Gamma_K_fs * self.ei_K + Gamma_L2_fs * self.ei_L2
        return 0.5 * (width[:, None] + width[None, :])

    def detuning_matrix(self, detuning_fs, L2_split_fs):
        """Delta_ij = f_i - f_j in the frame rotating at the Kalpha1 frequency, with each level at
        f = minus its energy offset: 0 for 2p3/2, -detuning for 1s, -L2_split for 2p1/2. A line between
        levels i and j then sits at -(f_upper - f_lower) from Kalpha1."""
        f = -detuning_fs * self.ei_K - L2_split_fs * self.ei_L2
        return f[:, None] - f[None, :]

    def _build_satellite_blocks(self):
        """
        One namespace per satellite channel (satellite_channels, then double_satellite_channels). A
        single-spectator channel is fed by 2s-hole Auger decay (Gamma_A_2s*), by the K-hole's own
        non-radiative decay (Gamma_A_K*), and by photoionisation of the spectator shell of base-block
        ions (sigma_Ka1_from_*). A double-spectator channel is fed only by a parent channel's decay
        (feed_from).
        """
        index = {}
        for k, channel in enumerate(self.satellite_channels):
            index.setdefault(channel['name'], []).append(k)

        self.satellite_channel_params = []
        for channel in self.satellite_channels:
            name = channel['name']
            is_double = bool(channel.get('feed_from'))
            required = ('detuning_eV_L2_split', 'Gamma_L2_eV') if is_double else (
                'detuning_eV_L2_split', 'Gamma_A_2s_to_L2_eV', 'Gamma_L2_eV', 'sigma_Ka1_from_2p1')
            missing = [key for key in required if key not in channel]
            if missing:
                raise ValueError(f"satellite channel {name!r} is missing {missing}")

            feed_from = []
            for feed in channel.get('feed_from', []):
                matches = index.get(feed['channel'], [])
                if len(matches) != 1:
                    raise ValueError(f"satellite channel {name!r} feeds from {feed['channel']!r}, which "
                                     f"matches {len(matches)} channels (must be exactly 1)")
                manifold = feed.get('manifold', 'lower')
                if manifold not in ('lower', 'upper', 'L2'):
                    raise ValueError(f"feed_from manifold must be 'lower', 'upper' or 'L2', got {manifold!r}")
                feed_from.append((matches[0], feed['Gamma_feed_eV'] / self.hbar, manifold))

            Gamma_L_fs = channel.get('Gamma_L_eV', self.GammaL3eVN) / self.hbar
            Gamma_K_fs = channel.get('Gamma_K_eV', self.GammaKeVN) / self.hbar
            Gamma_L2_fs = channel['Gamma_L2_eV'] / self.hbar

            S_ion_Fi = np.zeros((2, self.nlevel))
            S_ion_Fi[:, self.ei_L3.astype(bool)] = channel.get('sigma_ion_from_2p', 0.0)
            S_ion_Fi[:, self.ei_K.astype(bool)] = channel.get('sigma_ion_from_1s', 0.0)
            S_ion_Fi[:, self.ei_L2.astype(bool)] = channel.get('sigma_ion_from_2p1', 0.0)

            def per_polarisation(key):
                return np.full(2, channel.get(key, 0.0))

            self.satellite_channel_params.append(types.SimpleNamespace(
                name=name,
                Mij=self.decay_matrix(Gamma_L_fs, Gamma_K_fs, Gamma_L2_fs),
                Delta_ij=self.detuning_matrix(channel['detuning_eV'] / self.hbar,
                                              channel['detuning_eV_L2_split'] / self.hbar),
                S_ion_Fi=S_ion_Fi,
                Gamma_A_fs=channel.get('Gamma_A_2s_eV', 0.0) / self.hbar,
                Gamma_A_L2_fs=channel.get('Gamma_A_2s_to_L2_eV', 0.0) / self.hbar,
                Gamma_A_K_fs=channel.get('Gamma_A_K_eV', 0.0) / self.hbar,
                Gamma_A_K_to_L2_fs=channel.get('Gamma_A_K_to_L2_eV', 0.0) / self.hbar,
                S_feed_2p=per_polarisation('sigma_Ka1_from_2p'),
                S_feed_1s=per_polarisation('sigma_Ka1_from_1s'),
                S_feed_2p1=per_polarisation('sigma_Ka1_from_2p1'),
                feed_from=feed_from,
                Gamma_CK_L2_fs=0.0,
                mid_share=0.0,
            ))

        # The KLM-type feeds redirect part of the K hole's own non-radiative decay into the satellites.
        Gamma_K_nonradiative_eV = self.GammaKeVN - self.GammarKalpha1eVN - self.GammarKalpha2eVN
        Gamma_A_K_total_eV = sum(channel.get('Gamma_A_K_eV', 0.0) + channel.get('Gamma_A_K_to_L2_eV', 0.0)
                                 for channel in self.satellite_channels)
        if Gamma_A_K_total_eV > Gamma_K_nonradiative_eV:
            raise ValueError(f"the satellite channels' Gamma_A_K_eV + Gamma_A_K_to_L2_eV sum to {Gamma_A_K_total_eV:.4f} "
                             f"eV, more than the K hole's non-radiative width {Gamma_K_nonradiative_eV:.4f} eV")

        # A double-satellite feed only adds to the daughter (Model.feed_diag_satellite_block), so it must
        # be a branch of the parent's own width, never more than it.
        manifold_mask = {'lower': self.ei_L3, 'upper': self.ei_K, 'L2': self.ei_L2}
        self.feed_out = {}
        for chan in self.satellite_channel_params:
            for parent_index, Gamma_feed_fs, manifold in chan.feed_from:
                key = (parent_index, manifold)
                self.feed_out[key] = self.feed_out.get(key, 0.0) + Gamma_feed_fs
        for (parent_index, manifold), total_fs in self.feed_out.items():
            parent = self.satellite_channel_params[parent_index]
            level = int(np.argmax(manifold_mask[manifold]))
            width_fs = float(np.real(parent.Mij[level, level]))
            if total_fs > width_fs * (1.0 + 1e-9):
                raise ValueError(f"double_satellite_channels feed {total_fs * self.hbar:.4f} eV out of "
                                 f"{parent.name!r}'s {manifold!r} manifold, which only decays at "
                                 f"{width_fs * self.hbar:.4f} eV")

        # Metal-only L2-L3M45 Coster-Kronig: a branch of the base 2p1/2 width into the 3d satellites.
        self.Gamma_CK_L2_total_fs = 0.0
        l2_ck = self.config.get('L2_CK_feed')
        if l2_ck:
            check_keys(l2_ck, 'L2_CK_feed', {'rate_eV', 'targets'})
            rate_fs = l2_ck['rate_eV'] / self.hbar
            weights = l2_ck['targets']
            if rate_fs * sum(weights.values()) > self.GammaL2fsm1N * (1.0 + 1e-9):
                raise ValueError(f"L2_CK_feed rate {l2_ck['rate_eV']} eV exceeds GammaL2eVN")
            for name, weight in weights.items():
                self.satellite_channel(name, 'L2_CK_feed').Gamma_CK_L2_fs += rate_fs * weight
            self.Gamma_CK_L2_total_fs = rate_fs * sum(weights.values())

    def satellite_channel(self, name, what):
        matches = [chan for chan in self.satellite_channel_params if chan.name == name]
        if len(matches) != 1:
            names = [chan.name for chan in self.satellite_channel_params]
            raise ValueError(f"{what} references channel {name!r}, which matches {len(matches)} satellite "
                             f"channels (must be exactly 1; available: {names})")
        return matches[0]

    def _build_untracked_outflow(self):
        """
        The part of each tracked population's decay and further photoionisation that has no tracked
        destination. Without middlemen these atoms leave the model; with them they join the middleman
        pool. The electron ladder counts the electrons these processes emit.
        """
        n_base = 6
        params = self.satellite_channel_params
        manifold_mask = {'lower': self.ei_L3, 'upper': self.ei_K, 'L2': self.ei_L2}

        radiative_return = np.real(np.sum(self.Gamma_sp_Gij, axis=0))
        decay_base = np.real(np.diag(self.Mij)) - radiative_return
        klm_fs = sum(chan.Gamma_A_K_fs + chan.Gamma_A_K_to_L2_fs for chan in params)
        self.decay_untracked_base = decay_base - klm_fs * self.ei_K - self.Gamma_CK_L2_total_fs * self.ei_L2

        # A photoionisation feed out of a base level is a branch of that level's further ionisation, so
        # it carries the same sublevel pattern, normalised to a manifold mean of 1.
        S_ion = np.array(np.real(self.S_ion_Fi), dtype=float)
        self.pi_feed_pattern_Fi = np.ones_like(S_ion)
        for mask in (self.ei_L3, self.ei_K, self.ei_L2):
            m = mask.astype(bool)
            mean = S_ion[:, m].mean(axis=1, keepdims=True)
            self.pi_feed_pattern_Fi[:, m] = np.where(mean > 0, S_ion[:, m] / np.where(mean > 0, mean, 1.0), 1.0)
        self.S_untracked_base = S_ion.copy()
        for chan in params:
            feed_F = np.outer(chan.S_feed_2p, self.ei_L3[:n_base]) + np.outer(chan.S_feed_1s, self.ei_K[:n_base])
            self.S_untracked_base[:, :n_base] -= feed_F * self.pi_feed_pattern_Fi[:, :n_base]
            self.S_untracked_base[:, n_base:] -= chan.S_feed_2p1[:, None] * self.pi_feed_pattern_Fi[:, n_base:]

        # Electrons emitted per unit population and time, by source: non-radiative 1s decay (KLL/KLM,
        # ~7 keV), untracked L-shell Auger (~0.9 keV), and tracked Coster-Kronig decay (< 0.1 keV).
        self.e_emit_base = {
            'KLL': (np.real(np.diag(self.Mij)) - self.Gamma_sp_fsm1N) * self.ei_K,
            'LMM': self.decay_untracked_base * (self.ei_L3 + self.ei_L2),
            'CK': self.Gamma_CK_L2_total_fs * self.ei_L2,
        }
        for k, chan in enumerate(params):
            decay_sat = np.real(np.diag(chan.Mij)) - radiative_return
            fed = sum(self.feed_out.get((k, m), 0.0) * manifold_mask[m] for m in manifold_mask)
            chan.decay_untracked = decay_sat - fed
            chan.S_untracked = np.array(np.real(chan.S_ion_Fi), dtype=float)
            chan.e_emit = {
                'KLL': (np.real(np.diag(chan.Mij)) - self.Gamma_sp_fsm1N - fed) * self.ei_K,
                'LMM': chan.decay_untracked * (self.ei_L3 + self.ei_L2),
                'CK': fed,
            }
        self.twos_decay_tracked_fs = (float(np.sum(np.real(np.diag(self.auger_feeding_matrix))))
                                      + sum(chan.Gamma_A_fs + chan.Gamma_A_L2_fs for chan in params))
        self.twos_decay_untracked_fs = self.GammaL1fsm1N - self.twos_decay_tracked_fs

        for label, arr in [('base block', self.decay_untracked_base), ('base block PI', self.S_untracked_base),
                           ('2s', np.array([self.twos_decay_untracked_fs]))] + \
                          [(f'satellite {chan.name!r}', chan.decay_untracked) for chan in params]:
            if np.min(arr) < -1e-9 * max(1.0, float(np.max(np.abs(arr)))):
                raise ValueError(f"tracked feeds out of the {label} exceed its own decay/ionisation "
                                 f"(untracked outflow {np.min(arr):.4g} < 0): population would be created")

    def _build_middlemen(self):
        """
        The middleman pool: ions with valence (3d) holes only. They absorb like neutral atoms, and when
        photoionised in the 2p or 2s shell they become satellite ions, split over middlemen.targets.
        """
        mid_cfg = self.config.get('middlemen') or {}
        check_keys(mid_cfg, 'middlemen', {'targets'})
        params = self.satellite_channel_params
        self.sigma_mid_F = np.real(np.sum(self.S_ground_Fi, axis=1)) * np.ones(2)
        self.mid_ck_L3 = self.mid_ck_L2 = 0.0
        self.mid_S_out = np.zeros(2)
        if not self.use_middlemen:
            return
        targets = mid_cfg.get('targets')
        if targets is None:
            raise ValueError("use_middlemen needs middlemen: {targets: ...}, either 'none' or "
                             "{satellite channel name: weight}")
        if targets == 'none':
            return
        for name, weight in targets.items():
            self.satellite_channel(name, 'middlemen.targets').mid_share += float(weight)
        # A 2s hole on a middleman Coster-Kronig decays within 0.1 fs, with the same branching as the
        # tracked 2s -> single-satellite decay.
        self.mid_ck_L3 = sum(chan.Gamma_A_fs for chan in params) / self.GammaL1fsm1N
        self.mid_ck_L2 = sum(chan.Gamma_A_L2_fs for chan in params) / self.GammaL1fsm1N
        share = sum(chan.mid_share for chan in params)
        sigma_2p3 = np.real(self.S_ground_Fi[:, :4]).sum(axis=1)
        sigma_2s = np.real(self.S_ground_Fi[:, self.nlevel])
        routed = sigma_2p3 + sigma_2s * self.mid_ck_L3
        routed = routed + np.real(self.S_ground_Fi[:, 6:self.nlevel]).sum(axis=1) + sigma_2s * self.mid_ck_L2
        self.mid_S_out = routed * share

    def _build_eii(self):
        """
        Free-electron slowing-down ladder (eii.py): energy groups the electrons cascade down, and the
        electron-impact ionisation rate of each target subshell per electron per atom in each group.
        """
        from . import eii
        e_cfg = self.config.get('eii') or {}
        check_keys(e_cfg, 'eii', {'n_groups', 'E_top_eV', 'E_bottom_eV', 'spatial_factor', 'M_shell_scale'})
        self.eii_ladder = eii.build_ladder(self.n, n_groups=int(e_cfg.get('n_groups', 24)),
                                           E_top_eV=float(e_cfg.get('E_top_eV', 7950.0)),
                                           E_bottom_eV=float(e_cfg.get('E_bottom_eV', 16.5)))
        # plus one bin below E_bottom for electrons too slow to ionise anything
        self.eii_G = len(self.eii_ladder['E_centres']) + 1
        self.eii_k_down = np.append(self.eii_ladder['k_down_fs'], 0.0)
        # spatial_factor: the share of each electron's ionisations that happen inside the focus
        spatial = float(e_cfg.get('spatial_factor', 0.5))
        M_scale = float(e_cfg.get('M_shell_scale', 0.0))

        # rows: EII into 2p3/2, 2p1/2, 2s and the M shell (3s + 3p + 3d); the last column is the bin
        R = self.eii_ladder['rates_fs']
        table = np.zeros((4, self.eii_G))
        table[0, :-1] = spatial * R['2p3/2']
        table[1, :-1] = spatial * R['2p1/2']
        table[2, :-1] = spatial * R['2s'] * float(self.use_2s_pathway)
        table[3, :-1] = spatial * M_scale * (R['3s'] + R['3p'] + R['3d'])
        self.eii_rate_table = table
        self.eii_birth = dict(self.eii_ladder['birth_group'])
        # (G, G - 1): secondary electrons set into each group by one electron per atom in each group
        self.eii_secondary_matrix = spatial * eii.secondary_matrix(self.eii_ladder)

    def configure(self, seed_field=None):
        """Set the incident field (p, s, t, x, y), in the linear polarisation basis (pulses.py); by default
        the pulse the config names."""
        if seed_field is None:
            seed_field = pulses.make_pulse(self)
        self.sample = Sample(self, seed_field)

    def run(self):
        """
        March the field and the atoms through the sample, then copy the results onto self. With
        keep_z_history (default True) every array keeps all (t, x, y, z) points, which plot.py needs; set
        it False for sweeps, which keeps only what analysis.compute_run_outputs reads. A recorder
        (scripts/movie.py, scripts/population_budget.py) set as self.recorder is handed every step of the
        lean path.
        """
        self.sample.run(self)
        self.rho_ground_txyz = self.sample.rho_ground_txyz
        self.rho_other_txyz = self.sample.rho_other_txyz
        self.rho_2s_txyz = self.sample.rho_2s_txyz
        self.rho_ijtxyz = self.sample.rho_ijtxyz
        self.rho_sat_ijtxyz = self.sample.rho_sat_ijtxyz
        self.Omega_pstxyz = self.sample.Omega_pstxyz
        self.rho_mid_txyz = self.sample.rho_mid_txyz   # None without use_middlemen
        self.rho_e_gtxyz = self.sample.rho_e_gtxyz     # None without use_eii; last group is the bin

        Omega_pqtxy = pulses.circular_to_linear(self, self.Omega_pstxyz[:, :, :, :, :, -1])
        self.Omega_qtxy = (Omega_pqtxy[0, :, :, :, :] + np.conj(Omega_pqtxy[1, :, :, :, :])) / 2.0


def check_keys(cfg, block, allowed):
    """The config sub-blocks reject unknown keys, so a typo can't silently fall back to a default."""
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        raise ValueError(f"{block}: unknown key(s) {unknown} (allowed: {sorted(allowed)})")
