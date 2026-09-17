"""
Where the atoms and electrons are during a pulse: every tracked population, at every z plane, vs time,
for sweep-sized runs (scripts/run_population_budget.py).

The sweep outputs (tools.compute_run_outputs) keep the exit plane's centre pixel only, the satellite
blocks only as Tijs-weighted dipole contractions (not populations) and the free electrons only as two
totals. The movie recorder (movie.py) keeps everything at every (t, x, y, z) point for one shot, far
too much to repeat for a sweep. This recorder sits in between. It rides on
Sample._evaluate_n_level_3D_lean like the movie recorder, so the numerics are unchanged. Per z plane,
every `stride` time steps, it keeps two reductions of each population:

  centre   the centre pixel (the pixel compute_run_outputs and run_population_record.py use)
  beam     the average over (x, y) weighted by the incident fluence of each pixel, i.e. the
           population seen by the average incident photon, which is what a transmission measures

Channels (the `names` array; every entry is a population per atom, except flux):
  flux                      photons nm^-2 fs^-1 of the field that drove this plane
  ground, other, 2s, middlemen
  base/L3, base/K, base/L2  base block diagonal summed per manifold (2p3/2 hole, 1s hole, 2p1/2 hole)
  sat/<name>/L3, /K, /L2    each satellite block summed per manifold, in satellite_channel_params order
  electrons/<g>             free-electron ladder group g (use_eii); the last group is the thermalised bin
Unweighted diagonal sums, so ground + other + 2s + middlemen + base/* + sat/*/* = 1 to RK4 accuracy
when use_middlemen is on.

Plane iz is driven by the field after max(iz - 1, 0) absorption steps (the solver's readout convention,
see Sample._evaluate_n_level_3D_lean), so its depth is max(iz - 1, 0) * dz. Populations at time index it
are the state after the RK4 step that starts at t[it].
"""

import numpy as np


def channel_names(X):
    names = ['flux', 'ground', 'other', '2s', 'middlemen', 'base/L3', 'base/K', 'base/L2']
    for chan in X.satellite_channel_params:
        names += [f'sat/{chan.name}/L3', f'sat/{chan.name}/K', f'sat/{chan.name}/L2']
    if X.use_eii:
        names += [f'electrons/{g}' for g in range(X.eii_G)]
    return names


def incident_fluence_weights(X, seed_field_pstxy):
    """Normalised incident fluence per (x, y) pixel, from the seed field. sum_s Re(Omega[0,s] Omega[1,s])
    is invariant under the unitary linear -> circular transform that configure() applies."""
    F_xy = np.real(np.einsum('stxy,stxy->xy', seed_field_pstxy[0], seed_field_pstxy[1]))
    return F_xy / F_xy.sum()


class PopulationBudgetRecorder:
    """
    Attach as X.movie_recorder before X.run_3D() with X.keep_z_history = False.
    After the run, `centre` and `beam` hold (zgrid, n_channels, n_samples) float64 arrays and `t` the
    sampled times.
    """

    def __init__(self, X, weights_xy, stride=1):
        self.X = X
        self.w = np.asarray(weights_xy, dtype=float)
        self.stride = max(1, int(stride))
        self.cx, self.cy = int(X.xgrid / 2), int(X.ygrid / 2)  # same centre pixel as the lean path
        self.names = channel_names(X)
        self.index = {name: i for i, name in enumerate(self.names)}
        self.sample_it = np.arange(0, X.tgrid, self.stride)
        self.t = np.asarray(X.t, dtype=float)[self.sample_it]
        shape = (X.zgrid, len(self.names), len(self.sample_it))
        self.centre = np.zeros(shape)
        self.beam = np.zeros(shape)
        self.base_manifolds = np.stack([X.ei_L3, X.ei_K, X.ei_L2]).astype(float)
        self.sat_manifolds = np.stack([X.ei_L3_satellite, X.ei_K_satellite, X.ei_L2_satellite]).astype(float)
        self.diag = np.arange(X.nlevel)
        self.sat_diag = np.arange(X.satellite_nlevel)
        self.i_sat0 = self.index['base/L2'] + 1
        self.i_e0 = self.index.get('electrons/0')
        self.iz = 0

    def _put(self, i, s, q_xy):
        """Store channel i (or the block of channels starting at i, if q has leading axes) at sample s."""
        q = np.real(q_xy)
        self.centre[self.iz, i:i + (q.shape[0] if q.ndim == 3 else 1), s] = q[..., self.cx, self.cy]
        self.beam[self.iz, i:i + (q.shape[0] if q.ndim == 3 else 1), s] = np.tensordot(q, self.w, axes=2)

    def record_step(self, it, Omega_it, rho_ijxy, rho_sat_ijxy, rho_ground_xy, rho_other_xy, rho_2s_xy,
                    rho_mid_xy, rho_e_gxy):
        if it % self.stride:
            return
        s = it // self.stride
        X = self.X
        flux_xy = np.real(Omega_it[0, 0] * Omega_it[1, 0] + Omega_it[0, 1] * Omega_it[1, 1]) / X.flux_factor
        self._put(0, s, flux_xy)
        self._put(1, s, rho_ground_xy)
        self._put(2, s, rho_other_xy)
        self._put(3, s, rho_2s_xy)
        if rho_mid_xy is not None:
            self._put(4, s, rho_mid_xy)
        self._put(5, s, np.tensordot(self.base_manifolds, np.real(rho_ijxy[self.diag, self.diag]), axes=1))
        for k, rho_sat in enumerate(rho_sat_ijxy):
            self._put(self.i_sat0 + 3 * k, s,
                      np.tensordot(self.sat_manifolds, np.real(rho_sat[self.sat_diag, self.sat_diag]), axes=1))
        if rho_e_gxy is not None:
            self._put(self.i_e0, s, rho_e_gxy)

    def end_plane(self, iz, Omega_pstxy):
        self.iz = iz + 1

    def trace(self, view='centre'):
        """Total tracked population per (z, sample); 1 to RK4 accuracy with use_middlemen."""
        a = getattr(self, view)
        pops = [i for name, i in self.index.items() if name != 'flux' and not name.startswith('electrons/')]
        return a[:, pops, :].sum(axis=1)
