"""
Full (t, x, y, z) history of one run, streamed to HDF5 one z-plane at a time, for movies and 3D
renders (scripts/run_movie.py).

_evaluate_n_level_3D_full keeps every z-plane of every complex density matrix in memory (~8 kB per
grid point with L2 + 7 satellite blocks), which rules out grids fine enough to look good. The
recorder instead rides on _evaluate_n_level_3D_lean (identical physics) and, per (t, x, y, z)
point, keeps only what a picture needs in single precision (~240 B): the driving field, level
populations by manifold, the base-block Kalpha coherences and the free-electron ladder. The full
density matrices are kept at the centre pixel only. Each finished z-plane is written and flushed,
so a run killed partway still leaves every completed plane readable.

Layout (every (t, x, y) array is indexed [z, ..., t, x, y]):
  t, x, y, z                      axes (fs, nm)
  field           complex64 (z, 2, 2, t, x, y)  Omega_pstxy driving plane z (Omega_pstxyz[..., iz]
                                               of the full-history path; index 0 is the seed)
  flux            float32   (z, t, x, y)        photon flux of that field, photons nm^-2 fs^-1
  field_exit      complex64 (2, 2, t, x, y)     field leaving the last plane (not kept by full/lean)
  pop/ground, pop/other, pop/2s, pop/middlemen  float32 (z, t, x, y)
  pop/base        float32   (z, nlevel, t, x, y)        base-block diagonal, per sublevel
  pop/satellite   float32   (z, n_sat, 3, t, x, y)      each satellite block summed per manifold
  electrons       float32   (z, eii_G, t, x, y)         free-electron ladder (use_eii only)
  coherence/base  complex64 (z, n_pairs, t, x, y)       rho[i, j] for every Kalpha-coupled pair
  center/rho_base       complex64 (z, nlevel, nlevel, t)            centre pixel, full matrix
  center/rho_satellite  complex64 (z, n_sat, n_sat_level, n_sat_level, t)

Populations at index it are the state after the RK4 step starting at t[it] (the history
convention of both z-marching paths); the field at index it is the one that drove that step.
"""

import time

import h5py
import numpy as np


def _plan(X):
    """(name, leading shape, dtype) of every per-plane dataset, in (t, x, y) and centre-pixel groups."""
    n_sat = len(X.satellite_channel_params)
    n_pairs = int(np.count_nonzero(np.abs(X.Tijs_plus).sum(axis=2)))
    txy = [('field', (2, 2), np.complex64),
           ('flux', (), np.float32),
           ('pop/ground', (), np.float32),
           ('pop/other', (), np.float32),
           ('pop/2s', (), np.float32),
           ('pop/base', (X.nlevel,), np.float32),
           ('coherence/base', (n_pairs,), np.complex64)]
    if n_sat:
        txy.append(('pop/satellite', (n_sat, 3), np.float32))
    if X.use_middlemen:
        txy.append(('pop/middlemen', (), np.float32))
    if X.use_eii:
        txy.append(('electrons', (X.eii_G,), np.float32))
    center = [('center/rho_base', (X.nlevel, X.nlevel), np.complex64)]
    if n_sat:
        center.append(('center/rho_satellite', (n_sat, X.satellite_nlevel, X.satellite_nlevel), np.complex64))
    return txy, center


def estimate_size_gb(X):
    """Uncompressed size of the file MovieRecorder(X, ...) writes, in GB (1e9 bytes)."""
    txy, center = _plan(X)
    n_txy = X.zgrid * X.tgrid * X.xgrid * X.ygrid
    size = sum(np.prod(lead, dtype=int) * np.dtype(dt).itemsize * n_txy for _, lead, dt in txy)
    size += sum(np.prod(lead, dtype=int) * np.dtype(dt).itemsize * X.zgrid * X.tgrid for _, lead, dt in center)
    size += 4 * X.tgrid * X.xgrid * X.ygrid * np.dtype(np.complex64).itemsize  # field_exit
    return size / 1e9


class MovieRecorder:
    """
    Attach as X.movie_recorder before X.run_3D() with X.keep_z_history = False.
    Sample._evaluate_n_level_3D_lean then calls record_step once per (it, iz) and end_plane once per
    iz; neither touches the simulation state.
    """

    def __init__(self, X, path, attrs=None, compression='gzip', compression_opts=1):
        self.X = X
        self.path = path
        nt, nx, ny, nz = X.tgrid, X.xgrid, X.ygrid, X.zgrid
        self.cx, self.cy = int(nx / 2), int(ny / 2)  # same centre pixel as the lean path
        self.diag = np.arange(X.nlevel)
        self.sat_diag = np.arange(X.satellite_nlevel)
        self.n_sat = len(X.satellite_channel_params)
        # rows: 2p3/2 (L3), 1s (K), 2p1/2 (L2) manifold of each satellite block's own local levels
        self.sat_manifolds = np.stack([X.ei_L3_satellite, X.ei_K_satellite, X.ei_L2_satellite]).astype(float)
        # (i, j) with i the upper (K) and j the lower (2p-hole) level of a Kalpha-coupled pair
        self.coh_pairs = np.argwhere(np.abs(X.Tijs_plus).sum(axis=2) > 0)

        self.f = h5py.File(path, 'w')
        for name, axis in (('t', X.t), ('x', np.linspace(-X.xmax, X.xmax, nx)),
                           ('y', np.linspace(-X.ymax, X.ymax, ny)), ('z', X.z)):
            self.f.create_dataset(name, data=np.asarray(axis, dtype=float))

        txy, center = _plan(X)
        self.buf, self.dsets = {}, {}
        for name, lead, dtype in txy:
            self.buf[name] = np.zeros(lead + (nt, nx, ny), dtype=dtype)
            self.dsets[name] = self.f.create_dataset(
                name, shape=(nz,) + lead + (nt, nx, ny), dtype=dtype,
                chunks=(1,) * (1 + len(lead)) + (nt, nx, ny),  # one (t, x, y) cube per chunk
                compression=compression, compression_opts=compression_opts, shuffle=True)
        for name, lead, dtype in center:
            self.buf[name] = np.zeros(lead + (nt,), dtype=dtype)
            self.dsets[name] = self.f.create_dataset(name, shape=(nz,) + lead + (nt,), dtype=dtype,
                                                     chunks=(1,) + lead + (nt,))

        self.dsets['pop/base'].attrs['levels'] = [
            ('L3' if X.ei_L3[i] else 'K' if X.ei_K[i] else 'L2') + f'[{i}]' for i in range(X.nlevel)]
        self.dsets['coherence/base'].attrs['pairs_upper_lower'] = self.coh_pairs
        if self.n_sat:
            self.dsets['pop/satellite'].attrs['channels'] = [chan.name for chan in X.satellite_channel_params]
            self.dsets['pop/satellite'].attrs['manifolds'] = ['2p3/2 (L3)', '1s (K)', '2p1/2 (L2)']
        if X.use_eii:
            self.dsets['electrons'].attrs['E_edges_eV'] = X.eii_ladder['E_edges']
            self.dsets['electrons'].attrs['note'] = 'electrons per atom; last group = thermalised bin'
        self.dsets['flux'].attrs['note'] = 'sum_s Re(Omega[0,s] Omega[1,s]) / flux_factor'

        self.f.attrs['flux_factor'] = X.flux_factor
        self.f.attrs['center_pixel'] = (self.cx, self.cy)
        self.f.attrs['units'] = 't in fs, x/y/z in nm, field in fs^-1 (Rabi), flux in photons nm^-2 fs^-1'
        for key, value in (attrs or {}).items():
            self.f.attrs[key] = value
        self.f.flush()
        self._t_plane = time.perf_counter()

    def record_step(self, it, Omega_it, rho_ijxy, rho_sat_ijxy, rho_ground_xy, rho_other_xy, rho_2s_xy,
                    rho_mid_xy, rho_e_gxy):
        b = self.buf
        b['field'][:, :, it] = Omega_it
        b['flux'][it] = np.real(Omega_it[0, 0] * Omega_it[1, 0] + Omega_it[0, 1] * Omega_it[1, 1]) / self.X.flux_factor
        b['pop/ground'][it] = np.real(rho_ground_xy)
        b['pop/other'][it] = np.real(rho_other_xy)
        b['pop/2s'][it] = np.real(rho_2s_xy)
        b['pop/base'][:, it] = np.real(rho_ijxy[self.diag, self.diag])
        b['coherence/base'][:, it] = rho_ijxy[self.coh_pairs[:, 0], self.coh_pairs[:, 1]]
        b['center/rho_base'][:, :, it] = rho_ijxy[:, :, self.cx, self.cy]
        for k in range(self.n_sat):
            b['pop/satellite'][k, :, it] = np.tensordot(
                self.sat_manifolds, np.real(rho_sat_ijxy[k][self.sat_diag, self.sat_diag]), axes=1)
            b['center/rho_satellite'][k, :, :, it] = rho_sat_ijxy[k][:, :, self.cx, self.cy]
        if rho_mid_xy is not None:
            b['pop/middlemen'][it] = np.real(rho_mid_xy)
        if rho_e_gxy is not None:
            b['electrons'][:, it] = rho_e_gxy

    def end_plane(self, iz, Omega_pstxy):
        for name, buf in self.buf.items():
            self.dsets[name][iz] = buf
        if iz == self.X.zgrid - 1:
            self.f.create_dataset('field_exit', data=Omega_pstxy.astype(np.complex64))
        self.f.attrs['planes_written'] = iz + 1
        self.f.flush()
        now = time.perf_counter()
        print(f"z plane {iz + 1}/{self.X.zgrid} written ({now - self._t_plane:.1f} s)", flush=True)
        self._t_plane = now

    def close(self):
        if self.f.id.valid:
            self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
