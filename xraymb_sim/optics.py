import numpy as np
import scipy.fft as sp_fft


class Optics:
    """Transverse grids and FFT helpers, and the split-step Fresnel propagator between z planes."""

    def __init__(self, X):
        domains_xy, meshes_xy, step_sizes_xy = self.nd_space((-X.xmax, -X.ymax), (X.xmax, X.ymax), (X.xgrid, X.ygrid))
        self.xx, self.yy = meshes_xy
        self.dx, self.dy = step_sizes_xy
        self.xgrid, self.ygrid = X.xgrid, X.ygrid
        self.xpad, self.ypad = X.xpad, X.ypad

        domains_kxky, meshes_kxky, step_sizes_kxky = self.nd_kspace(
            (1.0, 1.0), (self.xgrid, self.ygrid), (self.xpad, self.ypad), (self.dx, self.dy))
        self.kx, self.ky = meshes_kxky
        self._drift_kernel = None

    def nd_space(self, mins=(), maxs=(), sizes=()):
        domains = [np.linspace(lo, hi, n) for lo, hi, n in zip(mins, maxs, sizes)]
        meshes = np.meshgrid(*domains, indexing='ij')
        step_sizes = [domain[1] - domain[0] for domain in domains]
        return domains, meshes, step_sizes

    def nd_kspace(self, coeffs=(), sizes=(), pads=(), steps=()):
        """FFT frequency grids (times coeff) of axes of the given sizes, each zero-padded by pad on both sides."""
        domains = [coeff * np.fft.fftfreq(n + 2 * pad, step) for coeff, n, pad, step in zip(coeffs, sizes, pads, steps)]
        meshes = np.meshgrid(*domains, indexing='ij')
        step_sizes = [domain[1] - domain[0] for domain in domains]
        return domains, meshes, step_sizes

    def fft2(self, wavefront):
        """2D FFT over the last two axes (one worker: the sweeps already run one process per core)."""
        return sp_fft.fft2(wavefront, workers=1, overwrite_x=True)

    def ifft2(self, wavefront):
        return sp_fft.ifft2(wavefront, workers=1, overwrite_x=True)

    def fft_phased(self, array, axes, phasors):
        """Orthonormal FFT over `axes`, multiplied by the phase factors that move the origin of each axis
        to the centre of the simulation window, then fftshifted."""
        array_fft = np.fft.fftn(array, axes=axes, norm='ortho')
        for phasor in phasors:
            array_fft *= phasor
        return np.fft.fftshift(array_fft, axes=axes)

    def pad(self, wavefront, shape):
        return np.pad(wavefront, shape, mode='constant', constant_values=(0.0 + 1j * 0.0, 0.0 + 1j * 0.0))

    def drift_kernel_tensor(self, X):
        """Paraxial free-space propagator over one z step, exp(-i pi lambda dz (kx^2 + ky^2)), for the field
        (p = 0) and its conjugate (p = 1). The same for every step, so it is built once."""
        if self._drift_kernel is None:
            kernel = np.exp(-1j * X.dz * np.pi * X.lambdaKalpha1N * (self.kx**2 + self.ky**2))
            kernel_plus = np.einsum('s, xy->sxy', X.e_pol, kernel)
            self._drift_kernel = np.asarray([kernel_plus, np.conj(kernel_plus)])
        return self._drift_kernel

    def propagate(self, X, wavefront, kappa):
        """
        Propagate the field (p, s, x, y) over one z step. Split step: half the absorption with the
        absorption coefficient of the plane the step starts from, free-space diffraction on a
        zero-padded grid, then the other half with that of the plane it ends on (kappa[..., 0] and
        kappa[..., 1]). kappa is the intensity absorption coefficient, hence dz/4 on the amplitude.
        """
        pad_shape = [(0, 0), (0, 0), (self.xpad, self.xpad), (self.ypad, self.ypad)]
        kappa_exp = np.exp(-kappa * X.dz / 4.0)
        wavefront_pad = self.pad(wavefront * kappa_exp[:, :, :, :, 0], pad_shape)
        wavefront_fft = self.fft2(wavefront_pad) * self.drift_kernel_tensor(X)
        wavefront_interior = self.ifft2(wavefront_fft)[:, :, self.xpad:self.xpad + X.xgrid, self.ypad:self.ypad + X.ygrid]
        return wavefront_interior * kappa_exp[:, :, :, :, 1]
