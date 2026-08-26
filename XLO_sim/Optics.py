import numpy as np
from scipy import interpolate
import scipy.fft as sp_fft
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator


class XLO_optics:
    
    
    def __init__(self, X):
        
        domains_xy, meshes_xy, step_sizes_xy = self.nd_space((-X.xmax, -X.ymax), (X.xmax, X.ymax), (X.xgrid, X.ygrid))
        self.xx, self.yy = meshes_xy
        self.dx, self.dy = step_sizes_xy

        self.xgrid = X.xgrid
        self.ygrid = X.ygrid
        self.xmax = X.xmax
        self.ymax = X.ymax
        
        self.xpad = X.xpad
        self.ypad = X.ypad
        self.tpad = X.tpad
        
        domains_kxky, meshes_kxky, step_sizes_kxky = self.nd_kspace((1.0, 1.0), (self.xgrid, self.ygrid), (self.xpad, self.ypad), (self.dx, self.dy))
        self.kx, self.ky = meshes_kxky
        self.dkx, self.dky = step_sizes_kxky
        
        if (self.dkx<self.dky):
            self.dk = self.dkx
        else:
            self.dk = self.dky

        self._drift_kernel_tensor_cache = {}


    def nd_space(self, mins=(), maxs=(), sizes=()):

        domains = [np.linspace(min, max, n)  for min, max, n in zip(mins, maxs, sizes)]
        meshes = np.meshgrid(*domains, indexing='ij')
        step_sizes = [domain[1] - domain[0] for domain in domains]

        return domains, meshes, step_sizes

    
    def nd_kspace(self, coeffs=(), sizes=(), pads=(), steps=(), shifted=False):

        if (shifted==False):
            domains = [coeff * np.fft.fftfreq(n + 2 * pad, step)  for coeff, n, pad, step in zip(coeffs, sizes, pads, steps)]
        else:
            domains = [np.fft.fftshift(coeff * np.fft.fftfreq(n + 2 * pad, step))  for coeff, n, pad, step in zip(coeffs, sizes, pads, steps)]

        
        meshes = np.meshgrid(*domains, indexing='ij')
        step_sizes = [domain[1] - domain[0] for domain in domains]

        return domains, meshes, step_sizes    
          
  
    def interpolate_wavefront_2D(self, wavefront, xnew, ynew):
        
        x = np.linspace(-self.xmax, self.xmax, self.xgrid)
        y = np.linspace(-self.ymax, self.ymax, self.ygrid)
        wvf_real = interpolate.interp2d(x, y, np.real(wavefront), kind='cubic')
        wvf_imag = interpolate.interp2d(x, y, np.imag(wavefront), kind='cubic')
        
        wvf_intrp = wvf_real(xnew, ynew) + 1j * wvf_imag(xnew, ynew)

        return wvf_intrp

    
    def interpolate_wavefront_3D(self, wavefront3D, old_domain, new_mesh):

        wvf3D_real = RegularGridInterpolator(old_domain, np.real(wavefront3D), method='linear', bounds_error=False, fill_value=None)
        wvf3D_imag = RegularGridInterpolator(old_domain, np.imag(wavefront3D), method='linear', bounds_error=False, fill_value=None)

        wvf3D_intrp = (wvf3D_real(new_mesh) + 1j * wvf3D_imag(new_mesh))
        

        return wvf3D_intrp

 

    def my_fft(self, wavefront):
        """
        Compute the 2-dimensional discrete forward Fourier transform.

        Parameters
        ----------
        wavefront: np.ndarray
            Input array

        Returns
        -------
        np.ndarray

        """
        # scipy.fft's pocketfft is consistently ~1.5x faster than numpy's here (verified
        # empirically at these array sizes), with workers=1 kept fixed since this runs
        # under SLURM with one process per core already -- workers=-1 would oversubscribe
        # cores rather than help. overwrite_x=True is safe: both call sites (the Fresnel
        # propagators) pass in a freshly created array that is never read again afterward.
        return sp_fft.fft2(wavefront, workers=1, overwrite_x=True)


    def my_fft_phased(self, array, axes, phasors):
        
        array_fft = np.fft.fftn(array, axes=axes, norm='ortho')
        for phasor in phasors:
            array_fft *= phasor
            
        #return array_fft
        return np.fft.fftshift(array_fft, axes=axes)
    

    def my_ifft_phased(self, array, axes, phasors):
        
        array_fft = np.fft.ifftn(array, axes=axes, norm='ortho')
        for phasor in phasors:
            array_fft *= np.conj(phasor)
            
        #return array_fft
        return np.fft.ifftshift(array_fft, axes=axes)


    def my_ifft(self, wavefront):
        """
        Compute the 2-dimensional discrete backward Fourier transform.

        Parameters
        ----------
        wavefront: np.ndarray
            Input array

        Returns
        -------
        np.ndarray

        """
        return sp_fft.ifft2(wavefront, workers=1, overwrite_x=True)


    def my_pad(self, wavefront, shape):
        """
        Pad an array with complex zero elements.

        Parameters
        ----------
        wavefront: np.ndarray
            The array to pad
        shape: array_like
            Number of values padded to the edges of each axis

        Returns
        -------
        np.ndarray

        """

        return np.pad(wavefront, shape, mode='constant', constant_values=(0.0 + 1j*0.0, 0.0 + 1j*0.0))
    
    
    def k_filter(self, X, wavefront_fft_p, kmax_x, kmax_y):
        
        kmax_ind_x = kmax_x / self.dkx
        kmax_ind_y = kmax_y / self.dky

        kpad_m_x = int((X.xgrid + 2*X.xpad)/2 - kmax_ind_x)
        kpad_p_x = int((X.xgrid + 2*X.xpad)/2 + kmax_ind_x)
        
        kpad_m_y = int((X.ygrid + 2*X.ypad)/2 - kmax_ind_y)
        kpad_p_y = int((X.ygrid + 2*X.ypad)/2 + kmax_ind_y)
        
        if (kpad_m_x>0):

            wavefront_fft_p = np.fft.fftshift(wavefront_fft_p, axes=(2,3))
            wavefront_fft_p[:, :, 0:kpad_m_x, :] = 0.0 + 0.0 * 1j
            wavefront_fft_p[:, :, kpad_p_x:X.xgrid + 2 * X.xpad, :] = 0.0 + 0.0 * 1j
            wavefront_fft_p = np.fft.ifftshift(wavefront_fft_p, axes=(2,3))

        if (kpad_m_y>0):
            
            wavefront_fft_p = np.fft.fftshift(wavefront_fft_p, axes=(2,3))
            wavefront_fft_p[:, :, :, 0:kpad_m_y] = 0.0 + 0.0 * 1j
            wavefront_fft_p[:, :, :, kpad_p_y:X.ygrid + 2 * X.ypad] = 0.0 + 0.0 * 1j
            wavefront_fft_p = np.fft.ifftshift(wavefront_fft_p, axes=(2,3))
            

        return wavefront_fft_p

    
    
    def drift_kernel(self, z, lambda_rad):

        return np.exp( -1j * z * np.pi * lambda_rad * (self.kx**2  + self.ky**2))

    
    def drift_propagator(self, wavefront_fft, z, lambda_rad):

        return wavefront_fft * self.drift_kernel(z, lambda_rad)
    
    
    def _drift_kernel_tensor(self, X, z, lambda_rad):
        """
        (2, 2, kx, ky) drift-kernel tensor, cached per (z, lambda_rad).

        Callers marching in z/t (Sample.py) hit this with the same (z, lambda_rad) pair
        every single (iz, it) step -- z is always the fixed z-grid step X.dz, never the
        running z-position, so the kernel is a true constant for the life of a run.
        Recomputing the exp() over the full padded k-grid on every call was previously
        the single biggest cost in the propagation step; memoizing it turns that into a
        one-time computation plus a dict lookup.
        """
        key = (z, lambda_rad)
        tensor = self._drift_kernel_tensor_cache.get(key)
        if tensor is None:
            drift_kernel_tensor_plus = np.einsum('s, xy->sxy', X.e_pol, self.drift_kernel(z, lambda_rad))
            tensor = np.asarray([drift_kernel_tensor_plus, np.conj(drift_kernel_tensor_plus)])
            self._drift_kernel_tensor_cache[key] = tensor
        return tensor


    def drift_propagator_tensorial(self, X, wavefront_fft, z, lambda_rad):

        return wavefront_fft * self._drift_kernel_tensor(X, z, lambda_rad)
    
    
    def thin_lens_kernel(self, k0, f_lens_x, f_lens_y):

        return np.exp(- 1j * k0 / 2.0 * (self.xx**2 / f_lens_x  + self.yy**2 / f_lens_y ))


    def thin_lens_propagator_tensorial(self, X, wavefront):

        thin_lens_kernel_tensor_plus = np.einsum('s, xy->sxy', X.e_pol, self.thin_lens_kernel(X.k0, X.f_lens_x, X.f_lens_y))
        thin_lens_kernel_tensor = np.asarray([thin_lens_kernel_tensor_plus, np.conj(thin_lens_kernel_tensor_plus)])
        
        return wavefront * thin_lens_kernel_tensor
    

    def Fresnel_propagator_no_absorption(self, X, wavefront, zstep, zpos):
        """
        Calculate the regular part of the ASE field propagation for a step in z direction without absorption.

        Parameters
        ----------
        X
            XLO_sim object
        wavefront: np.ndarray
            Complex field amplitudes of the ASE at given t, z
        zstep: float
            Size of the grid step in the z direction
        zpos: float
            Current grid position in the z direction

        Returns
        -------
        np.ndarray

        """

        if (X.enable_self_diffraction == False):
            return wavefront
    
        if (zpos==0):
            zpos = X.dz
        pad_shape = [(0, 0), (0, 0), (self.xpad, self.xpad), (self.ypad, self.ypad)]
        wavefront_pad = self.my_pad(wavefront, pad_shape)
        wavefront_fft = self.my_fft(wavefront_pad)        
        wavefront_fft_p = self.drift_propagator_tensorial(X, wavefront_fft, zstep, X.lambdaKalpha1N)

        return  self.my_ifft(wavefront_fft_p)[:, :, self.xpad: self.xpad + X.xgrid, self.ypad:self.ypad + X.ygrid]
    
    
    def Fresnel_propagator_with_absorption(self, X, wavefront, zstep, zpos, kappa, lambda_rad):
        """
        Calculate the regular part of the pump or seed field propagation for a step in z direction with absorption.

        Parameters
        ----------
        X
            XLO_sim object
        wavefront: np.ndarray
            Complex field amplitudes at given t,z
        zstep: float
            Size of the grid step in the z direction
        zpos: float
            Current grid position in the z direction
        kappa: np.ndarray
            Absorption coefficient of the chosen field at given t, and beginning and end of the current step in z direction
        lambda_rad: float
            Field wavelength

        Returns
        -------
        np.ndarray

        """
        
        if (zpos == 0):
            zpos = X.dz
        
            
        if (X.enable_self_diffraction == False):
            return wavefront * np.exp(-kappa[:, :, :, :, 0] * X.dz / 2.0)
        
        pad_shape = [(0, 0), (0, 0), (self.xpad, self.xpad), (self.ypad, self.ypad)]

        # kappa is only defined on the interior (xgrid, ygrid) grid, and exp(-kappa*dz/4)
        # would just evaluate to 1.0 across the padding either way -- so exponentiate the
        # small interior array once instead of zero-padding kappa first and running exp()
        # over the whole (often much larger) padded array.
        kappa_exp = np.exp(-kappa * X.dz / 4.0)

        wavefront_pad = self.my_pad(wavefront * kappa_exp[:, :, :, :, 0], pad_shape)
        wavefront_fft = self.my_fft(wavefront_pad)

        wavefront_fft_p = self.drift_propagator_tensorial(X, wavefront_fft, zstep, lambda_rad)

        wavefront_interior = self.my_ifft(wavefront_fft_p)[:, :, self.xpad: self.xpad + X.xgrid, self.ypad:self.ypad + X.ygrid]
        return wavefront_interior * kappa_exp[:, :, :, :, 1]
        
        
    def Greens_function_numerical_3D(self, X):
        
        Omega_psxyz = np.zeros((2, 2, X.xgrid, X.ygrid, X.zgrid), dtype=complex)
        dS = X.dx * X.dy
        Omega_psxyz[0, 0, int(X.xgrid/2), int(X.ygrid/2), 0] += 1.0 / dS

        for iz in range(1, X.zgrid):
            Omega_psxyz[:, :, :, :, iz] = self.Fresnel_propagator_no_absorption(X, Omega_psxyz[:, :, :, :, iz - 1], X.dz, iz * X.dz)

        self.Gxyz = Omega_psxyz[0, 0, :, :, :]

