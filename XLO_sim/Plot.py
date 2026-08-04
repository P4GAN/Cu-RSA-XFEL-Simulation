import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from . import tools
import scipy.constants as sp_const


class XLO_plot:
    
    def __init__(self):
        print('Activated X-Plot')
        matplotlib.rcParams['axes.labelsize'] = 12
        matplotlib.rcParams['xtick.labelsize'] = 12
        matplotlib.rcParams['ytick.labelsize'] = 12
        matplotlib.rcParams['legend.fontsize'] = 12
        
        self.sunrise_at_SLAC = ["010520","010c3a","011959","033185","024cab","017fd1","0499db","17baca","dfc49c","e65342"]
        pass
    
    def plot_complex2d(self, img, extent, labels, limits=None, cmap='viridis'):
    
        xmin, xmax, ymin, ymax = extent
        xlabel, ylabel = labels
        cmap = cmap

        if (limits is None):
            limits = extent

        xlim_min, xlim_max, ylim_min, ylim_max = limits

        fig = plt.figure(figsize=(14, 8))

        plt.subplot(2, 2, 1)
        plt.imshow(np.real(img).T, extent=[xmin, xmax, ymin, ymax], cmap=cmap, aspect='auto')
        plt.xlim(xlim_min, xlim_max)
        plt.ylim(ylim_min, ylim_max)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Re')
        plt.grid()
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.imshow(np.imag(img).T, extent=[xmin, xmax, ymin, ymax], cmap=cmap, aspect='auto')
        plt.xlim(xlim_min, xlim_max)
        plt.ylim(ylim_min, ylim_max)
        plt.xlabel(xlabel)
       # plt.ylabel(ylabel)
        plt.title('Im')
        plt.grid()
        plt.colorbar()
        
        plt.subplot(2, 2, 3)
        plt.imshow(np.abs(img).T, extent=[xmin, xmax, ymin, ymax], cmap=cmap, aspect='auto')
        plt.xlim(xlim_min, xlim_max)
        plt.ylim(ylim_min, ylim_max)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Abs')
        plt.grid()
        plt.colorbar()
        
        plt.subplot(2, 2, 4)
        plt.imshow(np.angle(img).T, extent=[xmin, xmax, ymin, ymax], cmap=cmap, aspect='auto')
        plt.xlim(xlim_min, xlim_max)
        plt.ylim(ylim_min, ylim_max)
        plt.xlabel(xlabel)
      #  plt.ylabel(ylabel)
        plt.title(r'$\phi$')
        plt.grid()
        plt.colorbar()
        
        plt.tight_layout()

        plt.show()
        
        
    def plot_WT(self, X, method, sig_tp, sig_tm, labels, limits=None):
        
        coeff = 2.0 * np.pi * X.hbar
        pad_shape = [(X.tpad, X.tpad)]
        shift = X.t0 + X.tpad * X.dt
        domain_w, mesh_w, dw = X.optics.nd_kspace([coeff], [X.tgrid], [X.tpad], [X.dt])
        
        sig_tp_pad = X.optics.my_pad(sig_tp, pad_shape)
        sig_tm_pad = X.optics.my_pad(sig_tm, pad_shape)
        
        extent = [np.min(domain_w[0][X.tpad:X.tgrid + X.tpad]), np.max(domain_w[0][X.tpad:X.tgrid + X.tpad]), 0, np.max(X.t)]
        
        if (limits is None):
            limits = extent

        xlim_min, xlim_max, ylim_min, ylim_max = limits
            
        ##Wigner's transform (rolling method)
        sig_t_corr = np.zeros((X.tgrid + 2 * X.tpad, X.tgrid + 2 * X.tpad), dtype=complex)
        
        for i in range(0, X.tgrid + 2 * X.tpad):
            j = int(X.tgrid / 2 + X.tpad) - i
            sig_t_corr[:, i] = np.roll(sig_tp_pad, j) * np.roll(sig_tm_pad, -j)
            #np.roll(sig_t_pad, j) * np.roll(np.conj(sig_t_pad), -j) 

        phasor = np.exp(1j * 2.0 * np.pi * domain_w[0] * shift / coeff)
        sig_corr_fft = X.optics.my_fft_phased(sig_t_corr, [1], [phasor])
        
        WT = sig_corr_fft[X.tpad:X.tgrid + X.tpad, X.tpad:X.tgrid + X.tpad]

        WT_proj_w = np.sum(np.real(WT), axis=0)
        WT_proj_t = np.sum(np.real(WT), axis=1)

        fudge_w = 2.5 
        fudge_t = 2.5 

        plt.figure(figsize=(8, 6))
        plt.imshow(np.real(WT), origin='lower', cmap='GnBu', aspect='auto', extent=extent)
        plt.plot(np.fft.fftshift(domain_w)[0][X.tpad:X.tgrid + X.tpad], fudge_w * WT_proj_w / np.max(WT_proj_w), color='blue', linewidth=2)
        plt.plot(fudge_t * WT_proj_t / np.max(WT_proj_t) + xlim_min, X.t, color='blue', linewidth=2)

        plt.xlabel('Photon energy (eV)')
        plt.ylabel('Time (fs)')
        plt.xlim(xlim_min, xlim_max)
        plt.ylim(ylim_min, ylim_max)
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        
    def Plot_pump(self, X):

        #print("2W0 beam size: ", 4.0 * X.sigma_r, "nm")
        #print("FWHM beam size: ", 2.355 * X.sigma_r, "nm")

        mx = 2
        mz = 2

        pump3D = X.pump_pulse_3D

        wavefront = np.zeros((2, 2, X.xgrid, X.ygrid), dtype=complex)
        wfz = np.zeros((mz * X.zgrid, X.xgrid, X.ygrid), dtype=complex)

        wavefront[0, 0, :, :] = pump3D[int(X.tgrid/2), :, :]
        wf = wavefront.copy()

        #plt.imshow(np.real(wavefront[0, 0, :, :]))
        #plt.imshow(np.imag(wavefront[0, 0, :, :]))

        w0 = np.sqrt(X.zR * X.lambdaKalpha1N / np.pi)
        z = np.linspace(-mz * X.zgrid * X.dz, mz * X.zgrid * X.dz, 2 * mz * X.zgrid)
        wz = w0 * np.sqrt(1 + (z/X.zR)**2)

        for zi in range(0, mz*X.zgrid):
            wf = X.optics.Fresnel_propagator_no_absorption(X, wf, X.dz, zi*X.dz)
            wfz[zi, :, :] = wf[0, 0, :, :].copy()

        wfz_wx = np.sum(wfz, axis=2)
        wfz_wx2 = np.vstack((np.flip(wfz_wx, axis=0), wfz_wx))

        plt.figure(figsize=(10, 4))
        
        plt.imshow(np.abs(wfz_wx2).T, extent=[-mz * X.zgrid * X.dz / 1e3, mz * X.zgrid * X.dz / 1e3, -X.ymax, X.ymax], aspect='auto')

        plt.axvline(x=-X.zmax/2/1e3, linewidth=0.61, ls='-.', color='magenta')
        plt.axvline(x=X.zmax/2/1e3, linewidth=0.61, ls='-.', color='magenta')
        plt.axhline(y=70, linewidth=1, ls='--', color='yellow')
        plt.axhline(y=-70, linewidth=1, ls='--', color='yellow')

        z /= 1e3
        
        plt.plot(z, wz, '-', linewidth=2, color='white')
        plt.plot(z, -wz, '-', linewidth=2, color='white')
        plt.plot(z, np.sqrt(2)*wz, '-', color='red')
        plt.plot(z, -np.sqrt(2)*wz, '-', color='red')
        
        plt.ylim(-X.ymax, X.ymax)

        plt.xlabel(r'z ($\mu m$)')
        plt.ylabel('y (nm)')
        plt.show()
        
        
 
        
    def Plot_reciprocal(self, X, limits_theta=None, limits_theta_w=None):
        
        if X.is_Cartesian_pol:
            field_stxy = X.Omega_qtxy
        else:
            field_stxy = (X.Omega_pstxyz[0, :, :, :, :, -1] + np.conj(X.Omega_pstxyz[1, :, :, :, :, -1])) / 2.0
    
        coeffs = (2.0 * np.pi * X.hbar, 2.0 * np.pi / X.k0, 2.0 * np.pi / X.k0)
        shifts = (X.t0 + X.tpad * X.dt, X.xmax + X.xpad * X.dx, X.ymax + X.ypad * X.dy)
        pad_shape = [(0, 0), (X.tpad, X.tpad), (X.xpad, X.xpad), (X.ypad, X.ypad)]
        domains_sim, meshes_sim, step_sizes_sim = X.optics.nd_kspace(coeffs, (X.tgrid, X.xgrid, X.ygrid), (X.tpad, X.xpad, X.ypad), (X.dt, X.dx, X.dy))
        phasors = [np.exp(1j * 2.0 * np.pi * mesh * shift / coeff) for coeff, mesh, shift in zip(coeffs, meshes_sim, shifts)]
    
        field_stxy_pad = X.optics.my_pad(field_stxy, pad_shape)
        field_fft_pad_phased = X.optics.my_fft_phased(field_stxy_pad, (1, 2, 3), phasors) 
        
        
        extent_theta = [1e3*np.min(domains_sim[2]), 1e3*np.max(domains_sim[2]), 1e3*np.min(domains_sim[1]), 1e3*np.max(domains_sim[1])]
        extent_theta_w = [1e3*np.min(domains_sim[2]), 1e3*np.max(domains_sim[2]), np.min(domains_sim[0][X.tpad:X.tgrid + X.tpad]), np.max(domains_sim[0][X.tpad:X.tgrid + X.tpad])]
        
        if (limits_theta is None):
            limits_theta = extent_theta
            
        if (limits_theta_w is None):
            limits_theta_w = extent_theta_w

        xlim_min_th, xlim_max_th, ylim_min_th, ylim_max_th = limits_theta
        xlim_min_th_w, xlim_max_th_w, ylim_min_th_w, ylim_max_th_w = limits_theta_w

    
        plt.figure(figsize=(12, 8))
    
        plt.subplot(2, 2, 1)
        plt.imshow(np.abs(field_fft_pad_phased[0, int(X.tgrid/2+X.tpad), :, :])**2, cmap=self.get_continuous_cmap(self.sunrise_at_SLAC), extent=extent_theta,aspect='auto')
        plt.xlabel(r'$\theta_x$ (mrad)')
        plt.ylabel(r'$\theta_y$ (mrad)')
        plt.xlim(xlim_min_th, xlim_max_th)
        plt.ylim(ylim_min_th, ylim_max_th)
        plt.colorbar()
        plt.tight_layout()
        
        plt.subplot(2, 2, 2)
        plt.imshow(np.abs(field_fft_pad_phased[1, int(X.tgrid/2+X.tpad), :, :])**2, cmap=self.get_continuous_cmap(self.sunrise_at_SLAC), extent=extent_theta,aspect='auto')
        plt.xlabel(r'$\theta_x$ (mrad)')
        plt.ylabel(r'$\theta_y$ (mrad)')
        plt.xlim(xlim_min_th, xlim_max_th)
        plt.ylim(ylim_min_th, ylim_max_th)
        plt.colorbar()
        plt.tight_layout()  
    
        plt.subplot(2, 2, 3)
        plt.imshow(np.abs(field_fft_pad_phased[0, :, :, int(X.ygrid/2+X.ypad)])**2, cmap=self.get_continuous_cmap(self.sunrise_at_SLAC), extent=extent_theta_w, aspect='auto')
        #plt.ylim(-2.5, 2.5)
        plt.ylabel('Photon energy (eV)')
        plt.xlabel(r'$\theta_x$ (mrad)')
        plt.xlim(xlim_min_th_w, xlim_max_th_w)
        plt.ylim(ylim_min_th_w, ylim_max_th_w)
        plt.colorbar()
        plt.tight_layout()

        plt.subplot(2, 2, 4)
        plt.imshow(np.abs(field_fft_pad_phased[1, :, :, int(X.ygrid/2+X.ypad)])**2, cmap=self.get_continuous_cmap(self.sunrise_at_SLAC), extent=extent_theta_w, aspect='auto')
        #plt.ylim(-2.5, 2.5)
        plt.ylabel('Photon energy (eV)')
        plt.xlabel(r'$\theta_x$ (mrad)')
        plt.xlim(xlim_min_th_w, xlim_max_th_w)
        plt.ylim(ylim_min_th_w, ylim_max_th_w)
        plt.colorbar()
        plt.tight_layout()
        
        plt.show()
        

    def get_continuous_cmap(self, hex_list, float_list=None):
        ''' Adopted from Kerry Halupka: https://gist.github.com/KerryHalupka/73af99afa8d4a4b7aaf1c05562ed4223
            creates and returns a color map that can be used in heat map figures.
            If float_list is not provided, colour map graduates linearly between each color in hex_list.
            If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 

            Parameters
            ----------
            hex_list: list of hex code strings
            float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

            Returns
            ----------
            colour map'''
        rgb_list = [self.rgb_to_dec(self.hex_to_rgb(i)) for i in hex_list]
        if float_list:
            pass
        else:
            float_list = list(np.linspace(0,1,len(rgb_list)))

        cdict = dict()
        for num, col in enumerate(['red', 'green', 'blue']):
            col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
            cdict[col] = col_list
        cmp = matplotlib.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
        return cmp

    def hex_to_rgb(self, value):
        '''
        Converts hex to rgb colours
        value: string of 6 characters representing a hex colour.
        Returns: list length 3 of RGB values'''
        value = value.strip("#") # removes hash symbol if present
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


    def rgb_to_dec(self, value):
        '''
        Converts rgb to decimal colours (i.e. divides each value by 256)
        value: list (length 3) of RGB values
        Returns: list (length 3) of decimal values'''
        return [v/256 for v in value]

    


    
    
#     def photon_total(X, pol, zpos):
    
#         return X.dt * X.dx * X.dy / (3.0 * X.lambdaKalpha1N**2 * X.Gamma_sp_fsm1N / 8.0 / np.pi) * np.sum(X.Omega_pstxyz[0,pol,:,:,:,zpos] * X.Omega_pstxyz[1,pol,:,:,:,zpos], axis=(0, 1, 2))

    
#     def sample_plot(self, X):

#         plt.figure()
#         plt.title(r'$\rho_0$')
#         plt.imshow(X.rho_0.T, origin=(0,0), aspect='auto', extent=[0, X.tmax, 0, X.zmax/1.0e3])
#         plt.ylabel('z ($\mu$m)')
#         plt.xlabel('t (s)')
#         plt.colorbar()


#         plt.figure()
#         plt.title(r'$\rho_i$')
#         plt.imshow(X.rho_i.T, origin=(0,0), aspect='auto', extent=[0, X.tmax, 0, X.zmax/1.0e3])
#         plt.ylabel(r'z ($\mu$m)')
#         plt.xlabel('t (s)')
#         plt.colorbar()


#         plt.figure()
#         plt.title('j')
#         plt.imshow(X.JA.T, origin=(0,0), aspect='auto', extent=[0, X.tmax, 0, X.zmax/1.0e3])
#         plt.ylabel(r'z ($\mu$m)')
#         plt.xlabel('t (s)')
#         plt.colorbar()

#         plt.show()

#     def nice_rho_plot(self, rho, X):

#         plt.figure()
#         plt.imshow(np.abs(rho.T), origin=(0,0), aspect='auto', extent=[0, X.tmax, 0, X.zmax/1.0e3])
#         plt.ylabel(r'z ($\mu$m)')
#         plt.xlabel('t (s)')
#         plt.colorbar()
#         plt.show()

#     def nice_rho_grid_plot(self, rho, X):

#         fig = plt.figure(figsize=(4, 4))
#         fig.subplots_adjust(hspace=0.4, wspace=0.4)
#         for i in range(1, len(rho)+1):
#             ax = fig.add_subplot(5, 4, i)
#             im = ax.imshow(np.real(rho[i-1].T), origin=(0,0), aspect='auto', extent=[0, X.tmax, 0, X.zmax/1.0e3])
#             fig.colorbar(im, ax=ax)
#         #plt.show()


#     def nice_Omega_grid_plot(self, Omega, X):

#         fig = plt.figure(figsize=(4, 4))
#         fig.subplots_adjust(hspace=0.4, wspace=0.4)
#         for i in range(1, len(Omega)+1):
#             ax = fig.add_subplot(1, 2, i)
#             im = ax.imshow(np.real(Omega[i-1].T), origin=(0,0), aspect='auto', extent=[0, X.tmax, 0, X.zmax/1.0e3])
#             fig.colorbar(im, ax=ax)
#         plt.show()
   
    
    
    
    def plot_pump_j_3D(self, X):
        # def find_fwhm(x, y):
        #     max_value = np.max(y)
        #     max_index = np.argmax(y)
        #     half_max = max_value / 2
        #     idx_left = 0
        #     idx_right = len(y) - 1
        #     for i in range(0, max_index, 1):
        #         if y[i] > half_max:
        #             idx_left = i
        #             break
        #     for i in range(len(y)-1, max_index, -1):
        #         if y[i] > half_max:
        #             idx_right = i
        #             break
        #     return x[idx_right] - x[idx_left]

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(X.t, X.convert_pump_phnm2fs_Wcm2 * X.j_3D[:,int(X.xgrid/2),int(X.ygrid/2),0])
        plt.xlabel('t (fs)')
        plt.ylabel(r'$j_{pump}$ (W/cm$^{2}$)')
        plt.grid(True)

        tint = np.abs(X.t - X.t_pump_max).argmin()
        plt.subplot(1, 2, 2)
        plt.imshow(X.convert_pump_phnm2fs_Wcm2 * X.j_3D[tint,:,:,0], aspect='auto', origin='lower', extent=[-X.xmax,X.xmax,-X.ymax,X.ymax])
        plt.xlabel('x (nm)')
        plt.ylabel('y (nm)')
        plt.title(f't = {(X.t[tint]):.2f} fs, z=0')
        plt.colorbar(label=r'$j_{pump}$ (W/cm$^{2}$)')

        plt.tight_layout()
        plt.show()

        x_x = np.linspace(-X.xmax,X.xmax,X.xgrid)
        print(f'FWHM x = {tools.find_fwhm(x_x, X.j_3D[int(X.tgrid/2),:,int(X.ygrid/2),0]):.0f} nm, nominal {X.pump_width_FWHM:.0f} nm')
        print(f'FWHM t = {tools.find_fwhm(X.t, X.j_3D[:,int(X.xgrid/2),int(X.ygrid/2),0]):.3f} fs, nominal {X.pump_duration_FWHM_t:.3f} fs')
        print(f'pump photons = {np.sum(X.dx * X.dy * X.dt * X.j_3D[:,:,:,0]):.2e}, nominal {X.N_pump_photons:.1e}')

        sigma_x = X.pump_width_FWHM / 2.355
        fwhm_kx = 1.1774 / sigma_x
        fwhm_th = fwhm_kx / X.kp
        print(f'expected pump intensity angular FWHM = {(fwhm_th * 1e3):.2f} mrad')

        fwhm_w_fsm1 = 1.1774 / (X.pump_duration_FWHM_t/2.355)
        fwhm_w_eV = 0.6582 * fwhm_w_fsm1
        print(f'expected pump intensity spectral FWHM = {fwhm_w_eV:.2f} eV')
        

    def plot_z_t__j_rho_g_rho_ee(self, X, folder_figs_path, folder_data_path):

        name = 'z-t__j-rho_g-rho_ee'

        j_tz    = X.convert_pump_phnm2fs_Wcm2 * X.j_3D[:,int(X.xgrid/2),int(X.ygrid/2),:]
        rho0_tz = X.rho_0_3D[:,int(X.xgrid/2),int(X.ygrid/2),:]
        E_tz    = np.einsum('ijs, jktz, kis-> tz', X.Tijs_minus, X.rho_ijtxyz[:,:,:,int(X.xgrid/2),int(X.ygrid/2),:], X.Tijs_plus)
        extentV = [0,X.tmax,0,X.zmax/1.0e3]

        plt.figure(figsize=(14, 4))
        plt.suptitle('x=0,y=0')
        plt.subplot(1, 3, 1)
        plt.title(r'$j_{pump}$ (W/cm$^2$)')
        plt.imshow(np.abs(j_tz.T), aspect='auto', origin='lower', extent=extentV)
        plt.xlabel('t (fs)')
        plt.ylabel(r'z ($\mu m$)')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.title(r'$\rho_{ground}$')
        plt.imshow(np.abs(rho0_tz.T), aspect='auto', origin='lower', extent=extentV)
        plt.xlabel('t (fs)')
        plt.ylabel(r'z ($\mu m$)')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.title(r'$\rho_{ee}$')
        plt.imshow(np.real(E_tz.T), aspect='auto', origin='lower', extent=extentV)
        plt.xlabel('t (fs)')
        plt.ylabel(r'z ($\mu m$)')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(folder_figs_path + '/' + name + '.pdf')
        plt.savefig(folder_figs_path + '/' + name + '.png')
        np.savez(folder_data_path + '/' + name + '.npz', j_tz=j_tz, rho0_tz=rho0_tz, E_tz=E_tz, extentV=extentV)
        plt.show()

    def plot_z__Nph(self, X, folder_figs_path, folder_data_path):

        name = 'z__Nph'

        nphot1 = np.real(X.nphoton_sz[0,:])
        nphot2 = np.real(X.nphoton_sz[1,:])
        nph_pump_z = X.dt * X.dx * X.dy * np.einsum('txyz -> z', X.j_3D)
        z_axis = 1e-3 * X.z

        plt.figure()
        plt.plot(z_axis, nphot1, '-', label = r'$N_{circ.pol.=+1}^{SF}$')
        plt.plot(z_axis, nphot2, '-', label = r'$N_{circ.pol.=-1}^{SF}$')
        # plt.plot(z_axis, nph_pump_z, '-', label = r'$N^{pump}$')
        # plt.yscale('log')
        plt.xlabel('z (um)')
        plt.ylabel('Photon number')
        plt.grid(True)
        plt.legend()

        plt.savefig(folder_figs_path + '/' + name + '.pdf')
        plt.savefig(folder_figs_path + '/' + name + '.png')
        np.savez(folder_data_path + '/' + name + '.npz', nphot1=nphot1, nphot2=nphot2, nph_pump_z=nph_pump_z, z_axis=z_axis)
        plt.show()
        
    def plot_t__j_I_rho_g_rho_ee(self, X, folder_figs_path, folder_data_path, nz):
        
        z_str_ar = [f'{(1e-3 * num):.3f}' for num in np.linspace(X.dz, X.zmax, nz)]
        name = "t__j-I-rho_g-rho_ee"
        np.savetxt(folder_data_path + '/' + name + '_zar', z_str_ar, fmt='%s')
        
        for z in z_str_ar:

            local_name = name + '_z=' + z + 'um'

            zint = np.abs(X.z - 1e3 * float(z)).argmin()

            I_t = np.einsum('stxy,stxy->t', X.Omega_pstxyz[0, :, :, :, :, zint], X.Omega_pstxyz[1, :, :, :, :, zint])
            E_t = np.einsum('ijs, jkt, kis-> t', X.Tijs_minus, X.rho_ijtxyz[:,:,:,int(X.xgrid/2),int(X.ygrid/2),zint], X.Tijs_plus)
            G_t = np.einsum('ijs, jkt, kis-> t', X.Tijs_plus, X.rho_ijtxyz[:,:,:,int(X.xgrid/2),int(X.ygrid/2),zint], X.Tijs_minus)
            j_t = X.j_3D[:,int(X.xgrid/2),int(X.ygrid/2),zint]
            r0_t = X.rho_0_3D[:,int(X.xgrid/2),int(X.ygrid/2),zint]
            t_axis = X.t

            plt.figure()
            plt.plot(t_axis, j_t/max(j_t)*max(np.real(E_t)),label = r'$j(x=y=0)$, norm.')
            plt.plot(t_axis, np.real(I_t)/max(np.real(I_t))*max(np.real(E_t)), label = r'$\int dx dy I(x,y)$, norm.')
            plt.plot(t_axis, np.imag(I_t)/max(np.real(I_t))*max(np.real(E_t)), label = r'Im I, norm.')
            plt.plot(t_axis, np.real(E_t),'--', label = r'$\rho_{ee}(x=y=0)$')
            plt.plot(t_axis, np.real(E_t - G_t),'--', label = r'$\rho_{inv}(x=y=0)$')
            plt.plot(t_axis, r0_t/max(r0_t)*max(np.real(E_t)),'--', label = r'$\rho_{g}(x=y=0)$, norm.')
            plt.xlabel('t (fs)')
            plt.ylabel(r'$\rho_{ee}$')
            plt.title(f'z={float(z):.0f} um')
            plt.grid(True)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.ylim(-0.05*max(np.real(E_t)), 1.05*max(np.real(E_t)))

            plt.savefig(folder_figs_path + '/' + local_name + '.pdf')
            plt.savefig(folder_figs_path + '/' + local_name + '.png')
            np.savez(folder_data_path + '/' + local_name + '.npz', I_t=I_t, E_t=E_t, G_t=G_t, j_t=j_t, r0_t=r0_t, t_axis=t_axis, zint=zint)
            plt.show()
      
    
    def plot_x_y__I_j_rho_ee(self, X, folder_figs_path, folder_data_path, nz):
         
        z_str_ar = [f'{(1e-3 * num):.3f}' for num in np.linspace(X.dz, X.zmax, nz)]
        name = "x-y__I-j-rho_ee"
        np.savetxt(folder_data_path + '/' + name + '_zar', z_str_ar, fmt='%s')
        
        tint_max_j = np.unravel_index(np.argmax(X.j_3D), X.j_3D.shape)[0]
        print(f'peak of pump at   {(X.t[tint_max_j]):.3f} fs')

        rho_ee_t = np.real(np.einsum('ijs, jkt, kis-> t', X.Tijs_minus, X.rho_ijtxyz[:,:,:,int(X.xgrid/2),int(X.ygrid/2),0], X.Tijs_plus))
        tint_max_rho_ee = np.unravel_index(np.argmax(rho_ee_t), rho_ee_t.shape)[0]
        print(f'peak of rho_ee at {(X.t[tint_max_rho_ee]):.3f} fs')

        I_tz = np.abs(np.einsum('stxyz,stxyz->tz', X.Omega_pstxyz[0, :, :, :, :, :], X.Omega_pstxyz[1, :, :, :, :, :]))
        tint_max_I = np.unravel_index(np.argmax(I_tz), I_tz.shape)[0]
        print(f'peak of SF at     {(X.t[tint_max_I]):.3f} fs')
        np.savetxt(folder_data_path + '/' + name + '_tint_max', np.asarray([tint_max_j, tint_max_rho_ee, tint_max_I]), fmt='%d')
        
        for z in z_str_ar:

            local_name = name + '_z=' + z + 'um'

            zint = np.abs(X.z - 1e3 * float(z)).argmin()

            conv_factor = 1 / (3.0 * X.lambdaKalpha1N**2 * X.Gamma_sp_fsm1N / 8.0 / np.pi)

            I_xy = (X.convert_SF_phnm2fs_Wcm2 * conv_factor * 
                    np.einsum('sxy,sxy->xy', X.Omega_pstxyz[0, :, tint_max_I, :, :, zint], X.Omega_pstxyz[1, :, tint_max_I, :, :, zint])
                   )
            E_xy = np.einsum('ijs, jkxy, kis-> xy', X.Tijs_minus, X.rho_ijtxyz[:,:,tint_max_rho_ee,:,:,zint], X.Tijs_plus)
            j_xy = X.convert_pump_phnm2fs_Wcm2 * X.j_3D[tint_max_j,:,:,zint]
            extentV = [-X.xmax,X.xmax,-X.ymax,X.ymax]
            t_max_I = X.t[tint_max_I]
            t_max_j = X.t[tint_max_j]
            t_max_rho_ee = X.t[tint_max_rho_ee]

            plt.figure(figsize=(14, 4))
            plt.suptitle(f"z = {float(z):.0f} um")    

            plt.subplot(1, 3, 1)
            plt.title(r'$I \quad (W/cm^2)$' + f' at {t_max_I:.3f} fs')
            plt.imshow(np.real(I_xy), aspect='auto', origin='lower', extent=extentV)
            plt.xlabel('x (nm)')
            plt.ylabel('y (nm)')
            plt.colorbar()  

            plt.subplot(1, 3, 2)
            plt.title(r'$j_{pump} \quad (W/cm^2)$' + f' at {t_max_j:.3f} fs')
            plt.imshow(j_xy, aspect='auto', origin='lower', extent=extentV)
            plt.xlabel('x (nm)')
            plt.ylabel('y (nm)')
            plt.colorbar()

            plt.subplot(1, 3, 3)
            plt.title(r'$\rho_{ee}$' + f' at {t_max_rho_ee:.3f} fs')
            plt.imshow(np.real(E_xy), aspect='auto', origin='lower', extent=extentV)
            plt.xlabel('x (nm)')
            plt.ylabel('y (nm)')
            plt.colorbar()    

            plt.tight_layout()

            plt.savefig(folder_figs_path + '/' + local_name + '.pdf')
            plt.savefig(folder_figs_path + '/' + local_name + '.png')
            np.savez(folder_data_path + '/' + local_name + '.npz', I_xy=I_xy, E_xy=E_xy, j_xy=j_xy, 
                     extentV=extentV, t_max_I=t_max_I, t_max_j=t_max_j, t_max_rho_ee=t_max_rho_ee)

            plt.show()
            
    def plot_x_y__I_j_int(self, X, folder_figs_path, folder_data_path, nz):
        
        z_str_ar = [f'{(1e-3 * num):.3f}' for num in np.linspace(X.dz, X.zmax, nz)]
        name = "x-y__I-j_int"
        np.savetxt(folder_data_path + '/' + name + '_zar', z_str_ar, fmt='%s')
        
        for z in z_str_ar:
            
            local_name = name + '_z=' + z + 'um'

            zint = np.abs(X.z - 1e3 * float(z)).argmin()

            conv_factor = 1 / (3.0 * X.lambdaKalpha1N**2 * X.Gamma_sp_fsm1N / 8.0 / np.pi)

            I_xy = conv_factor * X.dt * np.einsum('stxy,stxy->xy', X.Omega_pstxyz[0, :, :, :, :, zint], X.Omega_pstxyz[1, :, :, :, :, zint])
            j_xy = X.dt * np.einsum('txy->xy', X.j_3D[:,:,:,zint]) 
            Nph_SF   = np.real(X.dx * X.dy * np.sum(I_xy))
            Nph_pump = X.dx * X.dy * np.sum(j_xy)
            extentV = [-X.xmax,X.xmax,-X.ymax,X.ymax]

            plt.figure(figsize=(9, 4))
            plt.suptitle(r"z= "+f'{(1e-3 * X.z[zint]):.0f}'+" um, "+r'$N_{ph}^{SF}$'+' = '+f'{Nph_SF:.1e}'+r', $N_{ph}^{pump}$ = '+f'{Nph_pump:.1e}')

            plt.subplot(1, 2, 1)
            plt.title(r'$\int dt^{\prime} I(t^{\prime},x,y,z) \quad (ph/nm^2)$')
            plt.imshow(np.real(I_xy), aspect='auto', origin='lower', extent=extentV)
            plt.xlabel('x (nm)')
            plt.ylabel('y (nm)')
            plt.colorbar()  

            plt.subplot(1, 2, 2)
            plt.title(r'$\int dt^{\prime} j_{pump}(t^{\prime},x,y,z) \quad (ph/nm^2)$')
            plt.imshow(j_xy, aspect='auto', origin='lower', extent=extentV)
            plt.xlabel('x (nm)')
            plt.ylabel('y (nm)')
            plt.colorbar()

            plt.tight_layout()

            plt.savefig(folder_figs_path + '/' + local_name + '.pdf')
            plt.savefig(folder_figs_path + '/' + local_name + '.png')
            np.savez(folder_data_path + '/' + local_name + '.npz', I_xy=I_xy, j_xy=j_xy, Nph_SF=Nph_SF, Nph_pump=Nph_pump, extentV=extentV)

            plt.show()
            
            
    def plot_w_thx__I_j(self, X, folder_figs_path, folder_data_path, nz):
        
        z_str_ar = [f'{(1e-3 * num):.3f}' for num in np.linspace(X.dz, X.zmax, nz)]
        name = "w-thx__I-j"
        np.savetxt(folder_data_path + '/' + name + '_zar', z_str_ar, fmt='%s')        
        
        xpad = 128
        tpad = 428

        def FFT_theta_w(field_tx):
            coeffs = (2.0 * np.pi * X.hbar, 2.0 * np.pi / X.k0)
            # shifts = (X.t0 + tpad * X.dt, X.xmax + xpad * X.dx)
            shifts = (X.t_pump_max + tpad * X.dt, X.xmax + xpad * X.dx)
            pad_shape = [(tpad, tpad), (xpad, xpad)]
            domains_sim, meshes_sim, step_sizes_sim = X.optics.nd_kspace(coeffs, (X.tgrid, X.xgrid), (tpad, xpad), (X.dt, X.dx))
            phasors = [np.exp(1j * 2.0 * np.pi * mesh * shift / coeff) for coeff, mesh, shift in zip(coeffs, meshes_sim, shifts)]

            field_tx_pad = X.optics.my_pad(field_tx, pad_shape)
            field_fft_pad_phased = X.optics.my_fft_phased(field_tx_pad, (0, 1), phasors)

            extent_theta_w = [1e3*np.min(domains_sim[1]), 1e3*np.max(domains_sim[1]), np.min(domains_sim[0][tpad:X.tgrid + tpad]), 
                          np.max(domains_sim[0][tpad:X.tgrid + tpad])]

            return extent_theta_w, field_fft_pad_phased 


        def plotting_w_thx(I_w_thx, extent_theta_w, thx_show_min, thx_show_max, w_show_min, w_show_max, name, xgrid_pad, tgrid_pad):

            theta_x_ar = np.linspace(extent_theta_w[0], extent_theta_w[1], xgrid_pad)
            womega_ar  = np.linspace(extent_theta_w[2], extent_theta_w[3], tgrid_pad)

            dtheta_x = theta_x_ar[1] - theta_x_ar[0] 
            dwomega = womega_ar[1] - womega_ar[0] 
            I_w = dtheta_x * np.einsum('wa -> w', I_w_thx)
            I_thx = dwomega * np.einsum('wa -> a', I_w_thx)

            thx_FWHM = tools.find_fwhm(theta_x_ar, I_thx)
            w_FWHM   = tools.find_fwhm(womega_ar, I_w)
            plt.imshow(I_w_thx.T, origin='lower', extent=np.array(extent_theta_w)[[2, 3, 0, 1]],aspect='auto')
            plt.plot(womega_ar, I_w / max(I_w) * (thx_show_max - thx_show_min) * 0.3 + thx_show_min, color = 'blue')
            plt.plot(I_thx / max(I_thx) * (w_show_max - w_show_min) * 0.4 + w_show_min, theta_x_ar, color = 'red')
            plt.ylabel(r'$\theta_x$ (mrad)')
            plt.xlabel(r'$\omega$ (eV)')
            plt.title(name + '\n' + r'$FWHM_{\theta_x}$=' + f"{thx_FWHM:.1f}" + r' mrad, $FWHM_{\omega}$=' + f"{w_FWHM:.1f} eV")
            plt.ylim(thx_show_min, thx_show_max)
            plt.xlim(w_show_min, w_show_max)
            plt.colorbar()

        thx_show_min = -3
        thx_show_max = 3
        w_show_min = -20
        w_show_max = 20

        for z in z_str_ar:
            
            local_name = name + '_z=' + z + 'um'
    
            zint = np.abs(X.z - 1e3 * float(z)).argmin()

            OmegaL_pstxy = X.Omega_pstxyz[:, :, :, :, :, zint]
            OmegaLp0_tx = X.dy * np.einsum('txy -> tx', OmegaL_pstxy[0,0,:,:,:])
            OmegaLm0_tx = X.dy * np.einsum('txy -> tx', OmegaL_pstxy[1,0,:,:,:])
            OmegaLp1_tx = X.dy * np.einsum('txy -> tx', OmegaL_pstxy[0,1,:,:,:])
            OmegaLm1_tx = X.dy * np.einsum('txy -> tx', OmegaL_pstxy[1,1,:,:,:])
            extent_theta_w, OmegaLp0_w_thx = FFT_theta_w(OmegaLp0_tx)
            extent_theta_w, OmegaLm0_w_thx = FFT_theta_w(OmegaLm0_tx)
            extent_theta_w, OmegaLp1_w_thx = FFT_theta_w(OmegaLp1_tx)
            extent_theta_w, OmegaLm1_w_thx = FFT_theta_w(OmegaLm1_tx)
            I_w_thx = np.abs(OmegaLp0_w_thx * OmegaLm0_w_thx + OmegaLp1_w_thx * OmegaLm1_w_thx)

            Omega0_txy = np.sqrt(X.j_3D[:,:,:,zint])
            Omega0_tx = X.dy * np.einsum('txy -> tx', Omega0_txy)
            extent_theta_w, Omega0_w_thx = FFT_theta_w(Omega0_tx)

            xgrid_pad = X.xgrid + 2*xpad
            tgrid_pad = X.tgrid + 2*tpad


            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            plt.suptitle(f'z= {float(z):.0f} um')
            plotting_w_thx(np.abs(Omega0_w_thx)**2, extent_theta_w, thx_show_min, thx_show_max, w_show_min, w_show_max,
                           r'$I_{SF}(\omega,\theta_x,\theta_y=0,z)\quad (arb.u.)$', xgrid_pad, tgrid_pad)
            plt.sca(ax1)
            plotting_w_thx(I_w_thx, extent_theta_w, thx_show_min, thx_show_max, w_show_min, w_show_max, 
                           r'$j_{pump}(\omega,\theta_x,\theta_y=0,z)\quad (arb.u.)$', xgrid_pad, tgrid_pad)
            plt.sca(ax2)
            plt.tight_layout()

            plt.savefig(folder_figs_path + '/' + local_name + '.pdf')
            plt.savefig(folder_figs_path + '/' + local_name + '.png')
            np.savez(folder_data_path + '/' + local_name + '.npz', I_w_thx=I_w_thx, Omega0_w_thx=Omega0_w_thx,
                     extent_theta_w=extent_theta_w, 
                     thx_show_min=thx_show_min, thx_show_max=thx_show_max,
                     w_show_min=w_show_min, w_show_max=w_show_max,
                     xgrid_pad=xgrid_pad, tgrid_pad=tgrid_pad
                    )

            plt.show()
            
            
            
            
            
    def plot_thxy__w_thx_I_j(self, X, folder_figs_path, folder_data_path, nz, w_show, th_show):

        z_str_ar = [f'{(1e-3 * num):.3f}' for num in np.linspace(X.dz, X.zmax, nz)]
        name = "w-thx__I-j"
        np.savetxt(folder_data_path + '/' + name + '_zar', z_str_ar, fmt='%s')        

        tpad = 1000
        xpad = 64
        ypad = 64
        
        thx_show_min = -th_show
        thx_show_max = th_show
        thy_show_min = -th_show
        thy_show_max = th_show
        w_show_min = -w_show
        w_show_max = w_show


        def FFT_thxy(field_pstxy):
            coeffs = (2.0 * np.pi / X.k0, 2.0 * np.pi / X.k0)
            shifts = (X.xmax + xpad * X.dx, X.ymax + ypad * X.dy)
            pad_shape = [(0, 0), (0, 0), (0, 0), (xpad, xpad), (ypad, ypad)]
            domains_sim, meshes_sim, step_sizes_sim = X.optics.nd_kspace(coeffs, (X.xgrid, X.ygrid), (xpad, ypad), (X.dx, X.dy))
            phasors = [np.exp(1j * 2.0 * np.pi * mesh * shift / coeff) for coeff, mesh, shift in zip(coeffs, meshes_sim, shifts)]

            field_pstxy_pad = X.optics.my_pad(field_pstxy, pad_shape)
            field_fft_pad_phased = X.optics.my_fft_phased(field_pstxy_pad, (3, 4), phasors)

            extent_thxy = [
                1e3*np.min(domains_sim[0]), 1e3*np.max(domains_sim[0]),
                1e3*np.min(domains_sim[1]), 1e3*np.max(domains_sim[1])]

            return extent_thxy, field_fft_pad_phased 


        def FFT_w_thy(field_pstxy):
            coeffs = (2.0 * np.pi * X.hbar, 2.0 * np.pi / X.k0)
            shifts = (X.t_pump_max + tpad * X.dt, X.ymax + ypad * X.dy)
            field_psxty = np.einsum('pstxy->psxty',field_pstxy)
            pad_shape = [(0, 0), (0, 0), (0, 0), (tpad, tpad), (ypad, ypad)]
            domains_sim, meshes_sim, step_sizes_sim = X.optics.nd_kspace(coeffs, (X.tgrid, X.ygrid), (tpad, ypad), (X.dt, X.dy))
            phasors = [np.exp(1j * 2.0 * np.pi * mesh * shift / coeff) for coeff, mesh, shift in zip(coeffs, meshes_sim, shifts)]

            field_psxty_pad = X.optics.my_pad(field_psxty, pad_shape)
            field_fft_pad_phased = X.optics.my_fft_phased(field_psxty_pad, (3, 4), phasors)

            extent_w_thy = [
                np.min(domains_sim[0][tpad:X.tgrid + tpad]), np.max(domains_sim[0][tpad:X.tgrid + tpad]),
                1e3*np.min(domains_sim[1]), 1e3*np.max(domains_sim[1])]

            return extent_w_thy, field_fft_pad_phased 


        def FFT_pump_w_thy(field_txy):
            coeffs = (2.0 * np.pi * X.hbar, 2.0 * np.pi / X.k0)
            shifts = (X.t_pump_max + tpad * X.dt, X.ymax + ypad * X.dy)
            field_xty = np.einsum('txy->xty',field_txy)
            pad_shape = [(0, 0), (tpad, tpad), (ypad, ypad)]
            domains_sim, meshes_sim, step_sizes_sim = X.optics.nd_kspace(coeffs, (X.tgrid, X.ygrid), (tpad, ypad), (X.dt, X.dy))
            phasors = [np.exp(1j * 2.0 * np.pi * mesh * shift / coeff) for coeff, mesh, shift in zip(coeffs, meshes_sim, shifts)]

            field_xty_pad = X.optics.my_pad(field_xty, pad_shape)
            field_fft_pad_phased = X.optics.my_fft_phased(field_xty_pad, (1, 2), phasors)

            extent_w_thy = [
                np.min(domains_sim[0][tpad:X.tgrid + tpad]), np.max(domains_sim[0][tpad:X.tgrid + tpad]),
                1e3*np.min(domains_sim[1]), 1e3*np.max(domains_sim[1])]

            return extent_w_thy, field_fft_pad_phased


        def plotting_thxy(I_thxy, extent_thxy, thx_show_min, thx_show_max, thy_show_min, thy_show_max, name, xgrid_pad, ygrid_pad):

            theta_x_ar = np.linspace(extent_thxy[0], extent_thxy[1], xgrid_pad)
            theta_y_ar = np.linspace(extent_thxy[2], extent_thxy[3], ygrid_pad)

            dtheta_x = theta_x_ar[1] - theta_x_ar[0] 
            dtheta_y = theta_y_ar[1] - theta_y_ar[0] 
            I_thy = dtheta_x * np.einsum('xy -> y', I_thxy)
            I_thx = dtheta_y * np.einsum('xy -> x', I_thxy)

            thx_FWHM = tools.find_fwhm(theta_x_ar, I_thx)
            thy_FWHM = tools.find_fwhm(theta_y_ar, I_thy)
            plt.imshow(I_thxy.T, origin='lower', extent=extent_thxy,aspect='auto')
            plt.plot(theta_x_ar, I_thx / max(I_thx) * (thy_show_max - thy_show_min) * 0.3 + thy_show_min, color = 'blue')
            plt.plot(I_thy / max(I_thy) * (thx_show_max - thx_show_min) * 0.3 + thx_show_min, theta_y_ar, color = 'red')
            plt.ylabel(r'$\theta_y$ (mrad)')
            plt.xlabel(r'$\theta_x$ (mrad)')
            plt.title(name + '\n' + r'$FWHM_{\theta_x}$=' + f"{thx_FWHM:.1f}" + r' mrad, $FWHM_{\theta_y}$=' + f"{thy_FWHM:.1f} mrad")
            plt.ylim(thy_show_min, thy_show_max)
            plt.xlim(thx_show_min, thx_show_max)
            plt.colorbar()


        def plotting_w_thy(I_w_thy, extent_w_thy, thy_show_min, thy_show_max, w_show_min, w_show_max, name, ygrid_pad, tgrid_pad):

            womega_ar  = np.linspace(extent_w_thy[0], extent_w_thy[1], tgrid_pad)
            theta_y_ar = np.linspace(extent_w_thy[2], extent_w_thy[3], ygrid_pad)

            dtheta_y = theta_y_ar[1] - theta_y_ar[0] 
            dwomega = womega_ar[1] - womega_ar[0] 
            I_w = np.real(dtheta_y * np.einsum('wa -> w', I_w_thy))
            I_thy = np.real(dwomega * np.einsum('wa -> a', I_w_thy))

            thy_FWHM = tools.find_fwhm(theta_y_ar, I_thy)
            w_FWHM   = tools.find_fwhm(womega_ar, I_w)
            plt.imshow(I_w_thy.T, origin='lower', extent=extent_w_thy,aspect='auto')
            plt.plot(womega_ar, I_w / max(I_w) * (thy_show_max - thy_show_min) * 0.3 + thy_show_min, color = 'blue')
            plt.plot(I_thy / max(I_thy) * (w_show_max - w_show_min) * 0.3 + w_show_min, theta_y_ar, color = 'red')
            plt.ylabel(r'$\theta_y$ (mrad)')
            plt.xlabel(r'$\omega$ (eV)')
            plt.title(name + '\n' + r'$FWHM_{\omega}$=' + f"{w_FWHM:.1f}" + r' eV, $FWHM_{\theta_y}$=' + f"{thy_FWHM:.1f} mrad")
            plt.ylim(thy_show_min, thy_show_max)
            plt.xlim(w_show_min, w_show_max)
            plt.colorbar()

        
        for z in z_str_ar:

            local_name = name + '_z=' + z + 'um'

            zint = np.abs(X.z - 1e3 * float(z)).argmin()

            xgrid_pad = X.xgrid + 2*xpad
            ygrid_pad = X.ygrid + 2*ypad
            tgrid_pad = X.tgrid + 2*tpad

            OmegaSF_z_pstxy = X.Omega_pstxyz[:, :, :, :, :, zint]
            extent_thxy, OmegaSF_z_pstThxThy = FFT_thxy(OmegaSF_z_pstxy)
            extent_w_thy, OmegaSF_z_psxwThy = FFT_w_thy(OmegaSF_z_pstxy)

            OmegaPump_z_txy = X.Pfield_txyz[:,:,:,zint]#np.sqrt(X.j_3D[:,:,:,zint])
            extent_w_thy, OmegaPump_z_xwThy = FFT_pump_w_thy(OmegaPump_z_txy)


            I_SF_thxy = np.real(
                np.einsum('stxy,stxy->xy', OmegaSF_z_pstThxThy[0,:,:,:,:], OmegaSF_z_pstThxThy[1,:,:,::-1,::-1])
            )

            I_SF_w_thy = np.real(
                np.einsum('sxwy,sxwy->wy', OmegaSF_z_psxwThy[0,:,:,:,:], OmegaSF_z_psxwThy[1,:,:,::-1,::-1])
            )

            # I_Pump_w_thy = np.real(
            #     np.einsum('xwy,xwy->wy', OmegaPump_z_xwThy, np.conj(OmegaPump_z_xwThy[:,::-1,::-1]))
            # )
            I_Pump_w_thy = np.real(
                np.einsum('xwy,xwy->wy', OmegaPump_z_xwThy, np.conj(OmegaPump_z_xwThy))
            )



            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            plt.suptitle(f'z= {float(z):.0f} um')

            plotting_w_thy(I_Pump_w_thy, extent_w_thy, thy_show_min, thy_show_max, w_show_min, w_show_max,
                           r'$\int dx I_{pump}(\omega, x,\theta_y,z)\quad (arb.u.)$', ygrid_pad, tgrid_pad)
            plt.sca(ax1)

            plotting_thxy(I_SF_thxy,  extent_thxy, thx_show_min, thx_show_max, thy_show_min, thy_show_max, 
                          r'$\int dt I_{SF}(t,\theta_x,\theta_y,z)\quad (arb.u.)$', xgrid_pad, ygrid_pad)
            plt.sca(ax2)

            plotting_w_thy(I_SF_w_thy, extent_w_thy, thy_show_min, thy_show_max, w_show_min, w_show_max,
                           r'$\int dx I_{SF}(\omega, x,\theta_y,z)\quad (arb.u.)$', ygrid_pad, tgrid_pad)
            plt.sca(ax3)
            plt.tight_layout()

            plt.savefig(folder_figs_path + '/' + local_name + '.pdf')
            plt.savefig(folder_figs_path + '/' + local_name + '.png')
            np.savez(folder_data_path + '/' + local_name + '.npz', 
                     I_Pump_w_thy=I_Pump_w_thy, I_SF_w_thy=I_SF_w_thy, I_SF_thxy=I_SF_thxy,
                     extent_w_thy=extent_w_thy,
                     thx_show_min=thx_show_min, thx_show_max=thx_show_max,
                     thy_show_min=thx_show_min, thy_show_max=thx_show_max,
                     w_show_min=w_show_min, w_show_max=w_show_max,
                     tgrid_pad=tgrid_pad, xgrid_pad=xgrid_pad, ygrid_pad=ygrid_pad, 
                    )

            plt.show()
            
    def plot_t__Epump(self, X, folder_figs_path, folder_data_path):

             
        SASE_Pfield_txy = X.pump_pulse_3D


        np.savez(folder_data_path + '/' + 'SASE_field' + '.npz', 
                 SASE_Pfield_txy=SASE_Pfield_txy)
        
        name = 't_Epump'

        t_axis = X.t 
        Ep_Re_z0 = np.real(SASE_Pfield_txy[:,int(X.xgrid/2),int(X.ygrid/2)])
        Ep_Im_z0 = np.imag(SASE_Pfield_txy[:,int(X.xgrid/2),int(X.ygrid/2)])
        Ep_Re_zL = np.real(X.Pfield_txyz[:,int(X.xgrid/2),int(X.ygrid/2),-1])
        Ep_Re_zL = Ep_Re_zL / max(Ep_Re_zL) * max(Ep_Re_z0)
        Ep_Im_zL = np.imag(X.Pfield_txyz[:,int(X.xgrid/2),int(X.ygrid/2),-1])
        Ep_Im_zL = Ep_Im_zL / max(Ep_Re_zL) * max(Ep_Re_z0)
        plt.figure()
        plt.plot(t_axis, Ep_Re_z0,label = r'$Re(E_{pump}(t,x=0,y=0,z=0))$')
        plt.plot(t_axis, Ep_Im_z0,label = r'$Im(E_{pump}(t,x=0,y=0,z=0))$')
        plt.plot(t_axis, Ep_Re_zL,'--',label = r'$Re(E_{pump}(t,x=0,y=0,z=L)), norm$')
        plt.plot(t_axis, Ep_Im_zL,'--',label = r'$Im(E_{pump}(t,x=0,y=0,z=L)), norm$')
        plt.xlabel('t (fs)')
        plt.ylabel(r'$E$')
        plt.grid(True)
        plt.legend()
        plt.savefig(folder_figs_path + '/' + name + '.pdf')
        plt.savefig(folder_figs_path + '/' + name + '.png')
        np.savez(folder_data_path + '/' + name + '.npz', 
                 t_axis=t_axis, 
                 Ep_Re_z0=Ep_Re_z0, Ep_Im_z0=Ep_Im_z0, 
                 Ep_Re_zL=Ep_Re_zL, Ep_Im_zL=Ep_Im_zL)
        plt.show()
        
        

        
    def plot_t__Epump_0(self, X):

             
        SASE_Pfield_txy = X.pump_pulse_3D


#         np.savez(folder_data_path + '/' + 'SASE_field' + '.npz', 
#                  SASE_Pfield_txy=SASE_Pfield_txy)
        
        name = 't_Epump_abs_0'

        t_axis = X.t 
        Ep_abs_z0 = np.abs(SASE_Pfield_txy[:,int(X.xgrid/2),int(X.ygrid/2)])
        plt.figure()
        plt.plot(t_axis, Ep_abs_z0,label = r'$Abs(E_{pump}(t,x=0,y=0,z=0))$')
        plt.xlabel('t (fs)')
        plt.ylabel(r'$E$')
        plt.grid(True)
        plt.legend()
        # plt.savefig(folder_figs_path + '/' + name + '.pdf')
        # plt.savefig(folder_figs_path + '/' + name + '.png')
        # np.savez(folder_data_path + '/' + name + '.npz', 
        #          t_axis=t_axis, 
        #          Ep_abs_z0=Ep_abs_z0)
        plt.show()
        
        
        
        
        
        
        
        
        
######################################## seed plots #############################################################        
        
    def plot_t__Iseed_0(self, X):

             
        SASE_seed_field_pstxy = X.sample.seed_field


#         np.savez(folder_data_path + '/' + 'SASE_field' + '.npz', 
#                  SASE_Pfield_txy=SASE_Pfield_txy)
        
        name = 't_Eseed_abs_0'

        t_axis = X.t 
        I_s0 = SASE_seed_field_pstxy[0,0,:,int(X.xgrid/2),int(X.ygrid/2)] * SASE_seed_field_pstxy[1,0,:,int(X.xgrid/2),int(X.ygrid/2)]
        I_s1 = SASE_seed_field_pstxy[0,1,:,int(X.xgrid/2),int(X.ygrid/2)] * SASE_seed_field_pstxy[1,1,:,int(X.xgrid/2),int(X.ygrid/2)]
        plt.figure()
        plt.plot(t_axis, I_s0,label = r'$I_{seed}(s=0,t,x=0,y=0,z=0))$')
        plt.plot(t_axis, I_s1,label = r'$I_{seed}(s=1,t,x=0,y=0,z=0))$')
        plt.xlabel('t (fs)')
        plt.ylabel(r'$I$')
        plt.grid(True)
        plt.legend()
        # plt.savefig(folder_figs_path + '/' + name + '.pdf')
        # plt.savefig(folder_figs_path + '/' + name + '.png')
        # np.savez(folder_data_path + '/' + name + '.npz', 
        #          t_axis=t_axis, 
        #          Ep_abs_z0=Ep_abs_z0)
        plt.show()
        
        
    def plot_t__Eseed(self, X, folder_figs_path, folder_data_path):

             
        # np.savez(folder_data_path + '/' + 'SASE_field' + '.npz', 
        #          SASE_Pfield_txy=SASE_Pfield_txy)
        
        name = 't_Eseed'

        t_axis = X.t 
        
        Omega_pstxy = tools.circular_to_linear(X, X.Omega_pstxyz[:,:,:,:,:,0])
        Omega_t   = Omega_pstxy[0,1,:,int(X.xgrid/2),int(X.ygrid/2)]
        Ep_Re_z0 = np.real(Omega_t)
        Ep_Im_z0 = np.imag(Omega_t)
        Ep_Abs_z0 = np.abs(Omega_t)
        
        Omega_pstxy = tools.circular_to_linear(X, X.Omega_pstxyz[:,:,:,:,:,-1])
        Omega_t   = Omega_pstxy[0,1,:,int(X.xgrid/2),int(X.ygrid/2)]
        Ep_Re_zL = np.real(Omega_t)
        Ep_Re_zL = Ep_Re_zL / max(Ep_Re_zL) * max(Ep_Re_z0)
        Ep_Im_zL = np.imag(Omega_t)
        Ep_Im_zL = Ep_Im_zL / max(Ep_Re_zL) * max(Ep_Re_z0)
        Ep_Abs_zL = np.abs(Omega_t)
        Ep_Abs_zL = Ep_Abs_zL / max(Ep_Re_zL) * max(Ep_Re_z0)
        plt.figure()
        plt.plot(t_axis, Ep_Re_z0,label = r'$Re(E_{seed,y}(t,x=0,y=0,z=0))$')
        plt.plot(t_axis, Ep_Im_z0,label = r'$Im(E_{seed,y}(t,x=0,y=0,z=0))$')
        plt.plot(t_axis, Ep_Abs_z0,label = r'$Abs(E_{seed,y}(t,x=0,y=0,z=0))$')
        plt.plot(t_axis, Ep_Re_zL,'--',label = r'$Re(E_{seed,y}(t,x=0,y=0,z=L)), norm$')
        plt.plot(t_axis, Ep_Im_zL,'--',label = r'$Im(E_{seed,y}(t,x=0,y=0,z=L)), norm$')
        plt.plot(t_axis, Ep_Abs_zL,'--',label = r'$Abs(E_{seed,y}(t,x=0,y=0,z=L)), norm$')
        plt.xlabel('t (fs)')
        plt.ylabel(r'$E$')
        plt.grid(True)
        plt.legend()
        plt.savefig(folder_figs_path + '/' + name + '.pdf')
        plt.savefig(folder_figs_path + '/' + name + '.png')
        np.savez(folder_data_path + '/' + name + '.npz', 
                 t_axis=t_axis, 
                 Ep_Re_z0=Ep_Re_z0, Ep_Im_z0=Ep_Im_z0, 
                 Ep_Re_zL=Ep_Re_zL, Ep_Im_zL=Ep_Im_zL)
        plt.show()
        
        
        Omega_pstxy = tools.circular_to_linear(X, X.Omega_pstxyz[:,:,:,:,:,0])
        Omega_t   = Omega_pstxy[0,0,:,int(X.xgrid/2),int(X.ygrid/2)]
        Ep_Re_z0 = np.real(Omega_t)
        Ep_Im_z0 = np.imag(Omega_t)
        
        Omega_pstxy = tools.circular_to_linear(X, X.Omega_pstxyz[:,:,:,:,:,-1])
        Omega_t   = Omega_pstxy[0,0,:,int(X.xgrid/2),int(X.ygrid/2)]
        Ep_Re_zL = np.real(Omega_t)
        Ep_Re_zL = Ep_Re_zL / max(Ep_Re_zL) * max(Ep_Re_z0)
        Ep_Im_zL = np.imag(Omega_t)
        Ep_Im_zL = Ep_Im_zL / max(Ep_Re_zL) * max(Ep_Re_z0)
        plt.figure()
        plt.plot(t_axis, Ep_Re_z0,label = r'$Re(E_{seed,x}(t,x=0,y=0,z=0))$')
        plt.plot(t_axis, Ep_Im_z0,label = r'$Im(E_{seed,x}(t,x=0,y=0,z=0))$')
        plt.plot(t_axis, Ep_Re_zL,'--',label = r'$Re(E_{seed,x}(t,x=0,y=0,z=L)), norm$')
        plt.plot(t_axis, Ep_Im_zL,'--',label = r'$Im(E_{seed,x}(t,x=0,y=0,z=L)), norm$')
        plt.xlabel('t (fs)')
        plt.ylabel(r'$E$')
        plt.grid(True)
        plt.legend()
        plt.savefig(folder_figs_path + '/' + name + '.pdf')
        plt.savefig(folder_figs_path + '/' + name + '.png')
        np.savez(folder_data_path + '/' + name + '.npz', 
                 t_axis=t_axis, 
                 Ep_Re_z0=Ep_Re_z0, Ep_Im_z0=Ep_Im_z0, 
                 Ep_Re_zL=Ep_Re_zL, Ep_Im_zL=Ep_Im_zL)
        plt.show()

    def plot_z_t__Jseed__rho_g__rho_gg__rho_ee(self, X, folder_figs_path, folder_data_path):

        name = 'z_t__Jseed__rho_g__rho_gg__rho_ee'
        
        J_Omega_tz = np.einsum('stz,stz->tz', 
                               X.Omega_pstxyz[0,:,:,int(X.xgrid/2),int(X.ygrid/2),:], 
                               X.Omega_pstxyz[1,:,:,int(X.xgrid/2),int(X.ygrid/2),:]) / X.flux_factor

        j_tz    = X.convert_pump_phnm2fs_Wcm2 * J_Omega_tz
        rho0_tz = X.rho_0_3D[:,int(X.xgrid/2),int(X.ygrid/2),:]
        E_tz    = np.einsum('ijs, jktz, kis-> tz', X.Tijs_minus, X.rho_ijtxyz[:,:,:,int(X.xgrid/2),int(X.ygrid/2),:], X.Tijs_plus)
        G_tz    = np.einsum('ijs, jktz, kis-> tz', X.Tijs_plus , X.rho_ijtxyz[:,:,:,int(X.xgrid/2),int(X.ygrid/2),:], X.Tijs_minus)
        extentV = [0,X.tmax,0,X.zmax/1.0e3]

        plt.figure(figsize=(12, 8))
        plt.suptitle('x=0,y=0')
        plt.subplot(2, 2, 1)
        plt.title(r'$j_{seed}$ (W/cm$^2$)')
        plt.imshow(np.abs(j_tz.T), aspect='auto', origin='lower', extent=extentV)
        plt.xlabel('t (fs)')
        plt.ylabel(r'z ($\mu m$)')
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.title(r'$\rho_{ground}$')
        plt.imshow(np.abs(rho0_tz.T), aspect='auto', origin='lower', extent=extentV)
        plt.xlabel('t (fs)')
        plt.ylabel(r'z ($\mu m$)')
        plt.colorbar()

        plt.subplot(2, 2, 3)
        plt.title(r'$\rho_{gg}$')
        plt.imshow(np.real(G_tz.T), aspect='auto', origin='lower', extent=extentV)
        plt.xlabel('t (fs)')
        plt.ylabel(r'z ($\mu m$)')
        plt.colorbar()
        
        
        plt.subplot(2, 2, 4)
        plt.title(r'$\rho_{ee}$')
        plt.imshow(np.real(E_tz.T), aspect='auto', origin='lower', extent=extentV)
        plt.xlabel('t (fs)')
        plt.ylabel(r'z ($\mu m$)')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(folder_figs_path + '/' + name + '.pdf')
        plt.savefig(folder_figs_path + '/' + name + '.png')
        np.savez(folder_data_path + '/' + name + '.npz', j_tz=j_tz, rho0_tz=rho0_tz, E_tz=E_tz, extentV=extentV)
        plt.show()
        
        
    def plot_z__N_seed_ph(self, X, folder_figs_path, folder_data_path):

        name = 'z__Nph'

        nphot_z = X.dx * X.dy * X.dt * np.einsum('stxyz,stxyz->z', 
                               X.Omega_pstxyz[0,:,:,:,:], 
                               X.Omega_pstxyz[1,:,:,:,:]) / X.flux_factor
        
        N_seed_photons = X.E_seed_uJ * 1e-6 / (X.hwKalpha1N * sp_const.e)

        z_axis = 1e-3 * X.z

        plt.figure()
        plt.plot(z_axis, nphot_z, '-', label = r'$N_{seed}$')
        plt.plot(z_axis, nphot_z*0 + N_seed_photons, '-', label = r'$N_{seed}(z=0)$')
        # plt.plot(z_axis, nph_pump_z, '-', label = r'$N^{pump}$')
        # plt.yscale('log')
        plt.xlabel('z (um)')
        plt.ylabel('Photon number')
        plt.grid(True)
        plt.legend()

        plt.savefig(folder_figs_path + '/' + name + '.pdf')
        plt.savefig(folder_figs_path + '/' + name + '.png')
        # np.savez(folder_data_path + '/' + name + '.npz', nphot1=nphot1, nphot2=nphot2, nph_pump_z=nph_pump_z, z_axis=z_axis)
        plt.show()
        
    def plot_t__jseed_rho_g_rho_ee(self, X, folder_figs_path, folder_data_path, nz):
        
        z_str_ar = [f'{(1e-3 * num):.3f}' for num in np.linspace(X.dz, X.zmax, nz)]
        name = "t__jseed-rho_g-rho_ee"
        np.savetxt(folder_data_path + '/' + name + '_zar', z_str_ar, fmt='%s')
        
        for z in z_str_ar:

            local_name = name + '_z=' + z + 'um'

            zint = np.abs(X.z - 1e3 * float(z)).argmin()

            I_t = np.einsum('stxy,stxy->t', X.Omega_pstxyz[0, :, :, :, :, zint], X.Omega_pstxyz[1, :, :, :, :, zint])
            E_t = np.einsum('ijs, jkt, kis-> t', X.Tijs_minus, X.rho_ijtxyz[:,:,:,int(X.xgrid/2),int(X.ygrid/2),zint], X.Tijs_plus)
            G_t = np.einsum('ijs, jkt, kis-> t', X.Tijs_plus, X.rho_ijtxyz[:,:,:,int(X.xgrid/2),int(X.ygrid/2),zint], X.Tijs_minus)
            P_t = X.P_pstxyz[0,0,:,int(X.xgrid/2),int(X.ygrid/2),zint]
            # j_t = X.j_3D[:,int(X.xgrid/2),int(X.ygrid/2),zint]
            r0_t = X.rho_0_3D[:,int(X.xgrid/2),int(X.ygrid/2),zint]
            t_axis = X.t

            plt.figure()
            # plt.plot(t_axis, j_t/max(j_t)*max(np.real(E_t)),label = r'$j(x=y=0)$, norm.')
            plt.plot(t_axis, np.real(I_t)/max(np.real(I_t))*max(np.real(G_t)), label = r'$\int dx dy I(x,y)$, norm.')
            # plt.plot(t_axis, np.imag(I_t)/max(np.real(I_t))*max(np.real(E_t)), label = r'Im I, norm.')
            plt.plot(t_axis, np.real(G_t),'--', label = r'$\rho_{gg}(x=y=0)$')
            plt.plot(t_axis, np.real(E_t),'--', label = r'$\rho_{ee}(x=y=0)$')
            plt.plot(t_axis, np.real(P_t),'--', label = r'Re$\rho_{eg}(x=y=0)$')
            plt.plot(t_axis, np.imag(P_t),'--', label = r'Im$\rho_{eg}(x=y=0)$')
            plt.plot(t_axis, r0_t/max(r0_t)*max(np.real(G_t)),'--', label = r'$\rho_{g}(x=y=0)$, norm.')
            plt.xlabel('t (fs)')
            plt.ylabel(r'$\rho_{ee,gg}$')
            plt.title(f'z={float(z):.0f} um')
            plt.grid(True)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.ylim(-0.05*max(np.real(G_t)), 1.05*max(np.real(G_t)))

            plt.savefig(folder_figs_path + '/' + local_name + '.pdf')
            plt.savefig(folder_figs_path + '/' + local_name + '.png')
            # np.savez(folder_data_path + '/' + local_name + '.npz', I_t=I_t, E_t=E_t, G_t=G_t, j_t=j_t, r0_t=r0_t, t_axis=t_axis, zint=zint)
            plt.show()
            
            
            
    def plot_seed_thxy__w_thx_I_j(self, X, folder_figs_path, folder_data_path, nz, w_show, th_show):

        z_str_ar = [f'{(1e-3 * num):.3f}' for num in np.linspace(X.dz, X.zmax, nz)]
        name = "w-thx__I-j"
        np.savetxt(folder_data_path + '/' + name + '_zar', z_str_ar, fmt='%s')        

        tpad = 1000
        xpad = 64
        ypad = 64
        
        thx_show_min = -th_show
        thx_show_max = th_show
        thy_show_min = -th_show
        thy_show_max = th_show
        w_show_min = -w_show
        w_show_max = w_show


        def FFT_thxy(field_pstxy):
            coeffs = (2.0 * np.pi / X.k0, 2.0 * np.pi / X.k0)
            shifts = (X.xmax + xpad * X.dx, X.ymax + ypad * X.dy)
            pad_shape = [(0, 0), (0, 0), (0, 0), (xpad, xpad), (ypad, ypad)]
            domains_sim, meshes_sim, step_sizes_sim = X.optics.nd_kspace(coeffs, (X.xgrid, X.ygrid), (xpad, ypad), (X.dx, X.dy))
            phasors = [np.exp(1j * 2.0 * np.pi * mesh * shift / coeff) for coeff, mesh, shift in zip(coeffs, meshes_sim, shifts)]

            field_pstxy_pad = X.optics.my_pad(field_pstxy, pad_shape)
            field_fft_pad_phased = X.optics.my_fft_phased(field_pstxy_pad, (3, 4), phasors)

            extent_thxy = [
                1e3*np.min(domains_sim[0]), 1e3*np.max(domains_sim[0]),
                1e3*np.min(domains_sim[1]), 1e3*np.max(domains_sim[1])]

            return extent_thxy, field_fft_pad_phased 


        def FFT_w_thy(field_pstxy):
            coeffs = (2.0 * np.pi * X.hbar, 2.0 * np.pi / X.k0)
            shifts = (X.t_pump_max + tpad * X.dt, X.ymax + ypad * X.dy)
            field_psxty = np.einsum('pstxy->psxty',field_pstxy)
            pad_shape = [(0, 0), (0, 0), (0, 0), (tpad, tpad), (ypad, ypad)]
            domains_sim, meshes_sim, step_sizes_sim = X.optics.nd_kspace(coeffs, (X.tgrid, X.ygrid), (tpad, ypad), (X.dt, X.dy))
            phasors = [np.exp(1j * 2.0 * np.pi * mesh * shift / coeff) for coeff, mesh, shift in zip(coeffs, meshes_sim, shifts)]

            field_psxty_pad = X.optics.my_pad(field_psxty, pad_shape)
            field_fft_pad_phased = X.optics.my_fft_phased(field_psxty_pad, (3, 4), phasors)

            extent_w_thy = [
                np.min(domains_sim[0][tpad:X.tgrid + tpad]), np.max(domains_sim[0][tpad:X.tgrid + tpad]),
                1e3*np.min(domains_sim[1]), 1e3*np.max(domains_sim[1])]

            return extent_w_thy, field_fft_pad_phased 


        def FFT_pump_w_thy(field_txy):
            coeffs = (2.0 * np.pi * X.hbar, 2.0 * np.pi / X.k0)
            shifts = (X.t_pump_max + tpad * X.dt, X.ymax + ypad * X.dy)
            field_xty = np.einsum('txy->xty',field_txy)
            pad_shape = [(0, 0), (tpad, tpad), (ypad, ypad)]
            domains_sim, meshes_sim, step_sizes_sim = X.optics.nd_kspace(coeffs, (X.tgrid, X.ygrid), (tpad, ypad), (X.dt, X.dy))
            phasors = [np.exp(1j * 2.0 * np.pi * mesh * shift / coeff) for coeff, mesh, shift in zip(coeffs, meshes_sim, shifts)]

            field_xty_pad = X.optics.my_pad(field_xty, pad_shape)
            field_fft_pad_phased = X.optics.my_fft_phased(field_xty_pad, (1, 2), phasors)

            extent_w_thy = [
                np.min(domains_sim[0][tpad:X.tgrid + tpad]), np.max(domains_sim[0][tpad:X.tgrid + tpad]),
                1e3*np.min(domains_sim[1]), 1e3*np.max(domains_sim[1])]

            return extent_w_thy, field_fft_pad_phased


        def plotting_thxy(I_thxy, extent_thxy, thx_show_min, thx_show_max, thy_show_min, thy_show_max, name, xgrid_pad, ygrid_pad):

            theta_x_ar = np.linspace(extent_thxy[0], extent_thxy[1], xgrid_pad)
            theta_y_ar = np.linspace(extent_thxy[2], extent_thxy[3], ygrid_pad)

            dtheta_x = theta_x_ar[1] - theta_x_ar[0] 
            dtheta_y = theta_y_ar[1] - theta_y_ar[0] 
            I_thy = dtheta_x * np.einsum('xy -> y', I_thxy)
            I_thx = dtheta_y * np.einsum('xy -> x', I_thxy)

            thx_FWHM = tools.find_fwhm(theta_x_ar, I_thx)
            thy_FWHM = tools.find_fwhm(theta_y_ar, I_thy)
            plt.imshow(I_thxy.T, origin='lower', extent=extent_thxy,aspect='auto')
            plt.plot(theta_x_ar, I_thx / max(I_thx) * (thy_show_max - thy_show_min) * 0.3 + thy_show_min, color = 'blue')
            plt.plot(I_thy / max(I_thy) * (thx_show_max - thx_show_min) * 0.3 + thx_show_min, theta_y_ar, color = 'red')
            plt.ylabel(r'$\theta_y$ (mrad)')
            plt.xlabel(r'$\theta_x$ (mrad)')
            plt.title(name + '\n' + r'$FWHM_{\theta_x}$=' + f"{thx_FWHM:.1f}" + r' mrad, $FWHM_{\theta_y}$=' + f"{thy_FWHM:.1f} mrad")
            plt.ylim(thy_show_min, thy_show_max)
            plt.xlim(thx_show_min, thx_show_max)
            plt.colorbar()


        def plotting_w_thy(I_w_thy, extent_w_thy, thy_show_min, thy_show_max, w_show_min, w_show_max, name, ygrid_pad, tgrid_pad):

            womega_ar  = np.linspace(extent_w_thy[0], extent_w_thy[1], tgrid_pad)
            theta_y_ar = np.linspace(extent_w_thy[2], extent_w_thy[3], ygrid_pad)

            dtheta_y = theta_y_ar[1] - theta_y_ar[0] 
            dwomega = womega_ar[1] - womega_ar[0] 
            I_w = np.real(dtheta_y * np.einsum('wa -> w', I_w_thy))
            I_thy = np.real(dwomega * np.einsum('wa -> a', I_w_thy))

            thy_FWHM = tools.find_fwhm(theta_y_ar, I_thy)
            w_FWHM   = tools.find_fwhm(womega_ar, I_w)
            plt.imshow(I_w_thy.T, origin='lower', extent=extent_w_thy,aspect='auto')
            plt.plot(womega_ar, I_w / max(I_w) * (thy_show_max - thy_show_min) * 0.3 + thy_show_min, color = 'blue')
            plt.plot(I_thy / max(I_thy) * (w_show_max - w_show_min) * 0.3 + w_show_min, theta_y_ar, color = 'red')
            plt.ylabel(r'$\theta_y$ (mrad)')
            plt.xlabel(r'$\omega$ (eV)')
            plt.title(name + '\n' + r'$FWHM_{\omega}$=' + f"{w_FWHM:.1f}" + r' eV, $FWHM_{\theta_y}$=' + f"{thy_FWHM:.1f} mrad")
            plt.ylim(thy_show_min, thy_show_max)
            plt.xlim(w_show_min, w_show_max)
            plt.colorbar()

        
        for z in z_str_ar:

            local_name = name + '_z=' + z + 'um'

            zint = np.abs(X.z - 1e3 * float(z)).argmin()

            xgrid_pad = X.xgrid + 2*xpad
            ygrid_pad = X.ygrid + 2*ypad
            tgrid_pad = X.tgrid + 2*tpad

            OmegaSF_z_pstxy = X.Omega_pstxyz[:, :, :, :, :, zint]
            extent_thxy, OmegaSF_z_pstThxThy = FFT_thxy(OmegaSF_z_pstxy)
            extent_w_thy, OmegaSF_z_psxwThy = FFT_w_thy(OmegaSF_z_pstxy)

            OmegaPump_z_txy = X.Pfield_txyz[:,:,:,zint]#np.sqrt(X.j_3D[:,:,:,zint])
            extent_w_thy, OmegaPump_z_xwThy = FFT_pump_w_thy(OmegaPump_z_txy)


            I_SF_thxy = np.real(
                np.einsum('stxy,stxy->xy', OmegaSF_z_pstThxThy[0,:,:,:,:], OmegaSF_z_pstThxThy[1,:,:,::-1,::-1])
            )

            I_SF_w_thy = np.real(
                np.einsum('sxwy,sxwy->wy', OmegaSF_z_psxwThy[0,:,:,:,:], OmegaSF_z_psxwThy[1,:,:,::-1,::-1])
            )

            # I_Pump_w_thy = np.real(
            #     np.einsum('xwy,xwy->wy', OmegaPump_z_xwThy, np.conj(OmegaPump_z_xwThy[:,::-1,::-1]))
            # )
            # I_Pump_w_thy = np.real(
            #     np.einsum('xwy,xwy->wy', OmegaPump_z_xwThy, np.conj(OmegaPump_z_xwThy))
            # )



            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            plt.suptitle(f'z= {float(z):.0f} um')

            # plotting_w_thy(I_Pump_w_thy, extent_w_thy, thy_show_min, thy_show_max, w_show_min, w_show_max,
            #                r'$\int dx I_{pump}(\omega, x,\theta_y,z)\quad (arb.u.)$', ygrid_pad, tgrid_pad)
            # plt.sca(ax1)

            plotting_thxy(I_SF_thxy,  extent_thxy, thx_show_min, thx_show_max, thy_show_min, thy_show_max, 
                          r'$\int dt I_{seed}(t,\theta_x,\theta_y,z)\quad (arb.u.)$', xgrid_pad, ygrid_pad)
            plt.sca(ax1)

            plotting_w_thy(I_SF_w_thy, extent_w_thy, thy_show_min, thy_show_max, w_show_min, w_show_max,
                           r'$\int dx I_{seed}(\omega, x,\theta_y,z)\quad (arb.u.)$', ygrid_pad, tgrid_pad)
            plt.sca(ax2)
            plt.tight_layout()

            plt.savefig(folder_figs_path + '/' + local_name + '.pdf')
            plt.savefig(folder_figs_path + '/' + local_name + '.png')
            # np.savez(folder_data_path + '/' + local_name + '.npz', 
            #          I_Pump_w_thy=I_Pump_w_thy, I_SF_w_thy=I_SF_w_thy, I_SF_thxy=I_SF_thxy,
            #          extent_w_thy=extent_w_thy,
            #          thx_show_min=thx_show_min, thx_show_max=thx_show_max,
            #          thy_show_min=thx_show_min, thy_show_max=thx_show_max,
            #          w_show_min=w_show_min, w_show_max=w_show_max,
            #          tgrid_pad=tgrid_pad, xgrid_pad=xgrid_pad, ygrid_pad=ygrid_pad, 
            #         )

            plt.show()
            
            
            
    def plot_x_y__Iseed_j_rho_ee(self, X, folder_figs_path, folder_data_path, nz):

            z_str_ar = [f'{(1e-3 * num):.3f}' for num in np.linspace(X.dz, X.zmax, nz)]
            name = "x-y__I-j-rho_ee"
            np.savetxt(folder_data_path + '/' + name + '_zar', z_str_ar, fmt='%s')

            # tint_max_j = np.unravel_index(np.argmax(X.j_3D), X.j_3D.shape)[0]
            # print(f'peak of pump at   {(X.t[tint_max_j]):.3f} fs')

            rho_ee_t = np.real(np.einsum('ijs, jkt, kis-> t', X.Tijs_minus, X.rho_ijtxyz[:,:,:,int(X.xgrid/2),int(X.ygrid/2),1], X.Tijs_plus))
            tint_max_rho_ee = np.unravel_index(np.argmax(rho_ee_t), rho_ee_t.shape)[0]
            print(f'peak of rho_ee at {(X.t[tint_max_rho_ee]):.3f} fs')

            I_tz = np.abs(np.einsum('stxyz,stxyz->tz', X.Omega_pstxyz[0, :, :, :, :, :], X.Omega_pstxyz[1, :, :, :, :, :]))
            tint_max_I = np.unravel_index(np.argmax(I_tz), I_tz.shape)[0]
            print(f'peak of SF at     {(X.t[tint_max_I]):.3f} fs')
            # np.savetxt(folder_data_path + '/' + name + '_tint_max', np.asarray([tint_max_j, tint_max_rho_ee, tint_max_I]), fmt='%d')

            for z in z_str_ar:

                local_name = name + '_z=' + z + 'um'

                zint = np.abs(X.z - 1e3 * float(z)).argmin()

                conv_factor = 1 / (3.0 * X.lambdaKalpha1N**2 * X.Gamma_sp_fsm1N / 8.0 / np.pi)

                I_xy = (X.convert_SF_phnm2fs_Wcm2 * conv_factor * 
                        np.einsum('sxy,sxy->xy', X.Omega_pstxyz[0, :, tint_max_I, :, :, zint], X.Omega_pstxyz[1, :, tint_max_I, :, :, zint])
                       )
                E_xy = np.einsum('ijs, jkxy, kis-> xy', X.Tijs_minus, X.rho_ijtxyz[:,:,tint_max_rho_ee,:,:,zint], X.Tijs_plus)
                G_xy = np.einsum('ijs, jkxy, kis-> xy', X.Tijs_plus, X.rho_ijtxyz[:,:,tint_max_rho_ee,:,:,zint], X.Tijs_minus)
                # j_xy = X.convert_pump_phnm2fs_Wcm2 * X.j_3D[tint_max_j,:,:,zint]
                extentV = [-X.xmax,X.xmax,-X.ymax,X.ymax]
                t_max_I = X.t[tint_max_I]
                # t_max_j = X.t[tint_max_j]
                t_max_rho_ee = X.t[tint_max_rho_ee]

                plt.figure(figsize=(16, 4))
                plt.suptitle(f"z = {float(z):.0f} um") 
                # plt.suptitle(r"z= "+f'{(1e-3 * X.z[zint]):.0f}'+" um, "+r'$N_{ph}^{seed}$'+' = '+f'{Nph_SF:.1e}')

                plt.subplot(1, 4, 1)
                plt.title(r'$I \quad (W/cm^2)$' + f' at {t_max_I:.3f} fs')
                plt.imshow(np.real(I_xy), aspect='auto', origin='lower', extent=extentV)
                plt.xlabel('x (nm)')
                plt.ylabel('y (nm)')
                plt.colorbar()  

                plt.subplot(1, 4, 2)
                plt.title(r'$\rho_{gg}$' + f' at {t_max_rho_ee:.3f} fs')
                plt.imshow(np.real(G_xy), aspect='auto', origin='lower', extent=extentV)
                plt.xlabel('x (nm)')
                plt.ylabel('y (nm)')
                plt.colorbar() 

                plt.subplot(1, 4, 3)
                plt.title(r'$\rho_{ee}$' + f' at {t_max_rho_ee:.3f} fs')
                plt.imshow(np.real(E_xy), aspect='auto', origin='lower', extent=extentV)
                plt.xlabel('x (nm)')
                plt.ylabel('y (nm)')
                plt.colorbar()  
                
                
                I_xy = conv_factor * X.dt * np.einsum('stxy,stxy->xy', X.Omega_pstxyz[0, :, :, :, :, zint], X.Omega_pstxyz[1, :, :, :, :, zint])
                j_xy = X.dt * np.einsum('txy->xy', X.j_3D[:,:,:,zint]) 
                Nph_SF   = np.real(X.dx * X.dy * np.sum(I_xy))
                Nph_pump = X.dx * X.dy * np.sum(j_xy)
                extentV = [-X.xmax,X.xmax,-X.ymax,X.ymax]

                
                plt.suptitle(r"z= "+f'{(1e-3 * X.z[zint]):.0f}'+" um, "+r'$N_{seed}^{SF}$'+' = '+f'{Nph_SF:.1e}')

                plt.subplot(1, 4, 4)
                plt.title(r'$\int dt^{\prime} I(t^{\prime},x,y,z) \quad (ph/nm^2)$')
                plt.imshow(np.real(I_xy), aspect='auto', origin='lower', extent=extentV)
                plt.xlabel('x (nm)')
                plt.ylabel('y (nm)')
                plt.colorbar() 
             
            

                plt.tight_layout()

                plt.savefig(folder_figs_path + '/' + local_name + '.pdf')
                plt.savefig(folder_figs_path + '/' + local_name + '.png')
                # np.savez(folder_data_path + '/' + local_name + '.npz', I_xy=I_xy, E_xy=E_xy, j_xy=j_xy, 
                #          extentV=extentV, t_max_I=t_max_I, t_max_j=t_max_j, t_max_rho_ee=t_max_rho_ee)

                plt.show()
            
