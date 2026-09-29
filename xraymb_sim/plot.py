"""
Plots of one finished run. Everything except plot_complex2d and plot_WT needs the full (t, x, y, z) history,
i.e. a run with keep_z_history = True (the default).

The plot_* methods that take folder_figs_path / folder_data_path save each figure there as PDF and PNG
(and some the plotted data as .npz).
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sp_const

from .analysis import find_fwhm
from .pulses import circular_to_linear


class Plotter:

    def __init__(self):
        matplotlib.rcParams['axes.labelsize'] = 12
        matplotlib.rcParams['xtick.labelsize'] = 12
        matplotlib.rcParams['ytick.labelsize'] = 12
        matplotlib.rcParams['legend.fontsize'] = 12
        self.sunrise_at_SLAC = ["010520", "010c3a", "011959", "033185", "024cab", "017fd1", "0499db", "17baca",
                                "dfc49c", "e65342"]

    # ------------------------------------------------------------------------------------------ generic

    def plot_complex2d(self, img, extent, labels, limits=None, cmap='viridis'):
        """Real part, imaginary part, modulus and phase of a complex 2D array."""
        xmin, xmax, ymin, ymax = extent
        xlabel, ylabel = labels
        xlim_min, xlim_max, ylim_min, ylim_max = extent if limits is None else limits

        plt.figure(figsize=(14, 8))
        for k, (part, title) in enumerate([(np.real, 'Re'), (np.imag, 'Im'), (np.abs, 'Abs'), (np.angle, r'$\phi$')]):
            plt.subplot(2, 2, k + 1)
            plt.imshow(part(img).T, extent=[xmin, xmax, ymin, ymax], cmap=cmap, aspect='auto')
            plt.xlim(xlim_min, xlim_max)
            plt.ylim(ylim_min, ylim_max)
            plt.xlabel(xlabel)
            if k % 2 == 0:
                plt.ylabel(ylabel)
            plt.title(title)
            plt.grid()
            plt.colorbar()
        plt.tight_layout()
        plt.show()

    def plot_WT(self, X, sig_tp, sig_tm, limits=None):
        """Wigner transform of the time signal pair (sig_tp, sig_tm), e.g. a field component and its conjugate,
        with its projections on the energy and time axes."""
        coeff = 2.0 * np.pi * X.hbar
        pad_shape = [(X.tpad, X.tpad)]
        shift = X.t0 + X.tpad * X.dt
        domain_w, mesh_w, dw = X.optics.nd_kspace([coeff], [X.tgrid], [X.tpad], [X.dt])

        sig_tp_pad = X.optics.pad(sig_tp, pad_shape)
        sig_tm_pad = X.optics.pad(sig_tm, pad_shape)

        extent = [np.min(domain_w[0][X.tpad:X.tgrid + X.tpad]), np.max(domain_w[0][X.tpad:X.tgrid + X.tpad]), 0, np.max(X.t)]
        xlim_min, xlim_max, ylim_min, ylim_max = extent if limits is None else limits

        sig_t_corr = np.zeros((X.tgrid + 2 * X.tpad, X.tgrid + 2 * X.tpad), dtype=complex)
        for i in range(0, X.tgrid + 2 * X.tpad):
            j = int(X.tgrid / 2 + X.tpad) - i
            sig_t_corr[:, i] = np.roll(sig_tp_pad, j) * np.roll(sig_tm_pad, -j)

        phasor = np.exp(1j * 2.0 * np.pi * domain_w[0] * shift / coeff)
        sig_corr_fft = X.optics.fft_phased(sig_t_corr, [1], [phasor])
        WT = sig_corr_fft[X.tpad:X.tgrid + X.tpad, X.tpad:X.tgrid + X.tpad]
        WT_proj_w = np.sum(np.real(WT), axis=0)
        WT_proj_t = np.sum(np.real(WT), axis=1)

        scale = 2.5   # height of the projections, in axis units
        plt.figure(figsize=(8, 6))
        plt.imshow(np.real(WT), origin='lower', cmap='GnBu', aspect='auto', extent=extent)
        plt.plot(np.fft.fftshift(domain_w)[0][X.tpad:X.tgrid + X.tpad], scale * WT_proj_w / np.max(WT_proj_w),
                 color='blue', linewidth=2)
        plt.plot(scale * WT_proj_t / np.max(WT_proj_t) + xlim_min, X.t, color='blue', linewidth=2)
        plt.xlabel('Photon energy (eV)')
        plt.ylabel('Time (fs)')
        plt.xlim(xlim_min, xlim_max)
        plt.ylim(ylim_min, ylim_max)
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def plot_reciprocal(self, X, limits_theta=None, limits_theta_w=None):
        """Far field of the transmitted field in both polarisations: angular distribution at the pulse centre,
        and energy against theta_x on axis."""
        field_stxy = X.Omega_qtxy
        coeffs = (2.0 * np.pi * X.hbar, 2.0 * np.pi / X.k0, 2.0 * np.pi / X.k0)
        shifts = (X.t0 + X.tpad * X.dt, X.xmax + X.xpad * X.dx, X.ymax + X.ypad * X.dy)
        pad_shape = [(0, 0), (X.tpad, X.tpad), (X.xpad, X.xpad), (X.ypad, X.ypad)]
        domains, meshes, _ = X.optics.nd_kspace(coeffs, (X.tgrid, X.xgrid, X.ygrid), (X.tpad, X.xpad, X.ypad),
                                                (X.dt, X.dx, X.dy))
        phasors = [np.exp(1j * 2.0 * np.pi * mesh * shift / coeff) for coeff, mesh, shift in zip(coeffs, meshes, shifts)]
        field_fft = X.optics.fft_phased(X.optics.pad(field_stxy, pad_shape), (1, 2, 3), phasors)

        extent_theta = [1e3 * np.min(domains[2]), 1e3 * np.max(domains[2]), 1e3 * np.min(domains[1]), 1e3 * np.max(domains[1])]
        extent_theta_w = [1e3 * np.min(domains[2]), 1e3 * np.max(domains[2]),
                          np.min(domains[0][X.tpad:X.tgrid + X.tpad]), np.max(domains[0][X.tpad:X.tgrid + X.tpad])]
        limits_theta = extent_theta if limits_theta is None else limits_theta
        limits_theta_w = extent_theta_w if limits_theta_w is None else limits_theta_w
        cmap = self.get_continuous_cmap(self.sunrise_at_SLAC)

        plt.figure(figsize=(12, 8))
        for q in range(2):
            plt.subplot(2, 2, 1 + q)
            plt.imshow(np.abs(field_fft[q, int(X.tgrid / 2 + X.tpad), :, :])**2, cmap=cmap, extent=extent_theta, aspect='auto')
            plt.xlabel(r'$\theta_x$ (mrad)')
            plt.ylabel(r'$\theta_y$ (mrad)')
            plt.xlim(limits_theta[0], limits_theta[1])
            plt.ylim(limits_theta[2], limits_theta[3])
            plt.colorbar()
            plt.subplot(2, 2, 3 + q)
            plt.imshow(np.abs(field_fft[q, :, :, int(X.ygrid / 2 + X.ypad)])**2, cmap=cmap, extent=extent_theta_w, aspect='auto')
            plt.xlabel(r'$\theta_x$ (mrad)')
            plt.ylabel('Photon energy (eV)')
            plt.xlim(limits_theta_w[0], limits_theta_w[1])
            plt.ylim(limits_theta_w[2], limits_theta_w[3])
            plt.colorbar()
        plt.tight_layout()
        plt.show()

    def get_continuous_cmap(self, hex_list, float_list=None):
        """Colour map through the hex colours in hex_list, evenly spaced unless float_list (0 to 1) places them.
        Adapted from Kerry Halupka: https://gist.github.com/KerryHalupka/73af99afa8d4a4b7aaf1c05562ed4223"""
        rgb_list = [[int(h.strip('#')[i:i + 2], 16) / 256 for i in (0, 2, 4)] for h in hex_list]
        if not float_list:
            float_list = list(np.linspace(0, 1, len(rgb_list)))
        cdict = {col: [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
                 for num, col in enumerate(['red', 'green', 'blue'])}
        return matplotlib.colors.LinearSegmentedColormap('sunrise', segmentdata=cdict, N=256)

    # ------------------------------------------------------------------------------------------ one run

    def plot_t__Eseed(self, X, folder_figs_path, folder_data_path):
        """The y- and x-polarised field against time at the beam centre, entering and leaving the sample
        (the exit field rescaled to the entrance field's peak)."""
        name = 't_Eseed'
        cx, cy = int(X.xgrid / 2), int(X.ygrid / 2)
        field_0 = circular_to_linear(X, X.Omega_pstxyz[:, :, :, :, :, 0])
        field_L = circular_to_linear(X, X.Omega_pstxyz[:, :, :, :, :, -1])

        for s, axis in ((1, 'y'), (0, 'x')):
            E_z0 = field_0[0, s, :, cx, cy]
            E_zL = field_L[0, s, :, cx, cy]
            scale = max(np.real(E_z0)) / max(np.real(E_zL))
            plt.figure()
            plt.plot(X.t, np.real(E_z0), label=rf'$Re(E_{{seed,{axis}}}(t,x=0,y=0,z=0))$')
            plt.plot(X.t, np.imag(E_z0), label=rf'$Im(E_{{seed,{axis}}}(t,x=0,y=0,z=0))$')
            plt.plot(X.t, np.real(E_zL) * scale, '--', label=rf'$Re(E_{{seed,{axis}}}(t,x=0,y=0,z=L))$, norm')
            plt.plot(X.t, np.imag(E_zL) * scale, '--', label=rf'$Im(E_{{seed,{axis}}}(t,x=0,y=0,z=L))$, norm')
            plt.xlabel('t (fs)')
            plt.ylabel(r'$E$')
            plt.grid(True)
            plt.legend()
            plt.savefig(f'{folder_figs_path}/{name}_{axis}.pdf')
            plt.savefig(f'{folder_figs_path}/{name}_{axis}.png')
            np.savez(f'{folder_data_path}/{name}_{axis}.npz', t_axis=X.t, E_z0=E_z0, E_zL=E_zL)
            plt.show()

    def plot_z_t__Jseed__rho_g__rho_gg__rho_ee(self, X, folder_figs_path, folder_data_path):
        """Intensity, ground-state population, 2p-hole and 1s-hole populations at the beam centre against
        time and depth."""
        name = 'z_t__Jseed__rho_g__rho_gg__rho_ee'
        cx, cy = int(X.xgrid / 2), int(X.ygrid / 2)
        J_Omega_tz = np.einsum('stz,stz->tz', X.Omega_pstxyz[0, :, :, cx, cy, :],
                               X.Omega_pstxyz[1, :, :, cx, cy, :]) / X.flux_factor
        j_tz = X.convert_SF_phnm2fs_Wcm2 * J_Omega_tz
        rho0_tz = X.rho_ground_txyz[:, cx, cy, :]
        E_tz = np.einsum('ijs, jktz, kis-> tz', X.Tijs_minus, X.rho_ijtxyz[:, :, :, cx, cy, :], X.Tijs_plus)
        G_tz = np.einsum('ijs, jktz, kis-> tz', X.Tijs_plus, X.rho_ijtxyz[:, :, :, cx, cy, :], X.Tijs_minus)
        extent = [0, X.tmax, 0, X.zmax / 1.0e3]

        plt.figure(figsize=(12, 8))
        plt.suptitle('x=0,y=0')
        for k, (img, title) in enumerate([(np.abs(j_tz), r'$j_{seed}$ (W/cm$^2$)'), (np.abs(rho0_tz), r'$\rho_{ground}$'),
                                          (np.real(G_tz), r'$\rho_{gg}$'), (np.real(E_tz), r'$\rho_{ee}$')]):
            plt.subplot(2, 2, k + 1)
            plt.title(title)
            plt.imshow(img.T, aspect='auto', origin='lower', extent=extent)
            plt.xlabel('t (fs)')
            plt.ylabel(r'z ($\mu m$)')
            plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'{folder_figs_path}/{name}.pdf')
        plt.savefig(f'{folder_figs_path}/{name}.png')
        np.savez(f'{folder_data_path}/{name}.npz', j_tz=j_tz, rho0_tz=rho0_tz, E_tz=E_tz, extentV=extent)
        plt.show()

    def plot_z__N_seed_ph(self, X, folder_figs_path, folder_data_path):
        """Photon number of the pulse against depth."""
        name = 'z__Nph'
        nphot_z = X.dx * X.dy * X.dt * np.einsum('stxyz,stxyz->z', X.Omega_pstxyz[0], X.Omega_pstxyz[1]) / X.flux_factor
        N_seed_photons = X.E_seed_uJ * 1e-6 / (X.hwKalpha1N * sp_const.e)
        z_axis = 1e-3 * X.z

        plt.figure()
        plt.plot(z_axis, np.real(nphot_z), '-', label=r'$N_{seed}$')
        plt.plot(z_axis, np.real(nphot_z) * 0 + N_seed_photons, '-', label=r'$N_{seed}(z=0)$')
        plt.xlabel('z (um)')
        plt.ylabel('Photon number')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{folder_figs_path}/{name}.pdf')
        plt.savefig(f'{folder_figs_path}/{name}.png')
        plt.show()

    def plot_t__jseed_rho_g_rho_ee(self, X, folder_figs_path, folder_data_path, nz):
        """At nz depths: intensity, populations and the 1s-2p coherence at the beam centre against time."""
        z_str_ar = [f'{(1e-3 * num):.3f}' for num in np.linspace(X.dz, X.zmax, nz)]
        name = "t__jseed-rho_g-rho_ee"
        np.savetxt(f'{folder_data_path}/{name}_zar', z_str_ar, fmt='%s')
        cx, cy = int(X.xgrid / 2), int(X.ygrid / 2)

        for z in z_str_ar:
            zint = np.abs(X.z - 1e3 * float(z)).argmin()
            rho_ijt = X.rho_ijtxyz[:, :, :, cx, cy, zint]
            I_t = np.einsum('stxy,stxy->t', X.Omega_pstxyz[0, :, :, :, :, zint], X.Omega_pstxyz[1, :, :, :, :, zint])
            E_t = np.einsum('ijs, jkt, kis-> t', X.Tijs_minus, rho_ijt, X.Tijs_plus)
            G_t = np.einsum('ijs, jkt, kis-> t', X.Tijs_plus, rho_ijt, X.Tijs_minus)
            P_t = np.einsum('ij,jit->t', X.Tijs_minus[:, :, 0], rho_ijt)
            r0_t = X.rho_ground_txyz[:, cx, cy, zint]
            G_max = max(np.real(G_t))

            plt.figure()
            plt.plot(X.t, np.real(I_t) / max(np.real(I_t)) * G_max, label=r'$\int dx dy I(x,y)$, norm.')
            plt.plot(X.t, np.real(G_t), '--', label=r'$\rho_{gg}(x=y=0)$')
            plt.plot(X.t, np.real(E_t), '--', label=r'$\rho_{ee}(x=y=0)$')
            plt.plot(X.t, np.real(P_t), '--', label=r'Re$\rho_{eg}(x=y=0)$')
            plt.plot(X.t, np.imag(P_t), '--', label=r'Im$\rho_{eg}(x=y=0)$')
            plt.plot(X.t, r0_t / max(r0_t) * G_max, '--', label=r'$\rho_{g}(x=y=0)$, norm.')
            plt.xlabel('t (fs)')
            plt.ylabel(r'$\rho_{ee,gg}$')
            plt.title(f'z={float(z):.0f} um')
            plt.grid(True)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.ylim(-0.05 * G_max, 1.05 * G_max)
            plt.savefig(f'{folder_figs_path}/{name}_z={z}um.pdf')
            plt.savefig(f'{folder_figs_path}/{name}_z={z}um.png')
            plt.show()

    def plot_seed_thxy__w_thx_I_j(self, X, folder_figs_path, folder_data_path, nz, w_show, th_show):
        """At nz depths: the far-field angular distribution of the pulse, and its spectrum against theta_y,
        each with its FWHM."""
        z_str_ar = [f'{(1e-3 * num):.3f}' for num in np.linspace(X.dz, X.zmax, nz)]
        name = "w-thx__I-j"
        np.savetxt(f'{folder_data_path}/{name}_zar', z_str_ar, fmt='%s')
        tpad, xpad, ypad = 1000, 64, 64
        xgrid_pad, ygrid_pad, tgrid_pad = X.xgrid + 2 * xpad, X.ygrid + 2 * ypad, X.tgrid + 2 * tpad

        def fft_phased(field, coeffs, shifts, sizes, pads, steps, axes, pad_shape):
            domains, meshes, _ = X.optics.nd_kspace(coeffs, sizes, pads, steps)
            phasors = [np.exp(1j * 2.0 * np.pi * mesh * shift / coeff) for coeff, mesh, shift in zip(coeffs, meshes, shifts)]
            return domains, X.optics.fft_phased(X.optics.pad(field, pad_shape), axes, phasors)

        for z in z_str_ar:
            zint = np.abs(X.z - 1e3 * float(z)).argmin()
            field_pstxy = X.Omega_pstxyz[:, :, :, :, :, zint]

            k = 2.0 * np.pi / X.k0
            d, field_thxy = fft_phased(field_pstxy, (k, k), (X.xmax + xpad * X.dx, X.ymax + ypad * X.dy),
                                       (X.xgrid, X.ygrid), (xpad, ypad), (X.dx, X.dy), (3, 4),
                                       [(0, 0), (0, 0), (0, 0), (xpad, xpad), (ypad, ypad)])
            extent_thxy = [1e3 * np.min(d[0]), 1e3 * np.max(d[0]), 1e3 * np.min(d[1]), 1e3 * np.max(d[1])]
            d, field_w_thy = fft_phased(np.einsum('pstxy->psxty', field_pstxy), (2.0 * np.pi * X.hbar, k),
                                        (X.t_peak + tpad * X.dt, X.ymax + ypad * X.dy), (X.tgrid, X.ygrid),
                                        (tpad, ypad), (X.dt, X.dy), (3, 4),
                                        [(0, 0), (0, 0), (0, 0), (tpad, tpad), (ypad, ypad)])
            extent_w_thy = [np.min(d[0][tpad:X.tgrid + tpad]), np.max(d[0][tpad:X.tgrid + tpad]),
                            1e3 * np.min(d[1]), 1e3 * np.max(d[1])]

            I_thxy = np.real(np.einsum('stxy,stxy->xy', field_thxy[0], field_thxy[1, :, :, ::-1, ::-1]))
            I_w_thy = np.real(np.einsum('sxwy,sxwy->wy', field_w_thy[0], field_w_thy[1, :, :, ::-1, ::-1]))

            plt.figure(figsize=(10, 5))
            plt.suptitle(f'z= {float(z):.0f} um')

            plt.subplot(1, 2, 1)
            theta_x = np.linspace(extent_thxy[0], extent_thxy[1], xgrid_pad)
            theta_y = np.linspace(extent_thxy[2], extent_thxy[3], ygrid_pad)
            I_thx = (theta_y[1] - theta_y[0]) * I_thxy.sum(axis=1)
            I_thy = (theta_x[1] - theta_x[0]) * I_thxy.sum(axis=0)
            plt.imshow(I_thxy.T, origin='lower', extent=extent_thxy, aspect='auto')
            plt.plot(theta_x, I_thx / max(I_thx) * 2 * th_show * 0.3 - th_show, color='blue')
            plt.plot(I_thy / max(I_thy) * 2 * th_show * 0.3 - th_show, theta_y, color='red')
            plt.xlabel(r'$\theta_x$ (mrad)')
            plt.ylabel(r'$\theta_y$ (mrad)')
            plt.title(r'$\int dt I_{seed}(t,\theta_x,\theta_y,z)$' + '\n' + rf'FWHM$_{{\theta_x}}$ = {find_fwhm(theta_x, I_thx):.1f} mrad, '
                      + rf'FWHM$_{{\theta_y}}$ = {find_fwhm(theta_y, I_thy):.1f} mrad')
            plt.xlim(-th_show, th_show)
            plt.ylim(-th_show, th_show)
            plt.colorbar()

            plt.subplot(1, 2, 2)
            womega = np.linspace(extent_w_thy[0], extent_w_thy[1], tgrid_pad)
            theta_y = np.linspace(extent_w_thy[2], extent_w_thy[3], ygrid_pad)
            I_w = (theta_y[1] - theta_y[0]) * I_w_thy.sum(axis=1)
            I_thy = (womega[1] - womega[0]) * I_w_thy.sum(axis=0)
            plt.imshow(I_w_thy.T, origin='lower', extent=extent_w_thy, aspect='auto')
            plt.plot(womega, I_w / max(I_w) * 2 * th_show * 0.3 - th_show, color='blue')
            plt.plot(I_thy / max(I_thy) * 2 * w_show * 0.3 - w_show, theta_y, color='red')
            plt.xlabel(r'$\omega$ (eV)')
            plt.ylabel(r'$\theta_y$ (mrad)')
            plt.title(r'$\int dx I_{seed}(\omega,x,\theta_y,z)$' + '\n' + rf'FWHM$_\omega$ = {find_fwhm(womega, I_w):.1f} eV, '
                      + rf'FWHM$_{{\theta_y}}$ = {find_fwhm(theta_y, I_thy):.1f} mrad')
            plt.xlim(-w_show, w_show)
            plt.ylim(-th_show, th_show)
            plt.colorbar()

            plt.tight_layout()
            plt.savefig(f'{folder_figs_path}/{name}_z={z}um.pdf')
            plt.savefig(f'{folder_figs_path}/{name}_z={z}um.png')
            plt.show()

    def plot_x_y__Iseed_j_rho_ee(self, X, folder_figs_path, folder_data_path, nz):
        """At nz depths: transverse maps of the intensity at its peak, the 2p- and 1s-hole populations at the
        peak of the 1s population, and the time-integrated photon fluence."""
        z_str_ar = [f'{(1e-3 * num):.3f}' for num in np.linspace(X.dz, X.zmax, nz)]
        name = "x-y__I-j-rho_ee"
        np.savetxt(f'{folder_data_path}/{name}_zar', z_str_ar, fmt='%s')
        cx, cy = int(X.xgrid / 2), int(X.ygrid / 2)

        rho_ee_t = np.real(np.einsum('ijs, jkt, kis-> t', X.Tijs_minus, X.rho_ijtxyz[:, :, :, cx, cy, 1], X.Tijs_plus))
        it_rho_ee = int(np.argmax(rho_ee_t))
        I_tz = np.abs(np.einsum('stxyz,stxyz->tz', X.Omega_pstxyz[0], X.Omega_pstxyz[1]))
        it_I = int(np.unravel_index(np.argmax(I_tz), I_tz.shape)[0])
        print(f'peak of rho_ee at {X.t[it_rho_ee]:.3f} fs, peak of the intensity at {X.t[it_I]:.3f} fs')
        extent = [-X.xmax, X.xmax, -X.ymax, X.ymax]

        for z in z_str_ar:
            zint = np.abs(X.z - 1e3 * float(z)).argmin()
            I_xy = X.convert_SF_phnm2fs_Wcm2 / X.flux_factor * np.einsum(
                'sxy,sxy->xy', X.Omega_pstxyz[0, :, it_I, :, :, zint], X.Omega_pstxyz[1, :, it_I, :, :, zint])
            E_xy = np.einsum('ijs, jkxy, kis-> xy', X.Tijs_minus, X.rho_ijtxyz[:, :, it_rho_ee, :, :, zint], X.Tijs_plus)
            G_xy = np.einsum('ijs, jkxy, kis-> xy', X.Tijs_plus, X.rho_ijtxyz[:, :, it_rho_ee, :, :, zint], X.Tijs_minus)
            F_xy = X.dt / X.flux_factor * np.einsum('stxy,stxy->xy', X.Omega_pstxyz[0, :, :, :, :, zint],
                                                    X.Omega_pstxyz[1, :, :, :, :, zint])
            N_photons = np.real(X.dx * X.dy * np.sum(F_xy))

            plt.figure(figsize=(16, 4))
            plt.suptitle(f"z = {1e-3 * X.z[zint]:.0f} um, " + r'$N_{ph}$' + f' = {N_photons:.1e}')
            panels = [(I_xy, r'$I \quad (W/cm^2)$' + f' at {X.t[it_I]:.3f} fs'),
                      (G_xy, r'$\rho_{gg}$' + f' at {X.t[it_rho_ee]:.3f} fs'),
                      (E_xy, r'$\rho_{ee}$' + f' at {X.t[it_rho_ee]:.3f} fs'),
                      (F_xy, r'$\int dt^{\prime} I(t^{\prime},x,y,z) \quad (ph/nm^2)$')]
            for k, (img, title) in enumerate(panels):
                plt.subplot(1, 4, k + 1)
                plt.title(title)
                plt.imshow(np.real(img), aspect='auto', origin='lower', extent=extent)
                plt.xlabel('x (nm)')
                plt.ylabel('y (nm)')
                plt.colorbar()
            plt.tight_layout()
            plt.savefig(f'{folder_figs_path}/{name}_z={z}um.pdf')
            plt.savefig(f'{folder_figs_path}/{name}_z={z}um.png')
            plt.show()
