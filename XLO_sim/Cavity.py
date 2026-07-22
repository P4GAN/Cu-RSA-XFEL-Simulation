import numpy as np
import yaml
import tools
import Model
#import fourier
import Optics as XLO_optics
import Plot as XLO_plot
import scipy.constants as sp_const
from scipy.interpolate import RegularGridInterpolator
from scipy import optimize
import time
import matplotlib.pyplot as plt



class SVEA_field:
    
    def __init__(self, field_fft, x0, y0, zeff):
        
        self.field = field_fft.copy()
        self.x0 = x0
        self.y0 = y0
        self.zeff = zeff
        
    def drift_propagate(self, params):
        
        dz = params
        self.zeff += dz
            
    def CRL_propagate(self, params):
        
        def Transmission(alpha1, alpha2):
            
            prf = - 1.0 / M * np.exp(-1j * eta * Lp.dCRL * Lp.NCRL)
            return prf * np.exp(-gamma * f / 2.0 / Cavity.k0 * (Ft * alpha1 + Fs * alpha2 / 24.0 * (zL / f)**2))    
        
        def q_grid_before_vectorized(dw, theta_x_s_after, theta_y_s_after):
            
            dww, theta_xx_s_after, theta_yy_s_after = np.meshgrid(dw, theta_x_s_after, theta_y_s_after, indexing='ij')
            q_s_after = np.zeros((Cavity.omega_grid, Cavity.theta_x_grid, Cavity.theta_y_grid, 2))
            q_lens_after = np.zeros((Cavity.omega_grid, Cavity.theta_x_grid, Cavity.theta_y_grid, 2))
            q_s_before = np.zeros((Cavity.omega_grid, Cavity.theta_x_grid, Cavity.theta_y_grid, 2))

            q_s_after[:, :, :, 0] = theta_xx_s_after * ((Cavity.omega0 + dww) / (Cavity.hbar * Cavity.c))
            q_s_after[:, :, :, 1] = theta_yy_s_after * ((Cavity.omega0 + dww) / (Cavity.hbar * Cavity.c))
            
            q_lens_after[:, :, :, 0] = q_s_after[:, :, :, 0] - Cavity.k0 * Lp.dpsi
            q_lens_after[:, :, :, 1] = q_s_after[:, :, :, 1] + Cavity.k0 * Lp.chi

            q_lens_before = -q_lens_after / M
            
            q_s_before[:, :, :, 0] = q_lens_before[:, :, :, 0] + Cavity.k0 * Lp.dpsi
            q_s_before[:, :, :, 1] = q_lens_before[:, :, :, 1] - Cavity.k0 * Lp.chi
                        
            alpha1 = np.einsum('qkli, qkli->qkl', (q_lens_after - q_lens_before), (q_lens_after - q_lens_before))
            alpha2 = np.einsum('qkli, qkli->qkl', q_lens_after, q_lens_after) + np.einsum('qkli, qkli->qkl', q_lens_before, q_lens_before)
            
            return dww, q_s_before[:,:,:,0] / ((Cavity.omega0 + dww) / (Cavity.hbar * Cavity.c)), q_s_before[:,:,:,1] / ((Cavity.omega0 + dww) / (Cavity.hbar * Cavity.c)), alpha1, alpha2
            
        
        Cavity, type_def, geometry, Lens_num = params
        Lp = Lens_parameters(Cavity, type_def, geometry, Lens_num)
        
        eta = Lp.delta - 1j * Lp.beta
        gamma = Lp.beta / Lp.delta
        zcr = np.sqrt(Lp.pitch * Lp.Rcurv / 2.0 / Lp.delta)
        zL = Lp.NCRL * Lp.pitch
        xi = zL / zcr
        f = zcr / np.sin(xi)
        Ft = 0.5 * (1 + xi / np.tan(xi))
        Fs = (1 - np.sin(xi)/xi) / np.cos(0.5 * xi)**2 / xi / np.sin(xi) * 6.0
        t = zcr * np.tan(0.5 * xi)
        zeff_t = 1.0e9 * self.zeff + t
        M = (zeff_t - f) / f
        
        #print('numerical size: ', np.sqrt(gamma * f / Cavity.k0))
        
        dkx = Cavity.dtheta_x * Cavity.k0
        dky = Cavity.dtheta_y * Cavity.k0
        
        kappa_x = 1.0 / dkx * np.sqrt(Cavity.k0 / f / np.abs(M))
        kappa_y = 1.0 / dky * np.sqrt(Cavity.k0 / f / np.abs(M))
        
        old_domain = (Cavity.omega_domain, Cavity.theta_x_domain, Cavity.theta_y_domain)
        
        field_sigma_pol_real = RegularGridInterpolator(old_domain, np.real(self.field[0,:,:,:]), method='linear', bounds_error=False, fill_value=None)
        field_sigma_pol_imag = RegularGridInterpolator(old_domain, np.imag(self.field[0,:,:,:]), method='linear', bounds_error=False, fill_value=None)
        field_pi_pol_real = RegularGridInterpolator(old_domain, np.real(self.field[1,:,:,:]), method='linear', bounds_error=False, fill_value=None)
        field_pi_pol_imag = RegularGridInterpolator(old_domain, np.imag(self.field[1,:,:,:]), method='linear', bounds_error=False, fill_value=None)
        
        w_calc, th_x_calc, th_y_calc, alpha1_calc, alpha2_calc = q_grid_before_vectorized(Cavity.omega_domain, Cavity.theta_x_domain, Cavity.theta_y_domain)
        
        self.field[0, :, :, :] = Transmission(alpha1_calc, alpha2_calc) * (field_sigma_pol_real((w_calc, th_x_calc, th_y_calc)) + 1j * field_sigma_pol_imag((w_calc, th_x_calc, th_y_calc)))
        self.field[1, :, :, :] = Transmission(alpha1_calc, alpha2_calc) * (field_pi_pol_real((w_calc, th_x_calc, th_y_calc)) + 1j * field_pi_pol_imag((w_calc, th_x_calc, th_y_calc)))
        
        self.x0 = -self.x0 / M + Lp.xerr / M + Lp.xerr + Lp.dpsi * (zL - t * (1 - 1.0 / M))
        self.y0 = -self.y0 / M + Lp.yerr / M + Lp.yerr - Lp.chi * (zL - t * (1 - 1.0 / M))
        
        self.zeff = (t - zeff_t / M) / 1.0e9


    
    def crystal_propagate(self, params):
        
        def Reflectivity(alpha, P):
            
            refraction_shift = np.real(Xp.chi0)  / np.sin(2.0 * Xp.thetaB)
            #print('refraction shift: ', refraction_shift)
            
            #print(np.abs(Xp.chih) / np.sin(2.0 * Xp.thetaB))
            #1.2e-5 Si(444)

            eta = (alpha + refraction_shift + np.abs(Xp.chih) / np.sin(2.0 * Xp.thetaB) ) / 2.0 / np.sqrt(Xp.chih * Xp.chimh) / P 
            
            return eta - np.sign(np.real(eta)) * np.sqrt(eta**2 - 1.0)
            
        def q_grid_before_vectorized(dw, theta_x_s_after, theta_y_s_after):
            
            dww, theta_xx_s_after, theta_yy_s_after = np.meshgrid(dw, theta_x_s_after, theta_y_s_after, indexing='ij')
            q_s_after = np.zeros((Cavity.omega_grid, Cavity.theta_x_grid, Cavity.theta_y_grid, 3))
            q_s_after[:, :, :, 0] = theta_xx_s_after * ((Cavity.omega0 + dww) / (Cavity.hbar * Cavity.c))
            q_s_after[:, :, :, 1] = theta_yy_s_after * ((Cavity.omega0 + dww) / (Cavity.hbar * Cavity.c))        
            q_s_after[:, :, :, 2] = ((Cavity.omega0 + dww) / (Cavity.hbar * Cavity.c)) * np.sqrt(1.0 - theta_xx_s_after**2 - theta_yy_s_after**2)
            q_xtal_proj = np.einsum('ij, qklj->qkli', Proj_mat, (np.einsum('ij, qklj->qkli', T_xtal_from_s_after, q_s_after) - H_xtal))
            q_xtal = q_xtal_proj - np.einsum('i, qkl->qkli', ez, np.sqrt(-np.einsum('qkli, qkli->qkl', q_xtal_proj, q_xtal_proj) + ((Cavity.omega0 + dww) / (Cavity.hbar * Cavity.c))**2))
            q_s_before = np.einsum('ij, qklj->qkli', T_xtal_from_s_before.T, q_xtal)
            
            q_xtal_refracted = np.zeros((Cavity.omega_grid, Cavity.theta_x_grid, Cavity.theta_y_grid, 3), dtype=complex)
            q_xtal_refracted[:, :, :, 0] = q_xtal[:, :, :, 0]
            q_xtal_refracted[:, :, :, 1] = q_xtal[:, :, :, 1]
            q_xtal_refracted[:, :, :, 2] = -np.sqrt(((Cavity.omega0 + dww) / (Cavity.hbar * Cavity.c))**2 * (1.0 + Xp.chi0) - q_xtal[:, :, :, 0]**2 - q_xtal[:, :, :, 1]**2)
            
            alpha = (2.0 * np.einsum('qkli, i->qkl', q_xtal_refracted, H_xtal) + np.dot(H_xtal, H_xtal)) / ((Cavity.omega0 + dww) / (Cavity.hbar * Cavity.c))**2
            
            return dww, q_s_before[:,:,:,0] / ((Cavity.omega0 + dww) / (Cavity.hbar * Cavity.c)), q_s_before[:,:,:,1] / ((Cavity.omega0 + dww) / (Cavity.hbar * Cavity.c)), alpha

            
            
        Cavity, type_def, geometry, Xtal_num = params
        Xp = Xtal_parameters(Cavity, type_def, geometry, Xtal_num)
        
        old_domain = (Cavity.omega_domain, Cavity.theta_x_domain, Cavity.theta_y_domain)
        
        field_sigma_pol_real = RegularGridInterpolator(old_domain, np.real(self.field[0,:,:,:]), method='linear', bounds_error=False, fill_value=None)
        field_sigma_pol_imag = RegularGridInterpolator(old_domain, np.imag(self.field[0,:,:,:]), method='linear', bounds_error=False, fill_value=None)
        field_pi_pol_real = RegularGridInterpolator(old_domain, np.real(self.field[1,:,:,:]), method='linear', bounds_error=False, fill_value=None)
        field_pi_pol_imag = RegularGridInterpolator(old_domain, np.imag(self.field[1,:,:,:]), method='linear', bounds_error=False, fill_value=None)
        
        P_sigma = 1.0
        P_pi = np.cos(2.0 * Xp.thetaB)
        Inv_mat = np.eye(3)
        Inv_mat[2,2] *= -1
        Proj_mat = np.eye(3)
        Proj_mat[2,2] = 0
        ez = np.asarray([0, 0, 1])
        
        H_xtal = 2.0 * Cavity.k0 * np.sin(Xp.thetaB) * np.asarray([-np.sin(Xp.mu), 0, np.cos(Xp.mu)])
        
        T_xtal_from_glob = Cavity.T_psi_chi_phi(Xp.psi, Xp.chi, Xp.phi)
        T_s_before_from_glob = Cavity.R_psi(Xp.psi_s_before)
        T_s_after_from_glob = Cavity.R_psi(Xp.psi_s_after)
        T_xtal_from_s_before = T_xtal_from_glob @ T_s_before_from_glob.T
        T_s_after_from_s_before = T_s_after_from_glob @ T_s_before_from_glob.T
        T_xtal_from_s_after = T_xtal_from_s_before @ T_s_after_from_s_before.T

        w_calc, th_x_calc, th_y_calc, alpha_calc = q_grid_before_vectorized(Cavity.omega_domain, Cavity.theta_x_domain, Cavity.theta_y_domain)
        
        self.field[0, :, :, :] = Reflectivity(alpha_calc, P_sigma) * (field_sigma_pol_real((w_calc, th_x_calc, th_y_calc)) + 1j * field_sigma_pol_imag((w_calc, th_x_calc, th_y_calc)))
        self.field[1, :, :, :] = Reflectivity(alpha_calc, P_pi) * (field_pi_pol_real((w_calc, th_x_calc, th_y_calc)) + 1j * field_pi_pol_imag((w_calc, th_x_calc, th_y_calc)))

        self.x0 += 2.0 * self.zeff * (Xp.mu * np.cos(Xp.phi) - Xp.dpsi)
        self.y0 += np.sin(Xp.thetaB) * (Xp.chi + Xp.mu * np.sin(Xp.phi))

        
    
class Xtal_parameters:
    
    def __init__(self, Cavity, type_def, geometry, params):
        
        Xtal_num = params
        
        self.a0 = type_def['a0']
        self.Miller_h = type_def['Miller_h']
        self.Miller_k = type_def['Miller_k']
        self.Miller_l = type_def['Miller_l']

        self.thetaB = np.arcsin(2.0 * np.pi / self.a0 * np.sqrt(self.Miller_h**2 + self.Miller_k**2 + self.Miller_l**2) / 2.0 / Cavity.k0)

        self.chi0 = type_def['chi0_r'] + 1j * type_def['chi0_i']
        self.chih = type_def['chih_r'] + 1j * type_def['chih_i']
        self.chimh = self.chih
        
        self.psi0 = geometry['psi0']
        self.dpsi = geometry['dpsi']
        self.psi = self.psi0 + self.dpsi
        
        self.psi_s_before = Cavity.coord_rotation_angles[Xtal_num]
        self.psi_s_after = Cavity.coord_rotation_angles[Xtal_num + 1]
        
        self.chi = geometry['chi']
        self.phi = geometry['phi']
        self.mu = geometry['mu']

        
class Lens_parameters:
    
    def __init__(self, Cavity, type_def, geometry, params):
        
        Lens_num = params
        self.delta = type_def['delta']
        self.beta = type_def['beta']
        self.pitch = type_def['pitch']
        self.Rcurv = type_def['Rcurv']
        self.dCRL = type_def['dCRL']
        self.NCRL = type_def['NCRL']
        self.dpsi = geometry['dpsi']
        self.chi = geometry['chi']
        self.xerr = geometry['xerr']
        self.yerr = geometry['yerr']
    
    def optimize_NCRL(self, za, zb):
        
        #convert to nm
        za *= 1e9
        zb *= 1e9
        
        eta = self.delta - 1j * self.beta
        gamma = self.beta / self.delta
        zcr = np.sqrt(self.pitch * self.Rcurv / 2.0 / self.delta)
        
        zL0 = 2.0 * zcr * np.arctan((zcr**2 - za * zb + np.sqrt((zcr**2 + za * zb)**2 + zcr**2 * (za - zb)**2)) / (zcr * (za + zb)))
        
        def zL_func(zl):
            
            return (1.0 / (za - zl/2.0 + zcr * np.tan(zl / 2.0 / zcr)) + 1.0 / (zb - zl/2.0 + zcr * np.tan(zl / 2.0 / zcr))) -  np.sin(zl / zcr) / zcr
        
        zL = optimize.brentq(zL_func, 0.75 * zL0, 1.25 * zL0)
        
        return zL / self.pitch
        


class Cavity:

    def __init__(self, X, YAML, debug_mode = True):
        
        
        self.convr = 1.0
        self.config = yaml.load(open(YAML), Loader=yaml.FullLoader)
        
        self.types = self.config['element_types']

        self.c = X.c
        self.hbar = X.hbar
        self.k0 = X.k0
        self.omega0 = X.hwKalpha1N
        
        self.quiet_mode = self.config['quiet_mode']

        self.cavity_tof = self.config['cavity_tof']
        self.cavity_size = self.config['cavity_size']
        self.medium_width = self.config['medium_width']
        self.omega_grid = self.config['omega_grid']
        self.theta_x_grid = self.config['theta_x_grid']
        self.theta_y_grid = self.config['theta_y_grid']
        
        self.bandwidth = self.config['bandwidth']
        
        self.thetaB = np.arcsin(2.0 * np.pi / self.types['Bragg_crystal']['a0'] * np.sqrt(self.types['Bragg_crystal']['Miller_h']**2 + self.types['Bragg_crystal']['Miller_k']**2 + self.types['Bragg_crystal']['Miller_l']**2) / 2.0 / self.k0)
        
        self.Plot = XLO_plot.XLO_plot()
        self.optics = XLO_optics.XLO_optics(X)

        if self.bandwidth == 'auto':
            self.theta_x_max = self.config['theta_x_max']
            self.theta_y_max = self.config['theta_y_max']
            self.omega_max = self.config['omega_max']

        else:
            #what is this for?
            self.theta_x_max = self.config['theta_x_max']
            self.theta_y_max = self.config['theta_y_max']
            self.omega_max = self.config['omega_max']
            
        self.refraction_shift = self.types['Bragg_crystal']['chi0_r']  / np.sin(2.0 * self.thetaB)
        
        domains, meshes, step_sizes = self.optics.nd_space((-self.omega_max, -self.theta_x_max, -self.theta_y_max), (self.omega_max, self.theta_x_max, self.theta_y_max), (self.omega_grid, self.theta_x_grid, self.theta_y_grid))
        self.omega_domain, self.theta_x_domain, self.theta_y_domain = domains
        self.omega_mesh, self.theta_x_mesh, self.theta_y_mesh = meshes
        self.domega, self.dtheta_x, self.dtheta_y = step_sizes
        
        coeffs = (2.0 * np.pi * X.hbar, 2.0 * np.pi / X.k0, 2.0 * np.pi / X.k0)

        if debug_mode:
            print('Cavity simulation is in debug mode')
            #moments = [self.omega_max / 3.0, self.theta_x_max / 1.5, self.theta_y_max / 3.0]
            moments = [1.0, 1e-3, 1e-5]

            simulated_field = np.asarray([tools.Gaussian_from_mesh(X, meshes, moments), tools.Gaussian_from_mesh(X, meshes, moments)])
            
        else:
            shifts_sim = (X.t0 + X.tpad * X.dt, X.xmax + X.xpad * X.dx, X.ymax + X.ypad * X.dy)
            pad_shape = [(0, 0), (X.tpad, X.tpad), (X.xpad, X.xpad), (X.ypad, X.ypad)]
            domains_sim, meshes_sim, step_sizes_sim = self.optics.nd_kspace(coeffs, (X.tgrid, X.xgrid, X.ygrid), (X.tpad, X.xpad, X.ypad), (X.dt, X.dx, X.dy), shifted=True)
            domains_sim_us, meshes_sim_us, step_sizes_sim_us = self.optics.nd_kspace(coeffs, (X.tgrid, X.xgrid, X.ygrid), (X.tpad, X.xpad, X.ypad), (X.dt, X.dx, X.dy), shifted=False)
            self.omega_domain_sim, self.theta_x_domain_sim, self.theta_y_domain_sim = domains_sim

            transform_matrix = np.asarray([[1, -1], [1j, 1j]]) / np.sqrt(2.0) # transformation matrix from circular polarization vectors to Cartesian
            field_stxy = (X.Omega_pstxyz[0, :, :, :, :, -1] + np.conj(X.Omega_pstxyz[1, :, :, :, :, -1])) / 2.0
            field_qtxy = np.einsum('qs, stxy->qtxy', transform_matrix, field_stxy)
            phasors = [np.exp(1j * 2.0 * np.pi * mesh * shift / coeff) for coeff, mesh, shift in zip(coeffs, meshes_sim_us, shifts_sim)]
            field_qtxy_pad = self.optics.my_pad(field_qtxy, pad_shape)

            field_fft_pad_phased = self.optics.my_fft_phased(field_qtxy_pad, (1, 2, 3), phasors) 
            
            simulated_field = np.asarray([self.optics.interpolate_wavefront_3D(field_fft_pad_phased[0,:,:,:], domains_sim, (self.omega_mesh, self.theta_x_mesh, self.theta_y_mesh)), self.optics.interpolate_wavefront_3D(field_fft_pad_phased[1,:,:,:], domains_sim, (self.omega_mesh, self.theta_x_mesh, self.theta_y_mesh))])
            
            #print(np.shape(simulated_field))
        
        self.input_field = simulated_field * self.convr
        self.cavity_fields = []
        
        self.bow_tie_geometry = self.config['bow_tie_geometry']
        self.NCRL_optimize = self.config['NCRL_optimize']

        if self.bow_tie_geometry == 'auto':
            self.qprint('Auto-positioning elements in bow-tie geometry\n')
            self.lattice = self.config['lattice']
            self.lattice = self.auto_bow_tie(self.lattice)
            self.coord_rotation_angles = [0.0, -2.0 * self.thetaB, 0.0, 2.0 * self.thetaB, 0.0]
            
        else:
            self.lattice = self.config['lattice']
            

    def R_psi(self, psi): return np.asarray([[np.cos(psi), 0, -np.sin(psi)], [0, 1, 0], [np.sin(psi), 0, np.cos(psi)]])
    
    def R_chi(self, chi): return np.asarray([[1, 0, 0], [0, np.cos(chi), np.sin(chi)], [0, -np.sin(chi), np.cos(chi)]])
    
    def R_phi(self, phi): return np.asarray([[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    
    def T_psi_chi_phi(self, psi, chi, phi): return self.R_phi(phi) @ self.R_chi(chi) @ self.R_psi(psi)

    def auto_bow_tie(self, lat):           
        
        cav_L = sp_const.c * self.cavity_tof * 1.0e-9
        self.cav_L = cav_L
        Ll = cav_L / (1.0 + 1.0 / (np.cos(np.pi - 2.0 * self.thetaB))) - self.cavity_size
        
        c1pos = (self.cavity_size - self.medium_width) / 2.0
        c2pos = c1pos + (self.cavity_size + Ll) / 2.0 / np.cos(np.pi - 2.0 * self.thetaB)
        c3pos = c2pos + Ll
        c4pos = c3pos + (self.cavity_size + Ll) / 2.0 / np.cos(np.pi - 2.0 * self.thetaB)
        
        for ID, element in lat.items():
            
            if ID == 'crystal1':
                self.qprint(str(ID) + ' auto pos: ' + str(c1pos))
                element['geometry']['spos'] = c1pos
                element['geometry']['psi0'] = 3.0 * np.pi / 2.0 - self.thetaB 
                
            if ID == 'crystal2':
                self.qprint(str(ID) + ' auto pos: ' + str(c2pos))
                element['geometry']['spos'] = c2pos
                element['geometry']['psi0'] = np.pi / 2.0 - self.thetaB  
                
            if ID == 'crystal3':
                self.qprint(str(ID) + ' auto pos: ' + str(c3pos))
                element['geometry']['spos'] = c3pos
                element['geometry']['psi0'] = np.pi / 2.0 + self.thetaB 

                
            if ID == 'crystal4':
                self.qprint(str(ID) + ' auto pos: ' + str(c4pos))
                element['geometry']['spos'] = c4pos
                element['geometry']['psi0'] = -np.pi / 2.0 + self.thetaB 
            
        return lat
    
    
    def auto_screens(self, lat):
        
        active_lat = dict({ID:element for ID, element in lat.items() if element['type'] != 'Screen'})        
        el = 0    
        num_el = len(active_lat)
        pos_el = np.zeros((num_el, 2))
        
        for ID, element in lat.items():
             if element['type'] != 'Screen':
                    for name in element:
                        if name == 'geometry':
                            spos = element[name]['spos']
                            L = element[name]['L']
                            pos_el[el, 0] = spos
                            pos_el[el, 1] = L
                            el += 1 
                            
        for i in range(0, num_el):
            
            s_name_before = 'auto_screen_before' + str(i+1)
            s_name_after = 'auto_screen_after' + str(i+1)
            lat.update({s_name_before:{'type': 'Screen', 'geometry': {'spos': pos_el[i, 0] - 0.001, 'L': 0.0}}})
            lat.update({s_name_after:{'type': 'Screen', 'geometry': {'spos': pos_el[i, 0] + pos_el[i, 1] + 0.001, 'L': 0.0}}})
        
        new_lat = dict(sorted(lat.items(), key = lambda x: x[1]['geometry']['spos']))
                
        return new_lat
        
    
    def update_lens_length(self, lat):
        
        
        for ID, element in lat.items():
            
            if element['type'] == 'CRL':
                if self.NCRL_optimize == 'auto':
                    Lp = Lens_parameters(self, self.types['CRL'], element['geometry'], 0)
                    NCRL = Lp.optimize_NCRL(self.cav_L / 4.0, self.cav_L / 4.0)
                    self.qprint('Optimized number of lenses: ', NCRL)
                    element['geometry']['L'] = (NCRL * self.types['CRL']['pitch']) / 1.0e9
                    self.types['CRL']['NCRL'] = NCRL
                else:
                    element['geometry']['L'] = (self.types['CRL']['NCRL'] * self.types['CRL']['pitch']) / 1.0e9
                
        return lat
    

    def update_lattice_with_drifts(self, lat):

        el = 0    
        num_el = len(lat)
        pos_el = np.zeros((num_el, 2))

        for ID, element in lat.items():
            for name in element:

                if name == 'geometry':
                    spos = element[name]['spos']
                    L = element[name]['L']
                    pos_el[el, 0] = spos
                    pos_el[el, 1] = L        
                    el += 1  


        for i in range(0, num_el-1):

            d_start = pos_el[i, 0] + pos_el[i, 1]
            d_end = pos_el[i + 1, 0]
            d_L = d_end - d_start
            d_name = 'drift' + str(i+1)   
            lat.update({d_name:{'type': 'Drift', 'geometry': {'spos': d_start, 'L': d_L}}})
            
        pos_last = pos_el[-1, 0] + pos_el[-1, 1]            
        lat.update({'drift0':{'type': 'Drift', 'geometry': {'spos': 0.0, 'L': pos_el[0, 0]}}})
        lat.update({'drift' + str(i+2):{'type': 'Drift', 'geometry': {'spos': pos_last, 'L': self.cav_L - pos_last}}})

        new_lat = dict(sorted(lat.items(), key = lambda x: x[1]['geometry']['spos']))
        
        return new_lat
    

    def lattice_processor(self, SVEA_field, plot=False):

        types = self.types
        raw_lat = self.lattice

        lat = dict(sorted(raw_lat.items(), key = lambda x: x[1]['geometry']['spos']))
        lat = self.update_lens_length(lat)
        lat = self.auto_screens(lat)
        lat = self.update_lattice_with_drifts(lat)

        Xtal_num = 0
        Lens_num = 0

        self.qprint('\n')        
        self.qprint('Cavity tracking...')
        self.qprint('\n')        

        self.qprint('Bragg angle: (deg) ' + str(np.degrees(self.thetaB)))
        self.qprint('\n')

        for ID, element in lat.items():
            self.qprint("element ID: " + str(ID))

            el_type = lat[ID]['type']
            self.qprint("element type: " + el_type)

            for name in element:
                if name == 'geometry':
                    el_pos = element[name]['spos']
                    el_L =  element[name]['L']
                    self.qprint('Position: ' + str(el_pos))
                    self.qprint('Length: ' + str(el_L))

            if el_type == 'Bragg_crystal':
                self.qprint('Applying crystal kernel ' + str(Xtal_num+1))
                
                params = [self, self.types['Bragg_crystal'], element['geometry'], Xtal_num]
                SVEA_field.crystal_propagate(params)
                Xtal_num += 1

            if el_type == 'Drift':
                if plot:
                    self.Plot.plot_complex2d(SVEA_field.field[0, int(self.omega_grid/2), :, :], [-1e3*self.theta_x_max, 1e3*self.theta_x_max, -1e3*self.theta_y_max, 1e3*self.theta_y_max], [r'$\theta_x$ (mrad)', r'$\theta_y$ (mrad)'], [-1e3*self.theta_x_max, 1e3*self.theta_x_max, -1e3*self.theta_y_max, 1e3*self.theta_y_max])

                dz =  element['geometry']['L']
                self.qprint('Applying drift kernel by: ' + str(dz))
                #self.drift_cavity_element(SVEA_field, dz)
                SVEA_field.drift_propagate(dz)

            if el_type == 'CRL':
                self.qprint('Applying CRL kernel ' + str(Lens_num+1))
                params = [self, self.types['CRL'], element['geometry'], Lens_num]
                SVEA_field.CRL_propagate(params)
                Lens_num += 1

            if el_type == 'Screen':
                self.qprint('Saving screen data')
                self.cavity_fields.append([SVEA_field.field.copy(), SVEA_field.x0, SVEA_field.y0, SVEA_field.zeff, element['geometry']['spos']])

            self.qprint('\n') 
            self.qprint('zeff: ' + str(SVEA_field.zeff))
            self.qprint('\n') 

        self.qprint('Tracking complete...')
        
        
    def qprint(self, str): 
        
        if self.quiet_mode:
            pass
        else:
            print(str)     
            
            
#     def drift_cavity_element(self, SVEA_field, params):
        
#         dz = params
#         SVEA_field.zeff += dz
        
#         return SVEA_field
    
    
#     def lens_cavity_element(self, SVEA_field, params):
        
#         return SVEA_field * 1.0   
    
    
#     def crystal_cavity_element(self, SVEA_fieldd, params):
        
#         return SVEA_field * 1.0

            


        

#         def q_grid_before_direct(dw, theta_x_s_after, theta_y_s_after):
            
#             qx_s_after = theta_x_s_after * ((Cavity.omega0 + dw) / (Cavity.hbar * Cavity.c))
#             qy_s_after = theta_y_s_after * ((Cavity.omega0 + dw) / (Cavity.hbar * Cavity.c))
#             q_s_after = np.asarray([qx_s_after, qy_s_after, np.sqrt(((Cavity.omega0 + dw) / (Cavity.hbar * Cavity.c))**2 - qx_s_after**2 - qy_s_after**2)])
#             q_xtal_proj = np.dot(Proj_mat, (np.dot(T_xtal_from_s_after, q_s_after) - H_xtal))
#             q_xtal = q_xtal_proj - ez * np.sqrt(((Cavity.omega0 + dw) / (Cavity.hbar * Cavity.c))**2 - np.dot(q_xtal_proj, q_xtal_proj)) 
#             q_s_before = np.dot(T_xtal_from_s_before.T, q_xtal)    
#             q_xtal_refracted = np.asarray([q_xtal[0], q_xtal[1], -np.sqrt(((Cavity.omega0 + dw) / (Cavity.hbar * Cavity.c))**2 * (1 + Xp.chi0) - q_xtal[0]**2 - q_xtal[1]**2)])
            
#             alpha = (2.0 * np.dot(q_xtal_refracted, H_xtal) + np.dot(H_xtal, H_xtal)) / ((Cavity.omega0 + dw) / (Cavity.hbar * Cavity.c))**2            
            
#             return dw, q_s_before[0] / ((Cavity.omega0 + dw) / (Cavity.hbar * Cavity.c)), q_s_before[1] / ((Cavity.omega0 + dw) / (Cavity.hbar * Cavity.c)), alpha
#             ###### make work for mesh