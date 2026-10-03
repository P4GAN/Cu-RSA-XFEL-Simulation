"""
Right-hand sides of the Maxwell-Bloch equations for one z plane at one time step.

Every function returns d/dt of one population (or density matrix) over the transverse (x, y) grid.
The photon fluxes J_minus / J_plus of the two circular polarisations and the free-electron rates are
frozen over each RK4 step. Population feeds between blocks (photoionisation, Auger and Coster-Kronig
decay, electron-impact ionisation) use the start-of-step populations of their source, so every feed
is exactly the loss its source books.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _bloch_rhs_kernel(rho_ijxy, Omega_plus_sxy, Omega_minus_sxy, Tijs_plus, Tijs_minus,
                            Mij, Gamma_sp_Gij, S_ion_Fif, feed_diag_ixy, Delta_ij,
                            J_Omega_minus_xy, J_Omega_plus_xy):
    """
    d rho_ij / dt = -i [H, rho]_ij - M_ij rho_ij - i Delta_ij rho_ij - (gamma_i + gamma_j) / 2 rho_ij
                    + delta_ij (sum_s Gamma_sp G_is rho_ss + external feed_i)
    with H the dipole coupling to the field, M the decay matrix, Delta the detuning from the Kalpha1
    frame, and gamma_i the further-photoionisation rate out of level i.
    """
    nlevel = rho_ijxy.shape[0]
    s_dim = Tijs_plus.shape[2]
    nx = rho_ijxy.shape[2]
    ny = rho_ijxy.shape[3]

    Hint = np.zeros((nlevel, nlevel, nx, ny), dtype=np.complex128)
    for i in range(nlevel):
        for j in range(nlevel):
            for x in range(nx):
                for y in range(ny):
                    acc = 0.0j
                    for s in range(s_dim):
                        acc += Tijs_plus[i, j, s] * Omega_plus_sxy[s, x, y] + Tijs_minus[i, j, s] * Omega_minus_sxy[s, x, y]
                    Hint[i, j, x, y] = 1j * acc

    diag_feed = np.zeros((nlevel, nx, ny), dtype=np.complex128)
    gamma_ion = np.zeros((nlevel, nx, ny), dtype=np.complex128)
    for i in range(nlevel):
        for x in range(nx):
            for y in range(ny):
                diag_sum = 0.0j
                for s in range(nlevel):
                    diag_sum += Gamma_sp_Gij[i, s] * rho_ijxy[s, s, x, y]
                diag_feed[i, x, y] = diag_sum + feed_diag_ixy[i, x, y]
                gamma_ion[i, x, y] = S_ion_Fif[0, i] * J_Omega_minus_xy[x, y] + S_ion_Fif[1, i] * J_Omega_plus_xy[x, y]

    drho = np.zeros((nlevel, nlevel, nx, ny), dtype=np.complex128)
    for i in range(nlevel):
        for j in range(nlevel):
            for x in range(nx):
                for y in range(ny):
                    comm = 0.0j
                    for s in range(nlevel):
                        comm += Hint[i, s, x, y] * rho_ijxy[s, j, x, y] - rho_ijxy[i, s, x, y] * Hint[s, j, x, y]
                    val = comm - Mij[i, j] * rho_ijxy[i, j, x, y]
                    if i == j:
                        val += diag_feed[i, x, y]
                    else:
                        val += -1j * Delta_ij[i, j] * rho_ijxy[i, j, x, y]
                    val += -0.5 * (gamma_ion[i, x, y] + gamma_ion[j, x, y]) * rho_ijxy[i, j, x, y]
                    drho[i, j, x, y] = val

    return drho


def base_block_rhs(t, rho_ijxy, params):
    """Base block, fed from the ground state (photoionisation and EII) and the 2s hole (Auger decay)."""
    X, Omega_psxy, rho_ground_xy, rho_2s_xy, J_Omega_minus_xy, J_Omega_plus_xy, eii_R_xy = params
    feed_diag_ixy = base_block_feed(X, rho_ground_xy, rho_2s_xy, J_Omega_minus_xy, J_Omega_plus_xy, eii_R_xy)
    return _bloch_rhs_kernel(
        rho_ijxy, Omega_psxy[0], Omega_psxy[1], X.Tijs_plus, X.Tijs_minus,
        X.Mij, X.Gamma_sp_Gij, X.S_ion_Fi, feed_diag_ixy, X.Delta_ij,
        J_Omega_minus_xy, J_Omega_plus_xy)


def satellite_block_rhs(t, rho_ijxy, params):
    """One satellite block: the base block's equations with the channel's own widths and detuning."""
    (X, chan, Omega_psxy, rho_base_ijxy, rho_2s_xy, rho_sat_ijxy, J_Omega_minus_xy, J_Omega_plus_xy,
     rho_mid_xy, eii_R_xy) = params
    feed_diag_ixy = satellite_block_feed(X, chan, rho_2s_xy, rho_base_ijxy, rho_sat_ijxy, J_Omega_minus_xy,
                                              J_Omega_plus_xy, rho_mid_xy, eii_R_xy)
    return _bloch_rhs_kernel(
        rho_ijxy, Omega_psxy[0], Omega_psxy[1], X.Tijs_plus, X.Tijs_minus,
        chan.Mij, X.Gamma_sp_Gij, chan.S_ion_Fi, feed_diag_ixy, chan.Delta_ij,
        J_Omega_minus_xy, J_Omega_plus_xy)


def base_block_feed(X, rho_ground_xy, rho_2s_xy, J_Omega_minus_xy, J_Omega_plus_xy, eii_R_xy=None):
    """Population fed into each base-block level: photoionisation of the ground state, 2s-hole Auger
    decay, and electron-impact ionisation of ground atoms (spread evenly over the sublevels)."""
    feed = np.einsum('i,xy->ixy', X.S_ground_Fi[0, :X.nlevel], J_Omega_minus_xy * rho_ground_xy)
    feed += np.einsum('i,xy->ixy', X.S_ground_Fi[1, :X.nlevel], J_Omega_plus_xy * rho_ground_xy)
    feed += np.einsum('i,xy->ixy', np.diag(X.auger_feeding_matrix), rho_2s_xy)
    if eii_R_xy is not None:
        feed += np.einsum('i,xy->ixy', X.ei_L3 / np.sum(X.ei_L3), eii_R_xy[0] * rho_ground_xy)
        feed += np.einsum('i,xy->ixy', X.ei_L2 / np.sum(X.ei_L2), eii_R_xy[1] * rho_ground_xy)
    return feed


def satellite_block_feed(X, chan, rho_2s_xy, rho_base_ijxy, rho_sat_ijxy, J_Omega_minus_xy, J_Omega_plus_xy,
                              rho_mid_xy=None, eii_R_xy=None):
    """
    Population fed into each level of one satellite block. A single-spectator channel is fed by 2s-hole
    Auger decay, by the base block's non-radiative 1s decay, by photoionisation of the spectator shell
    of base-block ions, by the base 2p1/2 Coster-Kronig decay and from the middleman pool. A
    double-spectator channel is fed by the decay of its parent channel (chan.feed_from). Every feed is
    spread evenly over the sublevels of the manifold it lands in, except photoionisation feeds, which keep
    the source sublevel.
    """
    n_base = 6        # the 2p3/2 and 1s levels; 6, 7 are the 2p1/2 levels
    n_L2 = 2
    J_F = (J_Omega_minus_xy, J_Omega_plus_xy)
    ei_L3 = X.ei_L3[:n_base]
    ei_K = X.ei_K[:n_base]
    even_L3 = ei_L3 / np.sum(ei_L3)
    even_K = ei_K / np.sum(ei_K)
    even_L2 = np.ones(n_L2) / n_L2

    feed = np.zeros((X.nlevel,) + rho_2s_xy.shape, dtype=complex)
    rho_K_base_xy = sum(np.real(rho_base_ijxy[i, i]) for i in range(n_base) if ei_K[i] > 0)

    # 2s-hole Auger decay and K-hole KLM-type Auger decay of base-block ions
    feed[:n_base] += np.einsum('i,xy->ixy', even_L3 * chan.Gamma_A_fs, rho_2s_xy)
    feed[:n_base] += np.einsum('i,xy->ixy', even_L3 * chan.Gamma_A_K_fs, rho_K_base_xy)
    feed[n_base:] += np.einsum('i,xy->ixy', (chan.Gamma_A_L2_fs / n_L2) * np.ones(n_L2), rho_2s_xy)
    feed[n_base:] += np.einsum('i,xy->ixy', (chan.Gamma_A_K_to_L2_fs / n_L2) * np.ones(n_L2), rho_K_base_xy)

    # decay of a parent satellite channel (double-spectator channels)
    for parent_index, Gamma_feed_fs, manifold in chan.feed_from:
        parent_rho = rho_sat_ijxy[parent_index]
        if manifold == 'L2':
            parent_pop_xy = sum(np.real(parent_rho[i, i]) for i in range(n_base, n_base + n_L2))
            feed[n_base:] += np.einsum('i,xy->ixy', even_L2, Gamma_feed_fs * parent_pop_xy)
            continue
        src_mask, dst_weight = (ei_L3, even_L3) if manifold == 'lower' else (ei_K, even_K)
        parent_pop_xy = sum(np.real(parent_rho[i, i]) for i in range(n_base) if src_mask[i] > 0)
        feed[:n_base] += np.einsum('i,xy->ixy', dst_weight, Gamma_feed_fs * parent_pop_xy)

    # photoionisation of the spectator shell of a base-block ion, with the base further-ionisation
    # sublevel pattern: rate_i = sum_f sigma[f] pattern[f, i] J[f]
    pattern = X.pi_feed_pattern_Fi

    def pi_rate(sigma_F, i):
        return sum(sigma_F[f] * pattern[f, i] * J_F[f] for f in range(2))

    for i in range(n_base):
        sigma_F = chan.S_feed_2p if ei_L3[i] > 0 else chan.S_feed_1s
        feed[i] += pi_rate(sigma_F, i) * rho_base_ijxy[i, i]
    for offset in range(n_L2):
        i_base = n_base + offset
        feed[i_base] += pi_rate(chan.S_feed_2p1, i_base) * rho_base_ijxy[i_base, i_base]

    # base 2p1/2 -> 2p3/2 + 3d Coster-Kronig (metal only)
    if chan.Gamma_CK_L2_fs:
        rho_L2_base_xy = sum(np.real(rho_base_ijxy[i, i]) for i in range(X.nlevel) if X.ei_L2[i] > 0)
        feed[:n_base] += np.einsum('i,xy->ixy', even_L3 * chan.Gamma_CK_L2_fs, rho_L2_base_xy)

    # middlemen whose 2p or 2s shell is ionised (by photons or electrons; a 2s hole Coster-Kronig decays
    # at once), scaled by this channel's share of the pool
    if chan.mid_share and rho_mid_xy is not None:
        w = chan.mid_share
        feed[:4] += w * np.einsum('i,xy->ixy', X.S_ground_Fi[0, :4], J_Omega_minus_xy * rho_mid_xy)
        feed[:4] += w * np.einsum('i,xy->ixy', X.S_ground_Fi[1, :4], J_Omega_plus_xy * rho_mid_xy)
        rate_2s_xy = (X.S_ground_Fi[0, X.nlevel] * J_Omega_minus_xy + X.S_ground_Fi[1, X.nlevel] * J_Omega_plus_xy) * rho_mid_xy
        if eii_R_xy is not None:
            rate_2s_xy = rate_2s_xy + eii_R_xy[2] * rho_mid_xy
            feed[:n_base] += np.einsum('i,xy->ixy', even_L3 * w, eii_R_xy[0] * rho_mid_xy)
        feed[:n_base] += np.einsum('i,xy->ixy', even_L3 * w * X.mid_ck_L3, rate_2s_xy)
        feed[n_base:] += w * np.einsum('i,xy->ixy', X.S_ground_Fi[0, n_base:X.nlevel], J_Omega_minus_xy * rho_mid_xy)
        feed[n_base:] += w * np.einsum('i,xy->ixy', X.S_ground_Fi[1, n_base:X.nlevel], J_Omega_plus_xy * rho_mid_xy)
        feed[n_base:] += np.einsum('i,xy->ixy', (w * X.mid_ck_L2 / n_L2) * np.ones(n_L2), rate_2s_xy)
        if eii_R_xy is not None:
            feed[n_base:] += np.einsum('i,xy->ixy', (w / n_L2) * np.ones(n_L2), eii_R_xy[1] * rho_mid_xy)

    return feed


def other_rhs(t, rho_other_xy, params):
    """'Other' ions (every hole the model does not resolve): fed by photoionisation of the ground
    state, lost by further photoionisation. With middlemen the feed goes to the pool instead."""
    X, rho_ground_xy, J_Omega_minus_xy, J_Omega_plus_xy, eii_R_xy = params
    J_F = np.array([J_Omega_minus_xy, J_Omega_plus_xy])
    drho_ion = -np.einsum('f, fxy->xy', X.S_other_F, J_F) * rho_other_xy
    if X.use_middlemen:
        return drho_ion
    drho_pump = np.einsum('f, fxy->xy', X.S_ground_Fi[:, -1], J_F) * rho_ground_xy
    if eii_R_xy is not None:
        drho_pump = drho_pump + eii_R_xy[3] * rho_ground_xy   # M-shell EII of ground atoms
    return drho_pump + drho_ion


def twos_rhs(t, rho_2s_xy, params):
    """2s hole: fed by photoionisation and EII of the ground state, lost by decay and photoionisation."""
    X, rho_ground_xy, J_Omega_minus_xy, J_Omega_plus_xy, eii_R_xy = params
    J_F = np.array([J_Omega_minus_xy, J_Omega_plus_xy])
    drho_pump = np.einsum('f, fxy->xy', X.S_ground_Fi[:, -2], J_F) * rho_ground_xy
    if eii_R_xy is not None:
        drho_pump = drho_pump + eii_R_xy[2] * rho_ground_xy
    drho_ion = -np.einsum('f, fxy->xy', X.S_2s_F, J_F) * rho_2s_xy
    return drho_pump + drho_ion - X.GammaL1fsm1N * rho_2s_xy


def ground_rhs(t, rho_ground_xy, params):
    """Neutral ground state, lost by photoionisation and electron-impact ionisation."""
    X, J_Omega_minus_xy, J_Omega_plus_xy, eii_loss_xy = params
    drho = -np.einsum('fi, fxy->xy', X.S_ground_Fi, np.array([J_Omega_minus_xy, J_Omega_plus_xy])) * rho_ground_xy
    if eii_loss_xy is not None:
        # frozen at its start-of-step value, which is exactly what the EII destinations receive
        drho = drho - eii_loss_xy
    return drho


def eii_rates_xy(X, rho_e_gxy):
    """EII rate per target atom (fs^-1) into 2p3/2, 2p1/2, 2s and the M shell, from the electron ladder."""
    return np.einsum('rg,gxy->rxy', X.eii_rate_table, rho_e_gxy)


def middleman_gain_loss(X, rho_ground_xy, rho_other_xy, rho_2s_xy, rho_base_ijxy, rho_sat_ijxy,
                        J_Omega_minus_xy, J_Omega_plus_xy, eii_R_xy=None):
    """
    Gain and loss rate of the middleman pool, d rho_mid/dt = gain - loss_rate * rho_mid. The gain is the
    untracked outflow of every tracked population (Simulation._build_untracked_outflow), the ground-state
    photoionisation into 'other', and M-shell EII of ground atoms. The loss is the part of the pool's
    own ionisation that satellite_block_feed routes into satellite blocks.
    """
    Jm, Jp = J_Omega_minus_xy, J_Omega_plus_xy
    rho_ground_xy, rho_other_xy, rho_2s_xy = np.real(rho_ground_xy), np.real(rho_other_xy), np.real(rho_2s_xy)
    diag = np.real(np.einsum('iixy->ixy', rho_base_ijxy))
    gain = np.einsum('i,ixy->xy', X.decay_untracked_base, diag)
    gain += Jm * np.einsum('i,ixy->xy', X.S_untracked_base[0], diag) + Jp * np.einsum('i,ixy->xy', X.S_untracked_base[1], diag)
    for chan, rho_sat in zip(X.satellite_channel_params, rho_sat_ijxy):
        d = np.real(np.einsum('iixy->ixy', rho_sat))
        gain += np.einsum('i,ixy->xy', chan.decay_untracked, d)
        gain += Jm * np.einsum('i,ixy->xy', chan.S_untracked[0], d) + Jp * np.einsum('i,ixy->xy', chan.S_untracked[1], d)
    gain += (X.twos_decay_untracked_fs + X.S_2s_F[0] * Jm + X.S_2s_F[1] * Jp) * rho_2s_xy
    gain += (X.S_other_F[0] * Jm + X.S_other_F[1] * Jp) * rho_other_xy
    gain += (X.S_ground_Fi[0, -1] * Jm + X.S_ground_Fi[1, -1] * Jp) * rho_ground_xy

    loss_rate = X.mid_S_out[0] * Jm + X.mid_S_out[1] * Jp
    if eii_R_xy is not None:
        gain += eii_R_xy[3] * rho_ground_xy
        share = sum(chan.mid_share for chan in X.satellite_channel_params)
        if share:
            routed = eii_R_xy[0] + X.mid_ck_L3 * eii_R_xy[2]
            routed = routed + eii_R_xy[1] + X.mid_ck_L2 * eii_R_xy[2]
            loss_rate = loss_rate + share * routed
    return np.real(gain), loss_rate


def middleman_rhs(t, rho_mid_xy, params):
    gain_xy, loss_rate_xy = params
    return gain_xy - loss_rate_xy * rho_mid_xy


def electron_production_gxy(X, rho_ground_xy, rho_other_xy, rho_2s_xy, rho_mid_xy, rho_base_ijxy,
                            rho_sat_ijxy, J_Omega_minus_xy, J_Omega_plus_xy, rho_e_gxy):
    """
    Free electrons produced per atom per fs, by ladder birth group: one photoelectron per photoabsorption
    (M-shell photoelectrons of ground atoms and middlemen in their own group), one fast electron per
    non-radiative 1s decay, one ~0.9 keV electron per untracked L-shell Auger decay, one slow electron per
    tracked Coster-Kronig decay, and the secondaries of every electron-impact ionisation.
    """
    Jm, Jp = J_Omega_minus_xy, J_Omega_plus_xy
    birth = X.eii_birth
    rho_ground_xy, rho_other_xy, rho_2s_xy = np.real(rho_ground_xy), np.real(rho_other_xy), np.real(rho_2s_xy)
    prod = np.zeros((X.eii_G,) + rho_ground_xy.shape)
    diag = np.real(np.einsum('iixy->ixy', rho_base_ijxy))

    S_ground_total = np.sum(X.S_ground_Fi, axis=1)
    photo = (S_ground_total[0] * Jm + S_ground_total[1] * Jp) * rho_ground_xy
    photo += Jm * np.einsum('i,ixy->xy', X.S_ion_Fi[0], diag) + Jp * np.einsum('i,ixy->xy', X.S_ion_Fi[1], diag)
    photo += (X.S_2s_F[0] * Jm + X.S_2s_F[1] * Jp) * rho_2s_xy
    photo += (X.S_other_F[0] * Jm + X.S_other_F[1] * Jp) * rho_other_xy
    if rho_mid_xy is not None:
        photo += (X.sigma_mid_F[0] * Jm + X.sigma_mid_F[1] * Jp) * rho_mid_xy
    # "other" is the M shell alone when 2s is tracked; a middleman's cross section splits like the ground state's
    photo_M = 0.0
    if birth['photo_M'] != birth['photo'] and X.use_2s_pathway:
        S_M = np.real(X.S_ground_Fi[:, -1])
        photo_M = (S_M[0] * Jm + S_M[1] * Jp) * rho_ground_xy
        if rho_mid_xy is not None:
            f_M = S_M / np.real(S_ground_total)
            photo_M = photo_M + (f_M[0] * X.sigma_mid_F[0] * Jm + f_M[1] * X.sigma_mid_F[1] * Jp) * rho_mid_xy
    fast = np.einsum('i,ixy->xy', X.e_emit_base['KLL'], diag)
    lmm = np.einsum('i,ixy->xy', X.e_emit_base['LMM'], diag) + X.twos_decay_untracked_fs * rho_2s_xy
    slow = np.einsum('i,ixy->xy', X.e_emit_base['CK'], diag) + X.twos_decay_tracked_fs * rho_2s_xy
    for chan, rho_sat in zip(X.satellite_channel_params, rho_sat_ijxy):
        d = np.real(np.einsum('iixy->ixy', rho_sat))
        photo += Jm * np.einsum('i,ixy->xy', chan.S_ion_Fi[0], d) + Jp * np.einsum('i,ixy->xy', chan.S_ion_Fi[1], d)
        fast += np.einsum('i,ixy->xy', chan.e_emit['KLL'], d)
        lmm += np.einsum('i,ixy->xy', chan.e_emit['LMM'], d)
        slow += np.einsum('i,ixy->xy', chan.e_emit['CK'], d)
    targets = np.real(rho_ground_xy + (rho_mid_xy if rho_mid_xy is not None else 0.0))

    prod[birth['photo']] += np.real(photo - photo_M)
    prod[birth['photo_M']] += np.real(photo_M)
    prod[birth['KLL']] += fast
    prod[birth['LMM']] += lmm
    prod[birth['CK']] += slow
    prod += np.einsum('dg,gxy->dxy', X.eii_secondary_matrix, np.real(rho_e_gxy[:-1])) * targets
    return prod


def electron_rhs(t, rho_e_gxy, params):
    """Electron ladder: production, then transfer g -> g + 1 at the slowing-down rate. The last bin
    (below E_bottom) only accumulates."""
    X, prod_gxy = params
    flow = X.eii_k_down[:, None, None] * rho_e_gxy
    d = prod_gxy - flow
    d[1:] += flow[:-1]
    return d


def field_source(X, rho_ijxy):
    """Field radiated by the coherences of one block, per unit length (the source term of the field equation)."""
    rho_ijxy_hermitian = rho_ijxy + np.conj(np.swapaxes(rho_ijxy, 0, 1))
    Omega_plus_source = np.einsum('ijs, jixy-> sxy', X.Tijs_minus, rho_ijxy_hermitian)
    Omega_minus_source = np.einsum('jis, ijxy-> sxy', X.Tijs_plus, rho_ijxy_hermitian)
    return X.field_source_factor * np.einsum('p, psxy -> psxy', X.e_sign, np.asarray([Omega_plus_source, Omega_minus_source]))


def absorption(X, rho_ground_xyz, rho_other_xyz, rho_2s_xyz, rho_ijxyz, rho_sat_ijxyz_list, rho_mid_xyz=None):
    """Non-resonant absorption coefficient (nm^-1) from photoionisation of every population, at the two
    z planes bounding one propagation step (last axis)."""
    kappa_Omega_sxyz = X.n * np.einsum('xyz, si->sxyz', rho_ground_xyz, X.S_ground_Fi) + \
                       X.n * np.einsum('xyz, s->sxyz', rho_other_xyz, X.S_other_F) + \
                       X.n * np.einsum('xyz, s->sxyz', rho_2s_xyz, X.S_2s_F) + \
                       X.n * np.einsum('si, iixyz->sxyz', X.S_ion_Fi, rho_ijxyz)
    for chan, rho_sat_ijxyz in zip(X.satellite_channel_params, rho_sat_ijxyz_list):
        kappa_Omega_sxyz = kappa_Omega_sxyz + X.n * np.einsum('si, iixyz->sxyz', chan.S_ion_Fi, rho_sat_ijxyz)
    if rho_mid_xyz is not None:
        # middlemen photoabsorb like neutral Cu
        kappa_Omega_sxyz = kappa_Omega_sxyz + X.n * np.einsum('xyz, s->sxyz', rho_mid_xyz, X.sigma_mid_F)
    return np.array([kappa_Omega_sxyz, kappa_Omega_sxyz])
