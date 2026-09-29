"""Post-processing: transmitted spectra, the per-shot outputs the sweeps save, and loading a finished sweep."""

import os
from collections import OrderedDict

import numpy as np
import scipy.signal as sp_sign
import yaml


def find_fwhm(x, y):
    """Full width at half maximum of a single-peaked curve y(x), from the outermost points above half maximum."""
    half_max = np.max(y) / 2
    i_max = np.argmax(y)
    left = next((i for i in range(0, i_max) if y[i] > half_max), 0)
    right = next((i for i in range(len(y) - 1, i_max, -1) if y[i] > half_max), len(y) - 1)
    return x[right] - x[left]


def fft_field_t_y_to_w_thy(X, field_pstxy, ypad, tpad, window_alpha=0.25):
    """
    Fourier transform a field (p, s, t, x, y) from (t, y) to (photon energy, theta_y).

    A Tukey window tapers the field to zero at the edges of the t and y grids before zero padding, so the
    padding does not add a step (and Gibbs ringing in the spectrum wings). The phase is referred to the
    pulse centre t_peak and the beam axis.

    Returns [E_min, E_max, theta_min, theta_max] (eV relative to Kalpha1, mrad) and the transformed field
    (p, s, x, omega, theta_y).
    """
    coeffs = (2.0 * np.pi * X.hbar, 2.0 * np.pi / X.k0)
    shifts = (X.t_peak + tpad * X.dt, X.ymax + ypad * X.dy)
    field_psxty = np.einsum('pstxy->psxty', field_pstxy)

    window_t = sp_sign.windows.tukey(X.tgrid, alpha=window_alpha)
    window_y = sp_sign.windows.tukey(X.ygrid, alpha=window_alpha)
    field_psxty = field_psxty * window_t[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis] \
                              * window_y[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

    pad_shape = [(0, 0), (0, 0), (0, 0), (tpad, tpad), (ypad, ypad)]
    domains, meshes, _ = X.optics.nd_kspace(coeffs, (X.tgrid, X.ygrid), (tpad, ypad), (X.dt, X.dy))
    phasors = [np.exp(1j * 2.0 * np.pi * mesh * shift / coeff) for coeff, mesh, shift in zip(coeffs, meshes, shifts)]
    field_fft = X.optics.fft_phased(X.optics.pad(field_psxty, pad_shape), (3, 4), phasors)

    extent_w_thy = [np.min(domains[0][tpad:X.tgrid + tpad]), np.max(domains[0][tpad:X.tgrid + tpad]),
                    1e3 * np.min(domains[1]), 1e3 * np.max(domains[1])]
    return extent_w_thy, field_fft


def spectrum_w(X, zint, ypad, tpad, window_alpha=0.25):
    """
    Spectrum of the field at z-grid index zint (0: incident, -1: transmitted), |FFT(Omega)|^2 of the p = 0
    component. Returns the photon energy axis (eV relative to Kalpha1), the spectrum integrated over
    theta_y, and the on-axis (theta_y = 0) spectrum.
    """
    extent_w_thy, field_psxwThy = fft_field_t_y_to_w_thy(X, X.Omega_pstxyz[:, :, :, :, :, zint], ypad, tpad,
                                                         window_alpha)
    I_w_thy = np.real(np.einsum('sxwy,sxwy->wy', field_psxwThy[0], np.conj(field_psxwThy[0])))

    ygrid_pad = X.ygrid + 2 * ypad
    tgrid_pad = X.tgrid + 2 * tpad
    womega_ar = np.linspace(extent_w_thy[0], extent_w_thy[1], tgrid_pad)
    theta_y_ar = np.linspace(extent_w_thy[2], extent_w_thy[3], ygrid_pad)
    dtheta_y = theta_y_ar[1] - theta_y_ar[0]
    I_int_thy_w = np.real(dtheta_y * np.einsum('wa -> w', I_w_thy))
    I_thy0_w = np.real(I_w_thy[:, int(ygrid_pad / 2)])
    return womega_ar, I_int_thy_w, I_thy0_w


def compute_run_outputs(X, tpad=1000, ypad=64):
    """
    The per-shot outputs a sweep saves: incident and transmitted spectra and intensity, and the populations at
    the centre pixel of the exit plane.

    rho_K / rho_l3 / rho_l2 (and their _sat versions, one row per satellite channel) are dipole-weighted
    contractions Tijs_minus rho Tijs_plus, not plain populations. The plain populations are base_pop, sat_pop
    (summed over channels), ground, other, 2s and mid; with middlemen on, their sum total_population stays 1 to
    RK4 accuracy.
    """
    womega_ar, I_int_thy_w_0, I_thy0_w_0 = spectrum_w(X, 0, ypad, tpad)
    womega_ar, I_int_thy_w_last, I_thy0_w_last = spectrum_w(X, -1, ypad, tpad)

    # the lean path stores a single (x, y) pixel, the full path the whole grid
    cx = min(int(X.xgrid / 2), X.rho_ijtxyz.shape[3] - 1)
    cy = min(int(X.ygrid / 2), X.rho_ijtxyz.shape[4] - 1)
    rho_ijt = X.rho_ijtxyz[:, :, :, cx, cy, -1]

    Tijs_plus_L3 = X.Tijs_plus * X.ei_L3[None, :, None]
    Tijs_minus_L3 = X.Tijs_minus * X.ei_L3[:, None, None]
    Tijs_plus_L2 = X.Tijs_plus * X.ei_L2[None, :, None]
    Tijs_minus_L2 = X.Tijs_minus * X.ei_L2[:, None, None]

    def contractions(rho):
        return (np.einsum('ijs, jkt, kis-> t', X.Tijs_minus, rho, X.Tijs_plus, optimize=True),
                np.einsum('ijs, jkt, kis-> t', Tijs_plus_L3, rho, Tijs_minus_L3, optimize=True),
                np.einsum('ijs, jkt, kis-> t', Tijs_plus_L2, rho, Tijs_minus_L2, optimize=True))

    rho_K_t_last, rho_l3_t_last, rho_l2_t_last = contractions(rho_ijt)
    base_pop_t_last = np.einsum('iit->t', rho_ijt, optimize=True)

    n_sat = len(X.satellite_channel_params)
    rho_K_t_last_sat = np.zeros((n_sat, X.tgrid), dtype=complex)
    rho_l3_t_last_sat = np.zeros((n_sat, X.tgrid), dtype=complex)
    rho_l2_t_last_sat = np.zeros((n_sat, X.tgrid), dtype=complex)
    sat_pop_t_last = np.zeros(X.tgrid, dtype=complex)
    for k, rho_sat_ijtxyz in enumerate(X.rho_sat_ijtxyz):
        rho_sat_ijt = rho_sat_ijtxyz[:, :, :, cx, cy, -1]
        rho_K_t_last_sat[k], rho_l3_t_last_sat[k], rho_l2_t_last_sat[k] = contractions(rho_sat_ijt)
        sat_pop_t_last += np.einsum('iit->t', rho_sat_ijt, optimize=True)

    rho_ground_t_last = X.rho_ground_txyz[:, cx, cy, -1]
    rho_other_t_last = X.rho_other_txyz[:, cx, cy, -1]
    rho_2s_t_last = X.rho_2s_txyz[:, cx, cy, -1]
    rho_mid_t_last = X.rho_mid_txyz[:, cx, cy, -1] if X.rho_mid_txyz is not None else np.zeros(X.tgrid)
    total_population_t_last = (rho_ground_t_last + rho_other_t_last + rho_2s_t_last
                               + base_pop_t_last + sat_pop_t_last + rho_mid_t_last)

    # free electrons per atom: still able to ionise (every group) and ever produced (+ the bin below E_bottom)
    if X.rho_e_gtxyz is not None:
        n_e_hot_t_last = X.rho_e_gtxyz[:-1, :, cx, cy, -1].sum(axis=0)
        n_e_total_t_last = X.rho_e_gtxyz[:, :, cx, cy, -1].sum(axis=0)
    else:
        n_e_hot_t_last = n_e_total_t_last = np.zeros(X.tgrid)

    return {
        "womega_ar": womega_ar,
        "t_axis": X.t,
        "satellite_channel_names": tuple(chan.name for chan in X.satellite_channel_params),
        "I_int_thy_w_0": I_int_thy_w_0,
        "I_int_thy_w_last": I_int_thy_w_last,
        "I_thy0_w_0": I_thy0_w_0,
        "I_thy0_w_last": I_thy0_w_last,
        "I_t_0": np.einsum('stxy,stxy->t', X.Omega_pstxyz[0, :, :, :, :, 0], X.Omega_pstxyz[1, :, :, :, :, 0]),
        "I_t_last": np.einsum('stxy,stxy->t', X.Omega_pstxyz[0, :, :, :, :, -1], X.Omega_pstxyz[1, :, :, :, :, -1]),
        "rho_K_t_last": rho_K_t_last,
        "rho_l3_t_last": rho_l3_t_last,
        "rho_l2_t_last": rho_l2_t_last,
        "rho_K_t_last_sat": rho_K_t_last_sat,
        "rho_l3_t_last_sat": rho_l3_t_last_sat,
        "rho_l2_t_last_sat": rho_l2_t_last_sat,
        "rho_ground_t_last": rho_ground_t_last,
        "rho_other_t_last": rho_other_t_last,
        "rho_2s_t_last": rho_2s_t_last,
        "rho_mid_t_last": rho_mid_t_last,
        "base_pop_t_last": base_pop_t_last,
        "sat_pop_t_last": sat_pop_t_last,
        "total_population_t_last": total_population_t_last,
        "n_e_hot_t_last": n_e_hot_t_last,
        "n_e_total_t_last": n_e_total_t_last,
    }


# outputs that depend only on the grid, the same for every shot: saved once, not accumulated
RUN_OUTPUT_AXIS_KEYS = ("womega_ar", "t_axis", "satellite_channel_names")


def accumulate_run_outputs(results):
    """
    Fold a list of per-shot output dicts (compute_run_outputs) into '<key>_sum', '<key>_sumsq' (sum of
    |value|^2) and '<key>_count' (per element, excluding NaN), plus 'n_reps'. Sums and counts add, so the
    chunks of one sweep point, written by different array tasks, combine losslessly (data_from_folder).
    Saved in single precision, far below the shot-to-shot noise.
    """
    acc = {"n_reps": len(results)}
    for key in results[0]:
        if key in RUN_OUTPUT_AXIS_KEYS:
            value = results[0][key]
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
                value = value.astype(np.float32)
            acc[key] = value
            continue
        stacked = np.stack([r[key] for r in results])
        valid = ~np.isnan(stacked)
        n_bad = int((~valid).sum())
        if n_bad:
            print(f"Warning: {key} has {n_bad} NaN value(s) across {len(results)} repetitions "
                  f"-- excluded from the accumulated sum", flush=True)
        finite = np.where(valid, stacked, 0)
        key_sum = finite.sum(axis=0)
        acc[f"{key}_sum"] = key_sum.astype(np.complex64 if np.iscomplexobj(key_sum) else np.float32)
        acc[f"{key}_sumsq"] = (np.abs(finite) ** 2).sum(axis=0).astype(np.float32)
        acc[f"{key}_count"] = valid.sum(axis=0).astype(np.int32)
    return acc


def data_from_folder(folder_path, group_keys, aux_keys=(), reverse=True):
    """
    Combine the chunk files of every runs_.../ folder of one sweep into shot-averaged arrays.

    group_keys: config key(s), read from the YAML copied into each runs_.../ folder, to group by, e.g.
      'E_seed_uJ' or ('E_seed_uJ', 'monochromator_target_energy_eV'); a tuple nests one dict level per key.
    aux_keys: config keys returned as a flat list, from the last folder read (constant across a sweep).
    reverse: sort order of the outermost level (inner levels ascend).

    Returns (nested OrderedDict of {array name: mean} plus '<name>_std' for every accumulated array,
    aux values, number of shots of the last group). Unreadable chunk files (e.g. a task killed while
    writing) are skipped; a '.partial.npz' checkpoint left by such a task is picked up instead.
    """
    if isinstance(group_keys, str):
        group_keys = (group_keys,)

    accumulated = {}
    aux_data = None
    total_n_reps = None

    for runs_folder in os.listdir(folder_path):
        runs_path = os.path.join(folder_path, runs_folder)
        if not os.path.isdir(runs_path):
            continue
        yaml_file = next((f for f in os.listdir(runs_path) if f.endswith('.yaml')), None)
        if yaml_file is None:
            print(f"No YAML file found in {runs_path}")
            continue
        with open(os.path.join(runs_path, yaml_file), 'r') as f:
            yaml_data = yaml.safe_load(f)
        group_values = tuple(yaml_data[k] for k in group_keys)
        aux_data = [yaml_data[k] for k in aux_keys]

        sums, sumsqs, counts, axes = {}, {}, {}, {}
        n_reps = 0
        for file_name in os.listdir(runs_path):
            if not file_name.endswith('.npz'):
                continue
            file_path = os.path.join(runs_path, file_name)
            # read the whole file before folding it in, so a corrupt file is skipped cleanly
            try:
                data = np.load(file_path)
                file_n_reps = int(data['n_reps'])
                file_sums, file_counts, file_sumsqs, file_axes = {}, {}, {}, {}
                for key in {n[:-len('_sum')] for n in data.files if n.endswith('_sum')}:
                    chunk_sum = data[f"{key}_sum"]
                    # chunks written before per-element counts existed divide by the shot count
                    chunk_count = (data[f"{key}_count"] if f"{key}_count" in data.files
                                   else np.full(chunk_sum.shape, file_n_reps, dtype=float))
                    bad = np.isnan(chunk_sum)
                    file_sums[key] = np.where(bad, 0, chunk_sum)
                    file_counts[key] = np.where(bad, 0, chunk_count)
                    if f"{key}_sumsq" in data.files:
                        file_sumsqs[key] = np.where(bad, 0, data[f"{key}_sumsq"])
                for array_name in data.files:
                    if array_name != 'n_reps' and not array_name.endswith(('_sum', '_sumsq', '_count')):
                        file_axes[array_name] = data[array_name]
            except Exception as e:
                print(f"Warning: could not read chunk file {file_path} ({e}) -- skipping it")
                continue

            n_reps += file_n_reps
            for key, chunk_sum in file_sums.items():
                sums[key] = sums.get(key, 0) + chunk_sum
                counts[key] = counts.get(key, 0) + file_counts[key]
            for key, chunk_sumsq in file_sumsqs.items():
                sumsqs[key] = sumsqs.get(key, 0) + chunk_sumsq
            axes.update(file_axes)

        if n_reps == 0:
            print(f"No .npz chunk files found in {runs_path}")
            continue

        group_arrays = dict(axes)
        for key, total_sum in sums.items():
            count = counts[key]
            n_starved = int((count == 0).sum())
            if n_starved:
                print(f"Warning: {key} in {runs_path} has {n_starved} element(s) with no valid repetitions -- left as NaN")
            count_safe = np.where(count > 0, count, 1)
            mean = np.where(count > 0, total_sum / count_safe, np.nan)
            group_arrays[key] = mean
            if key in sumsqs:
                variance = np.where(count > 0, sumsqs[key] / count_safe - np.abs(mean) ** 2, np.nan)
                group_arrays[f"{key}_std"] = np.sqrt(np.clip(variance, 0, None))

        node = accumulated
        for gv in group_values[:-1]:
            node = node.setdefault(gv, {})
        node[group_values[-1]] = group_arrays
        total_n_reps = n_reps

    def sort_level(level_dict, depth):
        items = sorted(level_dict.items(), key=lambda kv: kv[0], reverse=(reverse if depth == 0 else False))
        if depth + 1 < len(group_keys):
            return OrderedDict((k, sort_level(v, depth + 1)) for k, v in items)
        return OrderedDict(items)

    return sort_level(accumulated, 0), aux_data, total_n_reps
