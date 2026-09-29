"""Satellite-channel parameters for the two double-L-hole absorbers (XATOM, Delta-SCF shifts).

Two species the model had no state for, both made by the K hole's own KLL Auger decay and by a
second photoionisation of a 2p-hole atom, and both resonant inside the measured spectrum:

  2s2p   2s^-1 2p^-1 (KL1L23 product, or a 2p-hole atom losing its 2s). Its Kalpha1-type line is
         ~+23 eV above Kalpha1; its Kalpha2-type line lands within ~1 eV of the bare Kalpha1.
  2p2    2p^-2 (KL23L23 product, or a 2p-hole atom losing a second 2p). Kalpha3,4 in emission:
         its lines sit ~+31/+32 eV above Kalpha2/Kalpha1, i.e. at 8059/8080 eV, where the seeded
         data show absorption switching on above ~10-20 uJ.

Block layout (the satellite block has one shared K manifold and separate L3/L2 manifolds):
  2s2p: L3 = 2s1_2p0,1  L2 = 2s1_2p1,0  K = 1s1_2s1
  2p2:  L3 = 2p0,2      L2 = 2p1,1      K = 1s1_2p0,1
        (the (2p1/2)^-2 product of KL2L2 is dropped; 2p1,1 is put in the L2 manifold because its
        2p1/2 hole is the transition at 8059 eV that the data resolve -- its 2p3/2 hole's Kalpha1-type
        line at 8080 eV is then not represented, a known approximation)

Feeds (all branches of existing widths / cross sections, never subtractions):
  Gamma_A_K_eV, Gamma_A_K_to_L2_eV   base K -> L3 / L2 manifold, from the 1s1 Auger table
  sigma_Ka1_from_2p                  base L3 -> L3 (2s2p: ionise 2s of 2p0,1; 2p2: ionise 2p+)
  sigma_Ka1_from_2p_to_L2            base L3 -> L2 (2p2 only: ionise 2p- of 2p0,1) -- new key
  sigma_Ka1_from_2p1                 base L2 -> L2 (2s2p: ionise 2s of 2p1,0; 2p2: ionise 2p+)
  sigma_Ka1_from_1s                  base K  -> K  (ionise 2s / 2p+ of 1s1)
  Gamma_A_2s_eV = 0                  a 2s hole cannot Coster-Kronig into either species
Not included: EII of core-holed atoms into these species, and photoionising the 2p of a 2s-hole
atom (J sigma tau ~ 1e-3 at 20 uJ).

The block reuses the base radiative rates and dipoles (spectator approximation, as every other
channel does); XATOM's own Kalpha1-type rate is 0.455 (2s2p) and 0.344 eV (2p2) against 0.445.

Run:  python xatom/double_L_hole_parameters.py [--base <config>]   (prints the YAML entries and a summary)
"""

import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xatom_tools as xt  # noqa: E402

KA1_EV = 8047.91
KA2_EV = 8027.98
HBAR_EV_FS = 0.6582119569


def energy(hole):
    return xt.parse_total_energy_rel(xt.run_xatom_cached(hole))


def width_eV(hole):
    return xt.state_total_decay_width_eV(hole)


def pcs_nm2(hole, subshell=None):
    """Photoabsorption cross section (nm^2) at the Kalpha1 energy: one subshell or the total."""
    res = xt.parse_photoabsorption(xt.run_xatom_cached(hole, photon_energy=KA1_EV, pcs=True))
    if subshell is None:
        return res.total_cs_Mb * 1e-4  # 1 Mb = 1e-22 m^2 = 1e-4 nm^2
    return res.per_subshell[subshell][0] * 1e-4


def branch_nm2(parent, subshell, config_total_nm2):
    """A feed out of a base population must be a branch of the cross section the config already
    drains that population with (sigma2_Ka1_*, original-paper totals), never more: take the
    branching fraction from XATOM's subshell table and the total from the config."""
    return pcs_nm2(parent, subshell) / pcs_nm2(parent) * config_total_nm2


def auger_eV(parent, initial, final1, final2):
    line = xt.parse_auger(xt.run_xatom_cached(parent, decay=True)).find(initial, final1, final2)
    if line is None:
        raise ValueError(f"no {initial} - {final1} {final2} row in the {parent} Auger table")
    return line.rate_eV


def fluorescence_eV(parent, initial, final):
    line = xt.parse_fluorescence(xt.run_xatom_cached(parent, decay=True)).find(initial, final)
    return line.rate_au * xt.HARTREE_TO_EV if line is not None else None


def channel(name, L3, L2, K, feeds):
    """One satellite_channels entry. `feeds` = (Gamma_A_K, Gamma_A_K_to_L2, s_2p, s_2p_to_L2, s_2p1, s_1s)."""
    ka1_base = energy('1s1') - energy('2p0,1')
    ka2_base = energy('1s1') - energy('2p1,0')
    ka1 = energy(K) - energy(L3)
    ka2 = energy(K) - energy(L2)
    detuning = ka1 - ka1_base                       # Kalpha1-type line relative to Kalpha1
    detuning_L2 = ka2 - ka2_base                    # Kalpha2-type line relative to Kalpha2
    # the schema stores the species' own Kalpha1-Kalpha2 separation
    l2_split = (KA1_EV + detuning) - (KA2_EV + detuning_L2)
    Gamma_A_K, Gamma_A_K_to_L2, s_2p, s_2p_to_L2, s_2p1, s_1s = feeds
    entry = {
        'name': name,
        'detuning_eV': round(detuning, 4),
        'Gamma_A_2s_eV': 0.0,
        'Gamma_A_K_eV': round(Gamma_A_K, 6),
        'Gamma_L_eV': round(width_eV(L3), 5),
        'Gamma_K_eV': round(width_eV(K), 5),
        'sigma_Ka1_from_2p': float(f"{s_2p:.6e}"),
        'sigma_Ka1_from_1s': float(f"{s_1s:.6e}"),
        'sigma_ion_from_2p': float(f"{pcs_nm2(L3):.6e}"),
        'sigma_ion_from_1s': float(f"{pcs_nm2(K):.6e}"),
        'detuning_eV_L2_split': round(l2_split, 4),
        'Gamma_A_2s_to_L2_eV': 0.0,
        'Gamma_A_K_to_L2_eV': round(Gamma_A_K_to_L2, 6),
        'Gamma_L2_eV': round(width_eV(L2), 5),
        'sigma_Ka1_from_2p1': float(f"{s_2p1:.6e}"),
        'sigma_ion_from_2p1': float(f"{pcs_nm2(L2):.6e}"),
    }
    if s_2p_to_L2:
        entry['sigma_Ka1_from_2p_to_L2'] = float(f"{s_2p_to_L2:.6e}")
    summary = dict(name=name, Ka1_type_eV=KA1_EV + detuning, Ka2_type_eV=KA2_EV + detuning_L2,
                   rate_Ka1_type_eV=fluorescence_eV(K, '1s0', '2p+'),
                   width_L3=entry['Gamma_L_eV'], width_L2=entry['Gamma_L2_eV'], width_K=entry['Gamma_K_eV'])
    return entry, summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                       'config', 'base', 'Cu-seed-mono-SASE-middlemen-eii.yaml'),
                        help='config whose sigma2_Ka1_2p3 / sigma2_Ka1_1s / sigma2_Ka1_2p1 totals the feeds branch from')
    args = parser.parse_args()
    with open(args.base) as f:
        base = yaml.safe_load(f)
    tot_L3, tot_K, tot_L2 = (float(base[k]) for k in ('sigma2_Ka1_2p3', 'sigma2_Ka1_1s', 'sigma2_Ka1_2p1'))
    print(f"# further-ionisation totals from {os.path.basename(args.base)}: 2p3/2-hole {tot_L3:.3e}, 1s-hole {tot_K:.3e}, "
          f"2p1/2-hole {tot_L2:.3e} nm^2; XATOM totals {pcs_nm2('2p0,1'):.3e}, {pcs_nm2('1s1'):.3e}, {pcs_nm2('2p1,0'):.3e}")

    # KLL branches of the bare 1s hole, read off the 1s1 Auger table
    kll = {pair: auger_eV('1s1', '1s0', *pair) for pair in
           [('2s0', '2p+'), ('2s0', '2p-'), ('2p+', '2p+'), ('2p-', '2p+'), ('2p-', '2p-'), ('2s0', '2s0')]}
    gamma_K = width_eV('1s1')
    print(f"# 1s hole: total width {gamma_K:.4f} eV; KLL branches (eV): " +
          ", ".join(f"{a}{b} {r:.4f}" for (a, b), r in kll.items()))

    two_s_two_p, s1 = channel(
        '2s2p', L3='2s1_2p0,1', L2='2s1_2p1,0', K='1s1_2s1',
        feeds=(kll[('2s0', '2p+')], kll[('2s0', '2p-')],
               branch_nm2('2p0,1', '2s0', tot_L3), 0.0, branch_nm2('2p1,0', '2s0', tot_L2), branch_nm2('1s1', '2s0', tot_K)))
    two_p_two, s2 = channel(
        '2p2', L3='2p0,2', L2='2p1,1', K='1s1_2p0,1',
        feeds=(kll[('2p+', '2p+')], kll[('2p-', '2p+')],
               branch_nm2('2p0,1', '2p+', tot_L3), branch_nm2('2p0,1', '2p-', tot_L3),
               branch_nm2('2p1,0', '2p+', tot_L2), branch_nm2('1s1', '2p+', tot_K)))

    for s in (s1, s2):
        print(f"# {s['name']}: Kalpha1-type at {s['Ka1_type_eV']:.2f} eV, Kalpha2-type at {s['Ka2_type_eV']:.2f} eV; "
              f"widths L3/L2/K = {s['width_L3']}/{s['width_L2']}/{s['width_K']} eV; "
              f"XATOM Kalpha1-type radiative rate {s['rate_Ka1_type_eV']}")
    print("\n# append to satellite_channels (xatom/double_L_hole_parameters.py, "
          f"XATOM {os.path.basename(xt.XATOM_PATH)}):")
    print(yaml.safe_dump([two_s_two_p, two_p_two], sort_keys=False))


if __name__ == '__main__':
    main()
