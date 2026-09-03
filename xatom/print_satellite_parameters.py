"""
Run XATOM for all four 2s-hole satellite channels (docs/theory-and-2s-satellite-pathways.md, Part
II), the 2p1/2 (L2, Kalpha2) pathway (Part III/IV), and the double-M-shell-spectator
("double-satellite") channels (docs/double-spectator-satellite-implementation-plan.md, Part V) --
and print the resulting parameters, both as readable tables and as ready-to-paste YAML blocks for
config/base/*.yaml.

Usage
-----
    python xatom/print_satellite_parameters.py [--Ka1-energy-eV 8047.91]

The first three sections (satellite_channels, their L2 extension, the base L2 pathway) take a
couple of minutes total: each channel needs ~8 XATOM invocations of order 1-10 seconds each (see
calculating_parameters.ipynb's timing footers). The double-satellite section is much slower --
each of the 4 distinct parent/manifold hole configurations (2p0,1_3p0,1, 2p0,1_3p1,0, 1s1_3p0,1,
1s1_3p1,0) needs its own live, uncached `-decay` call on an 8+-electron configuration, observed to
take on the order of 10 minutes each (~40 minutes total) -- run this script with that in mind, or
comment out the double-satellite section if only the first three are needed.
"""

import argparse

import yaml

import xatom_tools as xt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--Ka1-energy-eV', type=float, default=8047.91,
        help='Reference Kalpha1 diagram-line energy in eV (must match hwKalpha1N in the target '
             'config/base/*.yaml). Default: 8047.91.',
    )
    args = parser.parse_args()

    channels = []
    for spectator in ('3d+', '3d-', '3p+', '3p-'):
        print(f'Running XATOM for channel {spectator} ...', flush=True)
        params = xt.satellite_channel_parameters(spectator, spectator, args.Ka1_energy_eV)
        print(f'Running XATOM for channel {spectator}\'s 2p1/2-satellite (Kalpha2-satellite) extension ...', flush=True)
        params.update(xt.l2_satellite_channel_parameters(spectator, args.Ka1_energy_eV))
        channels.append(params)

        print(f'--- {spectator} ---')
        for key, value in params.items():
            if key == 'name':
                continue
            print(f'  {key:20s} = {value:.6g}')
        print()

    print('=' * 70)
    print('2p1/2-satellite (Kalpha2-satellite) Auger budget check:')
    total_Gamma_A_L2 = sum(c['Gamma_A_2s_to_L2_eV'] for c in channels)
    total_Gamma_A_L3 = sum(c['Gamma_A_2s_eV'] for c in channels)
    print(f'  sum of Gamma_A_2s_to_L2_eV over all 4 channels = {total_Gamma_A_L2:.4f} eV')
    print(f'  + sum of Gamma_A_2s_eV (Ka1-satellite)          = {total_Gamma_A_L3:.4f} eV')
    print(f'  = {total_Gamma_A_L2 + total_Gamma_A_L3:.4f} eV explicitly tracked out of GammaL1eVN '
          '(the remainder is the still-unmodeled L1 decay budget, e.g. L1-L2 Coster-Kronig)')
    print()

    print('=' * 70)
    print('Auger branching budget check (theory doc section 14):')
    total_Gamma_A = sum(c['Gamma_A_2s_eV'] for c in channels)
    print(f'  sum of Gamma_A_2s_eV over all 4 channels = {total_Gamma_A:.4f} eV')
    print('  (compare against GammaL1eVN in config/base/*.yaml -- the remainder is the '
          'still-unmodeled L1 decay budget, e.g. L1-L2 Coster-Kronig and other channels)')
    print()

    print('=' * 70)
    print('Ready to paste into config/base/*.yaml:')
    print()
    yaml_block = {'satellite_channels': channels}
    print(yaml.dump(yaml_block, sort_keys=False, default_flow_style=False))

    print('=' * 70)
    print('2p1/2 (L2, Kalpha2) pathway (docs/theory-and-2s-satellite-pathways.md, Part III):')
    l2_params = xt.l2_pathway_parameters(args.Ka1_energy_eV)
    for key, value in l2_params.items():
        print(f'  {key:20s} = {value:.6g}')
    print()
    print('Ready to paste into config/base/*.yaml (alongside the existing GammaL2eVN literature '
          'value, use_L2_pathway: True):')
    print()
    print(yaml.dump(l2_params, sort_keys=False, default_flow_style=False))

    print('=' * 70)
    print('Double-M-shell-spectator ("double-satellite") channels '
          '(docs/double-spectator-satellite-implementation-plan.md):')
    print('(slow -- each parent/manifold pair needs a live XATOM -decay call on a double-hole')
    print(' configuration, order 1-10 minutes each; not cached across runs of this script)')
    print()
    double_channels, carved_out = xt.build_double_satellite_channels(Ka1_energy_eV=args.Ka1_energy_eV)

    print('Budget check -- carve these out of the corresponding parent satellite_channels entry')
    print('(Gamma_L_eV for manifold=lower, Gamma_K_eV for manifold=upper):')
    for (parent, manifold), total in carved_out.items():
        print(f'  {parent} ({manifold}): carve out {total:.4f} eV')
    print()

    print('Ready to paste into config/base/*.yaml:')
    print()
    print(yaml.dump({'double_satellite_channels': double_channels}, sort_keys=False, default_flow_style=False))


if __name__ == '__main__':
    main()
