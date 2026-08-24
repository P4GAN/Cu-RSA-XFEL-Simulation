"""
Run XATOM for all four 2s-hole satellite channels (docs/theory-and-2s-satellite-pathways.md, Part
II) and print the resulting `satellite_channels` parameters -- both as a readable table and as a
ready-to-paste YAML block for config/base/*.yaml.

Usage
-----
    python xatom/print_satellite_parameters.py [--Ka1-energy-eV 8047.91]

Takes a couple of minutes: each channel needs ~8 XATOM invocations (detuning: 4 total-energy
calls; Gamma_A: 1, shared across channels via the "2s1" Auger table cache; Gamma_L/Gamma_K: 2
decay-width calls; cross sections: 4 -pcs calls), and XATOM itself takes order 1-10 seconds per
invocation (see calculating_parameters.ipynb's timing footers).
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
        channels.append(params)

        print(f'--- {spectator} ---')
        for key, value in params.items():
            if key == 'name':
                continue
            print(f'  {key:20s} = {value:.6g}')
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


if __name__ == '__main__':
    main()
