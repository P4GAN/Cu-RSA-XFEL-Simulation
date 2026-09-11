"""
Build clean, comment-free satellite_channels / double_satellite_channels YAML blocks from
recomputed_parameters.json (recompute_all_parameters.py), for direct splicing into
config/base/*.yaml. Supersedes assemble_configs.py's satellite-related output: no inline
provenance/carve-out comments, just the numbers.

Writes xatom/assembled/{satellite_channels_no_L2,satellite_channels_L2,
satellite_channels_L2_carved_for_double,double_satellite_channels}.yaml and a
carve_amounts.json (the sums needed to carve the base sigma2_Ka1_1s/2p3/2p1 cross sections in
config/base/*.yaml -- those live in the GRASP/RATIP base-cross-section block, not here).
"""

import json
from pathlib import Path

import yaml

HERE = Path(__file__).parent
IN = HERE / "recomputed_parameters.json"
OUT_DIR = HERE / "assembled"
OUT_DIR.mkdir(exist_ok=True)

SPECTATORS = ('3d+', '3d-', '3p+', '3p-')


def yaml_str(obj):
    return yaml.dump(obj, sort_keys=False, default_flow_style=False)


def main():
    data = json.loads(IN.read_text())
    sat = data['satellite_channels']
    klm = data['klm_feed']
    l2sat = data['l2_satellite']

    def channel_entry(spectator, include_L2):
        c = sat[spectator]
        k = klm[spectator]
        entry = {
            'name': spectator,
            'detuning_eV': round(c['detuning_eV'], 4),
            'Gamma_A_2s_eV': round(c['Gamma_A_2s_eV'], 4),
            'Gamma_A_K_eV': round(k['Gamma_A_K_eV'], 6),
            'Gamma_L_eV': round(c['Gamma_L_eV'], 5),
            'Gamma_K_eV': round(c['Gamma_K_eV'], 5),
            'sigma_Ka1_from_2p': c['sigma_Ka1_from_2p'],
            'sigma_Ka1_from_1s': c['sigma_Ka1_from_1s'],
            'sigma_ion_from_2p': c['sigma_ion_from_2p'],
            'sigma_ion_from_1s': c['sigma_ion_from_1s'],
        }
        if include_L2:
            l2 = l2sat[spectator]
            entry['detuning_eV_L2_split'] = round(l2['detuning_eV_L2_split'], 4)
            entry['Gamma_A_2s_to_L2_eV'] = round(l2['Gamma_A_2s_to_L2_eV'], 4)
            entry['Gamma_A_K_to_L2_eV'] = round(k['Gamma_A_K_to_L2_eV'], 6)
            entry['Gamma_L2_eV'] = round(l2['Gamma_L2_eV'], 5)
            entry['sigma_Ka1_from_2p1'] = l2['sigma_Ka1_from_2p1']
        return entry

    satellite_no_L2 = [channel_entry(s, include_L2=False) for s in SPECTATORS]
    satellite_L2 = [channel_entry(s, include_L2=True) for s in SPECTATORS]

    (OUT_DIR / 'satellite_channels_no_L2.yaml').write_text(
        yaml_str({'satellite_channels': satellite_no_L2}))
    (OUT_DIR / 'satellite_channels_L2.yaml').write_text(
        yaml_str({'satellite_channels': satellite_L2}))

    carve_amounts = {
        'sum_sigma_Ka1_from_1s': sum(sat[s]['sigma_Ka1_from_1s'] for s in SPECTATORS),
        'sum_sigma_Ka1_from_2p': sum(sat[s]['sigma_Ka1_from_2p'] for s in SPECTATORS),
        'sum_sigma_Ka1_from_2p1': sum(l2sat[s]['sigma_Ka1_from_2p1'] for s in SPECTATORS),
        'sum_Gamma_A_2s_eV': sum(sat[s]['Gamma_A_2s_eV'] for s in SPECTATORS),
    }

    if 'double_satellite' in data:
        ds = data['double_satellite']
        carved = ds['carved_out']
        double_channels = []
        for ch in ds['channels']:
            entry = {
                'name': ch['name'],
                'detuning_eV': round(ch['detuning_eV'], 2),
                'Gamma_L_eV': round(ch['Gamma_L_eV'], 6),
                'Gamma_K_eV': round(ch['Gamma_K_eV'], 5),
                'detuning_eV_L2_split': round(ch['detuning_eV_L2_split'], 2),
                'Gamma_L2_eV': round(ch['Gamma_L2_eV'], 6),
                'feed_from': [
                    {k: v for k, v in f.items()} for f in ch['feed_from']
                ],
                'sigma_ion_from_2p': ch['sigma_ion_from_2p'],
                'sigma_ion_from_1s': ch['sigma_ion_from_1s'],
            }
            double_channels.append(entry)
        (OUT_DIR / 'double_satellite_channels.yaml').write_text(
            yaml_str({'double_satellite_channels': double_channels}))

        satellite_L2_carved = [dict(e) for e in satellite_L2]
        for entry in satellite_L2_carved:
            name = entry['name']
            for manifold, key in (('lower', 'Gamma_L_eV'), ('upper', 'Gamma_K_eV'), ('L2', 'Gamma_L2_eV')):
                carve_key = f'{name}|{manifold}'
                if carve_key in carved and carved[carve_key] > 0.0:
                    entry[key] = round(entry[key] - carved[carve_key], 5)
        (OUT_DIR / 'satellite_channels_L2_carved_for_double.yaml').write_text(
            yaml_str({'satellite_channels': satellite_L2_carved}))

    (OUT_DIR / 'carve_amounts.json').write_text(json.dumps(carve_amounts, indent=2))
    print(json.dumps(carve_amounts, indent=2))
    print(f'Wrote clean snippets to {OUT_DIR}/')


if __name__ == '__main__':
    main()
