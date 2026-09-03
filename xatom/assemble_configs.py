"""
Turn recomputed_parameters.json (built by recompute_all_parameters.py) into ready-to-splice YAML
snippets for each of the 7 target config/base/*.yaml files (5 SASE variants + 2 monochromator
variants). Writes one snippet file per config under xatom/assembled/.

Carve-out conventions (docs/theory-and-2s-satellite-pathways.md sec 12.7, replicated from the
already-committed config/base/*.yaml files' own comments):
  - sigma1_Ka1_other / sigma2_Ka1_other: the ground-state -pcs decomposition's 6 "other" subshells
    (3s,3p+,3p-,4s,3d+,3d-) computed by xatom_tools.other_state_parameters() already excludes
    1s/2s/2p3/2p1 by construction (its own docstring). When use_L2_pathway is off, 2p1/2 isn't
    separately tracked, so its ground-state share must be folded back into "other" (sigma1: sum;
    sigma2: sigma1-weighted combine, same weighting other_state_parameters() itself uses).
  - sigma2_Ka1_1s / sigma2_Ka1_2p3: when satellite_channels are present, subtract the sum of the
    channels' sigma_Ka1_from_1s / sigma_Ka1_from_2p (that fraction is transferred into the explicit
    satellite states, not an additional loss on top of the bare further-ionization rate).
  - sigma2_Ka1_2p1: same transfer logic, subtracting sum(sigma_Ka1_from_2p1) -- the existing
    Cu-seed-SASE-satellite.yaml left this uncarved (likely an oversight, since sigma2_Ka1_1s/2p3
    *are* carved by the exact same argument); carved here for consistency.
  - GammaA_L1_to_L3M45eVN (lumped 2s-Auger-to-L3+M45 feed, used only when satellite_channels are
    NOT explicit): recomputed as sum of the 4 channels' Gamma_A_2s_eV (supersedes the old hand
    M45-only estimate), zeroed when satellite_channels are explicit (fully resolved per-channel).
  - Double-satellite channels' Gamma_L_eV/Gamma_K_eV/Gamma_L2_eV (3p+/3p- only): bare/uncarved value
    minus the sum of that parent+manifold's own feed_from entries in double_satellite_channels.
"""

import json
import sys
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
    base = data['base']
    sat = data['satellite_channels']
    klm = data['klm_feed']
    l2sat = data['l2_satellite']
    l2path = data['l2_pathway']

    sum_sigma_from_1s = sum(sat[s]['sigma_Ka1_from_1s'] for s in SPECTATORS)
    sum_sigma_from_2p = sum(sat[s]['sigma_Ka1_from_2p'] for s in SPECTATORS)
    sum_sigma_from_2p1 = sum(l2sat[s]['sigma_Ka1_from_2p1'] for s in SPECTATORS)
    sum_Gamma_A_2s = sum(sat[s]['Gamma_A_2s_eV'] for s in SPECTATORS)

    other6_sigma1 = base['other6']['sigma1_Ka1_other']
    other6_sigma2 = base['other6']['sigma2_Ka1_other']
    sigma1_2p1 = base['sigma1']['2p1']
    sigma2_2p1 = base['sigma2']['2p1']

    # "other" with 2p1/2 folded back in (non-L2 configs)
    other_noL2_sigma1 = other6_sigma1 + sigma1_2p1
    other_noL2_sigma2 = (other6_sigma1 * other6_sigma2 + sigma1_2p1 * sigma2_2p1) / other_noL2_sigma1

    report = {}
    report['sum_sigma_Ka1_from_1s'] = sum_sigma_from_1s
    report['sum_sigma_Ka1_from_2p'] = sum_sigma_from_2p
    report['sum_sigma_Ka1_from_2p1'] = sum_sigma_from_2p1
    report['sum_Gamma_A_2s_eV (lumped GammaA_L1_to_L3M45eVN)'] = sum_Gamma_A_2s
    report['other_noL2'] = {'sigma1_Ka1_other': other_noL2_sigma1, 'sigma2_Ka1_other': other_noL2_sigma2}
    report['other_L2'] = {'sigma1_Ka1_other': other6_sigma1, 'sigma2_Ka1_other': other6_sigma2}

    # ---------------- base cross-section blocks ----------------

    def base_block(with_L2, carve_satellite):
        d = {
            'sigma1_Ka1_2s': base['sigma1']['2s'],
            'sigma1_Ka1_2p3': base['sigma1']['2p3'],
        }
        if with_L2:
            d['sigma1_Ka1_2p1'] = sigma1_2p1
            d['sigma1_Ka1_other'] = other6_sigma1
        else:
            d['sigma1_Ka1_other'] = other_noL2_sigma1
        d['sigma2_Ka1_1s'] = base['sigma2']['1s'] - (sum_sigma_from_1s if carve_satellite else 0.0)
        d['sigma2_Ka1_2s'] = base['sigma2']['2s']
        d['sigma2_Ka1_2p3'] = base['sigma2']['2p3'] - (sum_sigma_from_2p if carve_satellite else 0.0)
        if with_L2:
            d['sigma2_Ka1_2p1'] = sigma2_2p1 - (sum_sigma_from_2p1 if carve_satellite else 0.0)
            d['sigma2_Ka1_other'] = other6_sigma2
        else:
            d['sigma2_Ka1_other'] = other_noL2_sigma2
        return d

    blocks = {}
    blocks['no_2s'] = base_block(with_L2=False, carve_satellite=False)
    blocks['lumped'] = dict(base_block(with_L2=False, carve_satellite=False),
                             GammaA_L1_to_L3M45eVN=sum_Gamma_A_2s)
    blocks['satellite_no_L2'] = dict(base_block(with_L2=False, carve_satellite=True),
                                      GammaA_L1_to_L3M45eVN=0.0)
    blocks['satellite_L2'] = dict(base_block(with_L2=True, carve_satellite=True),
                                   GammaA_L1_to_L3M45eVN=0.0)
    blocks['double_satellite'] = blocks['satellite_L2']  # same base cross sections

    for name, block in blocks.items():
        (OUT_DIR / f'base_{name}.yaml').write_text(yaml_str(block))

    # ---------------- satellite_channels blocks ----------------

    def channel_entry(spectator, include_L2):
        c = sat[spectator]
        k = klm[spectator]
        entry = {
            'name': spectator,
            'detuning_eV': c['detuning_eV'],
            'Gamma_A_2s_eV': c['Gamma_A_2s_eV'],
            'Gamma_A_K_eV': k['Gamma_A_K_eV'],
            'Gamma_L_eV': c['Gamma_L_eV'],
            'Gamma_K_eV': c['Gamma_K_eV'],
            'sigma_Ka1_from_2p': c['sigma_Ka1_from_2p'],
            'sigma_Ka1_from_1s': c['sigma_Ka1_from_1s'],
            'sigma_ion_from_2p': c['sigma_ion_from_2p'],
            'sigma_ion_from_1s': c['sigma_ion_from_1s'],
        }
        if include_L2:
            l2 = l2sat[spectator]
            entry['detuning_eV_L2_split'] = l2['detuning_eV_L2_split']
            entry['Gamma_A_2s_to_L2_eV'] = l2['Gamma_A_2s_to_L2_eV']
            entry['Gamma_A_K_to_L2_eV'] = k['Gamma_A_K_to_L2_eV']
            entry['Gamma_L2_eV'] = l2['Gamma_L2_eV']
            entry['sigma_Ka1_from_2p1'] = l2['sigma_Ka1_from_2p1']
        return entry

    satellite_no_L2 = [channel_entry(s, include_L2=False) for s in SPECTATORS]
    satellite_L2 = [channel_entry(s, include_L2=True) for s in SPECTATORS]

    (OUT_DIR / 'satellite_channels_no_L2.yaml').write_text(
        yaml_str({'satellite_channels': satellite_no_L2}))
    (OUT_DIR / 'satellite_channels_L2.yaml').write_text(
        yaml_str({'satellite_channels': satellite_L2}))

    # ---------------- double-satellite carve-out + channels ----------------

    if 'double_satellite' in data:
        ds = data['double_satellite']
        carved = ds['carved_out']  # {'3p+|lower': eV, ...}
        double_channels = ds['channels']

        satellite_L2_carved = [dict(e) for e in satellite_L2]
        for entry in satellite_L2_carved:
            name = entry['name']
            for manifold, key in (('lower', 'Gamma_L_eV'), ('upper', 'Gamma_K_eV'), ('L2', 'Gamma_L2_eV')):
                carve_key = f'{name}|{manifold}'
                if carve_key in carved and carved[carve_key] > 0.0:
                    entry[key] = entry[key] - carved[carve_key]

        (OUT_DIR / 'satellite_channels_L2_carved_for_double.yaml').write_text(
            yaml_str({'satellite_channels': satellite_L2_carved}))
        (OUT_DIR / 'double_satellite_channels.yaml').write_text(
            yaml_str({'double_satellite_channels': double_channels}))
        report['double_satellite_carved_out'] = carved
    else:
        print('NOTE: double_satellite stage not yet in recomputed_parameters.json -- '
              'skipping double-satellite-dependent snippets.', file=sys.stderr)

    (OUT_DIR / 'report.json').write_text(json.dumps(report, indent=2))
    print(f'Wrote snippets to {OUT_DIR}/')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
