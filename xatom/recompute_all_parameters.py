"""
Recompute, fresh, every XATOM-derived parameter needed for the 7 requested run configs
(5 SASE variants + 2 monochromator variants; see conversation / commit message for the mapping):

  base cross sections (sigma1_Ka1_{2s,2p3,2p1,other}, sigma2_Ka1_{1s,2s,2p3,2p1,other})
  satellite_channels (3d+, 3d-, 3p+, 3p-): detuning/Gamma_A_2s/Gamma_L/Gamma_K/sigma_Ka1_*/sigma_ion_*
    + the KLM-feed additions (Gamma_A_K_eV, Gamma_A_K_to_L2_eV) that print_satellite_parameters.py's
    CLI does not compute but the committed config/base/*.yaml files carry
  l2_satellite_channel_parameters per channel (Kalpha2-satellite extension)
  l2_pathway_parameters (base sigma1/2_Ka1_2p1 -- cross-checked against the base-cross-section stage)
  double_satellite_channels (3d+3d+, 3d-3d+, 3d-3d-) with feed_from at manifolds lower/upper/L2

Writes results/checkpoints incrementally to recomputed_parameters.json so a crash or interrupt
doesn't lose already-completed (and XATOM-cached-but-expensive) stages -- rerun the script and it
picks up where it left off.

Usage: python xatom/recompute_all_parameters.py [--Ka1-energy-eV 8047.91] [--skip-double-satellite]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import xatom_tools as xt

OUT = Path(__file__).parent / "recomputed_parameters.json"
SPECTATORS = ('3d+', '3d-', '3p+', '3p-')


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def load():
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {}


def save(results):
    OUT.write_text(json.dumps(results, indent=2, sort_keys=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--Ka1-energy-eV', type=float, default=8047.91)
    parser.add_argument('--skip-double-satellite', action='store_true',
                         help='Skip the slow (~40-90 min) double-satellite stage.')
    args = parser.parse_args()
    Ka1 = args.Ka1_energy_eV

    results = load()
    results['Ka1_energy_eV'] = Ka1

    # ---------------------------------------------------------------
    # Stage 1: base ground-state / singly-holed cross sections
    # ---------------------------------------------------------------
    if 'base' not in results:
        log('Stage 1/5: base cross sections (ground-state -pcs + singly-holed -pcs) ...')
        pa = xt.parse_photoabsorption(xt.run_xatom_cached('', photon_energy=Ka1, pcs=True))
        sigma1 = {
            '2s': pa.per_subshell['2s0'][0] * 1e-4,
            '2p3': pa.per_subshell['2p+'][0] * 1e-4,
            '2p1': pa.per_subshell['2p-'][0] * 1e-4,
        }
        sigma2 = {
            '1s': xt.total_photoionization_cross_section_nm2('1s1', Ka1),
            '2s': xt.total_photoionization_cross_section_nm2('2s1', Ka1),
            '2p3': xt.total_photoionization_cross_section_nm2(xt.BASE_LOWER_HOLE, Ka1),
            '2p1': xt.total_photoionization_cross_section_nm2(xt.L2_HOLE, Ka1),
        }
        other6 = xt.other_state_parameters(Ka1)
        results['base'] = {'sigma1': sigma1, 'sigma2': sigma2, 'other6': other6}
        save(results)
        log('Stage 1/5 done.')
    else:
        log('Stage 1/5: base cross sections already cached, skipping.')

    # ---------------------------------------------------------------
    # Stage 2: single-spectator satellite channels (Ka1-satellite keys)
    # ---------------------------------------------------------------
    if 'satellite_channels' not in results:
        log('Stage 2/5: satellite_channel_parameters for 3d+/3d-/3p+/3p- ...')
        channels = {}
        for spectator in SPECTATORS:
            log(f'  channel {spectator} ...')
            channels[spectator] = xt.satellite_channel_parameters(spectator, spectator, Ka1)
        results['satellite_channels'] = channels
        save(results)
        log('Stage 2/5 done.')
    else:
        log('Stage 2/5: satellite_channels already cached, skipping.')

    # ---------------------------------------------------------------
    # Stage 3: KLM feed (Gamma_A_K_eV / Gamma_A_K_to_L2_eV) -- the '1s1'-decay Auger table read,
    # not produced by satellite_channel_parameters() itself but carried in every committed config.
    # ---------------------------------------------------------------
    if 'klm_feed' not in results:
        log('Stage 3/5: KLM feed (Gamma_A_K_eV, Gamma_A_K_to_L2_eV) from 1s1-decay Auger table ...')
        klm = {}
        for spectator in SPECTATORS:
            log(f'  channel {spectator} ...')
            klm[spectator] = {
                'Gamma_A_K_eV': xt.auger_partial_rate_eV(
                    spectator, parent_hole_config='1s1', initial_label='1s0', final1_label='2p+'),
                'Gamma_A_K_to_L2_eV': xt.auger_partial_rate_eV(
                    spectator, parent_hole_config='1s1', initial_label='1s0', final1_label='2p-'),
            }
        results['klm_feed'] = klm
        save(results)
        log('Stage 3/5 done.')
    else:
        log('Stage 3/5: KLM feed already cached, skipping.')

    # ---------------------------------------------------------------
    # Stage 4: L2 (Kalpha2-satellite) extension per channel + base L2 pathway params
    # ---------------------------------------------------------------
    if 'l2_satellite' not in results:
        log('Stage 4/5: l2_satellite_channel_parameters (Kalpha2-satellite extension) ...')
        l2sat = {}
        for spectator in SPECTATORS:
            log(f'  channel {spectator} ...')
            l2sat[spectator] = xt.l2_satellite_channel_parameters(spectator, Ka1)
        results['l2_satellite'] = l2sat
        results['l2_pathway'] = xt.l2_pathway_parameters(Ka1)
        save(results)
        log('Stage 4/5 done.')
    else:
        log('Stage 4/5: L2 satellite extension already cached, skipping.')

    # ---------------------------------------------------------------
    # Stage 5: double-M-shell-spectator channels (SLOW: ~40-90+ min, 3 manifolds)
    # ---------------------------------------------------------------
    if args.skip_double_satellite:
        log('Stage 5/5: --skip-double-satellite passed, skipping.')
    elif 'double_satellite' not in results:
        log('Stage 5/5: double_satellite_channels (lower/upper/L2 manifolds) -- SLOW, be patient ...')
        double_channels, carved_out = xt.build_double_satellite_channels(
            parent_spectators=('3p+', '3p-'), Ka1_energy_eV=Ka1, manifolds=('lower', 'upper', 'L2'))
        results['double_satellite'] = {
            'channels': double_channels,
            'carved_out': {f'{p}|{m}': v for (p, m), v in carved_out.items()},
        }
        save(results)
        log('Stage 5/5 done.')
    else:
        log('Stage 5/5: double_satellite already cached, skipping.')

    log(f'All done. Results in {OUT}')


if __name__ == '__main__':
    main()
