#!/usr/bin/env python3
"""Derive a variant base config from an existing one by named physics transforms.

The sweep generators (generate_intensity_sweep_configs.py, generate_mono_sweep_configs.py) only
override top-level scalars (--set). Bracketing experiments need structured edits -- every
satellite detuning, the eii sub-block, the foil thickness with its comment, a second config's
extension blocks -- so this script applies a list of transforms and writes a comment-free YAML that
the generators then take as --base-yaml. Each transform is one line in the "derived" key of the
output, so a generated config records how it was made.

    python scripts/derive_config.py --base config/base/Cu-seed-SASE-middlemen-eii.yaml \\
        --out config/generated/bracket_bases/sase_spot71.yaml --apply spot_scale=0.7071

Transforms (KEY=VALUE, applied in order):
  spot_scale=<s>        multiply seed_width_FWHM_x and _y by s (s=0.7071 halves the focal area)
  foil_um=<L>           zmax = L um; zgrid unchanged (dz shrinks), so the per-step error only drops
  detunings_zero        every satellite_channels/double_satellite_channels entry absorbs at the bare
                        lines: detuning_eV = 0, detuning_eV_L2_split = hwKalpha1N - hwKalpha2N (the
                        "spectator hole delocalises before the 2p hole decays" picture)
  eii_spatial=<f>       eii.spatial_factor = f (requires an eii block)
  mixing=<rate>,<scope>,<coh>
                        L3 sublevel mixing at <rate> fs^-1 in scope 'sat' (satellite blocks) or
                        'all' (base too); <coh> = L3_sublevel_mixing_coherence_factor (1 Lindblad,
                        0 population exchange only)
  extensions_from=<yaml>
                        copy the Part VI/VII extension keys (use_middlemen, middlemen, L2_CK_feed,
                        GammaA_L1_to_L3M45eVN, GammaA_L1_to_L2eVN, L3_sublevel_mixing_*, use_eii,
                        eii) from another config, e.g. onto the GRASP-recomputed double-satellite
                        config
  set=<key>:<yaml>      any top-level scalar, value parsed as YAML
"""

import argparse
import os

import yaml

EXTENSION_KEYS = ('use_middlemen', 'middlemen', 'L2_CK_feed', 'GammaA_L1_to_L3M45eVN',
                  'GammaA_L1_to_L2eVN', 'L3_sublevel_mixing_fs_inv',
                  'L3_sublevel_mixing_satellite_fs_inv', 'L3_sublevel_mixing_coherence_factor',
                  'use_eii', 'eii')


def _channels(cfg):
    for key in ('satellite_channels', 'double_satellite_channels'):
        for chan in cfg.get(key) or []:
            yield chan


def apply(cfg, name, value):
    if name == 'spot_scale':
        s = float(value)
        cfg['seed_width_FWHM_x'] = float(cfg['seed_width_FWHM_x']) * s
        cfg['seed_width_FWHM_y'] = float(cfg['seed_width_FWHM_y']) * s
    elif name == 'foil_um':
        cfg['zmax'] = float(value) * 1.0e3
    elif name == 'detunings_zero':
        split = float(cfg['hwKalpha1N']) - float(cfg['hwKalpha2N'])
        for chan in _channels(cfg):
            chan['detuning_eV'] = 0.0
            if 'detuning_eV_L2_split' in chan:
                chan['detuning_eV_L2_split'] = split
    elif name == 'eii_spatial':
        if not cfg.get('use_eii'):
            raise SystemExit('eii_spatial needs a config with use_eii: true')
        cfg['eii']['spatial_factor'] = float(value)
    elif name == 'mixing':
        rate, scope, coh = value.split(',')
        rate, coh = float(rate), float(coh)
        cfg['L3_sublevel_mixing_satellite_fs_inv'] = rate
        cfg['L3_sublevel_mixing_fs_inv'] = rate if scope == 'all' else 0.0
        if scope not in ('sat', 'all'):
            raise SystemExit(f"mixing scope must be sat or all, got {scope!r}")
        cfg['L3_sublevel_mixing_coherence_factor'] = coh
    elif name == 'extensions_from':
        with open(value) as f:
            src = yaml.safe_load(f)
        for key in EXTENSION_KEYS:
            if key in src:
                cfg[key] = src[key]
        # the source's extension blocks name satellite channels; they must exist here too
        names = {chan['name'] for chan in _channels(cfg)}
        for key, block in (('middlemen', cfg.get('middlemen') or {}), ('L2_CK_feed', cfg.get('L2_CK_feed') or {})):
            targets = block.get('targets')
            if isinstance(targets, dict):
                missing = sorted(set(targets) - names)
                if missing:
                    raise SystemExit(f"{key}.targets names channels {missing} that {value} has but the base lacks")
    elif name == 'set':
        key, _, raw = value.partition(':')
        if key not in cfg:
            raise SystemExit(f"set={key}: not a top-level key of the base config")
        cfg[key] = yaml.safe_load(raw)
    else:
        raise SystemExit(f"unknown transform {name!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--apply', nargs='*', default=[], metavar='TRANSFORM',
                        help='NAME or NAME=VALUE, applied in order')
    args = parser.parse_args()

    with open(args.base) as f:
        cfg = yaml.safe_load(f)
    derived = [f"base: {os.path.relpath(args.base)}"]
    for item in args.apply:
        name, _, value = item.partition('=')
        apply(cfg, name, value)
        derived.append(item)
    cfg['derived'] = derived

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        yaml.safe_dump(cfg, f)
    print(f"wrote {args.out}  ({'; '.join(derived)})")


if __name__ == '__main__':
    main()
