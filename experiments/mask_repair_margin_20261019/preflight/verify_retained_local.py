"""Independent stdlib audit of stopped Run records; never estimates missing full-panel results."""
from pathlib import Path
import collections, hashlib, json, sys
STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'scripts'))
from verify_local import intersection_union


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def require(value, message):
    if not value:
        raise AssertionError(message)


def main():
    name = 'RUN_RSM_FULL_f02b00986ed14331871c00f9564d6426'
    root = STUDY / 'runs' / name
    run, cfg = read(root / 'run.json'), read(root / 'CONFIG.json')
    require(run['execution_status'] == 'error' and run['exit_code'] == 1, 'retain error status')
    rows = [json.loads(x) for x in (root / 'PAIR_RESULTS.jsonl').read_text(encoding='utf8').splitlines()]
    require(len(rows) == run['completed_pairs'] == 113 < cfg['panel_pairs'], 'partial denominator')
    require(not (root / 'STATISTICS.json').exists() and not (root / 'SUMMARY.json').exists(), 'do not invent a full result')
    mapping, manifest = read(root / 'SOURCE_SNAPSHOT_MAP.json'), read(root / 'SOURCE_MANIFEST.json')
    for key, path in mapping.items():
        require(hashlib.sha256((root / path).read_bytes()).hexdigest() == manifest['files'][key], 'source snapshot')
    count = 0
    for row in rows:
        require(row['endpoints']['C']['rle'] == row['endpoints']['W']['rle'], 'nominal equality')
        require(row['endpoints']['G_budget'] is None, 'reference gap')
        require(len(row['pressure']) == 24, 'pressure count')
        for item in list(row['endpoints'].values()) + [p[c] for p in row['pressure'] for c in ['C', 'W']]:
            if item is None:
                continue
            intersection, union = intersection_union(item['rle'], row['GT_rle'])
            require((intersection, union) == (item['intersection'], item['union']), 'RLE intersection and union')
            require(abs(intersection / union - item['iou']) < 1e-14, 'RLE IoU')
            count += 1
        if not row['accepted']:
            require(row['W_coefficients'] == row['C_coefficients'], 'fallback')
        require(abs(row['W_logit_rms'] / row['C_logit_rms'] - 1) <= cfg['numerics']['rms_relative_tolerance'], 'scale')
        for pressure in row['pressure']:
            for center in ['C', 'W']:
                require(abs(pressure[center + '_actual_rms'] / row['pressure_amplitude'] - 1) <= cfg['numerics']['rms_relative_tolerance'], 'pressure RMS')
    accepted = sum(x['accepted'] for x in rows)
    required = __import__('math').ceil(cfg['panel_pairs'] * cfg['gates']['min_valid_fraction'])
    best = accepted + cfg['panel_pairs'] - len(rows)
    report = {'status': 'passed_partial_records_only', 'source_run': name,
              'verified_pairs': len(rows), 'required_pairs': cfg['panel_pairs'],
              'images': len({x['image_id'] for x in rows}), 'independent_RLE_checks': count,
              'source_snapshots_verified': len(mapping), 'accepted_centers_in_retained_rows': accepted,
              'fallback_reasons': dict(collections.Counter(reason for row in rows for reason in row['fallback_reasons'])),
              'maximum_possible_accepted_if_all_remaining_accepted': best,
              'required_accepted': required, 'coverage_gate_cannot_be_met_this_run': best < required,
              'full_panel_IoU_gain': None, 'full_panel_bootstrap_interval': None,
              'scientific_conclusion': 'invalid_engineering_no_full_panel_mechanism_claim',
              'training_recommendation': 'do_not_enter_current_margin_training',
              'GPU_replayed_locally': False, 'new_training_steps': 0}
    out = STUDY / 'preflight/INDEPENDENT_LOCAL_RETAINED.json'
    require(not out.exists(), 'receipt already exists')
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
