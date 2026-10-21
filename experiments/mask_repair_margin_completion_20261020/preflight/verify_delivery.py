"""Local stdlib verification of transferred artifacts, reductions and failure flags."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
ORIGINAL = ROOT / 'experiments/mask_repair_margin_20261019'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


reports = []
for run in sorted((STUDY / 'runs').glob('RUN_RSMC_*')):
    meta = read(run / 'run.json')
    assert meta['execution_status'] == 'completed' and meta['exit_code'] == 0
    count = 0
    for item in read(run / 'ARTIFACT_MANIFEST.json')['artifacts']:
        file = run / item['relative_path']
        assert file.resolve().is_relative_to(run.resolve())
        assert file.stat().st_size == item['size_bytes']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == item['sha256']
        count += 1
    subprocess.run([sys.executable, str(ORIGINAL / 'scripts/verify_local.py'), str(run),
                    '--output', str(run / 'INDEPENDENT_LOCAL_STATISTICS.json')], check=True)
    rows = [json.loads(line) for line in (run / 'PAIR_RESULTS.jsonl').read_text(encoding='utf8').splitlines()]
    summary = read(run / 'STATISTICS.json')
    cfg = read(run / 'CONFIG.json')
    tol = cfg['numerics']['rms_relative_tolerance']
    diagnostics = {}
    for name in ['C', 'W']:
        linear = [abs(p[name + '_actual_rms'] / r['pressure_amplitude'] - 1)
                  for r in rows for p in r['pressure']]
        physical = [abs(p[name + '_decoded_difference_rms'] / r['pressure_amplitude'] - 1)
                    for r in rows for p in r['pressure']]
        diagnostics[name] = {'directions': len(linear), 'original_RMS_failed': sum(x > tol for x in linear),
                             'decoded_difference_failed': sum(x > tol for x in physical),
                             'original_max_relative_error': max(linear),
                             'decoded_max_relative_error': max(physical)}
    assert diagnostics == summary['pressure_diagnostics']
    for r in rows:
        original = all(abs(p[k] / r['pressure_amplitude'] - 1) <= tol
                       for p in r['pressure'] for k in ['C_actual_rms', 'W_actual_rms'])
        both = original and all(abs(p[k] / r['pressure_amplitude'] - 1) <= tol
                                for p in r['pressure']
                                for k in ['C_decoded_difference_rms', 'W_decoded_difference_rms'])
        assert original == r['original_pressure_precision_valid'] and both == r['pressure_precision_valid']
    assert summary['pressure_invalid_pairs'] == sum(not r['pressure_precision_valid'] for r in rows)
    assert summary['original_pressure_invalid_pairs'] == sum(not r['original_pressure_precision_valid'] for r in rows)
    if meta['scope'] == 'full':
        assert len(rows) == 256 and len({r['image_id'] for r in rows}) == 243
        assert summary['witness_retention']['C_witnesses'] == 159
        if not summary['numerical_valid']:
            assert summary['decision'] == 'INCONCLUSIVE_NUMERICAL'
        assert read(run / 'INDEPENDENT_LAPTOP.json')['original_prefix_pairs_exact'] == 113
    reports.append({'run_id': run.name, 'scope': meta['scope'], 'artifact_hashes_verified': count,
                    'pairs': len(rows), 'failure_flags_verified': True,
                    'independent_statistics': read(run / 'INDEPENDENT_LOCAL_STATISTICS.json')})
result = {'status': 'passed', 'runs': reports, 'GPU_replayed_locally': False,
          'scientific_gate_is_not_implied_by_integrity_verification': True}
(STUDY / 'preflight/INDEPENDENT_LOCAL.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
print(json.dumps(result))
