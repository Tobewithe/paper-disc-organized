"""Additive closeout hash audit; never changes frozen inputs or scientific results."""
import hashlib
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'experiments/mask_adapter_ray_audit_20261016'
BASE = ROOT / 'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170'
FULL = 'RUN_ADAPTER_RAY_FULL_362251e449e3493396e69c56562af96b'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    config = read(STUDY / 'CONFIG.json')
    assert socket.gethostname().upper() == config['executor']['host']
    assert str(ROOT).lower() == config['executor']['root'].lower()
    run = read(STUDY / 'runs' / FULL / 'run.json')
    assert run['execution_status'] == 'completed' and run['exit_code'] == 0
    frozen = read(STUDY / 'FREEZE_MANIFEST.json')
    expected = read(STUDY / 'INPUT_EXPECTATIONS.json')
    preparation = read(STUDY / 'PREPARATION.json')
    records = []

    def check(path, expected_sha, role):
        actual = sha(path)
        records.append(dict(path=path.relative_to(ROOT).as_posix(), role=role,
                            expected_sha256=expected_sha, actual_sha256=actual,
                            passed=actual == expected_sha))

    for rel, digest in frozen['files'].items():
        check(ROOT / rel, digest, 'frozen_source')
    for rel, digest in expected['files'].items():
        check(ROOT / rel, digest, 'original_input')
    check(STUDY / 'PREPARATION.json', frozen['preparation_sha256'], 'preparation')
    check(Path(preparation['engineering_test_report']), preparation['engineering_test_sha256'], 'engineering_test_receipt')
    for row in preparation['verified_cache_and_image_records']:
        check(BASE / 'trace' / row['cache'], row['cache_sha256'], 'original_cache')
        image = ROOT / 'assets/datasets/coco_native_s0_20261013/images/val2017' / f"{row['image_id']:012d}.jpg"
        check(image, row['image_sha256'], 'original_image')
    result = dict(status='passed' if all(r['passed'] for r in records) else 'failed',
                  at=datetime.now(timezone.utc).isoformat(), host=socket.gethostname(),
                  run_id=FULL, hash_checks=len(records),
                  counts={role: sum(r['role'] == role for r in records) for role in sorted({r['role'] for r in records})},
                  additive_closeout_check=True, scientific_run_changed=False, records=records)
    with (STUDY / 'POST_RUN_INPUT_AUDIT.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}))
    assert result['status'] == 'passed', 'post-run input/source drift'


if __name__ == '__main__':
    main()
