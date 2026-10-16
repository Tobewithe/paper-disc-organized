"""Read inputs and current resource ownership; freeze before any scientific solve."""
from mechanism_common import *
import argparse
import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-report', required=True)
    args = parser.parse_args()
    ownership_check()
    cfg, expected, profile, environment = inspect_inputs()
    test_path = Path(args.test_report).resolve()
    require(test_path.is_relative_to(STUDY), 'test receipt must be in this Study')
    tests = read_json(test_path)
    require(tests['status']=='passed', 'engineering tests did not pass')
    for rel, digest in tests['tested_sources'].items():
        require(sha256(ROOT / rel)==digest, 'tested implementation drift: ' + rel)
    panel = read_json(S1 / 'TARGET_PANEL.json')['targets']
    panel_ids = sorted({r['image_id'] for r in panel})
    require(len(panel)==cfg['panel']['pairs'] and len(panel_ids)==cfg['panel']['images'], 'S1 denominator changed')
    require(sha256(S1 / 'TARGET_PANEL.json')==cfg['panel']['sha256'], 'S1 target panel changed')
    panel_set = set(panel_ids)
    # Deterministic four distinct, panel-external images; selection uses only the old native ledger.
    eligible = sorted((r for r in read_jsonl(BASE / 'ledger/PER_GT_FAILURE.jsonl')
        if r['image_id'] not in panel_set and not r['is_ignored'] and r['task_mask75']=='failure'
        and r.get('bbox75_match_output_id') and r.get('box_iou_of_bbox_match', 0)>=.75
        and r.get('mask_iou_of_bbox_match', 1)<.75), key=lambda r:(r['image_id'],r['annotation_id']))
    annotations = {a['id']:a for a in read_json(GT)['annotations']}
    smoke, used = [], set()
    for row in eligible:
        if row['image_id'] in used:
            continue
        used.add(row['image_id'])
        ann = annotations[row['annotation_id']]
        require(ann['image_id']==row['image_id'], 'smoke GT/ledger image differs')
        smoke.append(dict(image_id=row['image_id'], annotation_id=row['annotation_id'],
            output_detection_id=row['bbox75_match_output_id'], baseline_mask_iou=row['mask_iou_of_bbox_match'],
            category_id=ann['category_id']))
        if len(smoke)==cfg['budget']['smoke_pairs']:
            break
    require(len(smoke)==4 and not (used & panel_set), 'invalid smoke allocation')
    all_ids = panel_set | used
    trace = read_json(BASE / 'trace/TRACE_MANIFEST.json')
    records = [r for r in trace['images'] if r['image_id'] in all_ids]
    require(len(records)==len(all_ids), 'missing input cache record')
    for row in records:
        require(sha256(BASE / 'trace' / row['cache'])==row['cache_sha256'], 'cache hash mismatch')
        image = ROOT / 'assets/datasets/coco_native_s0_20261013/images/val2017' / f"{row['image_id']:012d}.jpg"
        require(sha256(image)==row['image_sha256'], 'image hash mismatch')
    available, free = psutil.virtual_memory().available, shutil.disk_usage(ROOT).free
    require(available>=3*1024**3 and free>=4*1024**3, 'insufficient launch resources')
    write_new(STUDY / 'PREPARATION.json', dict(status='passed', at=now(), host=socket.gethostname(),
        environment=environment, profile_sha256=sha256(ROOT / 'paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json'),
        full_targets=panel, full_image_ids=panel_ids, smoke_targets=smoke,
        verified_cache_and_image_records=records, available_ram_bytes=available, free_disk_bytes=free,
        engineering_test_report=test_path.relative_to(STUDY).as_posix(), engineering_test_sha256=sha256(test_path),
        scientific_outcomes_computed=False, optimization_calls=0, new_training_steps=0, base_model_forward_calls=0))
    frozen = dict(frozen_at=now(), files={rel:sha256(ROOT / rel) for rel in CODE},
        preparation_sha256=sha256(STUDY / 'PREPARATION.json'), input_manifest_sha256=sha256(STUDY / 'INPUT_EXPECTATIONS.json'))
    write_new(STUDY / 'FREEZE_MANIFEST.json', frozen)
    study = read_json(STUDY / 'study.json')
    study.update(plan_status='frozen', execution_status='not_started', artifact_status='implementation_and_inputs_frozen',
        scientific_status='hypothesis_only', protocol='PROTOCOL.md', frozen_at=frozen['frozen_at'],
        protocol_sha256=sha256(STUDY / 'PROTOCOL.md'), current_phase='finite_mechanism_smoke_ready',
        execution_environment=dict(host=socket.gethostname(), root=str(ROOT), resources_verified_at=now()))
    save(STUDY / 'study.json', study)
    print(json.dumps(dict(status='frozen', files=len(CODE), full_pairs=len(panel), smoke_targets=smoke,
                         protocol_sha256=study['protocol_sha256'])))


if __name__ == '__main__':
    main()
