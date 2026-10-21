"""Check source and existing inputs before new gate outcomes are computed."""
from lea_common import *
import psutil


def main():
    check_ownership()
    cfg = read_json(STUDY / 'CONFIG.json')
    require(socket.gethostname().upper() == cfg['executor']['host'], 'Executor identity')
    require(ROOT.resolve() == Path(cfg['executor']['root']).resolve(), 'Executor workspace')
    require(Path(sys.executable).resolve() == Path(cfg['executor']['python']).resolve(), 'Interpreter identity')
    locked = read_json(STUDY / 'SOURCE_FREEZE.json')
    verify_files(locked['files'])
    tests = read_json(STUDY / 'TESTS.json')
    require(tests['status'] == 'passed', 'Engineering tests not passed')
    verify_files(tests['tested_sources'])
    profile, env = configure_native()
    old = read_json(RAY / 'PREPARATION.json')
    require(env == old['environment'], 'Native configuration drift')
    expected = dict(read_json(RAY / 'INPUT_EXPECTATIONS.json')['files'])
    verify_files(expected)
    refs = [BASE, S1, CFULL, CSMOKE, S3]
    for p in refs:
        require(read_json(p / 'run.json')['execution_status'] == 'completed', 'Incomplete reference')
    artifact = {r['relative_path']: r['sha256'] for r in read_json(S3 / 'ARTIFACT_MANIFEST.json')['artifacts']}
    require(sha256(S3 / 'predictions_mixed.json') == artifact['predictions_mixed.json'], 'S3 prediction hash')
    needed = [S3 / 'predictions_mixed.json', S3 / 'ARTIFACT_MANIFEST.json',
              CFULL / 'PAIR_RESULTS.jsonl', CSMOKE / 'PAIR_RESULTS.jsonl',
              PREV / 'PREPARATION.json', RAY / 'PREPARATION.json']
    expected.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in needed})
    full = read_json(S1 / 'TARGET_PANEL.json')['targets']
    smoke = old['smoke_targets']
    require(full == old['full_targets'], 'Original panel changed')
    require(len(full) == 256 and len({r['image_id'] for r in full}) == 243 and len(smoke) == 4, 'Panel denominator')
    require(not ({r['image_id'] for r in full} & {r['image_id'] for r in smoke}), 'Smoke overlaps full')
    records = old['verified_cache_and_image_records']
    for r in records:
        p = BASE / 'trace' / r['cache']
        require(sha256(p) == r['cache_sha256'], 'Cache changed')
        img = ROOT / f"assets/datasets/coco_native_s0_20261013/images/val2017/{r['image_id']:012d}.jpg"
        require(sha256(img) == r['image_sha256'], 'Image changed')
        expected[p.relative_to(ROOT).as_posix()] = r['cache_sha256']
        expected[img.relative_to(ROOT).as_posix()] = r['image_sha256']
    baseline = read_jsonl(BASE / 'ledger/PER_GT_FAILURE.jsonl')
    denoms = {}
    for scope, targets in [('smoke', smoke), ('full', full)]:
        ids = {r['image_id'] for r in targets}
        rows = [r for r in baseline if r['image_id'] in ids and not r['is_ignored']]
        denoms[scope] = dict(images=len(ids), pairs=len(targets), ordinary_gt=len(rows),
                             native_success_gt=sum(r['task_mask75'] == 'success' for r in rows),
                             native_prediction_rows=sum(len(r['prediction_indices']) for r in records if r['image_id'] in ids))
    require(psutil.virtual_memory().available >= 3 * 1024**3 and shutil.disk_usage(ROOT).free >= 4 * 1024**3, 'Insufficient resources')
    write_new(STUDY / 'INPUT_MANIFEST.json', dict(files=expected, environment=env, profile=profile,
              full_targets=full, smoke_targets=smoke, denominator=denoms, cache_and_image_records=records,
              zero_new_scientific_outcomes=True, source_context=read_json(STUDY / 'SOURCE_CONTEXT.json')))
    write_new(STUDY / 'PREPARATION.json', dict(status='passed', at=now(), pid=os.getpid(), host=socket.gethostname(),
              inputs_verified=len(expected), cached_images_verified=len(records), denominator=denoms,
              free_ram_bytes=psutil.virtual_memory().available, free_disk_bytes=shutil.disk_usage(ROOT).free,
              native_environment=env, new_training_steps=0, model_forward_calls=0, gate_outcomes_computed=False))
    files = dict(locked['files'])
    for name in ('INPUT_MANIFEST.json', 'PREPARATION.json', 'TESTS.json', 'SOURCE_FREEZE.json'):
        p = STUDY / name
        files[p.relative_to(ROOT).as_posix()] = sha256(p)
    write_new(STUDY / 'EXECUTION_FREEZE.json', dict(frozen_at=now(), files=files,
              protocol_sha256=sha256(STUDY / 'PROTOCOL.md'), input_manifest_sha256=sha256(STUDY / 'INPUT_MANIFEST.json')))
    study = read_json(STUDY / 'study.json')
    study.update(plan_status='frozen', execution_status='not_started', artifact_status='frozen_inputs_and_sources',
                 current_phase='smoke_ready', denominator=denoms, protocol_sha256=sha256(STUDY / 'PROTOCOL.md'))
    save(STUDY / 'study.json', study)
    print(json.dumps(dict(status='frozen', denominator=denoms, files=len(files), inputs=len(expected))))


if __name__ == '__main__':
    main()
