"""Package closed Runs and independently audit returned tables; never runs a model."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import zipfile

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def rows(path):
    with path.open(encoding='utf8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, data):
    with path.open('x', encoding='utf8') as out:
        json.dump(data, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write('\n')


def package(run):
    assert run.resolve().is_relative_to((STUDY/'runs').resolve())
    assert read(run/'run.json')['execution_status'] in ('completed', 'error')
    selected, omitted = {}, []
    for path in sorted(run.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(run).as_posix()
        if path.name in ('evidence.zip', 'transfer.json', 'manifest.sha256'):
            continue
        if path.name.endswith('predictions.json'):
            omitted.append({'relative_path': relative, 'sha256': digest(path), 'size_bytes': path.stat().st_size,
                            'availability': 'retained_on_execution_host'})
        else:
            selected[relative] = digest(path)
    receipt = {'source_host': socket.gethostname(), 'source_path': str(run.resolve()),
        'prepared_at': datetime.now(timezone.utc).isoformat(), 'status': 'prepared',
        'scope': 'all evidence and full GT/effect tables; large prediction JSON retained remotely',
        'files': selected, 'omitted': omitted, 'complete_artifact_inventory': 'ARTIFACT_MANIFEST.json'}
    save(run/'transfer.json', receipt)
    with (run/'manifest.sha256').open('x', encoding='utf8') as out:
        for name, sha in selected.items():
            out.write(sha+'  '+name+'\n')
    with zipfile.ZipFile(run/'evidence.zip', 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in [*selected, 'transfer.json', 'manifest.sha256']:
            archive.write(run/name, name)
    print(json.dumps({'run_id': run.name, 'files': len(selected), 'zip_bytes': (run/'evidence.zip').stat().st_size,
                      'zip_sha256': digest(run/'evidence.zip')}), flush=True)


def unpack_verify(run):
    with zipfile.ZipFile(run/'evidence.zip') as archive:
        for name in archive.namelist():
            target = (run/name).resolve()
            assert target.is_relative_to(run.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as out:
                out.write(archive.read(name))
    receipt = read(run/'transfer.json')
    for name, sha in receipt['files'].items():
        assert digest(run/name) == sha, name
    artifact_records = {r['relative_path']: r for r in read(run/'ARTIFACT_MANIFEST.json')['artifacts']}
    for name, sha in receipt['files'].items():
        if name in artifact_records:
            assert sha == artifact_records[name]['sha256'], name
            assert (run/name).stat().st_size == artifact_records[name]['size_bytes'], name
    save(run/'LOCAL_TRANSFER_VERIFICATION.json', {'status': 'passed', 'files_verified': len(receipt['files']),
        'zip_sha256': digest(run/'evidence.zip'), 'source': receipt['source_path'],
        'omitted': receipt['omitted'], 'verified_at': datetime.now(timezone.utc).isoformat()})


def audit(run):
    import numpy as np
    summary, contract = read(run/'SUMMARY.json'), read(run/'EVAL_CONTRACT.json')
    assert read(run/'run.json')['execution_status'] == 'completed'
    assert read(run/'CONFIG_AUDIT.json')['status'] == 'passed'
    source_manifest = read(run/'SOURCE_MANIFEST.json')['files']
    for i, (relative, sha) in enumerate(source_manifest.items()):
        assert digest(run/'source_snapshot'/(f'{i:02d}_'+Path(relative).name)) == sha, relative
    assert digest(run/'PROTOCOL.md') == read(run/'run.json')['protocol_sha256']
    assert digest(run/'EVAL_CONTRACT.json') == read(run/'run.json')['contract_sha256']
    assert contract['new_training_steps'] == 0
    state = read(run/'FROZEN_MODEL_STATE.json')
    assert state['before'] == state['after'] and state['new_optimizer_steps'] == 0
    assert not state['parameters_require_grad'] and not state['parameter_gradients_present']
    zero_checks = rows(run/'ZERO_DELTA_AND_JOIN_CHECKS.jsonl')
    expected_ids = set(contract['image_ids'])
    assert len(zero_checks) == len(expected_ids)
    assert {r['image_id'] for r in zero_checks} == expected_ids
    assert all(r['exact_zero_delta_RLE'] and r['saved_training_join_equal'] and r['nonmask_fields_unchanged'] for r in zero_checks)
    baseline, method = rows(run/'BASELINE_GT_LEDGER.jsonl'), rows(run/'B_GT_LEDGER.jsonl')
    old = {(r['image_id'], r['annotation_id']): r for r in baseline}
    new = {(r['image_id'], r['annotation_id']): r for r in method}
    assert len(old) == len(baseline) and len(new) == len(method) and set(old) == set(new)
    mapping = read(run/'ROW_MAPPING.json')
    assert len(mapping['original_global_indices']) == sum(r['outputs'] for r in zero_checks)
    effects = rows(run/'FIXED_PAIR_EFFECTS.jsonl')
    table = {(r['image_id'], r['annotation_id']): r for r in effects}
    assert len(table) == len(effects)
    target_groups, all_groups = defaultdict(list), defaultdict(list)
    success_keys, damaged_keys = [], []
    for key, before in old.items():
        after = new[key]
        assert before['is_ignored'] == after['is_ignored']
        oid = after['bbox75_match_output_id']
        mapped = None if oid is None else mapping['source_prediction_sha256']+':'+str(mapping['original_global_indices'][int(oid.rsplit(':', 1)[1])])
        assert mapped == before['bbox75_match_output_id']
        if before['is_ignored']:
            continue
        if before['task_mask75'] == 'success':
            success_keys.append(key)
            if after['task_mask75'] == 'failure':
                damaged_keys.append(key)
        if before['bbox75_match_output_id'] is None:
            continue
        effect = after['mask_iou_of_bbox_match']-before['mask_iou_of_bbox_match']
        assert abs(table[key]['delta_iou']-effect) < 1e-14
        all_groups[key[0]].append(effect)
        is_target = before['task_mask75'] == 'failure' and before['mask_iou_of_bbox_match'] < .75
        assert (table[key]['population'] == 'T') == is_target
        if is_target:
            target_groups[key[0]].append(effect)
    assert len(effects) == sum(map(len, all_groups.values()))
    damage = read(run/'DAMAGE_CENSUS.json')
    assert len(success_keys) == damage['population_N'] == damage['observed_n']
    assert len(damaged_keys) == damage['damaged_k']
    assert set(damaged_keys) == {tuple(k) for k in damage['damaged_GT_keys']}
    assert len(damaged_keys)/len(success_keys) == damage['fraction']
    recomputed = {}
    if summary['scope'] == 'full':
        for name, groups in [('target', target_groups), ('all_bbox75', all_groups)]:
            values = np.array([math.fsum(groups[i])/len(groups[i]) for i in sorted(groups)], dtype=np.float64)
            rng = np.random.default_rng(1729)
            samples = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(1)
            intervals = np.quantile(samples, [.025, .975], method='linear')
            assert abs(float(values.mean())-summary[name]['mean']) < 1e-14
            assert max(abs(float(v)-u) for v, u in zip(intervals, summary[name]['ci95'])) < 1e-14
            recomputed[name] = {'mean': float(values.mean()), 'ci95': [float(v) for v in intervals], 'images': len(values)}
        gates = {'target_effect': recomputed['target']['mean'] >= .005 and recomputed['target']['ci95'][0] > 0,
                 'all_bbox75_guardrail': recomputed['all_bbox75']['mean'] >= -.001,
                 'success_damage_census': damage['fraction'] <= .01}
        assert gates == summary['diagnostic_decision']['fit_diagnostic_gates']
    losses = rows(run/'LOSS_BY_IMAGE.jsonl')
    assert len(losses) == len(expected_ids) and {r['image_id'] for r in losses} == expected_ids
    rec_loss = {arm: {name: math.fsum(r[arm][name] for r in losses)/len(losses)
        for name in ('total_loss', 'gt_loss', 'preservation_loss')} for arm in ('zero', 'B')}
    for arm in ('zero', 'B'):
        for name, value in rec_loss[arm].items():
            assert abs(value-summary['fixed_checkpoint_loss'][arm][name]) < 1e-14
    pairs = rows(run/'TRAINING_PAIR_EFFECTS.jsonl')
    assert len(pairs) == sum(r['B']['matched_supervised_instances'] for r in losses)
    assert len({(r['image_id'], r['training_output_id']) for r in pairs}) == len(pairs)
    directions = Counter()
    pair_losses = defaultdict(lambda: {'zero': [], 'B': []})
    joins = read(ROOT/'experiments/mask_readout_mechanism_v2_20261012/runs/RUN_S2_TRAIN_B_6c7bc7e8c0dc48509a733d11e60f0ee2/TRAINING_JOIN.json')['images']
    # Source global indices preserve per-image order, so reconstruct the actual saved join rank.
    ranks, offset = {}, 0
    for check in zero_checks:
        image_indices = mapping['original_global_indices'][offset:offset+check['outputs']]
        for rank, index in enumerate(image_indices):
            ranks[(check['image_id'], index)] = rank
        offset += check['outputs']
    assert offset == len(mapping['original_global_indices'])
    for pair in pairs:
        k = (pair['image_id'], pair['annotation_id'])
        assert k in old
        assert pair['same_as_bbox75_pair'] == (pair['training_output_id'] == old[k]['bbox75_match_output_id'])
        index = int(pair['training_output_id'].rsplit(':', 1)[1])
        rank = ranks[(pair['image_id'], index)]
        assert joins[str(pair['image_id'])]['output_rank_to_annotation_id'][str(rank)] == pair['annotation_id']
        bce = pair['B_gt_bce']-pair['zero_gt_bce']
        iou = pair['B_full_iou']-pair['zero_full_iou']
        assert abs(bce-pair['delta_gt_bce']) < 1e-14 and abs(iou-pair['delta_full_iou']) < 1e-14
        direction = 'loss_down' if bce < -1e-6 else 'loss_up' if bce > 1e-6 else 'loss_tie'
        direction += '_iou_down' if iou < -1e-12 else '_iou_up' if iou > 1e-12 else '_iou_tie'
        assert direction == pair['direction']
        directions[direction] += 1
        pair_losses[pair['image_id']]['zero'].append(pair['zero_gt_bce'])
        pair_losses[pair['image_id']]['B'].append(pair['B_gt_bce'])
    assert dict(directions) == summary['fixed_checkpoint_loss']['per_instance_BCE_IoU_direction_counts']
    for image in losses:
        for arm in ('zero', 'B'):
            group = pair_losses[image['image_id']][arm]
            assert len(group) == image[arm]['matched_supervised_instances']
            gt = math.fsum(group)/len(group) if group else 0.
            assert abs(gt-image[arm]['gt_loss']) < 5e-7
            assert abs(image[arm]['gt_loss']+.1*image[arm]['preservation_loss']-image[arm]['total_loss']) < 5e-7
    result = {'status': 'passed', 'independent_recomputation': True, 'images': len(expected_ids),
        'all_zero_delta_outputs': sum(r['outputs'] for r in zero_checks), 'ordinary_GT': sum(not r['is_ignored'] for r in baseline),
        'target_GT': sum(map(len, target_groups.values())), 'fixed_bbox75_pairs': len(effects),
        'target_and_overall': recomputed, 'damage': {'k': len(damaged_keys), 'N': len(success_keys), 'fraction': damage['fraction']},
        'fixed_checkpoint_losses': rec_loss, 'training_pairs': len(pairs), 'BCE_IoU_directions': dict(directions),
        'new_training_steps': 0, 'configuration_audit_sha256': digest(run/'CONFIG_AUDIT.json'),
        'summary_sha256': digest(run/'SUMMARY.json'), 'audit_script_sha256': digest(Path(__file__)),
        'claim_limit': 'Recomputes saved tables and verifies identities; no independent re-inference or new experimental result.'}
    save(run/'LOCAL_RESULT_RECHECK.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('package', 'verify', 'audit'))
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    for path in args.runs:
        {'package': package, 'verify': unpack_verify, 'audit': audit}[args.action](path)
