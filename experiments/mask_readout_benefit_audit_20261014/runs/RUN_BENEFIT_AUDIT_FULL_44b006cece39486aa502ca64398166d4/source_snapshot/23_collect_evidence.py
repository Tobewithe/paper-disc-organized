"""Transfer closed Runs and independently recompute serialized evidence."""
import argparse
import json
from pathlib import Path
import sys
from pilot_common import *
sys.path.insert(0, str(ROOT/'experiments/mask_readout_fit_diagnostic_20261014/scripts'))
import collect_diagnostic_evidence as transfer
transfer.STUDY = STUDY


def equal(a, b):
    if isinstance(a, dict):
        require(set(a) == set(b), 'independent dictionary fields differ')
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, list):
        require(len(a) == len(b), 'independent list length differs')
        for x, y in zip(a, b):
            equal(x, y)
    elif isinstance(a, float):
        require(abs(a-b) <= 1e-12, 'independent numerical difference')
    else:
        require(a == b, 'independent value differs')


def audit(path):
    meta = read_json(path/'run.json')
    contract = read_json(path/'EVAL_CONTRACT.json')
    require(meta['execution_status'] == 'completed' and read_json(path/'CONFIG_AUDIT.json')['status'] == 'passed', 'unverified run')
    for i, (rel, digest) in enumerate(read_json(path/'SOURCE_MANIFEST.json')['files'].items()):
        require(sha256(path/'source_snapshot'/(f'{i:02d}_'+Path(rel).name)) == digest, 'returned source changed')
    require(sha256(path/'PROTOCOL.md') == meta['protocol_sha256'], 'protocol snapshot mismatch')
    require(sha256(path/'EVAL_CONTRACT.json') == meta['contract_sha256'], 'contract snapshot mismatch')
    report = {'status': 'passed', 'run_id': path.name, 'frozen_sources_protocol_contract_verified': True}
    if meta['kind'] == 'SOFT_IOU_TRAIN':
        from audit_s2_completed import audit as audit_training
        audit_training(path)
        from soft_iou_loss import LOSS_SPEC
        require(contract['gt_loss'] == LOSS_SPEC, 'GT definition changed')
        original = read_json(B_RUN/'EVAL_CONTRACT.json')
        require(all(contract[k] == v for k, v in original.items()), 'original training config mismatch')
        for name in ('initial.pt', 'IMAGE_ORDERS.json', 'TRAINING_JOIN.json'):
            require(sha256(path/name) == sha256(B_RUN/name), 'training artifact not identical to old B: '+name)
        report.update(original_initial_order_join_byte_identical=True, original_training_config_fields_identical=True,
                      optimizer_steps=1500, images_seen=6000, independent_checkpoint_audit_sha256=sha256(path/'INDEPENDENT_CONFIG_AUDIT.json'))
    elif meta['kind'] == 'SOFT_IOU_EVAL':
        from posttrain_audit import read_jsonl
        from damage_acceptance import score_damage
        summary = read_json(path/'SUMMARY.json')
        effects = read_jsonl(path/'B_FIXED_PAIR_EFFECTS.jsonl')
        target = original_evaluation.bootstrap(original_evaluation.image_effects(effects))
        overall = original_evaluation.bootstrap(original_evaluation.image_effects(effects, target_only=False))
        baseline = read_jsonl(BASE/'ledger/PER_GT_FAILURE.jsonl')
        require(sha256(BASE/'ledger/PER_GT_FAILURE.jsonl') == contract['ledger_sha256'], 'baseline identity changed')
        method = read_jsonl(path/'B_GT_LEDGER.jsonl')
        panel = read_json(path/'DAMAGE_PANEL.json')
        require(panel['checkpoint_sha256'] == contract['checkpoints']['B']['sha256'] and
                panel['inference_contract_sha256'] == sha256(path/'EVAL_CONTRACT.json'), 'panel binding mismatch')
        damage = score_damage(panel, baseline, [r for r in method if r['image_id'] in set(panel['image_ids'])])
        equal(target, summary['B']['target'])
        equal(overall, summary['B']['all_bbox75'])
        equal(damage, summary['B']['damage'])
        indices = read_json(path/'B_ROW_MAPPING.json')['original_global_indices']
        wanted = {(r['image_id'], r['annotation_id']): r for r in baseline if r['image_id'] in set(read_json(path/'EVALUATION_IMAGES.json'))}
        require({(r['image_id'], r['annotation_id']) for r in method} == set(wanted), 'GT coverage differs')
        for row in method:
            old = wanted[(row['image_id'], row['annotation_id'])]
            oid = row['bbox75_match_output_id']
            mapped = None if oid is None else panel['source_prediction_sha']+':'+str(indices[int(oid.rsplit(':', 1)[1])])
            require(mapped == old['bbox75_match_output_id'] and row['is_ignored'] == old['is_ignored'], 'GT/bbox identity differs')
        old_effects = read_jsonl(DEV_RUN/'B_FIXED_PAIR_EFFECTS.jsonl')
        a = original_evaluation.image_effects(effects)
        b = original_evaluation.image_effects(old_effects)
        contrast = original_evaluation.bootstrap({i: a[i]-b[i] for i in a})
        equal(contrast, summary['soft_IoU_minus_BCE_target'])
        gates = {'target_effect': target['mean'] >= .005 and target['ci95'][0] > 0,
                 'all_bbox75_guardrail': overall['mean'] >= -.001,
                 'success_damage': damage['gate'] == 'damage_guardrail_passed'}
        equal(gates, summary['gates'])
        state = 'passed' if all(gates.values()) else ('blocked' if gates['target_effect'] and gates['all_bbox75_guardrail'] and damage['gate'] == 'blocked_insufficient_precision' else 'failed')
        require(state == read_json(path/'DECISION.json')['status'], 'adjudication mismatch')
        report.update(recomputed_target=target, recomputed_all_bbox75=overall, recomputed_damage=damage,
                      recomputed_gates=gates, scientific_status=state, exact_GT_and_bbox_identity=True)
    else:
        require(meta['kind'] == 'SOFT_IOU_SMOKE', 'unknown run kind')
        checks = read_json(path/'ENGINEERING_CONTROLS.json')
        require(all(r['exact_zero_delta_RLE'] for rows in checks.values() for r in rows), 'smoke replay failed')
        report.update(zero_delta_images=sum(map(len, checks.values())), smoke_optimizer_steps=2)
    output = path/'LOCAL_INDEPENDENT_RECHECK.json'
    if output.exists():
        equal(read_json(output), report)
    else:
        write_new(output, report)
    print('INDEPENDENT_RECHECK_PASSED', path.name, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('package', 'verify', 'audit'))
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    if args.mode == 'package':
        transfer.package(args.run)
    elif args.mode == 'verify':
        transfer.unpack_verify(args.run)
    else:
        audit(args.run)
