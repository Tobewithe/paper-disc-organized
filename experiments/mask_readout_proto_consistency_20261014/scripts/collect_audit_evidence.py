"""Package closed Runs and independently recheck integer-count audit tables."""
import argparse
from collections import Counter
import math
from pathlib import Path
import sys

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
PRIOR_STUDY = ROOT/'experiments/mask_readout_fit_diagnostic_20261014'
PRIOR_RUN = PRIOR_STUDY/'runs/RUN_FIT_DIAG_FULL_1de2b1149b9d485eadf6afd98872c926'
sys.path.insert(0, str(PRIOR_STUDY/'scripts'))
import collect_diagnostic_evidence as transfer
transfer.STUDY = STUDY
read, rows, digest, save = transfer.read, transfer.rows, transfer.digest, transfer.save


def audit(run):
    summary, contract = read(run/'SUMMARY.json'), read(run/'EVAL_CONTRACT.json')
    assert read(run/'run.json')['execution_status'] == 'completed'
    assert read(run/'CONFIG_AUDIT.json')['status'] == 'passed'
    assert read(run/'LOCAL_TRANSFER_VERIFICATION.json')['status'] == 'passed'
    for i, (relative, sha) in enumerate(read(run/'SOURCE_MANIFEST.json')['files'].items()):
        assert digest(run/'source_snapshot'/(f'{i:02d}_'+Path(relative).name)) == sha, relative
    assert digest(run/'PROTOCOL.md') == read(run/'run.json')['protocol_sha256']
    assert digest(run/'EVAL_CONTRACT.json') == read(run/'run.json')['contract_sha256']
    assert contract['new_training_steps'] == 0 and contract['binary_prediction'] == 'strict logit>0'
    state = read(run/'FROZEN_MODEL_STATE.json')
    assert state['before'] == state['after'] and state['new_optimizer_steps'] == 0 and state['model_eval']
    assert not state['parameters_require_grad'] and not state['parameter_gradients_present']
    expected_ids = set(contract['image_ids'])
    key = lambda r: (r['image_id'], r['annotation_id'], r['training_output_id'])
    previous = {key(r): r for r in rows(PRIOR_RUN/'TRAINING_PAIR_EFFECTS.jsonl') if r['image_id'] in expected_ids}
    actual = rows(run/'PAIR_PROTO_AUDIT.jsonl')
    assert len(actual) == len(previous) == len({key(r) for r in actual})
    assert {key(r) for r in actual} == set(previous)
    classify = lambda x: 'down' if x < -1e-12 else 'up' if x > 1e-12 else 'tie'
    stage, directions = Counter(), Counter()
    empty = {'zero_proto': 0, 'B_proto': 0}
    discrepancies = 0
    gt_empty = 0
    max_pair_error = 0.
    for r in actual:
        old = previous[key(r)]
        assert all(r[k] == v for k, v in old.items())
        for field, prior_field in [('recomputed_zero_gt_bce', 'zero_gt_bce'), ('recomputed_B_gt_bce', 'B_gt_bce')]:
            error = abs(r[field]-old[prior_field])
            max_pair_error = max(max_pair_error, error)
            assert error <= 1e-7
        for arm in ('zero_proto', 'B_proto'):
            c = r[arm]
            assert all(type(c[k]) is int and c[k] >= 0 for k in ('tp', 'fp', 'fn', 'tn'))
            assert c['roi_pixels'] == c['tp']+c['fp']+c['fn']+c['tn'] > 0
            assert c['roi_pixels'] <= r['proto_hw'][0]*r['proto_hw'][1]
            assert c['gt_pixels'] == c['tp']+c['fn']
            union = c['tp']+c['fp']+c['fn']
            assert union == c['union_pixels'] and c['empty_union'] == (union == 0)
            assert c['iou'] == (c['tp']/union if union else 1.)
            empty[arm] += c['empty_union']
        assert r['zero_proto']['roi_pixels'] == r['B_proto']['roi_pixels']
        assert r['zero_proto']['gt_pixels'] == r['B_proto']['gt_pixels']
        gt_empty += r['zero_proto']['gt_pixels'] == 0
        difference = r['B_proto']['iou']-r['zero_proto']['iou']
        assert difference == r['delta_proto_iou']
        d = classify(difference)
        assert d == r['proto_direction']
        bce_difference = r['B_gt_bce']-r['zero_gt_bce']
        loss_direction = 'down' if bce_difference < -1e-6 else 'up' if bce_difference > 1e-6 else 'tie'
        assert r['direction'] == 'loss_'+loss_direction+'_iou_'+classify(r['B_full_iou']-r['zero_full_iou'])
        stage[r['direction']+'_proto_'+d] += 1
        if loss_direction == 'down' and r['B_full_iou']-r['zero_full_iou'] < -1e-12:
            discrepancies += 1
            directions[d] += 1
    directions = {d: directions[d] for d in ('down', 'tie', 'up')}
    assert directions == summary['discrepant_proto_directions']
    assert dict(stage) == summary['all_stage_direction_counts']
    assert empty == summary['empty_union_counts'] and gt_empty == summary['empty_GT_in_ROI_pairs']
    assert discrepancies == summary['known_discrepant_pairs'] and len(actual) == summary['supervised_pairs']
    assert math.isclose(math.fsum(r['delta_proto_iou'] for r in actual)/len(actual), summary['mean_delta_proto_iou'], abs_tol=1e-15)
    fraction = directions['down']/discrepancies if discrepancies else None
    assert fraction == summary['early_reversal_fraction']
    images = rows(run/'IMAGE_CHECKS.jsonl')
    old_images = {r['image_id']: r for r in rows(PRIOR_RUN/'LOSS_BY_IMAGE.jsonl')}
    original_checks = {r['image_id']: r for r in rows(PRIOR_RUN/'ZERO_DELTA_AND_JOIN_CHECKS.jsonl')}
    assert len(images) == len(expected_ids) and {r['image_id'] for r in images} == expected_ids
    pairs_per_image = Counter(r['image_id'] for r in actual)
    max_image_error = 0.
    for image in images:
        assert image['saved_training_join_equal'] and image['prior_endpoint_output_coverage_equal']
        assert image['outputs'] == original_checks[image['image_id']]['outputs']
        assert image['valid_supervision_pairs'] == pairs_per_image[image['image_id']]
        for arm in ('zero', 'B'):
            for k, value in old_images[image['image_id']][arm].items():
                if k in ('total_loss', 'gt_loss', 'preservation_loss'):
                    error = abs(image[arm][k]-value)
                    max_image_error = max(max_image_error, error)
                    assert error <= 1e-7
                else:
                    assert image[arm][k] == value
    assert max_pair_error == summary['measurement']['max_pair_BCE_abs_error']
    assert max_image_error == summary['measurement']['max_image_loss_abs_error']
    if summary['scope'] == 'full':
        assert len(actual) == 14268 and discrepancies == 3950
        early = fraction > .5
        expected_localization = 'majority_already_reverses_on_same_proto_ROI' if early else 'majority_first_negative_binary_change_after_proto_ROI'
        assert summary['diagnostic_decision']['majority_early_reversal'] == early
        assert summary['diagnostic_decision']['localization'] == expected_localization
        assert summary['diagnostic_decision']['prior_S2_negative_decision_retained']
    result = {'status': 'passed', 'scope': summary['scope'], 'images': len(images), 'supervised_pairs': len(actual),
        'discrepant_pairs': discrepancies, 'discrepant_proto_directions': directions, 'early_reversal_fraction': fraction,
        'empty_union_counts': empty, 'empty_GT_in_ROI_pairs': gt_empty, 'new_training_steps': 0,
        'max_pair_BCE_abs_error': max_pair_error, 'max_image_loss_abs_error': max_image_error,
        'integer_count_IoU_and_classification_independently_recomputed': True,
        'all_old_supervision_and_endpoint_fields_equal': True, 'summary_sha256': digest(run/'SUMMARY.json'),
        'configuration_audit_sha256': digest(run/'CONFIG_AUDIT.json'), 'audit_script_sha256': digest(Path(__file__)),
        'claim_limit': 'independent arithmetic and evidence identity check; GPU masks not re-inferred locally'}
    save(run/'LOCAL_RESULT_RECHECK.json', result)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('package', 'verify', 'audit'))
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    for path in args.runs:
        {'package': transfer.package, 'verify': transfer.unpack_verify, 'audit': audit}[args.action](path)
