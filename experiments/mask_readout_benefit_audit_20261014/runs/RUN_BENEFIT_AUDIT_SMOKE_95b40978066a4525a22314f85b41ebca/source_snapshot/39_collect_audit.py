"""Return frozen evidence and independently recompute counts and rank statistics."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT/'experiments/mask_readout_benefit_audit_20261014'
sys.path.insert(0, str(ROOT/'experiments/mask_readout_fit_diagnostic_20261014/scripts'))
import collect_diagnostic_evidence as transfer
transfer.STUDY = STUDY
from audit_statistics import FEATURES, STAGES, summarize, decide


def rank_auc(scores, labels):
    """Independent expanded-observation average-rank U statistic."""
    labels = np.asarray(labels, dtype=bool)
    nplus, nminus = int(labels.sum()), int((~labels).sum())
    if not nplus or not nminus:
        return None
    order = np.argsort(scores, kind='stable')
    sorted_scores = np.asarray(scores)[order]
    _, starts, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    ranks = np.repeat(starts+1+(counts-1)/2, counts)
    return float((ranks[labels[order]].sum()-nplus*(nplus+1)/2)/(nplus*nminus))


def audit(run):
    read = transfer.read
    meta, contract, summary = [read(run/name) for name in ('run.json', 'EVAL_CONTRACT.json', 'SUMMARY.json')]
    assert meta['execution_status'] == 'completed' and read(run/'CONFIG_AUDIT.json')['status'] == 'passed'
    assert transfer.digest(run/'PROTOCOL.md') == meta['protocol_sha256']
    assert transfer.digest(run/'EVAL_CONTRACT.json') == meta['contract_sha256']
    for i, (rel, sha) in enumerate(read(run/'SOURCE_MANIFEST.json')['files'].items()):
        assert transfer.digest(run/'source_snapshot'/(f'{i:02d}_'+Path(rel).name)) == sha, rel
    state = read(run/'FROZEN_MODEL_STATE.json')
    assert state['before'] == state['after'] and state['new_optimizer_steps'] == contract['new_training_steps'] == 0
    checks = transfer.rows(run/'IMAGE_REPLAY_CHECKS.jsonl')
    assert {r['image_id'] for r in checks} == set(contract['image_ids']) and len(checks) == len(contract['image_ids'])
    assert all(r['zero_RLE_equal'] and r['method_RLE_equal'] and r['nonmask_fields_equal'] for r in checks)
    rows = transfer.rows(run/'PAIR_MEASUREMENTS.jsonl')
    for row in rows:
        for name in STAGES:
            stage = row['stages'][name]
            if stage['delta'] is None:
                assert row['empty_roi'] and name in ('soft_roi', 'binary_roi')
                continue
            for side in ('before', 'after'):
                values = stage[side]
                if name == 'soft_roi':
                    iou = (values['intersection']+1e-6)/(values['union']+1e-6)
                    assert abs(iou-values['iou']) < 1e-7
                else:
                    union = values['tp']+values['fp']+values['fn']
                    iou = values['tp']/union if union else 1.0
                    assert abs(iou-values['iou']) < 1e-12
                    assert values['pixels'] == sum(values[k] for k in ('tp', 'fp', 'fn', 'tn'))
            assert abs(stage['delta']-(stage['after']['iou']-stage['before']['iou'])) < 1e-12
        signal = row['signals']
        denominator = signal['native_foreground_pixels']
        assert signal['preserved_foreground'] == (signal['retained_foreground_pixels']/denominator if denominator else 1.)
        assert all(0 <= signal[f] <= 1+1e-12 for f in FEATURES)
    recalculated = summarize(rows, contract['dev_ids'])
    for key in recalculated:
        assert recalculated[key] == summary[key], key
    dev = [r for r in rows if r['role'] == 'dev_bbox75' and abs(r['stages']['binary_original']['delta']) > 1e-12]
    labels = np.array([r['stages']['binary_original']['delta'] > 0 for r in dev], dtype=bool)
    scores = {f: np.array([r['signals'][f] for r in dev]) for f in FEATURES}
    ids = sorted(contract['dev_ids'])
    by_image = {image: np.array([i for i, r in enumerate(dev) if r['image_id'] == image], dtype=int) for image in ids}
    rng = np.random.default_rng(1729)
    draws = {f: [] for f in FEATURES}
    for _ in range(2000):
        sampled = rng.integers(0, len(ids), len(ids))
        expanded = np.concatenate([by_image[ids[i]] for i in sampled])
        for feature in FEATURES:
            val = rank_auc(scores[feature][expanded], labels[expanded])
            if val is not None: draws[feature].append(val)
    comparisons = {}
    for feature in FEATURES:
        point = rank_auc(scores[feature], labels)
        old = summary['signals']['features'][feature]
        assert point is None and old['auc'] is None or abs(point-old['auc']) <= 1e-12
        ci = list(np.quantile(draws[feature], [.05/6, 1-.05/6], method='linear')) if draws[feature] else None
        assert ci is None and old['familywise95_interval'] is None or np.allclose(ci, old['familywise95_interval'], rtol=0, atol=1e-12)
        comparisons[feature] = {'independent_auc': point, 'independent_familywise95_interval': ci}
    if contract['scope'] == 'full':
        decision = decide(recalculated)
        assert decision == summary['route_decision']
        assert summary['pair_roles'] == {'dev_bbox75': 2925, 'success_segm75': 1024}
        assert summary['damaged_success_GT'] == 53
    receipt = {'status': 'passed', 'run_id': run.name, 'source_snapshot_and_transfer_verified': True,
               'all_integer_IoU_recomputed': True, 'independent_average_rank_AUC_and_expanded_image_cluster_bootstrap': comparisons,
               'model_steps': 0, 'decision_recomputed': contract['scope'] == 'full'}
    transfer.save(run/'LOCAL_INDEPENDENT_RECHECK.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('package', 'verify', 'audit'))
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    {'package': transfer.package, 'verify': transfer.unpack_verify, 'audit': audit}[args.mode](args.run)
