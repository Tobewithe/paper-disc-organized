"""Recompute saved RLE identities, bounds, statistics and complete COCO matching."""
from lea_common import *
import argparse, contextlib, copy, hashlib, io, math, time
from collections import defaultdict
import numpy as np
from pycocotools import mask as masks


def dec(r):
    return masks.decode(r).astype(bool)


def score(a, b):
    denominator = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / denominator) if denominator else 0.


def close(a, b, label):
    require(abs(float(a) - float(b)) <= 1e-12, label)


def verify(run, replay=False):
    started = time.monotonic()
    run = Path(run)
    cfg = read_json(run / 'CONFIG.json')
    summary = read_json(run / 'STATISTICS.json')
    pairs = read_jsonl(run / 'PAIRS.jsonl')
    gate_list = read_jsonl(run / 'GATES.jsonl')
    gates = {r['global_index']: r for r in gate_list}
    require(len(gates) == len(gate_list), 'Duplicate gate row')
    source_map = read_json(run / 'SOURCE_SNAPSHOT_MAP.json')
    for rel, digest in read_json(run / 'SOURCE_MANIFEST.json')['files'].items():
        require(sha256(run / source_map[rel]) == digest, 'Snapshot hash mismatch')
    mapping = read_json(run / 'ROW_MAPPING.json')['global_indices']
    preds = {k: read_json(run / (k + '_predictions.json')) for k in ('N', 'G', 'G_B')}
    for k in preds:
        require(len(preds[k]) == len(mapping), 'Full prediction denominator')
    positions = {gi: j for j, gi in enumerate(mapping)}
    outside_changes = 0
    for j, gi in enumerate(mapping):
        n, g, m = [preds[k][j] for k in ('N', 'G', 'G_B')]
        require({k: v for k, v in n.items() if k != 'segmentation'} ==
                {k: v for k, v in g.items() if k != 'segmentation'} ==
                {k: v for k, v in m.items() if k != 'segmentation'}, 'Nonmask identity')
        if gi not in gates:
            require(n == g == m, 'Modified row has no gate receipt')
            continue
        b = dec(gates[gi]['B_rle'])
        nm, gm, mm = [dec(x['segmentation']) for x in (n, g, m)]
        require(np.array_equal(mm, np.where(b, gm, nm)), 'Mixed mask composition differs')
        outside_changes += int(np.count_nonzero((mm != nm) & ~b))
    require(outside_changes == 0, 'Region exterior changed')
    names = ['N', 'G', 'C', 'O_B', 'C_B', 'G_B', 'O_R']
    calculated = []
    mask_checks = 0
    for r in pairs:
        gi = r['global_index']
        b, roi = dec(r['B_rle']), dec(r['ROI_rle'])
        require(np.array_equal(b, dec(gates[gi]['B_rle'])) and not np.any(b & ~roi), 'Gate mismatch')
        require(int(b.sum()) == r['editable_pixels'] and int(roi.sum()) == r['roi_pixels'], 'Support count mismatch')
        n, c, g, y = [dec(r[k]) for k in ('native_rle', 'teacher_rle', 'existing_rle', 'GT_rle')]
        require(preds['N'][positions[gi]]['segmentation'] == r['native_rle'], 'Pair native identity')
        require(preds['G'][positions[gi]]['segmentation'] == r['existing_rle'], 'Pair G identity')
        actual = dict(N=n, C=c, G=g, O_B=np.where(b, y, n),
                      C_B=np.where(b, c, n), G_B=np.where(b, g, n))
        scores = {}
        for k, value in actual.items():
            require(np.array_equal(value, dec(r['arms'][k]['rle'])), 'Arm RLE differs: ' + k)
            scores[k] = score(value, y)
            close(scores[k], r['arms'][k]['iou'], 'Pair IoU')
            require(int(np.count_nonzero((value != n) & ~b)) == r['arms'][k]['changed_outside'], 'Pixel audit differs')
            mask_checks += 1
        require(scores['O_B'] + 1e-12 >= scores['N'], 'Exact oracle below native')
        possible = np.flatnonzero(roi)
        random_scores = []
        for repeat in r['random_oracles']:
            key = f"{cfg['seed']}:{r['image_id']}:{r['output_detection_id']}:{repeat['repeat']}".encode('ascii')
            seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')
            require(seed == repeat['seed'], 'Random support seed changed')
            rb = np.zeros(roi.size, dtype=bool)
            if b.sum():
                selected = np.random.Generator(np.random.PCG64(seed)).choice(possible, int(b.sum()), replace=False)
                rb[selected] = True
            rb = rb.reshape(roi.shape)
            val = score(np.where(rb, y, n), y)
            close(val, repeat['iou'], 'Random oracle IoU')
            random_scores.append(val)
        scores['O_R'] = float(np.mean(random_scores))
        close(scores['O_R'], r['O_R'], 'Random oracle reduction')
        close(r['editable_fraction'], b.sum()/roi.sum() if roi.any() else 0., 'Editable fraction')
        calculated.append(dict(image_id=r['image_id'], scores=scores, editable=r['editable_fraction']))
    ids = sorted({r['image_id'] for r in calculated})
    by = defaultdict(list)
    for r in calculated:
        by[r['image_id']].append(r)
    matrix = np.array([[sum(r['scores'][k] for r in by[i])/len(by[i]) for k in names] for i in ids])
    boot = np.load(run / 'BOOTSTRAP.npz')
    require(boot['image_ids'].tolist() == ids, 'Bootstrap image order')
    fresh = np.random.Generator(np.random.PCG64(cfg['seed'])).integers(0, len(ids), size=boot['indices'].shape)
    require(fresh.shape == (2000, len(ids)) and np.array_equal(fresh, boot['indices']), 'Bootstrap draws')
    for j, k in enumerate(names):
        close(matrix[:, j].mean(), summary['absolute'][k], 'Absolute macro IoU')
        if k != 'O_R':
            require(sum(r['scores'][k] >= .75 for r in calculated) == summary['mask75'][k], 'Mask75 count')
    for key, recorded in summary['comparisons'].items():
        a, b = key.split('-')
        v = matrix[:, names.index(a)] - matrix[:, names.index(b)]
        close(v.mean(), recorded['mean'], 'Paired mean')
        ci = np.quantile(np.mean(v[fresh], axis=1), [.025, .975], method='linear')
        for x, y in zip(ci, recorded['ci95']):
            close(x, y, 'Bootstrap interval')
    require(len(calculated) == summary['pairs'] and len(ids) == summary['images'], 'Summary denominator')
    witness = [r for r in calculated if r['scores']['C'] >= .75]
    retention = sum(r['scores']['O_B'] >= .75 for r in witness)
    require(len(witness) == summary['support']['C_witnesses'] and retention == summary['support']['C_witness_retained_by_O_B'], 'Witness retention')
    frac = np.mean([np.mean([r['editable'] for r in by[i]]) for i in ids])
    close(frac, summary['support']['editable_fraction_image_macro'], 'Localization reduction')
    # Independent evaluator setup uses full default IoU thresholds, then selects .75.
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    dataset = read_json(run / 'ANNOTATIONS_SUBSET.json')
    matches = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for arm, prediction in preds.items():
            coco = COCO()
            coco.dataset = copy.deepcopy(dataset)
            coco.createIndex()
            res = coco.loadRes(copy.deepcopy(prediction))
            ev = COCOeval(coco, res, 'segm')
            ev.params.imgIds = ids
            ev.params.maxDets = [1, 10, 100]
            ev.evaluate()
            t = next(i for i, v in enumerate(ev.params.iouThrs) if abs(v-.75) < 1e-12)
            all_rows = []
            for item in ev.evalImgs:
                if item is None or item['aRng'] != ev.params.areaRng[0]:
                    continue
                for j, aid in enumerate(item['gtIds']):
                    did = int(item['gtMatches'][t, j])
                    ignored = bool(item['gtIgnore'][j])
                    all_rows.append(dict(image_id=int(item['image_id']), annotation_id=int(aid),
                                         ignored=ignored, success=bool(did) and not ignored,
                                         matched_global_index=mapping[did-1] if did else None))
            all_rows.sort(key=lambda r: (r['image_id'], r['annotation_id']))
            require(all_rows == read_jsonl(run / (arm + '_MATCHES.jsonl')), 'Independent COCO matching differs')
            matches[arm] = {(r['image_id'], r['annotation_id']): r for r in all_rows}
    old_ledger = {(r['image_id'], r['annotation_id']): r for r in read_jsonl(run / 'SOURCE_GT_LEDGER.jsonl')}
    require(set(old_ledger) == set(matches['N']), 'Original ledger coverage')
    for key, r in matches['N'].items():
        old = old_ledger[key]
        require(r['ignored'] == old['is_ignored'] and r['success'] == (old['task_mask75'] == 'success'), 'Original S0 status changed')
        if r['success']:
            require(r['matched_global_index'] == int(old['segm75_match_output_id'].rsplit(':', 1)[1]), 'Original match identity')
    success_keys = [key for key, r in matches['N'].items() if r['success']]
    damage = {}
    for arm in ('G', 'G_B'):
        lost = sorted(key[1] for key in success_keys if not matches[arm][key]['success'])
        d = summary['task_damage'][arm]
        require(len(lost) == d['damaged'] and len(success_keys) == d['population'] and lost == sorted(d['damaged_annotation_ids']), 'Damage census')
        close(len(lost)/len(success_keys) if success_keys else 0., d['rate'], 'Damage rate')
        damage[arm] = len(lost)
    hard = [retention >= math.ceil(cfg['min_C_witness_retention'] * len(witness)),
            frac <= cfg['max_editable_fraction'],
            summary['task_damage']['G_B']['rate'] <= cfg['max_success_damage'], damage['G_B'] <= damage['G']]
    effects = [summary['comparisons'][k] for k in ('O_B-G', 'O_B-O_R')]
    if not all(hard) or any(e['ci95'][1] < cfg['min_effect'] for e in effects):
        decision = 'NEGATIVE_CURRENT_REGION'
    elif all(e['mean'] >= cfg['min_effect'] and e['ci95'][0] > 0 for e in effects):
        decision = 'NECESSARY_CONDITIONS_PASSED'
    else:
        decision = 'INCONCLUSIVE_NOT_READY'
    if read_json(run / 'EVAL_CONTRACT.json')['scope'] == 'smoke':
        decision = 'ENGINEERING_SMOKE_ONLY'
    require(decision == summary['decision'], 'Frozen decision differs')
    support = read_jsonl(run / 'SUCCESS_SUPPORT.jsonl')
    require(len(support) == len(success_keys), 'Success support denominator')
    worst_count = 0
    for r in support:
        n, y = dec(r['native_rle']), dec(r['GT_rle'])
        b = dec(gates[r['global_index']]['B_rle'])
        close(score(n, y), r['native_fixed_iou'], 'Native successful match IoU')
        close(score(np.where(b, ~y, n), y), r['worst_fixed_iou'], 'Worst support IoU')
        worst_count += int(r['worst_fixed_iou'] < .75)
    require(worst_count == summary['native_success_fixed_match_worst_below75'], 'Worst-case count')
    replayed = 0
    if replay:
        check_ownership()
        cfg2, manifest, profile, env = inspect_inputs()
        import gc, torch
        from adapter_data import FrozenCache
        from native_geometry import native_stages
        from lea_ops import gate_from_logits
        cache = FrozenCache(BASE / 'trace', ids)
        grouped = defaultdict(list)
        for r in gate_list:
            grouped[r['image_id']].append(r)
        with torch.inference_mode():
            for step, iid in enumerate(ids, 1):
                require(time.monotonic()-started <= cfg['budget']['verify_seconds'], 'Verification budget')
                data, native = cache.load(iid)
                _, _, z = native_stages(data, data['c0'], [r['rank'] for r in grouped[iid]])
                for r in grouped[iid]:
                    require(data['prediction_indices'][r['rank']] == r['global_index'] and native[r['rank']]['bbox'] == r['bbox'], 'Replay identity')
                    b, roi = gate_from_logits(z[r['rank']], data, r['bbox'], cfg['tau'])
                    require(np.array_equal(b, dec(r['B_rle'])) and np.array_equal(roi, dec(r['ROI_rle'])), 'Native gate replay differs')
                    replayed += 1
                del data, native, z
                gc.collect()
                torch.cuda.empty_cache()
                if step % 25 == 0 or step == len(ids):
                    print('INDEPENDENT_GATE_REPLAY', step, '/', len(ids), flush=True)
        cache.verify()
        verify_files(manifest['files'])
    return dict(status='passed', at=now(), interpreter=sys.executable, elapsed_seconds=time.monotonic()-started,
                pairs=len(pairs), images=len(ids), full_prediction_rows=len(mapping), pair_rle_checks=mask_checks,
                independent_random_support_repetitions=len(pairs)*cfg['random_support_repeats'],
                independent_COCO_match_arms=3, native_success_denominator=len(success_keys), damage_counts=damage,
                bootstrap_resamples=2000, decision=decision, outside_changed_pixels=outside_changes,
                native_gate_replays=replayed, new_training_steps=0, verifier_sha256=sha256(Path(__file__)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--replay', action='store_true')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = verify(args.run, args.replay)
    write_new(Path(args.run) / args.output, result)
    print(json.dumps(result))
