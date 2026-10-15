"""Independent row, census, percentile and AP-curve verification after execution."""
from common import *
import argparse
from collections import Counter
import gc
import hashlib
import json
import numpy as np
from coco_bootstrap import load_cache


def independently_accumulate(cache, draws):
    """101-point AP via vectorized precision envelope; no COCOeval.accumulate call."""
    curves = []
    recall_thresholds = np.linspace(0, 1, 101)
    for category in cache['entries']:
        rows = [category[int(i)] for i in draws if category[int(i)] is not None]
        if not rows:
            continue
        positive_gt = sum(np.count_nonzero(np.asarray(r['gtIgnore']) == 0) for r in rows)
        if not positive_gt:
            continue
        scores = np.concatenate([r['dtScores'][:100] for r in rows])
        order = np.argsort(-scores, kind='mergesort')
        matches = np.concatenate([r['dtMatches'][:, :100] for r in rows], axis=1)[:, order] != 0
        ignored = np.concatenate([r['dtIgnore'][:, :100] for r in rows], axis=1)[:, order]
        tp = np.cumsum(matches & ~ignored, axis=1).astype(float)
        fp = np.cumsum(~matches & ~ignored, axis=1).astype(float)
        for t in range(10):
            values = np.zeros(101)
            if scores.size:
                recall = tp[t] / positive_gt
                precision = tp[t] / (tp[t] + fp[t] + np.spacing(1))
                envelope = np.maximum.accumulate(precision[::-1])[::-1]
                indices = np.searchsorted(recall, recall_thresholds, side='left')
                valid = indices < len(envelope)
                values[valid] = envelope[indices[valid]]
            curves.append(values)
    # COCO averages valid T/R/category cells, not per-image AP.
    return float(np.asarray(curves).mean() * 100)


def raw_rle_area(rle):
    """COCO compressed RLE integer decoding, independent of mask.area."""
    counts, cursor = [], 0
    code = rle['counts']
    while cursor < len(code):
        value, shift, more = 0, 0, True
        while more:
            token = ord(code[cursor]) - 48
            cursor += 1
            value |= (token & 31) << (5 * shift)
            more = bool(token & 32)
            shift += 1
            if not more and token & 16:
                value |= -1 << (5 * shift)
        if len(counts) > 2:
            value += counts[-2]
        require(value >= 0, 'negative RLE run')
        counts.append(value)
    require(sum(counts) == rle['size'][0] * rle['size'][1], 'invalid RLE total')
    return sum(counts[1::2])


def verify_rows(run):
    from s1_geometry import iter_prediction_array
    old_indices = read_json(GATE/'ROW_MAPPING.json')['original_global_indices']
    old = {i: r for i, r in zip(old_indices, iter_prediction_array(GATE/'MIXED_predictions.json'), strict=True)}
    counter = Counter()
    with (run/'GATE_ROWS.jsonl').open(encoding='utf8') as gates:
        for i, (a, b, line) in enumerate(zip(iter_prediction_array(NATIVE), iter_prediction_array(run/'predictions_mixed.json'), gates, strict=True)):
            row = json.loads(line)
            require(row['global_index'] == i and a['image_id'] == b['image_id'] == row['image_id'], 'row ID/order changed')
            require(set(a) == set(b) and all(a[k] == b[k] for k in a if k != 'segmentation'), 'non-mask field changed')
            area = raw_rle_area(a['segmentation'])
            selected = area >= 9216
            changed = a['segmentation'] != b['segmentation']
            require(area == row['native_mask_area'] and selected == row['apply_delta'] and changed == row['mask_changed'], 'gate receipt incorrect')
            require(selected or a == b, 'unselected output changed')
            if i in old:
                require(b == old[i], '1314-image overlap differs from original candidate')
                counter['old_candidate_rows'] += 1
            counter['outputs'] += 1
            counter['selected_rows'] += selected
            counter['changed_masks'] += changed
    require(counter['outputs'] == 539128 and counter['old_candidate_rows'] == 185621, 'row coverage incomplete')
    del old
    return dict(counter)


def verify_checkpoint(run):
    import torch
    from adapter_model import CoefficientAdapter
    from adapter_data import FrozenCache
    from native_output_bridge import decode_native, encode_rle
    from s1_geometry import iter_prediction_array
    configure_native()
    saved = torch.load(TRAIN/'last.pt', map_location='cpu', weights_only=True)
    model = CoefficientAdapter('B').cuda().eval()
    model.load_state_dict(saved['state_dict'], strict=True)
    del saved
    delta = torch.load(run/'DELTA.pt', map_location='cpu', weights_only=True)
    ids = list(delta)[:8]
    require(len(ids) == 8, 'insufficient new images for checkpoint replay')
    source = FrozenCache(BASE/'trace', ids)
    modified = {i: p for i, p in enumerate(iter_prediction_array(run/'predictions_mixed.json')) if p['image_id'] in ids}
    rows_checked = 0
    with torch.inference_mode():
        for image_id in ids:
            cache, original = source.load(image_id)
            actual_delta = model(cache, 'eval:0')
            require(torch.equal(actual_delta.cpu(), delta[image_id]), 'checkpoint coefficient replay differs')
            zero = decode_native(cache)
            corrected = decode_native(cache, cache['c0'].to(actual_delta) + actual_delta)
            for rank, index in enumerate(cache['prediction_indices']):
                require(encode_rle(zero[rank]) == original[rank]['segmentation'], 'independent zero-delta replay differs')
                wanted = encode_rle(corrected[rank]) if raw_rle_area(original[rank]['segmentation']) >= 9216 else original[rank]['segmentation']
                require(modified[index]['segmentation'] == wanted, 'independent corrected RLE differs')
                rows_checked += 1
            del cache, zero, corrected, actual_delta
            torch.cuda.empty_cache()
    return dict(images=ids, output_rows=rows_checked, coefficient_bit_exact=True, RLE_exact=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--laptop-input-audit', action='store_true')
    a = parser.parse_args()
    run, smoke = a.run, a.smoke_run
    require(read_json(run/'run.json')['execution_status'] == 'completed', 'Run not complete')
    summary = read_json(run/'SUMMARY.json')
    for path in (run, smoke):
        require(read_json(path/'CONFIG_AUDIT.json')['status'] == 'passed', 'configuration audit failed')
        manifest = read_json(path/'ARTIFACT_MANIFEST.json')['artifacts']
        for item in manifest:
            require(sha256(path/item['relative_path']) == item['sha256'], 'artifact changed')
    census = read_jsonl(run/'GT_CENSUS.jsonl')
    require(len({(r['image_id'],r['annotation_id']) for r in census}) == len(census), 'duplicate GT')
    s = [r for r in census if r['baseline_success']]
    k = sum(not r['method_success'] for r in s)
    require(len(s) == 22957 and k == summary['damage']['damaged'], 'damage numerator/denominator mismatch')
    require(k/len(s) == summary['damage']['exact_rate'] == summary['damage']['upper95'], 'census rate mismatch')
    base = read_json(smoke/'BASE_FULL_POINT.json')
    method = read_json(run/'METHOD_POINT.json')
    matches = [{(r['image_id'],r['annotation_id']):r for r in p['matching']['segm']} for p in (base, method)]
    for row in census:
        key = row['image_id'], row['annotation_id']
        x, y = matches[0][key], matches[1][key]
        require(row['ignored'] == x['ignore'] == y['ignore'], 'ignore mismatch')
        require(row['baseline_success'] == (bool(x['d75']) and not x['ignore']), 'baseline status mismatch')
        require(row['method_success'] == (bool(y['d75']) and not y['ignore']), 'method status mismatch')
    require(base['matching']['bbox'] == method['matching']['bbox'] and base['metrics']['bbox'] == method['metrics']['bbox'], 'bbox changed')
    samples = read_jsonl(run/'AP_BOOTSTRAP.jsonl')
    require([r['replicate'] for r in samples] == list(range(2000)), 'bootstrap missing/duplicate')
    changes = [r['method_AP']-r['baseline_AP'] for r in samples]
    require(all(x == y['delta_AP'] for x,y in zip(changes,samples)), 'replicate delta incorrect')
    interval = [float(x) for x in np.quantile(changes, [.025,.975], method='linear')]
    require(interval == summary['uncertainty']['ci95'], 'CI recomputation differs')
    caches = [load_cache(smoke/'BASE_FULL_MATCH_CACHE.pkl'), load_cache(run/'METHOD_MATCH_CACHE.pkl')]
    require(caches[0]['image_ids'] == caches[1]['image_ids'] and len(caches[0]['image_ids']) == 5000, 'unpaired cache')
    rng = np.random.default_rng(1729)
    digest = hashlib.sha256()
    replicated = []
    for i in range(2000):
        draws = rng.integers(0,5000,size=5000)
        digest.update(draws.astype('<i8').tobytes())
        if i in (0,999,1999):
            values = [independently_accumulate(cache,draws) for cache in caches]
            error = max(abs(values[0]-samples[i]['baseline_AP']), abs(values[1]-samples[i]['method_AP']))
            require(error < 1e-10, 'independent AP-curve accumulation differs')
            replicated.append(dict(replicate=i, max_error_AP_points=error))
    require(digest.hexdigest() == summary['uncertainty']['image_draws_sha256'], 'bootstrap draws changed')
    point_delta = method['metrics']['segm']['AP']-base['metrics']['segm']['AP']
    gates = dict(mask_AP_point=point_delta >= .3, paired_AP_ci_lower=interval[0] > 0,
                 success_damage=k/22957 <= .01, frozen_Box_outputs=True)
    require(gates == summary['gates'], 'gate computation differs')
    require(read_json(run/'DECISION.json')['status'] == ('passed' if all(gates.values()) else 'failed'), 'wrong decision')
    result = dict(status='passed', all_artifact_hashes=True, census_population=len(s), damaged=k,
                  confidence_interval=interval, bootstrap_draws_sha256=digest.hexdigest(),
                  independent_full_dataset_AP_replicates=replicated, gates=gates,
                  coefficient_and_RLE_replay='not_requested_on_this_host')
    del caches
    gc.collect()
    if a.laptop_input_audit:
        result['rows'] = verify_rows(run)
        result['coefficient_and_RLE_replay'] = verify_checkpoint(run)
    write_new(a.out, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
