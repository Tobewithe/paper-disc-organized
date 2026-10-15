"""Exact paired image bootstrap: clone independent matches, rerun COCOeval accumulation.

COCO matching is independent for each image/category. Cloning its complete result
with fresh image/GT/detection IDs is equivalent to evaluating duplicated RLEs.
Every replicate reruns the pinned COCOeval.accumulate (including global stable
score ranking and 101-point interpolation); no average of image APs is used.
"""
import contextlib
import copy
import io
import pickle
from pathlib import Path
import numpy as np

ID_STRIDE = 10000000


def evaluate(gt, predictions, ids, task='segm', ap_only=False, quiet=False):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from build_native_ledger import load_results
    stream = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with stream:
        if isinstance(gt, dict):
            coco = COCO()
            coco.dataset = copy.deepcopy(gt)
            coco.createIndex()
        else:
            coco = COCO(str(gt))
        result = load_results(coco, predictions)
        ev = COCOeval(coco, result, task)
        ev.params.imgIds = list(ids)
        ev.params.useCats = 1
        ev.params.maxDets = [100] if ap_only else [1, 10, 100]
        if ap_only:
            ev.params.areaRng = [ev.params.areaRng[0]]
            ev.params.areaRngLbl = ['all']
        ev.evaluate()
        ev.accumulate()
        if not ap_only:
            ev.summarize()
    return ev


def compact(ev):
    """Keep only area=all/maxDets=100 sufficient statistics, with native IDs."""
    p = copy.deepcopy(ev._paramsEval)
    lookup = {(r['category_id'], r['image_id']): r for r in ev.evalImgs
              if r is not None and r['aRng'] == p.areaRng[0]}
    p.areaRng, p.areaRngLbl, p.maxDets = [p.areaRng[0]], ['all'], [100]
    entries = [[lookup.get((c, i)) for i in p.imgIds] for c in p.catIds]
    return dict(params=p, entries=entries, image_ids=list(p.imgIds), version=1)


def save_cache(path, cache):
    with Path(path).open('xb') as stream:
        pickle.dump(cache, stream, protocol=5)


def load_cache(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def clone_item(item, occurrence):
    if item is None:
        return None
    offset = (occurrence + 1) * ID_STRIDE
    # Native IDs are smaller than the stride; all synthetic IDs are exact float64 integers.
    if any(i >= ID_STRIDE for i in item['gtIds']) or any(i >= ID_STRIDE for i in item['dtIds']):
        raise ValueError('ID stride is insufficient')
    return {**item, 'image_id': occurrence + 1,
            'dtIds': [i + offset for i in item['dtIds']],
            'gtIds': [i + offset for i in item['gtIds']],
            'dtMatches': np.where(item['dtMatches'] != 0, item['dtMatches'] + offset, 0),
            'gtMatches': np.where(item['gtMatches'] != 0, item['gtMatches'] + offset, 0)}


def resampled_evaluator(cache, draws):
    from pycocotools.cocoeval import COCOeval
    draws = [int(i) for i in draws]
    if not draws or min(draws) < 0 or max(draws) >= len(cache['image_ids']):
        raise ValueError('invalid image draw')
    ev = COCOeval(iouType='segm')
    ev.params = copy.deepcopy(cache['params'])
    ev.params.imgIds = list(range(1, len(draws) + 1))
    ev._paramsEval = copy.deepcopy(ev.params)
    ev.evalImgs = [clone_item(category[index], occurrence)
                   for category in cache['entries'] for occurrence, index in enumerate(draws)]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.accumulate()
    return ev


def ap(ev):
    values = ev.eval['precision']
    values = values[values >= 0]
    if not values.size:
        raise ValueError('no valid category AP')
    return float(values.mean() * 100)


def literal_duplicate(gt_dataset, predictions, image_ids, draws):
    """Independent check: materialize duplicated COCO data and evaluate from RLE."""
    images = {r['id']: r for r in gt_dataset['images']}
    from collections import defaultdict
    annotations, detections = defaultdict(list), defaultdict(list)
    for row in gt_dataset['annotations']:
        annotations[row['image_id']].append(row)
    for row in predictions:
        detections[row['image_id']].append(row)
    dataset = dict(info=gt_dataset.get('info', {}), categories=gt_dataset['categories'], images=[], annotations=[])
    output = []
    for occurrence, index in enumerate(draws):
        old = image_ids[int(index)]
        image_id, offset = occurrence + 1, (occurrence + 1) * ID_STRIDE
        dataset['images'].append({**images[old], 'id': image_id})
        for row in annotations[old]:
            dataset['annotations'].append({**row, 'image_id': image_id, 'id': row['id'] + offset})
        for row in detections[old]:
            output.append({**row, 'image_id': image_id})
    # loadRes assigns a new unique dt ID to every copied detection; numeric labels
    # may differ from clone_item but correspondence, sorting and TP/FP are identical.
    return evaluate(dataset, output, list(range(1, len(draws) + 1)), ap_only=True, quiet=True)


def verify_replication(cache, gt_dataset, predictions, draws):
    reused = resampled_evaluator(cache, draws)
    literal = literal_duplicate(gt_dataset, predictions, cache['image_ids'], draws)
    for field in ('precision', 'recall', 'scores'):
        if not np.array_equal(reused.eval[field], literal.eval[field]):
            raise AssertionError('duplicate-image COCO parity failed: ' + field)
    return dict(status='passed', draw=[int(x) for x in draws], copies=len(draws),
                unique_source_images=len(set(draws)), exact_precision_recall_scores=True,
                AP_points=ap(reused))
