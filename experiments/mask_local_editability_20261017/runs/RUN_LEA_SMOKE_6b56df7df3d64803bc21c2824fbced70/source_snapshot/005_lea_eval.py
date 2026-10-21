"""Complete per-image COCO Mask75 matching; no AP or rematching target pairs."""
import contextlib, copy, io
import numpy as np


def match75(dataset, predictions, global_indices):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if len(predictions) != len(global_indices):
        raise ValueError('Prediction identity length')
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = copy.deepcopy(dataset)
        coco.createIndex()
        if predictions:
            result = coco.loadRes(copy.deepcopy(predictions))
        else:
            result = COCO()
            result.dataset = dict(images=copy.deepcopy(dataset['images']),
                                  categories=copy.deepcopy(dataset['categories']), annotations=[])
            result.createIndex()
        ev = COCOeval(coco, result, 'segm')
        ev.params.imgIds = sorted(i['id'] for i in dataset['images'])
        ev.params.iouThrs = np.array([.75])
        ev.params.maxDets = [100]
        ev.params.areaRng = [[0, 1e10]]
        ev.params.areaRngLbl = ['all']
        ev.params.useCats = 1
        ev.evaluate()
    records = []
    for item in ev.evalImgs:
        if item is None:
            continue
        for j, aid in enumerate(item['gtIds']):
            did = int(item['gtMatches'][0, j])
            ignored = bool(item['gtIgnore'][j])
            if did and not ignored:
                di = list(item['dtIds']).index(did)
                if item['dtIgnore'][0, di] or int(item['dtMatches'][0, di]) != int(aid):
                    raise ValueError('Inconsistent COCO matching')
            records.append(dict(image_id=int(item['image_id']), annotation_id=int(aid), ignored=ignored,
                                success=bool(did) and not ignored,
                                matched_global_index=global_indices[did - 1] if did else None))
    keys = [(r['image_id'], r['annotation_id']) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate GT row')
    return sorted(records, key=lambda r: (r['image_id'], r['annotation_id']))


def damage_records(matches):
    maps = {k: {(r['image_id'], r['annotation_id']): r for r in v} for k, v in matches.items()}
    if not all(set(v) == set(maps['N']) for v in maps.values()):
        raise ValueError('GT denominator drift')
    output = []
    for key, n in maps['N'].items():
        if n['ignored']:
            continue
        output.append(dict(image_id=key[0], annotation_id=key[1],
                           **{k: v[key]['success'] for k, v in maps.items()}))
    successes = [r for r in output if r['N']]
    damage = {}
    for k in ('G', 'G_B'):
        bad = [r['annotation_id'] for r in successes if not r[k]]
        damage[k] = dict(damaged=len(bad), population=len(successes),
                         rate=len(bad) / len(successes) if successes else 0.,
                         damaged_annotation_ids=bad,
                         recovered=sum(not r['N'] and r[k] for r in output))
    return damage, output
