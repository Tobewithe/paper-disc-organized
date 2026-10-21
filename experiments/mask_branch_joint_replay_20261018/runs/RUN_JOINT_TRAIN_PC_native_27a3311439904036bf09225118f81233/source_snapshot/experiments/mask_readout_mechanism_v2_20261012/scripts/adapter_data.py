"""Verified native-cache inputs and training-only class/box GT joins."""
from collections import defaultdict
from pathlib import Path
import torch
from posttrain_run import read_json, require, sha256
from s1_geometry import iter_prediction_array, project_gt


class FrozenCache:
    def __init__(self, trace, image_ids):
        self.trace = Path(trace)
        self.manifest_path = self.trace / 'TRACE_MANIFEST.json'
        self.manifest_sha = sha256(self.manifest_path)
        self.manifest = read_json(self.manifest_path)
        require(self.manifest['capture_zero_delta_rle_equal'] and self.manifest['base_state_unchanged'], 'unverified native cache')
        self.image_ids = list(image_ids)
        self.records = {r['image_id']: r for r in self.manifest['images'] if r['image_id'] in set(image_ids)}
        require(len(self.records) == len(image_ids) == len(set(image_ids)), 'missing or duplicate frozen cache images')
        self.prediction_path = Path(self.manifest['predictions_path'])
        require(sha256(self.prediction_path) == self.manifest['predictions_sha256'], 'source predictions changed')
        indices = {i for r in self.records.values() for i in r['prediction_indices']}
        self.predictions = {i: row for i, row in enumerate(iter_prediction_array(self.prediction_path)) if i in indices}
        require(set(self.predictions) == indices, 'prediction rows missing')
        self.verify()

    def verify(self):
        require(sha256(self.manifest_path) == self.manifest_sha, 'cache manifest changed')
        for record in self.records.values():
            path = (self.trace / record['cache']).resolve()
            require(path.is_relative_to(self.trace.resolve()), 'cache escaped root')
            require(sha256(path) == record['cache_sha256'], 'frozen cache bytes changed')

    def load(self, image_id):
        record = self.records[image_id]
        cache = torch.load(self.trace / record['cache'], map_location='cpu', weights_only=True)
        require(cache['image_id'] == image_id and cache['prediction_indices'] == record['prediction_indices'], 'cache identity changed')
        require(cache['input_hw'] == record['input_hw'], 'input geometry changed')
        rows = [self.predictions[i] for i in cache['prediction_indices']]
        require(all(r['image_id'] == image_id for r in rows), 'cross-image rows')
        require(not any(t.requires_grad for t in [cache['proto'], cache['c0'], *cache['features']]), 'base cache requires gradients')
        return cache, rows


def training_join(predictions, annotations):
    """Ordinary GT only; deterministic one-to-one, class + box IoU>=.50."""
    from pycocotools import mask as masks
    ordinary = sorted((a for a in annotations if not a.get('iscrowd', 0) and not a.get('ignore', 0)), key=lambda a:a['id'])
    if not ordinary or not predictions:
        return {}
    ious = masks.iou([p['bbox'] for p in predictions], [a['bbox'] for a in ordinary], [0] * len(ordinary))
    used, result = set(), {}
    for rank in sorted(range(len(predictions)), key=lambda i:(-predictions[i]['score'], i)):
        eligible = [j for j, a in enumerate(ordinary) if j not in used and a['category_id'] == predictions[rank]['category_id'] and ious[rank,j] >= .5]
        if eligible:
            j = min(eligible, key=lambda j:(-ious[rank,j], ordinary[j]['id']))
            result[rank] = ordinary[j]['id']
            used.add(j)
    return result


def projected_targets(coco, cache, join):
    from pycocotools import mask as masks
    return {rank: project_gt(masks.decode(coco.annToRLE(coco.anns[ann_id])), cache) for rank, ann_id in join.items()}
