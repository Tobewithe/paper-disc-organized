from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as cm
import torch

from common import dump, read, sha
from data_base import save_sample, contact_target, rle, letterbox


def prepare_real(root, run):
    source = read(root/'inputs/REAL_MANIFEST.json'); coco = read(root/'inputs/REAL_SOURCE_COCO.json')
    assert sha(root/'inputs/REAL_SOURCE_COCO.json') == source['source_annotation_sha256']
    annotations = {}
    for ann in coco['annotations']:
        annotations.setdefault(ann['image_id'], []).append(ann)
    rows = []; gt = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'pig'}], 'info': {}}
    for i, row in enumerate(source['rows']):
        path = root/row['image']; assert sha(path) == row['image_sha256']; image = cv2.imread(str(path))
        assert image.shape == (640, 640, 3)
        h, w = row['height'], row['width']; left, top, ww, hh = row['valid_rect']; masks = []
        for ann in annotations.get(row['id'], []):
            seg = ann['segmentation']; enc = cm.merge(cm.frPyObjects(seg, h, w)) if isinstance(seg, list) else seg
            mask = cm.decode(enc).astype(np.uint8); out = np.zeros((640, 640), np.uint8)
            out[top:top+hh, left:left+ww] = cv2.resize(mask, (ww, hh), interpolation=cv2.INTER_NEAREST)
            if out.sum() >= 16:
                masks.append(out)
        assert masks, f"No valid original annotations for {row['source_id']}"
        cached = save_sample(root/'cache', f"real_{row['id']}", image, masks,
                             {'source_id': row['source_id'], 'domain': 'real', 'split': row['split'], 'group': row['group'],
                              'image_id': row['id'], 'source_sha256': row['source_sha256']}, row['valid_rect'], existing_image=path)
        # Do not recompress the JPEG again; preserve the frozen RGB input bytes as model input.
        cached.update(image=str(path), image_sha256=row['image_sha256'], ordinal=i)
        rows.append(cached)
        if row['split'] == 'dev':
            gt['images'].append({'id': row['id'], 'width': 640, 'height': 640, 'file_name': str(path)})
            for enc in cached['rles']:
                gt['annotations'].append({'id': len(gt['annotations'])+1, 'image_id': row['id'], 'category_id': 1,
                                          'segmentation': enc, 'bbox': cm.toBbox(enc).tolist(),
                                          'area': int(cm.area(enc)), 'iscrowd': 0})
        if i % 100 == 0:
            print('REAL_MASK_CACHE', i+1, len(source['rows']), flush=True)
    dataset = {'rows': rows, 'counts': source['counts'], 'test_included': False,
               'real_source_manifest_sha256': sha(root/'inputs/REAL_MANIFEST.json'), 'canvas': 640}
    dump(root/'REAL_DATASET.json', dataset); dump(root/'DEV_COCO.json', gt)
    dump(run/'SUMMARY.json', {'real_train': 1272, 'real_dev': 432, 'images': len(rows), 'instances': sum(r['instances'] for r in rows),
                              'dataset_sha256': sha(root/'REAL_DATASET.json'), 'GT_sha256': sha(root/'DEV_COCO.json')})
    dump(run/'DECISION.json', {'status': 'full_real_cache_prepared', 'next_action': 'baseline_engineering_then_120_epochs'})


class Store:
    def __init__(self, rows, max_cache=384):
        self.rows = rows; self.cache = OrderedDict(); self.max_cache = max_cache

    def get(self, row):
        key = row['source_id']
        if key not in self.cache:
            with np.load(row['label']) as labels:
                self.cache[key] = {'rgb': cv2.imread(row['image']), **{k: labels[k] for k in labels.files}}
            if len(self.cache) > self.max_cache:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def batch(self, rows, seed, augment=True, device='cuda'):
        rng = np.random.default_rng(seed); images, masks, boxes, batch_indices = [], [], [], []
        for i, row in enumerate(rows):
            sample = self.get(row); image = sample['rgb'].copy(); target = sample['masks'].copy(); bb = sample['boxes'].copy()
            if augment:
                if rng.random() < .5:
                    image = image[:, ::-1].copy(); target = target[:, :, ::-1].copy(); bb[:, 0] = 1-bb[:, 0]
                image = np.clip((image.astype(np.float32)-127.5)*rng.uniform(.85, 1.15)+127.5+rng.uniform(-10, 10), 0, 255).astype(np.uint8)
            images.append(image[:, :, ::-1].transpose(2, 0, 1).copy()); masks.extend(target); boxes.extend(bb); batch_indices.extend([i]*len(bb))
        return {'img': torch.from_numpy(np.stack(images)).to(device).float()/255,
                'masks': torch.tensor(np.asarray(masks, np.uint8).reshape(-1, 160, 160), device=device),
                'bboxes': torch.tensor(np.asarray(boxes, np.float32).reshape(-1, 4), device=device),
                'batch_idx': torch.tensor(batch_indices, dtype=torch.long, device=device),
                'cls': torch.zeros((len(boxes), 1), device=device),
                'sem_masks': torch.zeros((len(rows), 160, 160), dtype=torch.long, device=device)}
