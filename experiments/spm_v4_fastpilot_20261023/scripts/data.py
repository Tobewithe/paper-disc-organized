from collections import Counter
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as cm

from common import dump, read, sha


def rle(mask):
    x = cm.encode(np.asfortranarray(mask, dtype=np.uint8))
    return {'size': x['size'], 'counts': x['counts'].decode()}


def letterbox(image, masks):
    h, w = image.shape[:2]; scale = 640/max(h, w)
    hh, ww = round(h*scale), round(w*scale); top, left = (640-hh)//2, (640-ww)//2
    rgb = np.full((640, 640, 3), 114, np.uint8)
    rgb[top:top+hh, left:left+ww] = cv2.resize(image, (ww, hh), interpolation=cv2.INTER_LINEAR)
    targets = []
    for mask in masks:
        out = np.zeros((640, 640), np.uint8)
        out[top:top+hh, left:left+ww] = cv2.resize(mask, (ww, hh), interpolation=cv2.INTER_NEAREST)
        if out.sum() >= 16:
            targets.append(out)
    return rgb, targets, [left, top, ww, hh]


def save_sample(folder, key, image, masks, meta, valid_rect):
    folder.mkdir(parents=True, exist_ok=True)
    image_path, label_path = folder/(key+'.png'), folder/(key+'.npz')
    assert cv2.imwrite(str(image_path), image)
    ids = np.zeros((640, 640), np.uint16); overlap = np.zeros((640, 640), bool)
    bboxes, encoded, loss_masks = [], [], []
    for i, mask in enumerate(masks, 1):
        overlap |= (ids > 0) & (mask > 0); ids[mask > 0] = i
        box = cm.toBbox(cm.encode(np.asfortranarray(mask, dtype=np.uint8))).astype(float)
        bboxes.append([(box[0]+box[2]/2)/640, (box[1]+box[3]/2)/640, box[2]/640, box[3]/640])
        encoded.append(rle(mask)); loss_masks.append(cv2.resize(mask, (160, 160), interpolation=cv2.INTER_NEAREST))
    loss_masks = np.stack(loss_masks) if loss_masks else np.zeros((0, 160, 160), np.uint8)
    np.savez_compressed(label_path, ids=ids, masks=loss_masks, boxes=np.asarray(bboxes, np.float32).reshape(-1, 4),
                        overlap=overlap.astype(np.uint8), valid_rect=np.asarray(valid_rect))
    return {**meta, 'image': str(image_path), 'label': str(label_path), 'image_sha256': sha(image_path),
            'label_sha256': sha(label_path), 'rles': encoded, 'bboxes': bboxes, 'instances': len(encoded),
            'width': 640, 'height': 640, 'valid_rect': valid_rect}


def prepare_dataset(root, run):
    prior = Path('/srv/gpu-lab/users/a25023104/projects/spm-prototype-training-20261022/INPUT_MANIFEST.json')
    real_root = Path('/srv/gpu-lab/users/a25023104/projects/spm-yoloseg-20261022/inputs')
    real_rows = [r for r in read(prior)['rows'] if r['domain'] == 'real']
    rows = []; folder = root/'cache'
    for i, row in enumerate(real_rows):
        path = real_root/row['file_name']; assert sha(path) == row['source_sha256']
        image = cv2.imread(str(path)); assert image is not None
        h, w = image.shape[:2]; targets = []
        for ann in row['annotations']:
            seg = ann['segmentation']
            enc = cm.merge(cm.frPyObjects(seg, h, w)) if isinstance(seg, list) else seg
            targets.append(cm.decode(enc).astype(np.uint8))
        image, targets, rect = letterbox(image, targets)
        rows.append(save_sample(folder, f"real_{row['id']}", image, targets,
                                {'source_id': row['source_id'], 'domain': 'real', 'split': row['split'],
                                 'group': row['group'], 'image_id': row['id'], 'source_sha256': row['source_sha256']}, rect))
        if i % 40 == 0:
            print('REAL_CACHE', i+1, len(real_rows), flush=True)
    ptr = read(root/'FORMAL_POINTER.json'); render = root/'runs'/ptr['run_id']
    for j, row in enumerate(read(render/'DATA_MANIFEST.json')['rows']):
        image = cv2.imread(row['rgb_path']); ids = np.load(row['label_path'])['instance_id']
        targets = [(ids == k).astype(np.uint8) for k in np.unique(ids) if k > 0 and (ids == k).sum() >= 16]
        image, targets, rect = letterbox(image, targets)
        rows.append(save_sample(folder, f'syn_{j:04d}', image, targets,
                                {'source_id': row['source_id'], 'domain': 'synthetic', 'split': row['split'],
                                 'group': row['group'], 'image_id': 100000+j, 'appearance_id': row['appearance_id'],
                                 'source_sha256': row['rgb_sha256']}, rect))
    # Controlled cutout: keep all original base instances, subtract pasted pixels, add the donor.
    rng = np.random.default_rng(231023); train_real = [r for r in rows if r['domain'] == 'real' and r['split'] == 'train']
    for j in range(64):
        base, donor = [train_real[int(rng.integers(len(train_real)))] for _ in range(2)]
        base_rgb = cv2.imread(base['image']); donor_rgb = cv2.imread(donor['image'])
        base_ids = np.load(base['label'])['ids']; donor_ids = np.load(donor['label'])['ids']
        eligible = []
        for k in np.unique(donor_ids):
            if k == 0:
                continue
            ys, xs = np.where(donor_ids == k)
            area = len(xs); bw, bh = xs.max()-xs.min()+1, ys.max()-ys.min()+1
            if area/(bw*bh) >= .48 and min(bw, bh) >= 12:
                eligible.append(int(k))
        assert eligible, donor['source_id']
        identity = int(rng.choice(eligible)); ys, xs = np.where(donor_ids == identity)
        x0, x1, y0, y1 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
        crop, mask = donor_rgb[y0:y1, x0:x1], (donor_ids[y0:y1, x0:x1] == identity).astype(np.uint8)
        factor = float(rng.uniform(.8, 1.2)); hh, ww = max(8, round(len(crop)*factor)), max(8, round(crop.shape[1]*factor))
        hh, ww = min(300, hh), min(300, ww)
        crop = cv2.resize(crop, (ww, hh)); mask = cv2.resize(mask, (ww, hh), interpolation=cv2.INTER_NEAREST)
        left, top, valid_w, valid_h = base['valid_rect']
        x = int(rng.integers(left, max(left+1, left+valid_w-ww+1))); y = int(rng.integers(top, max(top+1, top+valid_h-hh+1)))
        hh, ww = min(hh, 640-y), min(ww, 640-x); crop, mask = crop[:hh, :ww], mask[:hh, :ww]
        pasted = np.zeros_like(base_ids, bool); pasted[y:y+hh, x:x+ww] = mask > 0
        base_rgb[y:y+hh, x:x+ww][mask > 0] = crop[mask > 0]
        targets = [(base_ids == k).astype(np.uint8) for k in np.unique(base_ids) if k > 0]
        targets = [np.where(pasted, 0, target).astype(np.uint8) for target in targets]
        targets = [target for target in targets if target.sum() >= 16]+[pasted.astype(np.uint8)]
        rows.append(save_sample(folder, f'cutout_{j:04d}', base_rgb, targets,
                                {'source_id': f'cutout/{j}', 'domain': 'cutout', 'split': 'train',
                                 'group': f'cutout/{j}', 'image_id': 200000+j,
                                 'base_source_id': base['source_id'], 'donor_source_id': donor['source_id']}, base['valid_rect']))
    for i, row in enumerate(rows):
        row['ordinal'] = i
    train_groups = {r['domain']+'/'+r['group'] for r in rows if r['split'] == 'train'}
    dev_groups = {r['domain']+'/'+r['group'] for r in rows if r['split'] == 'dev'}
    assert not train_groups & dev_groups
    dataset = {'rows': rows, 'prior_real_manifest_sha256': sha(prior), 'render_run_id': render.name,
               'counts': dict(Counter(r['split']+'/'+r['domain'] for r in rows)),
               'task_canvas': 'common_640_letterbox', 'test_included': False}
    dump(root/'DATASET.json', dataset)
    coco = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'pig'}], 'info': {}}
    for row in rows:
        if row['domain'] != 'real' or row['split'] != 'dev':
            continue
        coco['images'].append({'id': row['image_id'], 'width': 640, 'height': 640, 'file_name': row['image']})
        for enc in row['rles']:
            decoded = cm.decode(enc); box = cm.toBbox(enc).tolist()
            coco['annotations'].append({'id': len(coco['annotations'])+1, 'image_id': row['image_id'], 'category_id': 1,
                                         'segmentation': enc, 'bbox': box, 'area': int(decoded.sum()), 'iscrowd': 0})
    dump(root/'DEV_COCO.json', coco); dump(run/'SUMMARY.json', {'counts': dataset['counts'], 'dataset_sha256': sha(root/'DATASET.json')})
    print('DATA_READY', dataset['counts'], flush=True)
    return dataset


class Store:
    def __init__(self, rows):
        self.rows = rows; self.cache = {}

    def get(self, row):
        key = row['source_id']
        if key not in self.cache:
            with np.load(row['label']) as data:
                self.cache[key] = {'rgb': cv2.imread(row['image']), **{k: data[k] for k in data.files}}
        return self.cache[key]

    def batch(self, rows, seed, augment=True, device='cuda'):
        rng = np.random.default_rng(seed); images, masks, boxes, indices = [], [], [], []
        for i, row in enumerate(rows):
            sample = self.get(row); im = sample['rgb'].copy(); target = sample['masks'].copy(); bb = sample['boxes'].copy()
            if augment:
                if rng.random() < .5:
                    im = im[:, ::-1].copy(); target = target[:, :, ::-1].copy(); bb[:, 0] = 1-bb[:, 0]
                im = np.clip((im.astype(np.float32)-127.5)*rng.uniform(.85, 1.15)+127.5+rng.uniform(-10, 10), 0, 255).astype(np.uint8)
            images.append(im[:, :, ::-1].transpose(2, 0, 1).copy())
            masks.extend(target); boxes.extend(bb); indices.extend([i]*len(bb))
        return {'img': torch.from_numpy(np.stack(images)).to(device).float()/255,
                'masks': torch.from_numpy(np.asarray(masks, np.uint8).reshape(-1, 160, 160)).to(device),
                'bboxes': torch.tensor(np.asarray(boxes, np.float32).reshape(-1, 4), device=device),
                'batch_idx': torch.tensor(indices, dtype=torch.long, device=device),
                'cls': torch.zeros((len(boxes), 1), device=device),
                'sem_masks': torch.zeros((len(rows), 160, 160), dtype=torch.long, device=device)}


def contact_target(sample):
    ids = sample['ids']; fg = ids > 0
    contact = np.zeros_like(fg); kernel = np.ones((9, 9), np.uint8)
    for k in np.unique(ids):
        if k > 0:
            own = ids == k
            contact |= own & (cv2.dilate((fg & ~own).astype(np.uint8), kernel) > 0)
    boundary = fg & (cv2.dilate((~fg).astype(np.uint8), kernel) > 0)
    valid = np.zeros_like(fg); left, top, w, h = sample['valid_rect']
    valid[top:top+h, left:left+w] = True
    valid &= cv2.dilate(sample['overlap'], np.ones((3, 3), np.uint8)) == 0
    resize = lambda x: cv2.resize(x.astype(np.float32), (80, 80), interpolation=cv2.INTER_AREA)
    return resize(fg), resize(contact), resize(boundary), resize(valid) > .999
