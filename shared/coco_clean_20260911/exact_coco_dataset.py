"""COCO annToMask supervision, preserving annotation identity and disconnected parts.

One original non-crowd annotation is one row, even with multiple polygons.
No polygon concatenation, nearest-neighbor polygon resampling, bbox deduplication,
or YOLO text-label cache is involved. Training uses letterbox, HSV and flips.
"""
import json
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools import mask as mask_utils
from ultralytics.data.augment import RandomHSV

def ann_mask(ann, height, width):
    seg = ann['segmentation']
    if isinstance(seg, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(seg, height, width))
    elif isinstance(seg['counts'], list):
        rle = mask_utils.frPyObjects(seg, height, width)
    else:
        rle = seg
    mask = mask_utils.decode(rle)
    assert mask.ndim == 2 and mask.shape == (height, width)
    return mask

class ExactCOCODataset(Dataset):
    def __init__(self, root, split, imgsz=640, augment=False, mask_ratio=4, limit=None):
        self.root = Path(root)
        self.split, self.imgsz, self.augment, self.mask_ratio = split, imgsz, augment, mask_ratio
        self.rect = False
        self.mosaic = False
        self.hsv = RandomHSV(0.015, 0.7, 0.4)
        source = self.root/'annotations'/f'instances_{split}_selected.json'
        obj = json.loads(source.read_text())
        cats = sorted(obj['categories'], key=lambda c: c['id'])
        self.cat_to_cls = {c['id']: i for i, c in enumerate(cats)}
        self.names = {i: c['name'] for i, c in enumerate(cats)}
        self.images = sorted(obj['images'], key=lambda im: im['id'])
        if limit is not None: self.images = self.images[:limit]
        ids = {im['id'] for im in self.images}
        self.by_image = {i: [] for i in ids}
        for ann in obj['annotations']:
            if ann['image_id'] in ids and not ann.get('iscrowd', 0):
                assert ann['bbox'][2] > 0 and ann['bbox'][3] > 0, ann['id']
                self.by_image[ann['image_id']].append(ann)
        self.labels = []
        self.im_files = []
        for im in self.images:
            anns = sorted(self.by_image[im['id']], key=lambda a: a['id'])
            self.by_image[im['id']] = anns
            assert len({a['id'] for a in anns}) == len(anns)
            boxes = np.asarray([a['bbox'] for a in anns], np.float32).reshape(-1, 4)
            boxes[:, :2] += boxes[:, 2:] / 2
            boxes /= np.array([im['width'], im['height'], im['width'], im['height']], np.float32)
            cls = np.asarray([self.cat_to_cls[a['category_id']] for a in anns], np.float32).reshape(-1, 1)
            path = str(self.root/'images'/split/im['file_name'])
            self.im_files.append(path)
            self.labels.append(dict(im_file=path, shape=(im['height'], im['width']), cls=cls,
                bboxes=boxes, segments=[], normalized=True, bbox_format='xywh'))
        self.ni = len(self.images)

    def __len__(self): return self.ni

    def close_mosaic(self, hyp=None): self.mosaic = False

    def __getitem__(self, index):
        im = self.images[index]
        anns = self.by_image[im['id']]
        image = cv2.imread(self.im_files[index])
        assert image is not None, self.im_files[index]
        h, w = image.shape[:2]
        assert (h, w) == (im['height'], im['width']), (im['id'], image.shape)
        size = self.imgsz
        gain = min(size/h, size/w)
        nh, nw = round(h*gain), round(w*gain)
        top, left = round((size-nh)/2-0.1), round((size-nw)/2-0.1)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        image = cv2.copyMakeBorder(image, top, size-nh-top, left, size-nw-left, cv2.BORDER_CONSTANT, value=(114,114,114))
        n = len(anns)
        masks = np.zeros((n, size//self.mask_ratio, size//self.mask_ratio), np.uint8)
        for k, ann in enumerate(anns):
            raw = ann_mask(ann, h, w)
            resized = cv2.resize(raw, (nw, nh), interpolation=cv2.INTER_NEAREST)
            padded = cv2.copyMakeBorder(resized, top, size-nh-top, left, size-nw-left, cv2.BORDER_CONSTANT, value=0)
            masks[k] = cv2.resize(padded, (size//self.mask_ratio, size//self.mask_ratio), interpolation=cv2.INTER_NEAREST)
        # The same gain/offset as Ultralytics LetterBox bbox transformation.
        boxes = np.asarray([a['bbox'] for a in anns], np.float32).reshape(-1,4)
        boxes[:, :2] += boxes[:, 2:] / 2
        boxes *= gain
        boxes[:, 0] += left
        boxes[:, 1] += top
        boxes /= size
        if self.augment:
            image = self.hsv({'img': image})['img']
            if random.random() < 0.5:
                image = np.fliplr(image)
                masks = masks[:, :, ::-1]
                boxes[:, 0] = 1-boxes[:, 0]
        cls = self.labels[index]['cls'].copy()
        # Standard non-overlap semantic target: smallest instance wins overlaps.
        sem = np.zeros(masks.shape[1:], np.float32)
        for k in sorted(range(n), key=lambda i: (-int(masks[i].sum()), i)):
            sem[masks[k].astype(bool)] = cls[k, 0]
        return dict(img=torch.from_numpy(np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1))),
            cls=torch.from_numpy(cls), bboxes=torch.from_numpy(boxes), masks=torch.from_numpy(np.ascontiguousarray(masks)),
            sem_masks=torch.from_numpy(sem), batch_idx=torch.zeros(n),
            ann_ids=torch.tensor([a['id'] for a in anns], dtype=torch.int64), image_id=im['id'],
            im_file=self.im_files[index], ori_shape=(h,w), resized_shape=(size,size), ratio_pad=((gain,gain),(left,top)))

    @staticmethod
    def collate_fn(samples):
        out = {}
        for key in samples[0]:
            vals = [s[key] for s in samples]
            if key in {'img', 'sem_masks'}: out[key] = torch.stack(vals)
            elif key == 'batch_idx': out[key] = torch.cat([v+i for i,v in enumerate(vals)])
            elif key in {'cls','bboxes','masks','ann_ids'}: out[key] = torch.cat(vals)
            else: out[key] = tuple(vals)
        assert len(out['cls']) == len(out['masks']) == len(out['ann_ids']) == len(out['bboxes'])
        return out
