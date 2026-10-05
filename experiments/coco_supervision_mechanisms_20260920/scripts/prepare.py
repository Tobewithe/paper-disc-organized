"""Freeze disjoint COCO train cohorts and bind official converted label rows to annotation IDs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from ultralytics.data.converter import convert_coco, coco91_to_coco80_class, merge_multi_segment


def key(cls, segment):
    return str(int(cls)) + ':' + hashlib.sha256(np.asarray(segment, np.float32).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = json.loads((a.data/'annotations/instances_train2017.json').read_text())
    ids = sorted(v['id'] for v in data['images'])
    random.Random(20260920).shuffle(ids)
    explore, confirm = ids[:500], ids[500:1000]
    assert not set(explore) & set(confirm)
    for name, values in [('explore', explore), ('confirm_reserved', confirm)]:
        (a.root/(name+'_ids.json')).write_text(json.dumps(values))
    selected = set(explore)
    subset = {**data, 'images': [v for v in data['images'] if v['id'] in selected],
              'annotations': [v for v in data['annotations'] if v['image_id'] in selected]}
    folder = a.root/'conversion_input'
    folder.mkdir(exist_ok=True)
    (folder/'instances_train2017.json').write_text(json.dumps(subset))
    convert_coco(str(folder), str(a.root/'converted'), use_segments=True)
    cmap = coco91_to_coco80_class()
    byimage = defaultdict(list)
    for ann in subset['annotations']:
        byimage[ann['image_id']].append(ann)
    ledger = {}
    excluded = []
    imagepaths = []
    for image in subset['images']:
        im_id = image['id']
        w, h = image['width'], image['height']
        seenboxes = []
        expected = []
        for ann in byimage[im_id]:
            if ann.get('iscrowd', False):
                excluded.append({'id': ann['id'], 'reason': 'crowd'}); continue
            box = np.array(ann['bbox'], np.float64)
            box[:2] += box[2:]/2
            box[[0, 2]] /= w; box[[1, 3]] /= h
            cls = cmap[ann['category_id']-1]
            bb = [cls, *box.tolist()]
            if box[2] <= 0 or box[3] <= 0 or bb in seenboxes:
                excluded.append({'id': ann['id'], 'reason': 'degenerate_or_duplicate_converter_bbox'}); continue
            seenboxes.append(bb)
            seg = ann['segmentation']
            if not isinstance(seg, list) or not seg:
                raise RuntimeError(f'Non-polygon ordinary annotation {ann["id"]}: requires explicit handling')
            xy = np.concatenate(merge_multi_segment(seg), axis=0) if len(seg)>1 else np.array(seg[0]).reshape(-1, 2)
            values = [cls, *(xy/np.array([w, h])).reshape(-1).tolist()]
            line = ('%g '*len(values)).rstrip() % tuple(values)
            expected.append((ann['id'], line))
        labelpath = a.root/'converted/labels/train2017'/Path(image['file_name']).with_suffix('.txt')
        lines = labelpath.read_text().splitlines() if labelpath.exists() else []
        assert lines == [v[1] for v in expected], f'Official conversion ledger mismatch {im_id}'
        records = []
        for row, (ann_id, line) in enumerate(expected):
            values = np.asarray(line.split(), np.float32)
            records.append({'row': row, 'annotation_id': ann_id, 'key': key(values[0], values[1:].reshape(-1, 2))})
        ledger[str(im_id)] = records
        source = a.data/'images/train2017'/image['file_name']
        assert source.is_file(), source
        dest = a.root/'converted/images/train2017'/image['file_name']
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(source)
        imagepaths.append(str(dest))
    (a.root/'explore_images.txt').write_text('\n'.join(imagepaths)+'\n')
    (a.root/'LABEL_LEDGER.json').write_text(json.dumps(ledger))
    result = {'explore_images': len(explore), 'confirm_reserved_images': len(confirm),
              'split_seed': 20260920, 'source': 'COCO train2017 full image list',
              'ordinary_annotations': sum(not v.get('iscrowd', 0) for v in subset['annotations']),
              'converted_rows': sum(map(len, ledger.values())), 'exclusions': excluded,
              'confirmation_images_not_converted_or_evaluated': True}
    (a.out/'SUMMARY.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='exclusions'}), flush=True)


if __name__ == '__main__':
    main()
