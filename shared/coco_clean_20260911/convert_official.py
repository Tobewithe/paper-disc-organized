"""Run the official converter unchanged and audit source-ID to row counts."""
import argparse
from collections import defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import os
import numpy as np
import yaml
from ultralytics.data.converter import convert_coco, coco91_to_coco80_class
from pycocotools.coco import COCO

ROOT = Path('/root/autodl-tmp/coco_clean_20260911')

def convert(split):
    src = ROOT/'data/annotations'/f'instances_{split}_selected.json'
    obj = json.loads(src.read_text())
    dest = ROOT/'conversion'/split
    receipt_path = ROOT/'audits'/f'official_conversion_{split}.json'
    if dest.exists():
        if receipt_path.exists():
            print('ALREADY_AUDITED', split, flush=True)
            return
        raise RuntimeError(f'Partial conversion exists; inspect before retry: {dest}')
    inp = ROOT/'conversion_input'/split
    inp.mkdir(parents=True, exist_ok=True)
    input_json = inp/f'instances_{split}.json'
    if not input_json.exists(): os.link(src, input_json)
    # All actual annotation conversion is done by the official public function.
    convert_coco(labels_dir=str(inp), save_dir=str(dest), use_segments=True)
    by_image = defaultdict(list)
    image_info = {im['id']: im for im in obj['images']}
    for ann in obj['annotations']: by_image[ann['image_id']].append(ann)
    classes = coco91_to_coco80_class()
    removed = []
    raw_non_crowd = kept = multi_kept = 0
    source_map = ROOT/'audits'/f'official_label_source_ids_{split}.jsonl'
    with source_map.open('w') as sidecar:
        for im_id, im in image_info.items():
            expected_ids, expected_classes, seen = [], [], {}
            for ann in by_image[im_id]:
                if ann.get('iscrowd', False): continue
                raw_non_crowd += 1
                box = np.array(ann['bbox'], dtype=np.float64)
                box[:2] += box[2:]/2
                box[[0,2]] /= im['width']; box[[1,3]] /= im['height']
                if box[2]<=0 or box[3]<=0:
                    removed.append(dict(image_id=im_id, annotation_id=ann['id'], reason='invalid_bbox'))
                    continue
                cls = classes[ann['category_id']-1]
                key = (cls, *box.tolist())
                if key in seen:
                    removed.append(dict(image_id=im_id, annotation_id=ann['id'], kept_id=seen[key], reason='official_class_bbox_dedup'))
                    continue
                seen[key] = ann['id']
                seg = ann['segmentation']
                assert isinstance(seg,list) and all(isinstance(p,list) and len(p)>=6 and len(p)%2==0 for p in seg) and seg, ann['id']
                expected_ids.append(ann['id']); expected_classes.append(cls)
                multi_kept += len(seg)>1
            path = dest/'labels'/split/Path(im['file_name']).with_suffix('.txt')
            lines = path.read_text().splitlines() if path.exists() else []
            assert len(lines)==len(expected_ids), (im_id,len(lines),len(expected_ids))
            assert len(expected_ids)==len(set(expected_ids))
            for line, cls in zip(lines,expected_classes):
                nums = [float(s) for s in line.split()]
                assert int(nums[0])==cls and len(nums)>=7 and len(nums)%2==1
                assert np.isfinite(nums).all() and min(nums[1:])>=0 and max(nums[1:])<=1
            kept += len(lines)
            sidecar.write(json.dumps(dict(image_id=im_id,file_name=im['file_name'],annotation_ids_by_line=expected_ids))+'\n')
    assert kept+len(removed)==raw_non_crowd
    # Public interface regression: one annotation containing two disjoint polygons.
    fixture_dir = ROOT/'conversion_fixture_input'
    if split=='train2017':
        fixture_dir.mkdir(exist_ok=True)
        sample=dict(images=[dict(id=1,width=64,height=64,file_name='fixture.jpg')],categories=[dict(id=1,name='person')],annotations=[dict(id=99,image_id=1,category_id=1,iscrowd=0,bbox=[4,4,48,48],area=200,segmentation=[[4,4,14,4,14,14,4,14],[42,42,52,42,52,52,42,52]])])
        (fixture_dir/'instances_fixture.json').write_text(json.dumps(sample))
        fixture_out=ROOT/'conversion_fixture'
        convert_coco(str(fixture_dir),str(fixture_out),use_segments=True)
        assert len((fixture_out/'labels/fixture/fixture.txt').read_text().splitlines())==1
        coco=COCO(str(fixture_dir/'instances_fixture.json'))
        mask=coco.annToMask(coco.anns[99])
        assert mask[8,8] and mask[46,46] and not mask[30,30]
        (ROOT/'audits/official_converter_source.py.txt').write_text(inspect.getsource(convert_coco))
        (ROOT/'audits/coco_annToMask_source.py.txt').write_text(inspect.getsource(COCO.annToMask)+inspect.getsource(COCO.annToRLE))
    out = ROOT/'official_yolo'
    (out/'labels').mkdir(parents=True,exist_ok=True)
    (out/'images').mkdir(parents=True,exist_ok=True)
    for kind, target in [('labels', dest/'labels'/split),('images',ROOT/'data/images'/split)]:
        link=out/kind/split
        if not link.exists():link.symlink_to(target,target_is_directory=True)
    ordered=sorted(obj['images'],key=lambda im:im['id'])
    (out/f'{split}.txt').write_text(''.join(str(out/'images'/split/im['file_name'])+'\n' for im in ordered))
    names={i:c['name'] for i,c in enumerate(sorted(obj['categories'],key=lambda c:c['id']))}
    (ROOT/'coco_clean.yaml').write_text(yaml.safe_dump(dict(path=str(out),train='train2017.txt',val='val2017.txt',names=names),sort_keys=False))
    receipt=dict(split=split,images=len(obj['images']),raw_non_crowd=raw_non_crowd,emitted_instances=kept,multipolygon_instances_kept=multi_kept,removed_count=len(removed),removed=removed,source_json_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),converter='ultralytics.data.converter.convert_coco',version='8.4.143',status='PASS')
    receipt_path.write_text(json.dumps(receipt,indent=2))
    print('OFFICIAL_CONVERSION_PASS',json.dumps({k:v for k,v in receipt.items() if k!='removed'}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('split',choices=['train2017','val2017']);args=parser.parse_args();convert(args.split)
