"""Select train2017 images before inference and acquire official COCO image files."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time
import urllib.request


def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    os.replace(temp,path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--annotations',required=True)
    p.add_argument('--val-annotations',required=True)
    p.add_argument('--images',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=2000)
    p.add_argument('--fit-count',type=int,default=1500)
    p.add_argument('--seed',type=int,default=20260915)
    args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    image_root=Path(args.images);image_root.mkdir(parents=True,exist_ok=True)
    raw=json.loads(Path(args.annotations).read_text(encoding='utf-8'))
    val_ids={r['id'] for r in json.loads(Path(args.val_annotations).read_text(encoding='utf-8'))['images']}
    ordered=sorted(raw['images'],key=lambda x:x['id'])
    random.Random(args.seed).shuffle(ordered)
    images=ordered[:args.count]
    selected={r['id'] for r in images}
    assert len(selected)==args.count and not selected&val_ids
    fit_ids=[r['id'] for r in images[:args.fit_count]]
    selection_ids=[r['id'] for r in images[args.fit_count:]]
    subset={k:v for k,v in raw.items() if k not in ('images','annotations')}
    subset['images']=sorted(images,key=lambda r:r['id'])
    subset['annotations']=[a for a in raw['annotations'] if a['image_id'] in selected]
    atomic(out/'calibration_annotations.json',subset)
    atomic(out/'split.json',dict(seed=args.seed,fit_ids=fit_ids,selection_ids=selection_ids,
        image_selection='uniform shuffle of sorted original train2017 image IDs, no failure/class/crowding filtering',
        val_image_intersection=0,model_saw_train2017=True))
    def acquire(image):
        target=image_root/image['file_name']
        if not target.exists():
            error=None
            for attempt in range(3):
                try:
                    url=image['coco_url']
                    req=urllib.request.Request(url,headers={'User-Agent':'COCO-research-calibration/1.0'})
                    with urllib.request.urlopen(req,timeout=40) as response:
                        content=response.read()
                    if not content.startswith(b'\xff\xd8'):raise ValueError('Response is not JPEG')
                    temp=target.with_suffix('.jpg.part');temp.write_bytes(content);os.replace(temp,target)
                    break
                except Exception as exc:
                    error=str(exc)
                    time.sleep(attempt+1)
            else:raise RuntimeError(f"{image['file_name']}: {error}")
        from PIL import Image
        with Image.open(target) as im:
            assert im.size==(image['width'],image['height'])
            im.verify()
        data=target.read_bytes()
        return dict(image_id=image['id'],file_name=image['file_name'],bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    manifest=[];errors=[]
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures={pool.submit(acquire,im):im['id'] for im in images}
        for f in as_completed(futures):
            try:manifest.append(f.result())
            except Exception as exc:errors.append(dict(image_id=futures[f],error=str(exc)))
            if (len(manifest)+len(errors))%100==0:
                progress=dict(downloaded=len(manifest),failed=len(errors),total=len(images))
                atomic(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    atomic(out/'image_manifest.json',sorted(manifest,key=lambda r:r['image_id']))
    atomic(out/'errors.json',errors)
    if errors:raise RuntimeError(f'{len(errors)} images missing; do not alter the selected split')
    counts=Counter('crowd' if a.get('iscrowd') else 'ordinary' for a in subset['annotations'])
    result=dict(run_id=os.environ.get('RESEARCH_RUN_ID'),images=len(images),fit_images=len(fit_ids),selection_images=len(selection_ids),
        annotation_counts=counts,complete=True,image_root=str(image_root),source_annotations=str(Path(args.annotations).resolve()))
    atomic(out/'SUMMARY.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
