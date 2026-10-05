"""Post-lock cache source witness, no parameter selection or model modification."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse,contextlib,io,json
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics import YOLO
from frozen_mechanism_probe import ROOT,Capture,ownership,sha,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out
    cache=ROOT/'diagnostics/full_val_cache_20260911';protocol=json.loads((cache/'protocol.json').read_text());receipt=json.loads((cache/'COMPLETE.json').read_text())
    experiment=json.loads((out/'protocol.json').read_text());ids=json.loads((out/'selection.json').read_text())['evaluation']
    assert receipt['status']=='COMPLETE' and receipt['val_images']==5000
    assert protocol['official_weight_sha256']==experiment['weight_sha256']==sha(ROOT/'weights/yolo26m-seg.pt')
    assert protocol['script_sha256']==receipt['script_sha256']==sha(ROOT/'cache_full_val.py')
    recorded=json.loads((out/'EVALUATION_CACHE_HASHES.json').read_text());assert set(map(int,recorded))==set(ids)
    for iid in ids:assert recorded[str(iid)]==sha(cache/'val'/f'{iid}.npz')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    model=YOLO(str(ROOT/'weights/yolo26m-seg.pt'));model.model.eval().requires_grad_(False);rows=[]
    for iid in ids[:3]:
        with torch.inference_mode():model.predict(str(ROOT/'data/images/val2017'/gt.imgs[iid]['file_name']),predictor=Capture,imgsz=640,conf=.001,iou=.7,max_det=300,rect=False,half=False,retina_masks=False,device=0,verbose=False)
        cap=model.predictor.capture;mapping=ownership(gt,iid,cap['detections']);errors={}
        with np.load(cache/'val'/f'{iid}.npz') as old:
            for k in ['proto','coeff','boxes','detections']:
                now=cap[k].cpu().numpy();assert now.shape==old[k].shape;err=float(np.max(np.abs(now-old[k]))) if now.size else 0.;errors[k]=err;assert err==0,(iid,k,err)
            assert dict(zip(map(int,old['mapping_gt']),map(int,old['mapping_pred'])))==mapping
            assert tuple(old['shape'])==cap['shape'] and tuple(old['input_shape'])==cap['input_shape']
        rows.append(dict(image_id=iid,max_abs=errors,mapping_equal=True))
    write_json(out/'CACHE_PROVENANCE_WITNESS.json',dict(status='PASS',timing='Post-lock supplementary verification; no change to masks/parameters/selection',
        source_protocol_sha256=sha(cache/'protocol.json'),source_complete_sha256=sha(cache/'COMPLETE.json'),source_script_sha256=sha(ROOT/'cache_full_val.py'),
        weight_sha256=protocol['official_weight_sha256'],all_selected_cache_hashes_rechecked=len(ids),source_creation_had_per_file_manifest=False,
        exact_forward_replay=rows,limits='Fresh forward parity is only three fixed selected images. Cache source receipt did not precommit per-file hashes; all400 checks establish present-time read consistency, not historical bitwise immutability.',source_sha256=sha(__file__)))
    print(json.dumps(dict(status='PASS',cache_files=len(ids),replay=rows)),flush=True)


if __name__=='__main__':main()
