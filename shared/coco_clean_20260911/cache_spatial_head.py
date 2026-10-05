"""Re-run official train predictions to cache GT-free box-pooled head inputs."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO
from frozen_mechanism_probe import ROOT,Capture,write_json,sha
from spatial_coefficient_head import features

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=YOLO(str(ROOT/'weights/yolo26m-seg.pt'));model.model.eval().requires_grad_(False);rows=[];identities=[];start=time.monotonic()
    paths=sorted((a.cache/'train').glob('*.npz'))
    for n,path in enumerate(paths,1):
        iid=int(path.stem);q=np.load(path)
        with torch.inference_mode():
            model.predict(str(ROOT/'data/images/train2017'/f'{iid:012d}.jpg'),predictor=Capture,imgsz=640,conf=.001,max_det=300,iou=.7,device=0,rect=False,half=False,retina_masks=False,verbose=False)
            cap=model.predictor.capture
            np.testing.assert_allclose(cap['coeff'].cpu().numpy(),q['coeff'],rtol=0,atol=1e-6);np.testing.assert_allclose(cap['boxes'].cpu().numpy(),q['boxes'],rtol=0,atol=1e-6)
            x=features(torch.tensor(q['features'],device='cuda'),torch.tensor(q['level'],device='cuda'),cap['boxes'],cap['input_shape'],cap['proto'])
            idx=q['sample_prediction'];rows.append(x[torch.tensor(idx,device='cuda')].cpu().numpy());identities.extend((iid,int(aid)) for aid in q['sample_annotation'])
        if n%200==0:print(json.dumps(dict(phase='cache',completed=n,total=len(paths),seconds=time.monotonic()-start)),flush=True)
    np.savez(out/'features.npz',features=np.concatenate(rows),identities=np.array(identities))
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',targets=len(identities),images=len(paths),seconds=time.monotonic()-start,script_sha256=sha(__file__),feature_script_sha256=sha(Path(__file__).with_name('spatial_coefficient_head.py')),cache_protocol_sha256=sha(a.cache/'protocol.json'),features_sha256=sha(out/'features.npz'),GT_features=False,original_prediction_replay='coeff and boxes <=1e-6 absolute'))

if __name__=='__main__':main()
