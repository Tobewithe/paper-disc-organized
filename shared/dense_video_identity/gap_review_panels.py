"""Create model-blind GT-gap panels; no tracker outputs are opened."""
from pathlib import Path
import cv2
import numpy as np
from .common import read_json, write_json, sha256


def main(data, out):
    candidates=read_json(out/'public_gap_candidates.json');selected=[]
    for seq in sorted({e['sequence'] for e in candidates}):
        selected.extend(sorted([e for e in candidates if e['sequence']==seq],key=lambda e:(e['before'],e['gt_id']))[:2])
    for i,e in enumerate(selected):
        a=np.loadtxt(data/'gt'/(e['sequence']+'_gt.txt'),delimiter=',',ndmin=2)
        b=a[(a[:,0]==e['before'])&(a[:,1]==e['gt_id'])][0,2:6].copy();b[2:]+=b[:2]
        c=a[(a[:,0]==e['after'])&(a[:,1]==e['gt_id'])][0,2:6].copy();c[2:]+=c[:2]
        x0,y0=np.minimum(b[:2],c[:2]);x1,y1=np.maximum(b[2:],c[2:]);w=x1-x0;h=y1-y0
        x0=max(0,int(x0-w*.8-60));y0=max(0,int(y0-h*.7-60));x1=min(2560,int(x1+w*.8+60));y1=min(1440,int(y1+h*.7+60))
        fs=sorted(set([max(1,e['before']-3),e['before'],e['before']+1,(e['before']+e['after'])//2,e['after']-1,e['after'],e['after']+3]))
        canvas=np.full((560,1280,3),245,np.uint8);cap=cv2.VideoCapture(str(data/'videos'/(e['sequence']+'.mp4')))
        cv2.putText(canvas,f"Event {i+1} | {e['sequence']} | GT {e['gt_id']} | {e['before']} -> {e['after']}",(12,27),cv2.FONT_HERSHEY_SIMPLEX,.7,(30,30,30),2)
        for j,f in enumerate(fs):
            cap.set(cv2.CAP_PROP_POS_FRAMES,f-1);ok,img=cap.read();assert ok
            box=a[(a[:,0]==f)&(a[:,1]==e['gt_id'])]
            if len(box):
                z=box[0,2:6].copy();z[2:]+=z[:2];cv2.rectangle(img,tuple(z[:2].astype(int)),tuple(z[2:].astype(int)),(50,230,60),3)
            crop=img[y0:y1,x0:x1];ch,cw=crop.shape[:2];ratio=min(310/cw,220/ch)
            thumb=cv2.resize(crop,(int(cw*ratio),int(ch*ratio)));xx=(j%4)*320+5;yy=(j//4)*255+60
            canvas[yy:yy+len(thumb),xx:xx+thumb.shape[1]]=thumb
            cv2.putText(canvas,f"f{f} | {'GT box' if len(box) else 'no GT box'}",(xx,yy-8),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1)
        cap.release();path=out/f'gap_event_{i+1:02d}.jpg';cv2.imwrite(str(path),canvas)
        e.update(event_id=f'G{i+1:02d}',review_frames=fs,panel=path.name,panel_sha256=sha256(path))
    write_json(out/'selected_gap_review.json',dict(selection='first two chronological 2-60 frame GT gaps in each historical sequence with gaps',tracker_outputs_used=False,events=selected))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.data,a.out)
