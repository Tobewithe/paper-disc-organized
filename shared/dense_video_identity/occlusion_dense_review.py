"""Dense chronological raw-frame evidence for candidate physical occlusions."""
from pathlib import Path
import cv2
import numpy as np
from .common import Run,read_json,write_json,load_gt,sha256


def main(root):
    m=read_json(root/'input_manifest.json');seqs={s['name']:s for s in m['sequences']}
    review=read_json(root/'annotations/initial_visual_review.json');ids={e['id'] for e in review['events'] if e['category']=='candidate_full'}|{'O002','O005'}
    candidates=read_json(root/'annotations/review_manifest.json')['events'];dest=root/'annotations/dense';dest.mkdir(exist_ok=True)
    with Run(root,'occlusion_dense_frames',dict(events=sorted(ids),tracker_outputs_read=False,review_source='AI initial review',interpolated_GT_created=False)) as run:
        items=[]
        for e in candidates:
            if e['event_id'] not in ids:continue
            s=seqs[e['sequence']];gt=load_gt(root,s);a=max(1,e['before']-3);b=min(s['frames'],e['after']+12)
            cap=cv2.VideoCapture(s['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,a-1);tiles=[];ranges=[]
            x0,y0,x1,y1=e['crop'];ww=x1-x0;hh=y1-y0
            for f in range(a,b+1):
                ok,img=cap.read();assert ok
                rows=gt[f][gt[f][:,1]==e['gt_id']]
                crop=img[y0:y1,x0:x1].copy()
                if len(rows):
                    bb=rows[0,2:6].astype(int)-np.array([x0,y0,x0,y0]);cv2.rectangle(crop,tuple(bb[:2]),tuple(bb[2:]),(30,230,30),2)
                scale=min(312/ww,180/hh);small=cv2.resize(crop,(round(ww*scale),round(hh*scale)))
                tile=np.full((214,320,3),248,np.uint8);tile[28:28+len(small),4:4+small.shape[1]]=small
                cv2.putText(tile,f'f{f} '+('GT' if len(rows) else 'NO LABEL'),(4,20),cv2.FONT_HERSHEY_SIMPLEX,.52,(20,20,20),1);tiles.append(tile)
            cap.release()
            for start in range(0,len(tiles),30):
                group=tiles[start:start+30];canvas=np.full((50+214*((len(group)+4)//5),1600,3),248,np.uint8)
                cv2.putText(canvas,f"{e['event_id']} {s['name']} GT {e['gt_id']} | original f{a+start}..{a+start+len(group)-1}",(10,29),cv2.FONT_HERSHEY_SIMPLEX,.7,(10,10,10),2)
                for j,tile in enumerate(group):x=j%5*320;y=50+j//5*214;canvas[y:y+214,x:x+320]=tile
                path=dest/f"{e['event_id']}_{start//30+1:02d}.jpg";cv2.imwrite(str(path),canvas,[cv2.IMWRITE_JPEG_QUALITY,93]);ranges.append(dict(path=path.name,frames=[a+start,a+start+len(group)-1],sha256=sha256(path)))
            items.append(dict(event_id=e['event_id'],sequence=s['name'],gt_id=e['gt_id'],frames=[a,b],pages=ranges))
        write_json(dest/'manifest.json',items);write_json(run.path/'SUMMARY.json',dict(events=len(items),reviewed_frame_evidence=sum(b['frames'][1]-b['frames'][0]+1 for b in items),pages=sum(len(b['pages']) for b in items)))
        print('DENSE_EVIDENCE',len(items),sum(len(b['pages']) for b in items),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
