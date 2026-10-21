"""V3 registration and raw-video occlusion review; never reads tracker outputs."""
from pathlib import Path
import cv2
import numpy as np
from .common import Run, read_json, write_json, sha256, load_gt


def register(root):
    parent=Path(r'D:\thesis-video-v2-20261018')
    manifest=read_json(parent/'input_manifest.json')
    for s in manifest['sequences']:
        if not s.get('cache_path'):
            p=read_json(parent/('demo_features_pointer.json' if s['split']=='demo' else 'features_pointer.json'))
            s['cache_path']=str(parent/p['cache']/s['name'])
        s['previous_split']=s['split']
        s['split']='annotation_only' if s['split']=='demo' else 'engineering' if s['split']=='train' else 'retrospective'
    with Run(root,'reversible_registration',dict(new_training=0,new_sequences=['zebra_5','duck_1','duck_2'],new_outcomes_read=False)) as run:
        for name,cls in [('zebra_5',22),('duck_1',14),('duck_2',14)]:
            v=root/'data'/(name+'.mp4');g=root/'data'/(name+'_gt.txt');cap=cv2.VideoCapture(str(v));assert cap.isOpened()
            manifest['sequences'].append(dict(name=name,split='independent',model='horse',classes=[cls],video=str(v),gt=str(g),frames=int(cap.get(7)),fps=cap.get(5),width=int(cap.get(3)),height=int(cap.get(4)),video_sha256=sha256(v),gt_sha256=sha256(g),gt_kind='official_AnimalTrack_bbox_identity',visibility_labels=False))
            cap.release()
        for s in manifest['sequences']:
            assert sha256(Path(s['video']))==s['video_sha256'],s['name']
            if s.get('gt'): assert sha256(Path(s['gt']))==s['gt_sha256'],s['name']
        for k,p in manifest['models'].items():assert sha256(Path(p))==manifest['model_sha256'][k]
        write_json(root/'input_manifest.json',manifest);write_json(run.path/'input_manifest.json',manifest)
        write_json(run.path/'SUMMARY.json',dict(sequences=[dict(name=s['name'],frames=s['frames'],split=s['split']) for s in manifest['sequences']],hashes_passed=True))
        print('REGISTERED',len(manifest['sequences']),flush=True)


def panels(root):
    manifest=read_json(root/'input_manifest.json');out=root/'annotations';out.mkdir(exist_ok=True)
    with Run(root,'occlusion_candidate_review',dict(GT='proposal for raw-video review only',predictions_read=False,gap_range=[2,180],limit_per_sequence=8,max_total=128)) as run:
        selected=[];all_candidates=[]
        for s in sorted([s for s in manifest['sequences'] if s.get('gt')],key=lambda s:s['name']):
            gt=load_gt(root,s);ids=sorted({int(r[1]) for a in gt.values() for r in a});cand=[]
            for gid in ids:
                fs=[f for f,a in gt.items() if gid in a[:,1]]
                for before,after in zip(fs,fs[1:]):
                    if 2<=after-before-1<=180:cand.append(dict(sequence=s['name'],gt_id=gid,before=before,after=after,gap=after-before-1))
            cand.sort(key=lambda e:(-e['gap'],e['before'],e['gt_id']));all_candidates+=cand
            for e in cand[:8]:
                if len(selected)>=128:break
                e=dict(e,event_id=f'O{len(selected)+1:03d}')
                b=gt[e['before']][gt[e['before']][:,1]==e['gt_id']][0,2:6];c=gt[e['after']][gt[e['after']][:,1]==e['gt_id']][0,2:6]
                lo=np.minimum(b[:2],c[:2]);hi=np.maximum(b[2:],c[2:]);size=hi-lo
                lo=np.maximum(0,(lo-size*.7-30)).astype(int);hi=np.minimum([s['width'],s['height']],hi+size*.7+30).astype(int)
                fs=sorted(set([max(1,e['before']-2),e['before'],e['before']+1,(2*e['before']+e['after'])//3,(e['before']+2*e['after'])//3,e['after']-1,e['after'],min(s['frames'],e['after']+2)]))
                cap=cv2.VideoCapture(s['video']);canvas=np.full((650,1600,3),245,np.uint8)
                cv2.putText(canvas,f"{e['event_id']} {s['name']} GT {e['gt_id']}  {e['before']} -> {e['after']} gap {e['gap']}",(15,28),cv2.FONT_HERSHEY_SIMPLEX,.75,(20,20,20),2)
                for j,f in enumerate(fs):
                    cap.set(0 if False else cv2.CAP_PROP_POS_FRAMES,f-1);ok,im=cap.read();assert ok
                    rows=gt[f][gt[f][:,1]==e['gt_id']]
                    if len(rows):
                        bb=rows[0,2:6].astype(int);cv2.rectangle(im,tuple(bb[:2]),tuple(bb[2:]),(30,220,30),3)
                    crop=im[lo[1]:hi[1],lo[0]:hi[0]];h,w=crop.shape[:2];scale=min(390/w,260/h);thumb=cv2.resize(crop,(round(w*scale),round(h*scale)))
                    x=(j%4)*400+5;y=(j//4)*300+68;canvas[y:y+len(thumb),x:x+thumb.shape[1]]=thumb
                    cv2.putText(canvas,f'f{f} '+('GT' if len(rows) else 'NO LABEL'),(x,y-9),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1)
                cap.release();p=out/(e['event_id']+'.jpg');cv2.imwrite(str(p),canvas,[cv2.IMWRITE_JPEG_QUALITY,92]);e.update(panel=p.name,review_frames=fs,crop=[*lo.tolist(),*hi.tolist()],panel_sha256=sha256(p));selected.append(e)
        write_json(out/'candidates.json',all_candidates);write_json(out/'review_manifest.json',dict(selection='eight longest 2..180-frame gaps per sequence; no prediction-based selection',events=selected))
        write_json(run.path/'SUMMARY.json',dict(all_candidates=len(all_candidates),review_events=len(selected),by_sequence={s:sum(e['sequence']==s for e in selected) for s in sorted({e['sequence'] for e in selected})}))
        print('PANELS',len(selected),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('mode',choices=['register','panels']);a=p.parse_args();(register if a.mode=='register' else panels)(a.root)
