"""Unmodified high resolution crops for visual annotation, no tracker access."""
from pathlib import Path
import cv2
from .common import read_json,write_json,Run,sha256

def main(root):
    seq={s['name']:s for s in read_json(root/'input_manifest.json')['sequences']}
    events={e['event_id']:e for e in read_json(root/'annotations/review_manifest.json')['events']}
    choices={'O032':[51,70,80,90,100,110,120,127], 'O043':list(range(71,87)), 'O051':list(range(225,244))}
    out=root/'annotations/zoom';out.mkdir(exist_ok=True)
    with Run(root,'occlusion_zoom',dict(tracker_outputs_read=False,frames=choices)) as run:
        manifest=[]
        for name,frames in choices.items():
            e=events[name];cap=cv2.VideoCapture(seq[e['sequence']]['video']);x0,y0,x1,y1=e['crop']
            for f in frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES,f-1);ok,img=cap.read();assert ok
                crop=img[y0:y1,x0:x1];path=out/f'{name}_{f:04d}.png';cv2.imwrite(str(path),crop)
                manifest.append(dict(event=name,frame=f,crop=e['crop'],shape=list(crop.shape),path=path.name,sha256=sha256(path)))
            cap.release()
        write_json(out/'manifest.json',manifest);write_json(run.path/'SUMMARY.json',dict(frames=len(manifest)))
        print('Raw zoom evidence:',len(manifest))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
