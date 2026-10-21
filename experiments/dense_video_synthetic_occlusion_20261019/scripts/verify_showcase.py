"""Consumer-format and full media decode verification for the final delivery."""
import hashlib
import json
import pathlib
import re

import cv2
import numpy as np
from pycocotools.coco import COCO

REPO=pathlib.Path('C:/Dpan/codexproject/paper-disc-organized')
STUDY=REPO/'experiments/dense_video_synthetic_occlusion_20261019'
SHOW=REPO/'paper/presentations/三维遮挡场景与自动标签_20261019'
BASE=pathlib.Path('C:/Users/Public/occlusion-scene-20261019/delivery')
DATA=BASE/'verified/experiments/dense_video_synthetic_occlusion_20261019/runs/RUN_SCENE_PILOT_20261019T092948_31ae46d0/dataset'
EXPORT=BASE/'exports_verified/exports/v1'


def main():
    result={'status':'running','coco_masks_checked':0,'videos':[],'html_browser_rendered':False,'visual_inspection':['labels_example.png','scene_overview.png','visibility_curves.png']}
    for kind in ['visible','amodal']:
        coco=COCO(str(EXPORT/('annotations_'+kind+'.coco.json')))
        for annotation in coco.anns.values():
            image=coco.imgs[annotation['image_id']]
            frame=image['frame_id']
            scene=image['scene_id']
            with np.load(DATA/scene/'labels'/('%04d.npz'%frame)) as arrays:
                target=arrays['instance_id']==annotation['track_id'] if kind=='visible' else arrays['amodal_%d'%annotation['track_id']]>0
                actual=coco.annToMask(annotation)>0
                assert np.array_equal(actual,target),'Official COCO consumer mask mismatch'
                assert int(actual.sum())==annotation['area']
            result['coco_masks_checked']+=1
    assert result['coco_masks_checked']==532
    for path in sorted((SHOW/'videos').glob('*.mp4')):
        video=cv2.VideoCapture(str(path))
        assert video.isOpened()
        fps=video.get(cv2.CAP_PROP_FPS)
        count=0
        while True:
            ok,frame=video.read()
            if not ok:break
            assert frame.shape==(900,1600,3)
            assert frame.std()>10
            count+=1
        video.release()
        assert count==24 and abs(fps-6)<1e-6
        result['videos'].append({'file':path.name,'decoded_frames':count,'fps':fps,'width':1600,'height':900})
    assert len(result['videos'])==6
    html=(SHOW/'index.html').read_text(encoding='utf-8')
    targets=re.findall(r'(?:src|href)="([^"]+)"',html)
    for target in targets:
        if not target.startswith(('http:','https:','#')):
            assert (SHOW/target).exists(),target
    result['html_assets_checked']=len(targets)
    result['decoded_video_frames']=sum(v['decoded_frames'] for v in result['videos'])
    result['status']='passed'
    result['artifact_manifest']=[{'path':p.relative_to(SHOW).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size} for p in sorted(SHOW.rglob('*')) if p.is_file()]
    (STUDY/'FINAL_VERIFICATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_manifest'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
