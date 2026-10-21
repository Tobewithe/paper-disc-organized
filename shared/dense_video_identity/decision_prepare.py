from pathlib import Path
import cv2
from .common import Run, read_json, write_json, sha256
from .infer import extract

FIT=['pig_3','pig_5','horse_4','horse_6','pig_2','horse_3','zebra_1','duck_1']
CAL=['pig_4','horse_5','zebra_2','duck_2']
NEW=['duck_3','duck_4','duck_5','duck_6']

def main(root):
    assert read_json(Path(read_json(root/'evidence_probe_pointer.json')['path'])/'DECISION.json')['status']=='eligible_for_frozen_learning_pilot'
    manifest=read_json(root/'input_manifest.json')
    assert not any(s['name'] in NEW for s in manifest['sequences'])
    with Run(root,'decision_split_registration',dict(fit=FIT,calibration=CAL,confirmation=NEW,source_archive='AnimalTrack duck.zip previously downloaded official source',outcomes_read=False)) as run:
        for s in manifest['sequences']:s['decision_split']='fit' if s['name'] in FIT else 'calibration' if s['name'] in CAL else 'seen_holdout'
        for name in NEW:
            v=root/'data'/(name+'.mp4');g=root/'data'/(name+'_gt.txt');cap=cv2.VideoCapture(str(v));assert cap.isOpened()
            manifest['sequences'].append(dict(name=name,split='confirmation',decision_split='confirmation',model='horse',classes=[14],video=str(v),gt=str(g),frames=int(cap.get(7)),fps=cap.get(5),width=int(cap.get(3)),height=int(cap.get(4)),video_sha256=sha256(v),gt_sha256=sha256(g),gt_kind='official_AnimalTrack_bbox_identity',visibility_labels=False))
            cap.release()
        write_json(root/'input_manifest.json',manifest);write_json(run.path/'input_manifest.json',manifest)
        write_json(run.path/'SUMMARY.json',dict(sequences=[dict(name=s['name'],split=s['decision_split'],frames=s['frames']) for s in manifest['sequences']]))
    extract(root,read_json(root/'config.json')['detector'],only_names=NEW,pointer_name='new_features_pointer.json')
    pointer=read_json(root/'new_features_pointer.json');manifest=read_json(root/'input_manifest.json')
    for s in manifest['sequences']:
        if s['name'] in NEW:s['cache_path']=str(root/pointer['cache']/s['name'])
    write_json(root/'input_manifest.json',manifest)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
