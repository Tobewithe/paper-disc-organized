"""Register and extract the two protocol-selected zebra videos after model freeze."""
from pathlib import Path
import cv2
from .common import Run,read_json,write_json,sha256
from .infer import extract


def main(root):
    assert (root/'trajectory_selector_pointer.json').exists(),'Model must be frozen before transfer extraction'
    manifest=read_json(root/'input_manifest.json');names=['zebra_1','zebra_2']
    assert not any(s['name'] in names for s in manifest['sequences'])
    with Run(root,'transfer_registration',dict(sequences=names,selection='lexicographically first two entire zebra videos',model=read_json(root/'trajectory_selector_pointer.json'),outcomes_read=False)) as run:
        write_json(run.path/'input_manifest_before.json',manifest)
        for name in names:
            video=root/'data'/(name+'.mp4');gt=root/'data'/(name+'_gt.txt');cap=cv2.VideoCapture(str(video));assert cap.isOpened()
            seq=dict(name=name,split='transfer',model='horse',classes=[22],video=str(video),gt=str(gt),frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),fps=cap.get(cv2.CAP_PROP_FPS),width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),video_sha256=sha256(video),gt_sha256=sha256(gt),gt_kind='official_AnimalTrack_bbox_identity',visibility_labels=False,model_note='horse key points to unmodified COCO YOLO26m-seg; class22 zebra')
            cap.release();manifest['sequences'].append(seq);print('TRANSFER_FROZEN',seq,flush=True)
        write_json(root/'input_manifest.json',manifest);write_json(run.path/'input_manifest_after.json',manifest)
    extract(root,read_json(root/'config.json')['detector'],only_names=names,pointer_name='transfer_features_pointer.json')
    ptr=read_json(root/'transfer_features_pointer.json')
    for seq in manifest['sequences']:
        if seq['name'] in names:seq['cache_path']=str(root/ptr['cache']/seq['name'])
    write_json(root/'input_manifest.json',manifest)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
