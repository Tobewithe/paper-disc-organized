import collections
import datetime
import hashlib
import json
import os
import pathlib
import platform
import shutil
import sys
import traceback
import uuid

import numpy as np

ROOT=pathlib.Path('/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion')
STUDY=ROOT/'experiments/dense_video_synthetic_occlusion_20261019'
PARENT='RUN_SCENE_PILOT_20261019T092948_31ae46d0'


def rle(mask):
    a=np.asarray(mask,dtype=np.uint8).ravel(order='F')
    indices=np.flatnonzero(a[1:]!=a[:-1])+1
    bounds=np.r_[0,indices,len(a)]
    counts=np.diff(bounds).tolist()
    if a[0]: counts=[0]+counts
    decoded=np.repeat(np.arange(len(counts))%2,counts).astype(np.uint8)
    assert np.array_equal(decoded,a)
    return {'size':list(mask.shape),'counts':counts}


def main():
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
    rid='RUN_SCENE_LABEL_EXPORT_'+stamp+'_'+uuid.uuid4().hex[:8]
    run=STUDY/'runs'/rid
    run.mkdir(parents=True,exist_ok=False)
    (run/'export_labels.py').write_bytes(pathlib.Path(__file__).read_bytes())
    (run/'EXPORT_CONTRACT.md').write_bytes((STUDY/'EXPORT_CONTRACT.md').read_bytes())
    rec={'study_id':'STUDY_DENSE_VIDEO_SYNTHETIC_OCCLUSION_20261019','run_id':rid,'parent_run_id':PARENT,'stage_id':'label_format_export','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'executor':platform.node(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'execution_status':'running','command':[sys.executable,*sys.argv],'training_steps':0,'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'contract_sha256':hashlib.sha256((STUDY/'EXPORT_CONTRACT.md').read_bytes()).hexdigest()}
    try:
        dataset=STUDY/'runs'/PARENT/'dataset'
        out=ROOT/'exports/v1'
        out.mkdir(parents=True,exist_ok=False)
        visible={'info':{'description':'procedural engineering pilot; visible masks'},'images':[],'annotations':[],'categories':[{'id':1,'name':'procedural_quadruped'}]}
        amodal={'info':{'description':'amodal projected masks; supervision only'},'images':visible['images'],'annotations':[],'categories':visible['categories']}
        ledger=[]
        mot_count=0
        for si,folder in enumerate(sorted(x for x in dataset.iterdir() if x.is_dir())):
            data=json.loads((folder/'annotations.json').read_text())
            records={(r['frame'],r['track_id']):r for r in data['records']}
            mot=[]
            for frame in range(24):
                image_id=si*24+frame+1
                visible['images'].append({'id':image_id,'file_name':str((folder/'rgb'/('%04d.png'%frame)).relative_to(dataset)).replace('\\','/'),'width':512,'height':320,'scene_id':folder.name,'frame_id':frame})
                with np.load(folder/'labels'/('%04d.npz'%frame)) as a:
                    for track in [1,2]:
                        row=dict(records[frame,track],scene_id=folder.name,image_id=image_id)
                        ledger.append(row)
                        for kind,mask,container in [('visible',a['instance_id']==track,visible),('amodal',a['amodal_%d'%track]>0,amodal)]:
                            area=int(mask.sum())
                            if not area: continue
                            ys,xs=np.where(mask)
                            bbox=[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)]
                            ann={'id':len(container['annotations'])+1,'image_id':image_id,'category_id':1,'track_id':track,'scene_id':folder.name,'asset_id':row['asset_id'],'segmentation':rle(mask),'area':area,'bbox':bbox,'iscrowd':0,'annotation_type':kind}
                            container['annotations'].append(ann)
                            if kind=='visible':
                                visibility=area/row['full_projected_pixels']
                                mot.append(','.join(map(str,[frame+1,track,bbox[0]+1,bbox[1]+1,bbox[2],bbox[3],1,1,round(visibility,8)])))
            (out/'mot'/folder.name/'gt').mkdir(parents=True)
            (out/'mot'/folder.name/'gt/gt.txt').write_text('\n'.join(mot)+'\n')
            mot_count+=len(mot)
        assert len(visible['images'])==144
        assert len(visible['annotations'])==252 and len(amodal['annotations'])==280
        assert len(ledger)==288 and mot_count==252
        for name,data in [('annotations_visible.coco.json',visible),('annotations_amodal.coco.json',amodal)]:
            (out/name).write_text(json.dumps(data))
        (out/'occlusion_states.jsonl').write_text('\n'.join(json.dumps(r) for r in ledger)+'\n')
        summary={'status':'passed','images':144,'visible_coco_annotations':252,'amodal_coco_annotations':280,'mot_observations':252,'state_records':288,'rle_roundtrips':532,'output':str(out),'training_steps':0}
        (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
        (run/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
        rec.update(execution_status='completed',exit_code=0)
    except Exception:
        (run/'ERROR.txt').write_text(traceback.format_exc())
        rec.update(execution_status='failed',exit_code=1)
    finally:
        rec['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        (run/'run.json').write_text(json.dumps(rec,indent=2))
        print(json.dumps(rec),flush=True)
    raise SystemExit(rec['exit_code'])


if __name__=='__main__':main()
