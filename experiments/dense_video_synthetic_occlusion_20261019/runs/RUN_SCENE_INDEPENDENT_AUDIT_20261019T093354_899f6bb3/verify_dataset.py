"""Independent array/annotation audit; does not call the generator's label code."""
import argparse
import collections
import datetime
import hashlib
import json
import pathlib
import platform
import shutil
import sys
import traceback
import uuid

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def verify(dataset):
    summary=json.loads((dataset/'SUMMARY.json').read_text())
    assert summary['mode']=='pilot'
    names=['front_occlusion','rear_control','similar_crossing','partial_occlusion','out_of_view','mixed_occlusion_exit']
    assert sorted(x.name for x in dataset.iterdir() if x.is_dir())==sorted(names)
    state_counts=collections.Counter()
    per_scene=[]
    events=[]
    for name in names:
        folder=dataset/name
        data=json.loads((folder/'annotations.json').read_text())
        assert data['frames']==list(range(24))
        assert len(data['records'])==48
        records={(r['frame'],r['track_id']):r for r in data['records']}
        assert len(records)==48
        last={}
        counts=collections.Counter()
        for frame in range(24):
            rgb=folder/'rgb'/('%04d.png'%frame)
            assert rgb.exists() and rgb.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
            with np.load(folder/'labels'/('%04d.npz'%frame),allow_pickle=False) as arrays:
                visible=arrays['instance_id']
                distance=arrays['range_to_surface_m']
                assert visible.shape==(320,512) and distance.shape==visible.shape
                assert set(np.unique(visible)).issubset({0,1,2,100,900})
                assert np.all(np.isfinite(distance[visible!=0]))
                assert np.all(distance[visible!=0]>0)
                for track in [1,2]:
                    r=records[frame,track]
                    full=arrays['overscan_%d'%track]
                    inside=arrays['amodal_%d'%track]
                    assert full.shape==(640,1024)
                    assert np.array_equal(full[160:480,256:768],inside)
                    seen=visible==track
                    assert not np.any(seen & (inside==0))
                    a,v,f=int(inside.sum()),int(seen.sum()),int(full.sum())
                    assert (a,v,f)==(r['in_frame_pixels'],r['visible_pixels'],r['full_projected_pixels'])
                    occ=(a-v)/a if a else None
                    trunc=1-a/f if f else None
                    assert occ==r['occlusion_fraction'] and trunc==r['truncation_fraction']
                    hidden=(inside>0)&~seen
                    owners={str(int(k)):int(n) for k,n in zip(*np.unique(visible[hidden],return_counts=True))}
                    assert owners==r['occluder_pixels'] and sum(owners.values())==a-v
                    assert not owners.get('0',0)
                    assert not r['overscan_censored']
                    state='out_of_view' if a==0 else 'fully_occluded' if v==0 else 'partially_occluded' if occ>.01 else 'visible'
                    assert r['state']==state
                    counts[state]+=1
                    if track in last and last[track]!=state:
                        events.append({'scene':name,'track_id':track,'frame':frame,'from':last[track],'to':state})
                    last[track]=state
                    assert np.asarray(r['object_to_world']).shape==(4,4)
                    assert r['asset_id']=='procedural_quadruped_%d'%track
        assert len(data['independent_scene_raycast_checks'])==24
        assert all(len(x['checks'])==48 for x in data['independent_scene_raycast_checks'])
        assert len(data['gpu_devices'])==1 and data['gpu_devices'][0]['type']=='OPTIX'
        state_counts.update(counts)
        per_scene.append({'scene':name,'frames':24,'track_observations':48,'states':dict(counts)})
    missing=[state for state in ['visible','partially_occluded','fully_occluded','out_of_view'] if state_counts[state]==0]
    assert not missing,'Required event coverage missing: '+str(missing)
    # Paired control: identical query trajectory and appearance, obstacle placed behind it.
    front=json.loads((dataset/'front_occlusion/annotations.json').read_text())
    rear=json.loads((dataset/'rear_control/annotations.json').read_text())
    for r1,r2 in zip(front['records'],rear['records']):
        assert r1['track_id']==r2['track_id']
        assert r1['object_to_world']==r2['object_to_world']
    assert any(r['track_id']==1 and r['state']=='fully_occluded' for r in front['records'])
    assert all(r['state']=='visible' for r in rear['records'] if r['track_id']==1),'Rear control query should remain visible'
    return {'status':'passed','scenes':6,'frames':144,'track_observations':288,'independent_geometry_checks':6912,'state_counts':dict(state_counts),'per_scene':per_scene,'events':events,'paired_trajectory_control_passed':True,'model_training_steps':0,'scientific_scope':'synthetic geometry/labels only; real-video transfer untested'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True)
    p.add_argument('--study',required=True)
    p.add_argument('--parent-run',required=True)
    args=p.parse_args()
    study=pathlib.Path(args.study)
    dataset=pathlib.Path(args.dataset)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
    run_id='RUN_SCENE_INDEPENDENT_AUDIT_'+stamp+'_'+uuid.uuid4().hex[:8]
    run=study/'runs'/run_id
    run.mkdir(parents=True,exist_ok=False)
    record={'study_id':'STUDY_DENSE_VIDEO_SYNTHETIC_OCCLUSION_20261019','run_id':run_id,'parent_run_id':args.parent_run,'stage_id':'independent_array_audit','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution_status':'running','executor':platform.node(),'command':[sys.executable,*sys.argv],'dataset_path':str(dataset),'source_sha256':sha(__file__),'protocol_sha256':sha(study/'PROTOCOL.md'),'contract_sha256':sha(study/'GENERATION_CONTRACT.md'),'training_steps':0}
    shutil.copy2(__file__,run/'verify_dataset.py')
    shutil.copy2(study/'PROTOCOL.md',run/'PROTOCOL.md')
    shutil.copy2(study/'GENERATION_CONTRACT.md',run/'GENERATION_CONTRACT.md')
    try:
        result=verify(dataset)
        (run/'SUMMARY.json').write_text(json.dumps(result,indent=2))
        manifest=[{'path':str(x.relative_to(dataset)),'sha256':sha(x),'size_bytes':x.stat().st_size} for x in sorted(dataset.rglob('*')) if x.is_file()]
        (run/'DATASET_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
        (run/'DECISION.json').write_text(json.dumps({'status':'passed','next_action':'visual inspection followed by a separately frozen real-transfer training study','claim_scope':result['scientific_scope']},indent=2))
        record.update(execution_status='completed',exit_code=0)
    except Exception:
        (run/'ERROR.txt').write_text(traceback.format_exc())
        record.update(execution_status='failed',exit_code=1)
    finally:
        record['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        (run/'run.json').write_text(json.dumps(record,indent=2))
        print(json.dumps(record),flush=True)
    if record['exit_code']:
        raise SystemExit(record['exit_code'])


if __name__=='__main__':
    main()
