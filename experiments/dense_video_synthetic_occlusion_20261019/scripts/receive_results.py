"""Check transport hashes, extract into a new directory, and audit sampled labels."""
import argparse
import collections
import hashlib
import json
import pathlib
import shutil
import tarfile

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--delivery',required=True)
    p.add_argument('--repository',required=True)
    args=p.parse_args()
    delivery=pathlib.Path(args.delivery)
    receipt=json.loads((delivery/'DELIVERY.json').read_text(encoding='utf-8'))
    archive=delivery/'scene-pilot-v1.tar.gz'
    assert archive.stat().st_size==receipt['size_bytes']
    assert sha(archive)==receipt['sha256']
    extracted=delivery/'verified'
    extracted.mkdir(exist_ok=False)
    with tarfile.open(archive) as tf:
        for m in tf.getmembers():
            assert (extracted/m.name).resolve().is_relative_to(extracted.resolve())
            assert not m.issym() and not m.islnk()
        tf.extractall(extracted)
    manifest=json.loads((extracted/'ARTIFACT_MANIFEST.json').read_text())
    assert len(manifest)==receipt['files']
    assert sha(extracted/'ARTIFACT_MANIFEST.json')==receipt['manifest_sha256']
    for item in manifest:
        file=extracted/item['path']
        assert file.stat().st_size==item['size_bytes'] and sha(file)==item['sha256'],item['path']
    study=extracted/'experiments/dense_video_synthetic_occlusion_20261019'
    pilot=study/'runs/RUN_SCENE_PILOT_20261019T092948_31ae46d0'
    dataset=pilot/'dataset'
    checked=0
    full_intervals=[]
    for folder in sorted(p for p in dataset.iterdir() if p.is_dir()):
        data=json.loads((folder/'annotations.json').read_text())
        for entry in data['independent_scene_raycast_checks']:
            with np.load(folder/'labels'/('%04d.npz'%entry['frame'])) as a:
                for c in entry['checks']:
                    assert c['track_id']==int(a['instance_id'][c['y'],c['x']])
                    checked+=1
        for track in [1,2]:
            rows=[r for r in data['records'] if r['track_id']==track]
            start=None
            for r in rows:
                if r['state']=='fully_occluded' and start is None:
                    start=r['frame']
                elif r['state']!='fully_occluded' and start is not None:
                    full_intervals.append({'scene':folder.name,'track_id':track,'start_frame':start,'end_frame':r['frame']-1,'reappearance_frame':r['frame'],'right_censored':False})
                    start=None
            if start is not None:
                full_intervals.append({'scene':folder.name,'track_id':track,'start_frame':start,'end_frame':rows[-1]['frame'],'reappearance_frame':None,'right_censored':True})
    assert checked==6912
    for run in (study/'runs').iterdir():
        source=run/'SOURCE_MANIFEST.json'
        if source.exists():
            for item in json.loads(source.read_text()):
                assert sha(run/'source'/item['path'])==item['sha256']
    repository=pathlib.Path(args.repository)
    local_study=repository/'experiments/dense_video_synthetic_occlusion_20261019'
    copied=0
    for file in (study/'runs').rglob('*'):
        if file.is_file() and 'dataset' not in file.relative_to(study).parts:
            target=local_study/file.relative_to(study)
            target.parent.mkdir(parents=True,exist_ok=True)
            assert not target.exists(),'Refuse overwriting prior Run evidence: '+str(target)
            shutil.copy2(file,target)
            copied+=1
    shutil.copy2(study/'SLURM_RECEIPT.txt',local_study/'SLURM_RECEIPT.txt')
    result={'status':'passed','archive_sha256':receipt['sha256'],'files_verified':len(manifest),'sampled_geometry_ids_verified':checked,'full_occlusion_intervals':full_intervals,'reappearance_intervals':sum(not x['right_censored'] for x in full_intervals),'repository_evidence_files_copied':copied,'verified_root':str(extracted),'dataset':str(dataset),'software_not_in_delivery':True}
    (local_study/'TRANSFER_VERIFICATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
