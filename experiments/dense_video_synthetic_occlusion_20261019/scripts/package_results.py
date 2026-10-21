"""Package actual finite-stage evidence; software distributions stay on executor."""
import datetime
import hashlib
import json
import pathlib
import subprocess
import tarfile

root=pathlib.Path('/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion')
study=root/'experiments/dense_video_synthetic_occlusion_20261019'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def main():
    runs=[json.loads(p.read_text()) for p in (study/'runs').glob('*/run.json')]
    assert any(r['stage_id']=='pilot' and r['execution_status']=='completed' for r in runs)
    audits=[r for r in runs if r['stage_id']=='independent_array_audit' and r['execution_status']=='completed']
    assert audits,'Independent audit must pass before delivery'
    jobs=subprocess.check_output(['sacct','-j','115,116,117,118,119,120','--noheader','--format=JobID,State,ExitCode,Elapsed,AllocTRES','-P'],text=True)
    (study/'SLURM_RECEIPT.txt').write_text(jobs)
    save_dir=root/'delivery'
    save_dir.mkdir(exist_ok=True)
    candidates=[*study.rglob('*'),*(root/'shared/dense_video_synthetic').rglob('*'),*(root/'logs').rglob('*'),root/'INSTALLATION.json',root/'software/SOFTWARE_MANIFEST.json',root/'software/runtime-rpms/MANIFEST.json']
    files=sorted({p for p in candidates if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.lock'})
    manifest=[{'path':p.relative_to(root).as_posix(),'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
    manifest_path=save_dir/'ARTIFACT_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest,indent=2))
    archive=save_dir/'scene-pilot-v1.tar.gz'
    assert not archive.exists(),'Do not overwrite an earlier delivery'
    with tarfile.open(archive,'w:gz',compresslevel=3) as tf:
        tf.add(manifest_path,arcname='ARTIFACT_MANIFEST.json')
        for p in files:
            tf.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
    receipt={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive':str(archive),'size_bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(files),'manifest_sha256':sha(manifest_path),'audit_run_ids':[r['run_id'] for r in audits]}
    (save_dir/'DELIVERY.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
