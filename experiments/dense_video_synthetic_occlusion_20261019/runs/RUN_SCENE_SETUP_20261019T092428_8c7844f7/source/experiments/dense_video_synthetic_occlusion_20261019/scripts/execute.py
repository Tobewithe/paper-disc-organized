"""Audited finite setup/render stages, called inside Slurm allocations."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import traceback
import uuid

ROOT = pathlib.Path('/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion')
STUDY = ROOT/'experiments/dense_video_synthetic_occlusion_20261019'
STUDY_ID = 'STUDY_DENSE_VIDEO_SYNTHETIC_OCCLUSION_20261019'
EXPECTED = 'be0fbaa0c1e52d4552023220b4c67351efbb707cac49bb381fcbee2182447005'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def setup(run):
    software=ROOT/'software'
    archive=software/'blender-4.2.1-linux-x64.tar.xz'
    assert sha(archive)==EXPECTED,'Official archive SHA mismatch'
    destination=software/'blender-4.2.1-linux-x64'
    assert not destination.exists(),'Existing install must be audited instead of overwritten'
    with tarfile.open(archive) as tf:
        base=software.resolve()
        for member in tf.getmembers():
            resolved=(base/member.name).resolve()
            assert resolved.is_relative_to(base),'Unsafe archive member'
            if member.issym():
                assert (resolved.parent/member.linkname).resolve().is_relative_to(base),'Unsafe archive symlink'
        tf.extractall(software)
    blender=destination/'blender'
    p=subprocess.run([str(blender),'--version'],capture_output=True,text=True,timeout=60)
    (run/'blender_version.log').write_text(p.stdout+p.stderr)
    assert p.returncode==0,'Blender executable failed; inspect version log'
    ldd=subprocess.run(['ldd',str(blender)],capture_output=True,text=True,timeout=30)
    (run/'ldd.log').write_text(ldd.stdout+ldd.stderr)
    assert 'not found' not in ldd.stdout,'Missing runtime libraries; keep isolated install'
    result={'archive_sha256':EXPECTED,'blender':str(blender),'version_output':p.stdout,'blender_executable_sha256':sha(blender),'setup_run_id':run.name}
    save(ROOT/'INSTALLATION.json',result)
    return result


def render(run, stage):
    install=json.loads((ROOT/'INSTALLATION.json').read_text())
    blender=pathlib.Path(install['blender'])
    assert sha(blender)==install['blender_executable_sha256']
    command=[str(blender),'--background','--factory-startup','--python-exit-code','1','--python',str(ROOT/'shared/dense_video_synthetic/render_scene.py'),'--','--mode',stage,'--output',str(run/'dataset')]
    save(run/'COMMAND.json',command)
    with open(run/'render.log','w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=43*60)
    assert result.returncode==0,'Renderer failed; see render.log'
    result=json.loads((run/'dataset/SUMMARY.json').read_text())
    result['installation_run_id']=install['setup_run_id']
    assert sum(x['frames'] for x in result['scenes'])==(2 if stage=='smoke' else 144)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['setup','smoke','pilot'])
    args=p.parse_args()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
    run_id='RUN_SCENE_%s_%s_%s'%(args.stage.upper(),stamp,uuid.uuid4().hex[:8])
    run=STUDY/'runs'/run_id
    run.mkdir(parents=True,exist_ok=False)
    lock=STUDY/(args.stage+'.lock')
    with open(lock,'x') as f:
        f.write(run_id)
    record={'study_id':STUDY_ID,'run_id':run_id,'stage_id':args.stage,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution_status':'running','executor':os.uname().nodename,'user':os.environ.get('USER'),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'command':[sys.executable,*sys.argv],'code_commit':'bc0507c5400b61935fe93f981a4ff5026964c271','training_steps':0,'protocol_sha256':sha(STUDY/'PROTOCOL.md'),'contract_sha256':sha(STUDY/'GENERATION_CONTRACT.md')}
    save(run/'run.json',record)
    sources=[STUDY/'PROTOCOL.md',STUDY/'GENERATION_CONTRACT.md',*sorted((STUDY/'scripts').glob('*.py')),*sorted((STUDY/'scripts').glob('*.sbatch')),*sorted((ROOT/'shared/dense_video_synthetic').glob('*.py'))]
    manifest=[]
    for source in sources:
        rel=source.relative_to(ROOT)
        dest=run/'source'/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
        manifest.append({'path':str(rel),'sha256':sha(source),'size_bytes':source.stat().st_size})
    save(run/'SOURCE_MANIFEST.json',manifest)
    record['source_manifest_sha256']=sha(run/'SOURCE_MANIFEST.json')
    save(run/'EVAL_CONTRACT.json',{'protocol_sha256':record['protocol_sha256'],'generation_contract_sha256':record['contract_sha256'],'unit':'scene/track/frame','label_backend':'pixel-center mesh ray intersections','renderer':'Blender 4.2.1 / Cycles OptiX','scientific_scope':'engineering validation only','training_steps':0})
    try:
        result=setup(run) if args.stage=='setup' else render(run,args.stage)
        save(run/'SUMMARY.json',result)
        save(run/'DECISION.json',{'status':'passed','scope':'finite engineering stage; no learned capability or real-video benefit claim','next_action':{'setup':'two-frame renderer smoke','smoke':'six-scene finite pilot','pilot':'independent dataset verification and visual inspection'}[args.stage]})
        record.update(execution_status='completed',exit_code=0)
    except Exception:
        (run/'ERROR.txt').write_text(traceback.format_exc())
        record.update(execution_status='failed',exit_code=1)
    finally:
        record['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(run/'run.json',record)
        if lock.exists() and lock.read_text()==run_id:
            lock.unlink()
        print(json.dumps(record),flush=True)
        if (run/'SUMMARY.json').exists():
            print((run/'SUMMARY.json').read_text(),flush=True)
    sys.exit(record['exit_code'])


if __name__=='__main__':
    main()
