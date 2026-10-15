"""Package completed or failed Runs without changing them; retain all large local artifacts."""
import argparse,hashlib,json,os,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]


def accessible(p):
    return Path('\\\\?\\'+str(p.resolve())) if os.name=='nt' and not str(p).startswith('\\\\?\\') else p


def sha(p):
    h=hashlib.sha256()
    with accessible(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(accessible(p).read_text(encoding='utf8'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    study=a.study.resolve();assert study.parent==ROOT/'experiments'
    assert not list(study.glob('.*.lock')) and not (study/'.AUTONOMOUS_OWNER.guard').exists()
    files=[];states=[]
    for run in sorted((study/'runs').iterdir()):
        if not run.is_dir():continue
        state=read(run/'run.json');assert state['execution_status'] in ('completed','error'),state['run_id'];states.append(state)
        manifest=run/'ARTIFACT_MANIFEST.json'
        if manifest.exists():
            for item in read(manifest)['artifacts']:
                assert sha(run/item['relative_path'])==item['sha256'],item['relative_path']
        else:assert state['execution_status']=='error'
        for parent,_,names in os.walk(accessible(run)):
            for name in names:
                path=Path(parent)/name
                normal=Path(str(path)[4:]) if str(path).startswith('\\\\?\\') else path
                if normal.suffix!='.pyc':files.append(normal)
    for pattern in ('INDEPENDENT*.json','AUDIT_*.json','AUTONOMOUS*.json','PAIRED_COMPARISON.json','*execution.log'):
        files.extend(study.glob(pattern))
    files=sorted(set(files))
    manifest=study/'TRANSFER_MANIFEST.json'
    value=dict(study_id=read(study/'study.json')['study_id'],runs=[dict(run_id=s['run_id'],execution_status=s['execution_status']) for s in states],
        files=[dict(path=v.relative_to(ROOT).as_posix(),size_bytes=accessible(v).stat().st_size,sha256=sha(v)) for v in files])
    with manifest.open('x',encoding='utf8') as f:json.dump(value,f,indent=2);f.write('\n')
    with zipfile.ZipFile(a.output,'x',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for path in files+[manifest]:z.writestr(path.relative_to(ROOT).as_posix(),accessible(path).read_bytes())
    result=dict(archive=str(a.output),archive_sha256=sha(a.output),bytes=a.output.stat().st_size,files=len(files))
    with (study/'PACKAGE_RECEIPT.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
