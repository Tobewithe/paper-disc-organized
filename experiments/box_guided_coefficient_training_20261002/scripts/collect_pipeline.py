"""Transport this running job's status and final artifacts; no model computation."""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile


def remote(code):
    enc=base64.b64encode(code.encode('utf-16le')).decode()
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','28358lan',
        'powershell','-NoProfile','-EncodedCommand',enc],capture_output=True,text=True,timeout=180)
    if r.returncode:raise RuntimeError(r.stderr[-1000:])
    return r.stdout


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    deadline=time.monotonic()+36*3600;errors=0
    while time.monotonic()<deadline:
        try:
            output=remote("$p='D:\\coco_wire\\bgcr_native_20261002\\PIPELINE_STATUS.json'; if(Test-Path -LiteralPath $p){ Get-Content -LiteralPath $p -Raw }")
            if not output.strip():time.sleep(30);continue
            status=json.loads(output)
            (root/'PIPELINE_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
            current=status.get('current_run')
            if current:
                payload=remote(f"$p='D:\\coco_wire\\bgcr_native_20261002\\runs\\{current}'; @{{run=(Get-Content -LiteralPath \"$p\\run.json\" -Raw | ConvertFrom-Json);progress=if(Test-Path -LiteralPath \"$p\\PROGRESS.json\"){{Get-Content -LiteralPath \"$p\\PROGRESS.json\" -Raw | ConvertFrom-Json}}else{{$null}}}} | ConvertTo-Json -Depth 12")
                data=json.loads(payload);dest=root/'runs'/current;dest.mkdir(parents=True,exist_ok=True)
                for name,key in [('run.json','run'),('PROGRESS.json','progress')]:
                    if data.get(key) is not None:(dest/name).write_text(json.dumps(data[key],indent=2,ensure_ascii=False),encoding='utf-8')
                (dest/'transfer.json').write_text(json.dumps(dict(status='metadata_snapshot_only',run_id=current,
                    source_host='28358lan',snapshot_at=datetime.now(timezone.utc).isoformat()),indent=2),encoding='utf-8')
            study=json.loads((root/'study.json').read_text(encoding='utf-8-sig'))
            study.update(status='running' if status['status']=='running' else status['status'],current_stage=status['stage'],latest_run_id=current)
            (root/'study.json').write_text(json.dumps(study,indent=2,ensure_ascii=False),encoding='utf-8')
            errors=0
            if status['status'] in ('completed','failed'):break
        except Exception as exc:
            errors+=1
            (root/'COLLECTION_WARNING.json').write_text(json.dumps(dict(consecutive_errors=errors,error=str(exc),
                training_not_interrupted=True),indent=2),encoding='utf-8')
        time.sleep(30)
    else:raise TimeoutError('Status transfer budget expired; remote job is not terminated')
    # Only after the pipeline exits are its child run records/artifacts stable.
    remote(r"""
$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$root='D:\coco_wire\bgcr_native_20261002'
Get-ChildItem -LiteralPath "$root\runs" -Directory | ForEach-Object {
 $base=$_.FullName
 $lines=Get-ChildItem -LiteralPath $base -Recurse -File | Where-Object {$_.Name -ne 'manifest.sha256'} | Sort-Object FullName | ForEach-Object {((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower()+'  '+$_.FullName.Substring($base.Length+1).Replace('\','/'))}
 $lines | Set-Content -LiteralPath "$base\manifest.sha256" -Encoding UTF8
}
Compress-Archive -LiteralPath "$root\runs" -DestinationPath "$root\FINAL_RUNS.zip"
""")
    archive=root/'FINAL_RUNS.zip'
    subprocess.run(['scp','28358lan:D:/coco_wire/bgcr_native_20261002/FINAL_RUNS.zip',str(archive)],check=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target=(root/info.filename).resolve()
            if not target.is_relative_to(root.resolve()):raise RuntimeError('Unsafe archive path')
        z.extractall(root)
    for manifest in (root/'runs').glob('*/manifest.sha256'):
        count=0
        for line in manifest.read_text(encoding='utf-8-sig').splitlines():
            digest,relative=line.split('  ',1);path=manifest.parent/relative;h=hashlib.sha256()
            with path.open('rb') as stream:
                for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
            if h.hexdigest()!=digest:raise RuntimeError(f'Transfer hash mismatch: {path}')
            count+=1
        (manifest.parent/'transfer.json').write_text(json.dumps(dict(status='complete',verified_files=count,
            run_id=manifest.parent.name,source_host='28358lan',transferred_at=datetime.now(timezone.utc).isoformat()),indent=2),encoding='utf-8')
    if status['status']=='completed':
        relative=f"runs/{status['runs']['evaluation']}/REPORT.md"
        study=json.loads((root/'study.json').read_text(encoding='utf-8-sig'));study['report']=relative
        (root/'study.json').write_text(json.dumps(study,indent=2,ensure_ascii=False),encoding='utf-8')
    (root/'COLLECTION_COMPLETE.json').write_text(json.dumps(dict(status=status['status'],completed_at=datetime.now(timezone.utc).isoformat()),indent=2),encoding='utf-8')


if __name__=='__main__':main()
