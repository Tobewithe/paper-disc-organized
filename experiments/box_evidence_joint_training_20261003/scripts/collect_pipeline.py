"""Desktop transport only: status mirror, then final run artifact transfer."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

REMOTE=r'D:\coco_wire\box_evidence_joint_training_20261003'


def hidden_process_options():
    """pythonw does not stop console children from creating their own windows."""
    options = dict(stdin=subprocess.DEVNULL)
    if os.name == 'nt':
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        options.update(creationflags=subprocess.CREATE_NO_WINDOW, startupinfo=startup)
    return options


def remote(code, timeout=180):
    code="[Console]::OutputEncoding=[Text.UTF8Encoding]::new();$OutputEncoding=[Text.UTF8Encoding]::new();"+code
    enc=base64.b64encode(code.encode('utf-16le')).decode()
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','28358lan',
                           'powershell','-NoProfile','-EncodedCommand',enc],capture_output=True,
                          text=True,encoding='utf-8',errors='replace',timeout=timeout,
                          **hidden_process_options())
    if result.returncode:raise RuntimeError(result.stderr[-2000:])
    return result.stdout


def dump(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);a=parser.parse_args()
    root=Path(a.root);deadline=time.monotonic()+24*3600
    while time.monotonic()<deadline:
        try:
            output=remote(f"$root='{REMOTE}';if(Test-Path -LiteralPath \"$root\\PIPELINE_STATUS.json\"){{Get-Content -LiteralPath \"$root\\PIPELINE_STATUS.json\" -Raw -Encoding UTF8}}")
            if not output.strip():time.sleep(30);continue
            state=json.loads(output);dump(root/'PIPELINE_STATUS.json',state)
            current=state.get('current_run')
            if current:
                payload=remote(f"$p='{REMOTE}\\runs\\{current}';@{{run=if(Test-Path \"$p\\run.json\"){{Get-Content \"$p\\run.json\" -Raw -Encoding UTF8|ConvertFrom-Json}};progress=if(Test-Path \"$p\\PROGRESS.json\"){{Get-Content \"$p\\PROGRESS.json\" -Raw -Encoding UTF8|ConvertFrom-Json}}}}|ConvertTo-Json -Depth 16")
                data=json.loads(payload);dest=root/'runs'/current;dest.mkdir(parents=True,exist_ok=True)
                for name,key in [('run.json','run'),('PROGRESS.json','progress')]:
                    if data.get(key) is not None:dump(dest/name,data[key])
                dump(dest/'transfer.json',dict(status='metadata_snapshot_only',source_host='28358lan',
                     snapshot_at=datetime.now(timezone.utc).isoformat(),run_id=current))
            study=json.loads((root/'study.json').read_text(encoding='utf-8-sig'))
            study.update(status=state['status'],current_stage=state['stage'],latest_run_id=current)
            dump(root/'study.json',study)
            if state['status'] in ('completed','failed'):break
        except Exception as exc:
            dump(root/'COLLECTION_WARNING.json',dict(error=str(exc),at=datetime.now(timezone.utc).isoformat(),training_not_interrupted=True))
        time.sleep(30)
    else:raise TimeoutError('Transport polling limit; remote computation not terminated')
    code=r"""
$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$root='__ROOT__'
Get-ChildItem -LiteralPath "$root\runs" -Directory | ForEach-Object {
 $base=$_.FullName
 $lines=Get-ChildItem -LiteralPath $base -Recurse -File | Where-Object {$_.Name -ne 'manifest.sha256'} | Sort-Object FullName | ForEach-Object {((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower()+'  '+$_.FullName.Substring($base.Length+1).Replace('\','/'))}
 $lines|Set-Content -LiteralPath "$base\manifest.sha256" -Encoding UTF8
}
Compress-Archive -LiteralPath "$root\runs" -DestinationPath "$root\FINAL_RUNS.zip" -Force
""".replace('__ROOT__',REMOTE)
    remote(code,timeout=180)
    archive=root/'FINAL_RUNS.zip'
    subprocess.run(['scp','-B','28358lan:D:/coco_wire/box_evidence_joint_training_20261003/FINAL_RUNS.zip',str(archive)],
                   check=True,capture_output=True,**hidden_process_options())
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if not (root/member.filename).resolve().is_relative_to(root.resolve()):raise RuntimeError('Unsafe archive path')
        z.extractall(root)
    for manifest in (root/'runs').glob('*/manifest.sha256'):
        count=0
        for line in manifest.read_text(encoding='utf-8-sig').splitlines():
            digest,relative=line.split('  ',1);file=manifest.parent/relative;h=hashlib.sha256()
            with file.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
            if h.hexdigest()!=digest:raise RuntimeError(f'Transfer mismatch: {file}')
            count+=1
        dump(manifest.parent/'transfer.json',dict(status='complete',verified_files=count,
             run_id=manifest.parent.name,source_host='28358lan',transferred_at=datetime.now(timezone.utc).isoformat()))
    study=json.loads((root/'study.json').read_text(encoding='utf-8-sig'))
    if state['status']=='completed':study['report']=f"runs/{state['runs']['evaluation']}/REPORT.md"
    dump(root/'study.json',study)
    dump(root/'COLLECTION_COMPLETE.json',dict(status=state['status'],completed_at=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
