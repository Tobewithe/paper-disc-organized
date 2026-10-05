"""Desktop FILE TRANSFER only. All tensors/training/evaluation run on 28358lan."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time


def remote(code):
    enc=base64.b64encode(code.encode('utf-16le')).decode()
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','28358lan',
        'powershell','-NoProfile','-EncodedCommand',enc],text=True,capture_output=True,timeout=300,check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root);stage=root/'runs/RUN_7f5d9d3a4f924602ba15ec72cb86e91c'
    desktop_cfg=json.loads((root/'DATA_STAGE_CONFIG.json').read_text(encoding='utf-8-sig'))
    log=root/'DELIVERY_STATUS.json';state=dict(stage='waiting_for_dataset',local_compute='network and file transport only')
    def save():log.write_text(json.dumps(state,indent=2),encoding='utf-8')
    save();start=time.monotonic()
    while not (stage/'COMPLETE.json').exists():
        record=stage/'run.json'
        if record.exists():
            run=json.loads(record.read_text(encoding='utf-8-sig'))
            if run.get('status') in ('failed','cancelled'):raise RuntimeError('Dataset staging failed; do not launch partial cohort')
        if time.monotonic()-start>7200:raise TimeoutError('Dataset staging exceeded two-hour transfer budget')
        time.sleep(15)
    split=json.loads((root/'SPLIT.json').read_text())
    archive=root/'DATA_IMAGES.tar';state['stage']='packing_fixed_images';save()
    with tarfile.open(archive,'w') as tar:
        for domain,ids in [('train2017',split['fit']+split['dev']),('val2017',split['val'])]:
            for iid in ids:
                relative=f'{domain}/{iid:012d}.jpg';tar.add(Path(desktop_cfg['images'])/relative,arcname=relative,recursive=False)
    h=hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    digest=h.hexdigest();state.update(stage='transferring',archive_bytes=archive.stat().st_size,sha256=digest);save()
    subprocess.run(['scp',str(archive),'28358lan:D:/coco_wire/bgcr_native_20261002/DATA_IMAGES.tar'],check=True)
    code=f"""
$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$root='D:\\coco_wire\\bgcr_native_20261002'
if((Get-FileHash -LiteralPath \"$root\\DATA_IMAGES.tar\" -Algorithm SHA256).Hash.ToLower() -ne '{digest}') {{ throw 'Image transport hash mismatch' }}
& 'C:\\Windows\\System32\\tar.exe' -xf \"$root\\DATA_IMAGES.tar\" -C 'D:\\coco_wire\\data\\images'
if($LASTEXITCODE -ne 0) {{ throw 'Image archive extraction failed' }}
$env:PYTHONPATH='D:\\coco_wire\\py'
$py='C:\\Users\\28358\\anaconda3\\envs\\pytorch\\pythonw.exe'
$active=Get-CimInstance Win32_Process | Where-Object {{$_.Name -like 'python*' -and $_.CommandLine -match 'bgcr_native_20261002.*pipeline.py'}}
if($active) {{ throw 'Existing pipeline; refuse duplicate launch' }}
$command='\"'+$py+'\" -u \"'+$root+'\\scripts\\pipeline.py\" --root \"'+$root+'\"'
$p=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{CommandLine=$command;CurrentDirectory=$root}}
if($p.ReturnValue -ne 0) {{ throw 'Windowless independent launch failed' }}
@{{pipeline_pid=$p.ProcessId;dataset_sha256='{digest}';transferred_at=(Get-Date).ToString('o');launch='Win32_Process.Create with pythonw'}} | ConvertTo-Json | Set-Content -LiteralPath \"$root\\DATA_TRANSFER.json\"
$p.ProcessId
"""
    result=remote(code);state.update(stage='pipeline_launched',remote_result=result.stdout.strip(),finished_at=time.time());save()


if __name__=='__main__':main()
