"""Desktop stdlib transport only: deploy code and launch hidden laptop pipeline."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from collect_pipeline import remote, hidden_process_options, REMOTE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    required = ['joint_head.py', 'train_joint.py', 'evaluate_joint.py', 'pipeline.py']
    for name in required:
        source = root / 'scripts' / name
        ast.parse(source.read_text(encoding='utf-8-sig'), filename=str(source))
    payload = list(root.glob('*.json')) + [root / 'PROTOCOL.md'] + sorted((root / 'scripts').glob('*.py'))
    manifest = {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in payload}
    manifest_file = root / 'CODE_MANIFEST.json'
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    archive = root / 'DEPLOY.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in [*payload, manifest_file]:
            z.write(path, path.relative_to(root).as_posix())
    check = remote(f"if(Test-Path -LiteralPath '{REMOTE}\\PIPELINE_STATUS.json'){{throw 'Existing pipeline state; inspect before any relaunch'}}; 'fresh' ")
    if 'fresh' not in check:
        raise RuntimeError('Remote launch guard did not pass')
    remote_archive = 'D:/coco_wire/box_evidence_joint_training_20261003_DEPLOY.zip'
    subprocess.run(['scp', '-B', str(archive), '28358lan:' + remote_archive], check=True,
                   capture_output=True, **hidden_process_options())
    code = r'''
$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$root='__ROOT__'
Expand-Archive -LiteralPath '__ZIP__' -DestinationPath $root -Force
$manifest=Get-Content -LiteralPath "$root\CODE_MANIFEST.json" -Raw -Encoding UTF8 | ConvertFrom-Json
foreach($entry in $manifest.PSObject.Properties){
 $actual=(Get-FileHash -LiteralPath (Join-Path $root $entry.Name) -Algorithm SHA256).Hash.ToLower()
 if($actual -ne $entry.Value){throw "Code transfer mismatch: $($entry.Name)"}
}
$command='"C:\Users\28358\anaconda3\envs\pytorch\pythonw.exe" "'+$root+'\scripts\pipeline.py" --root "'+$root+'"'
$started=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command;CurrentDirectory=$root}
if($started.ReturnValue -ne 0){throw "WMI launch failed: $($started.ReturnValue)"}
@{process_id=$started.ProcessId;return_value=$started.ReturnValue;command=$command;host=$env:COMPUTERNAME;started_at=(Get-Date).ToString('o')} | ConvertTo-Json
'''.replace('__ROOT__', REMOTE).replace('__ZIP__', remote_archive)
    result = json.loads(remote(code))
    result['local_recorded_at'] = datetime.now(timezone.utc).isoformat()
    result['code_manifest_sha256'] = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
    (root / 'LAUNCH.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
