"""Finite return of these five already-authorized runs over the LAN SSH link.

Does not launch experiments, change automations, or modify remote results.
"""
from pathlib import Path
import argparse
import base64
import json
import subprocess
import time
import zipfile
from common import atomic

STUDY=Path(__file__).resolve().parents[2]
RUNS=STUDY/'runs'
IDS=['RUN_7fb7971e45a64569a94ed6f194a89c2c','RUN_c70956b46a3d4e769b7982a4d9267afd',
     'RUN_7bfe6b97306e4a56b56c1bd538e628ad','RUN_9fa3374a7519409fb2b09995ce37affb',
     'RUN_a720b6dc7b7448a7badb5bfd94df1d9e']


def remote(script,timeout=60):
    encoded=base64.b64encode(script.encode('utf-16-le')).decode()
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','28358lan',
                      'powershell','-NoProfile','-EncodedCommand',encoded],capture_output=True,text=True,encoding='utf-8',timeout=timeout)
    if r.returncode:raise RuntimeError(r.stderr[-1500:])
    return r.stdout


def poll():
    names=','.join("'"+x+"'" for x in IDS)
    script="$runs='D:/coco_wire/runs/mask_boundary_route_20260914';$values=@();foreach($id in @("+names+")){"+r'''
        $d="$runs/$id";if(Test-Path "$d/run.json"){
            $r=Get-Content "$d/run.json" -Raw|ConvertFrom-Json
            $p=$null;if(Test-Path "$d/progress.json"){$p=Get-Content "$d/progress.json" -Raw|ConvertFrom-Json}
            $values+=@{id=$id;status=$r.status;exit=$r.return_code;progress=$p;run=$r}
        }else{$values+=@{id=$id;status='queued'}}
    };ConvertTo-Json -InputObject $values -Depth 30 -Compress
    '''
    return json.loads(remote(script))


def copy_finished(run_id):
    # Exclude raw tensor banks from the small return; keep them on the laptop.
    code='''import json,zipfile
from pathlib import Path
r=Path('D:/coco_wire/runs/mask_boundary_route_20260914')/RUNID
out=Path('D:/coco_wire/transfers/rcmc_mechanism_20260920')
out.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(out/(RUNID+'.zip'),'w',zipfile.ZIP_DEFLATED) as z:
    for p in r.rglob('*'):
        if p.is_file() and 'raw' not in p.relative_to(r).parts and p.name!='.run.claim':
            z.write(p,p.relative_to(r).as_posix())
'''.replace('RUNID',repr(run_id))
    script="$code=@'\n"+code+"\n'@\n$code | & 'C:/Users/28358/anaconda3/envs/pytorch/python.exe' -\nif($LASTEXITCODE -ne 0){throw 'Packing failed'}"
    remote(script,timeout=120)
    destination=RUNS/run_id;destination.mkdir(parents=True,exist_ok=True)
    zip_path=destination/'returned_artifacts.zip'
    subprocess.run(['scp','-q','-o','BatchMode=yes',
        '28358lan:D:/coco_wire/transfers/rcmc_mechanism_20260920/'+run_id+'.zip',str(zip_path)],check=True,timeout=120)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            target=(destination/member).resolve()
            if not target.is_relative_to(destination.resolve()):raise RuntimeError('Archive path escaped run')
        z.extractall(destination)
    print(json.dumps({'returned':run_id}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    status_dir=STUDY/'execution/rcmc_mechanism_20260920';status_dir.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+3*3600;returned=set();errors=[]
    while time.monotonic()<deadline:
        try:
            states=poll()
            for state in states:
                if 'run' in state:
                    directory=RUNS/state['id'];directory.mkdir(parents=True,exist_ok=True)
                    if state['id'] not in returned:
                        atomic(directory/'run.json',state['run'])
                        if state.get('progress') is not None:atomic(directory/'progress.json',state['progress'])
                    if state['status'] in ('completed','failed','interrupted','cancelled') and state['id'] not in returned:
                        copy_finished(state['id']);returned.add(state['id'])
            compact=[{k:v for k,v in s.items() if k!='run'} for s in states]
            atomic(status_dir/'remote_status.json',{'observed_at_local':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'states':compact,'returned':sorted(returned),'errors':errors[-5:],
                'raw_tensor_location':'28358lan:D:/coco_wire/runs/mask_boundary_route_20260914/<threshold Run>/raw'})
            if args.once or len(returned)==len(IDS):break
            if any(s['status'] in ('failed','interrupted','cancelled') for s in states):
                break
        except Exception as exc:
            errors.append(str(exc));atomic(status_dir/'return_error.json',{'errors':errors[-5:]})
            if args.once:raise
        time.sleep(30)
    print(json.dumps({'returned':sorted(returned),'finished':len(returned)==len(IDS)}),flush=True)


if __name__=='__main__':main()
