"""Stage one selected image bank over LAN and use the existing recorded launcher."""
import argparse,base64,json,os,subprocess,time,zipfile
from pathlib import Path

def remote(script,timeout=120):
    encoded=base64.b64encode(script.encode('utf-16-le')).decode('ascii')
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','28358lan',
        'powershell -NoProfile -EncodedCommand '+encoded],capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError(p.stderr.decode('utf-8',errors='replace'))
    return p.stdout.decode('utf-8-sig',errors='replace').strip()

def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--prepare-run',required=True);p.add_argument('--bank-run',required=True)
    a=p.parse_args();root=Path(a.project);study=root/'experiments/mask_boundary_route_20260914'
    prep=study/'runs'/a.prepare_run;out=study/'runs'/a.bank_run;out.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+3600
    while True:
        record=json.loads((prep/'run.json').read_text())
        if record['status']=='completed':break
        if record['status'] in ('failed','cancelled','interrupted'):raise RuntimeError('Selected dataset preparation failed')
        if time.monotonic()>deadline:raise TimeoutError('Preparation timeout')
        time.sleep(10)
    assert json.loads((prep/'SUMMARY.json').read_text())['complete']
    manifest=json.loads((prep/'image_manifest.json').read_text())
    archive=out/'calibration_input_transfer.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED) as z:
        for row in manifest:z.write(root/'assets/datasets/coco/images/train2017'/row['file_name'],'images/train2017/'+row['file_name'])
        z.write(prep/'calibration_annotations.json','annotations/calibration2000_train2017.json')
        z.write(prep/'split.json','annotations/calibration2000_split.json')
    remote("New-Item -ItemType Directory -Force -Path 'D:/coco_wire/scripts/risk_v4_20260915','D:/coco_wire/transfers/risk_v4_20260915' | Out-Null")
    scripts=['decoder_calibration_experiment.py','mask_calibration.py','launch_decoder_controls.ps1']
    subprocess.run(['scp',*[str(study/'scripts'/n) for n in scripts],str(study/'PROTOCOL_RISK_V4.json'),
        '28358lan:D:/coco_wire/scripts/risk_v4_20260915/'],check=True,timeout=120)
    subprocess.run(['scp',str(archive),'28358lan:D:/coco_wire/transfers/risk_v4_20260915/calibration_inputs.zip'],check=True,timeout=600)
    remote("$ErrorActionPreference='Stop'\nExpand-Archive -LiteralPath 'D:/coco_wire/transfers/risk_v4_20260915/calibration_inputs.zip' -DestinationPath 'D:/coco_wire/data' -Force",timeout=300)
    command=("& 'D:/coco_wire/scripts/risk_v4_20260915/launch_decoder_controls.ps1' -RunId '"+a.bank_run+
        "' -Start 0 -Limit 2000 -ScriptRoot 'D:/coco_wire/scripts/risk_v4_20260915' -ProtocolName 'PROTOCOL_RISK_V4.json'"
        " -Images 'D:/coco_wire/data/images/train2017' -Annotations 'D:/coco_wire/data/annotations/calibration2000_train2017.json'"
        " -Variants 'official_zero,smooth_gated,global_0.25,global_0.5,official_global,global_1.0' -BenchmarkRepeats 0 -ExportOnly")
    # SSH may keep stdout open while its descendant runs; allow the bounded job
    # to finish rather than killing an SSH process that still owns the job.
    result=remote("$ErrorActionPreference='Stop'\n"+command,timeout=10800)
    atomic(out/'staging_receipt.json',dict(bank_run=a.bank_run,prepare_run=a.prepare_run,images=len(manifest),
        launcher='existing launch_decoder_controls.ps1; SSH lifetime is not a durability guarantee',receipt=result))
    print(result,flush=True)
    archive.unlink()

if __name__=='__main__':main()
