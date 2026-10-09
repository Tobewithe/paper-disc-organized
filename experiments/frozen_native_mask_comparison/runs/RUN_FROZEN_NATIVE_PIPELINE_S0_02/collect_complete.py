from pathlib import Path
import hashlib,json,zipfile,datetime
root=Path('D:/coco_wire/experiments/frozen_native_mask_comparison')
rid='RUN_FROZEN_NATIVE_5K_INFERENCE_S0_02';run=root/'runs'/rid
record=json.loads((run/'run.json').read_text(encoding='utf-8'))
assert record['status']=='completed' and record['return_code']==0 and record['artifact_completeness']=='complete',record
summary=json.loads((run/'SUMMARY.json').read_text(encoding='utf-8'))
assert summary['passed'] is True and summary['image_count']==5000 and not summary['engineering']
parity=json.loads((run/'BASELINE_PARITY.json').read_text(encoding='utf-8'))
assert parity['passed'] is True and parity['covers_full5000'] is True
for arm in summary['arm_names']:
 assert len(list((run/arm/'images').glob('*.json')))==5000
 c=json.loads((run/arm/'COMPLETE.json').read_text(encoding='utf-8'))
 assert c['status']=='prediction_complete' and c['image_count']==5000
output=root/'runs/RUN_FROZEN_NATIVE_PIPELINE_S0_02/FULL_INFERENCE_RETURN.zip'
assert not output.exists(),output
files=sorted(p for p in run.rglob('*') if p.is_file())
manifest=[];total_bytes=0
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
 for file in files:
  rel=file.relative_to(run).as_posix(); data=file.read_bytes(); h=hashlib.sha256(data).hexdigest()
  total_bytes+=len(data);archive.writestr(rid+'/'+rel,data);manifest.append(h+'  '+rel)
 archive.writestr(rid+'/manifest.sha256','\n'.join(manifest)+'\n')
 receipt={'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'complete scientific consumption artifacts; excludes external images/vendor/official weights','run_id':rid,'source_run_status':record['status'],'source_return_code':record['return_code'],'source_artifact_completeness':record['artifact_completeness'],'file_count':len(files),'source_bytes':total_bytes,'image_json_files':35000,'all7arms_present':True,'images':5000,'source_run_json_preserved_verbatim':True,'summary_sha256':hashlib.sha256((run/'SUMMARY.json').read_bytes()).hexdigest()}
 archive.writestr('FULL_BUNDLE_RECEIPT.json',json.dumps(receipt,indent=2))
result={'path':str(output),'zip_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'zip_bytes':output.stat().st_size,**receipt}
(root/'runs/RUN_FROZEN_NATIVE_PIPELINE_S0_02/FULL_RETURN_PACKAGE.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
