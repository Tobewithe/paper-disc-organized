from pathlib import Path
import hashlib,json,zipfile,datetime
root=Path('D:/coco_wire/experiments/frozen_native_mask_comparison')
ids=['RUN_FROZEN_NATIVE_PIPELINE_S0_01','RUN_FROZEN_NATIVE_ENGINEERING_S0_01','RUN_FROZEN_NATIVE_5K_INFERENCE_S0_01','RUN_FROZEN_NATIVE_PIPELINE_S0_02','RUN_FROZEN_NATIVE_ENGINEERING_S0_02','RUN_FROZEN_NATIVE_5K_INFERENCE_S0_02']
output=root/'runs/RUN_FROZEN_NATIVE_PIPELINE_S0_02/MINIMAL_RETURN.zip'
if output.exists():raise FileExistsError(output)
known={'run.json','SUMMARY.json','BASELINE_PARITY.json','EVALUATION_INPUTS.json','EVALUATION_COMPLETE.json','EVALUATION_FAILURE.json','ORIGINAL_BASELINE_RECEIPT.json','PIPELINE_CONFIG.json','PIPELINE_STATUS.json','stdout.log','stderr.log','launcher.stdout.log','launcher.stderr.log'}
report={'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'minimal artifacts; full02/pipeline02 are active snapshots, not sealed results','runs':{}}
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
 for rid in ids:
  run=root/'runs'/rid; manifest=[]
  if not run.exists():continue
  for file in sorted(run.rglob('*')):
   if not file.is_file():continue
   rel=file.relative_to(run)
   include=(str(rel).replace('\\','/') in known or (rel.parts[0] in ('source','snapshots','assets')) or (rid.endswith('ENGINEERING_S0_02') and str(rel).replace('\\','/') in ('NATIVE_DECISIONS.jsonl','BASELINE_PARITY_IMAGES.jsonl','IMAGE_TIMINGS.jsonl')))
   if not include:continue
   data=file.read_bytes();digest=hashlib.sha256(data).hexdigest();archive.writestr(rid+'/'+rel.as_posix(),data);manifest.append(digest+'  '+rel.as_posix())
  archive.writestr(rid+'/manifest.sha256','\n'.join(manifest)+'\n')
  if (run/'run.json').exists():
   r=json.loads((run/'run.json').read_text(encoding='utf-8'));report['runs'][rid]={'status_observed':r.get('status'),'pid':r.get('pid'),'return_code':r.get('return_code'),'file_count':len(manifest),'terminal_snapshot':r.get('status') in ('completed','failed','cancelled')}
 archive.writestr('BUNDLE_RECEIPT.json',json.dumps(report,indent=2))
print(json.dumps({'zip':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'bytes':output.stat().st_size,'runs':report['runs']}))
