"""One-shot official input download and hash-verified SSH relay, no training."""
import concurrent.futures as cf
import hashlib,json,os,subprocess,time,traceback,urllib.request,uuid,zipfile
from pathlib import Path
from prepare_inputs import STUDY,read,write,sha,now

REMOTE='D:/coco_wire/baseline_native_20261013/experiments/mask_sampling_readout_20261020'
PY='C:/Users/28358/anaconda3/envs/pytorch/python.exe'

def main():
 run=STUDY/'runs'/('RUN_SSR_RELAY_'+uuid.uuid4().hex);run.mkdir(parents=True)
 t=time.monotonic();meta={'run_id':run.name,'study_id':'STUDY_MASK_SAMPLING_READOUT_20261020','stage':'input_relay',
  'executor':'local download to authorized laptop','pid':os.getpid(),'started_at':now(),'execution_status':'running',
  'training_steps':0,'protocol_sha256':sha(STUDY/'PREPARATION_PROTOCOL.md'),'script_sha256':sha(__file__)}
 def update(**kw):
  meta.update(kw);meta['elapsed_seconds']=time.monotonic()-t;meta['updated_at']=now()
  write(run/'run.json',meta);write(STUDY/'RELAY_STATUS.json',meta)
 try:
  update(phase='download')
  inv=read(STUDY/'inputs/REMOTE_INVENTORY.json');assert not inv['bad']
  assert all(r['split']=='train' for r in inv['missing'])
  ids=[r['image_id'] for r in inv['missing']]
  contract=read(STUDY/'inputs/PARENT_DATA_CONTRACT.json')
  local=STUDY/'data/relay';local.mkdir(parents=True,exist_ok=True)
  def fetch(iid):
   path=local/f'{iid:012d}.jpg';expected=contract['train_image_hashes'][str(iid)]
   if path.exists():
    assert sha(path)==expected;return {'image_id':iid,'sha256':expected,'path':str(path),'source':'existing_relay'}
   url=f'http://images.cocodataset.org/train2017/{iid:012d}.jpg';errors=[]
   for attempt in range(3):
    try:
     if time.monotonic()-t>10800:raise RuntimeError('relay deadline')
     with urllib.request.urlopen(url,timeout=35) as r:body=r.read()
     assert hashlib.sha256(body).hexdigest()==expected,'official image SHA mismatch'
     from PIL import Image
     import io
     with Image.open(io.BytesIO(body)) as im:im.verify()
     path.write_bytes(body)
     return {'image_id':iid,'sha256':expected,'bytes':len(body),'path':str(path),'url':url,'attempt':attempt+1,'previous_errors':errors}
    except Exception as e:errors.append(str(e))
   raise RuntimeError(f'failed image {iid}: {errors}')
  all_rows=[];pending=[];chunk=0
  def transfer(rows,number):
   archive=run/f'chunk_{number:04d}.zip'
   manifest={f'{r["image_id"]:012d}.jpg':r['sha256'] for r in rows}
   with zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED) as z:
    for r in rows:z.write(r['path'],f'{r["image_id"]:012d}.jpg')
    z.writestr('MANIFEST.json',json.dumps(manifest))
   remote_archive=REMOTE+f'/inputs/chunk_{number:04d}.zip'
   subprocess.run(['scp','-q',str(archive),'28358lan:'+remote_archive],check=True,timeout=300)
   code='''from pathlib import Path
import json,zipfile,hashlib,os
root=Path(REMOTE_ROOT)
archive=Path(ARCHIVE_PATH)
target=root/'data/train_images';target.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(archive) as z:
 manifest=json.loads(z.read('MANIFEST.json'))
 for name,digest in manifest.items():
  p=(target/name).resolve();assert p.parent==target.resolve()
  body=z.read(name);assert hashlib.sha256(body).hexdigest()==digest
  if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
  else:
   with p.open('xb') as f:f.write(body)
  assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
receipt=root/'inputs'/('relay_'+archive.stem+'.json')
receipt.write_text(json.dumps({'status':'passed','images':len(manifest),'files':manifest}),encoding='utf-8')
archive.unlink()
print(json.dumps({'status':'passed','images':len(manifest)}))
'''.replace('REMOTE_ROOT',repr(REMOTE)).replace('ARCHIVE_PATH',repr(remote_archive))
   result=subprocess.run(['ssh','-o','BatchMode=yes','28358lan',PY,'-'],input=code,text=True,capture_output=True,timeout=180)
   if result.returncode:raise RuntimeError(result.stderr)
   receipt=json.loads(result.stdout);assert receipt['status']=='passed' and receipt['images']==len(rows)
   write(run/f'TRANSFER_{number:04d}.json',{'archive_sha256':sha(archive),'images':len(rows),'files':manifest,'remote_result':receipt})
   archive.unlink()
  with (run/'DOWNLOAD_RECEIPTS.jsonl').open('x',encoding='utf-8') as log:
   with cf.ThreadPoolExecutor(max_workers=8) as pool:
    jobs={pool.submit(fetch,i):i for i in ids}
    for future in cf.as_completed(jobs):
     row=future.result();all_rows.append(row);pending.append(row)
     log.write(json.dumps(row)+'\n')
     if len(pending)>=256:
      transfer(pending,chunk);chunk+=1;pending=[];log.flush()
      update(completed_images=len(all_rows),total_images=len(ids),transferred_chunks=chunk)
      print('RELAY',len(all_rows),len(ids),flush=True)
   if pending:transfer(pending,chunk);chunk+=1
  write(run/'SUMMARY.json',{'status':'passed','images':len(all_rows),'chunks':chunk,'all_source_and_destination_sha_exact':True})
  update(execution_status='completed',phase='completed',completed_images=len(all_rows),finished_at=now(),exit_code=0)
 except BaseException as e:
  (run/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8');update(execution_status='failed',error=str(e),exit_code=1);raise

if __name__=='__main__':main()
