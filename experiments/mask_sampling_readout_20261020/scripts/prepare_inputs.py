"""Finite, hash-verified COCO input preparation; no model import or training."""
from __future__ import annotations
import concurrent.futures as futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import time
import traceback
import urllib.request
import uuid

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''): h.update(block)
    return h.hexdigest()

def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)

def now(): return dt.datetime.now(dt.timezone.utc).isoformat()

def main():
    import psutil
    from PIL import Image
    started=time.monotonic()
    run=STUDY/'runs'/('RUN_SSR_PREPARE_'+uuid.uuid4().hex)
    lock=STUDY/'.prepare.lock'
    with lock.open('x',encoding='utf-8') as f: json.dump({'pid':os.getpid(),'run_id':run.name},f)
    run.mkdir(parents=True)
    meta={'study_id':'STUDY_MASK_SAMPLING_READOUT_20261020','run_id':run.name,'stage':'input_preparation',
          'host':socket.gethostname(),'pid':os.getpid(),'started_at':now(),'execution_status':'running',
          'training_steps':0,'protocol_sha256':sha(STUDY/'PREPARATION_PROTOCOL.md'),'script_sha256':sha(__file__),
          'code_commit':'bc0507c5400b61935fe93f981a4ff5026964c271','command':__import__('sys').argv}
    def status(**kw):
        meta.update(kw);meta['updated_at']=now();meta['elapsed_seconds']=time.monotonic()-started
        write(run/'run.json',meta);write(STUDY/'PREPARATION_STATUS.json',meta)
    try:
        status(phase='hash_existing_images')
        contract=read(STUDY/'inputs/PARENT_DATA_CONTRACT.json')
        assert len(contract['train_ids'])==20000 and len(contract['val_ids'])==5000
        annotations=Path('D:/coco_wire/data/annotations')
        for split,expected in [('train','610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d'),('val','e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f')]:
            assert sha(annotations/f'instances_{split}2017.json')==expected
        annotation_images={r['id']:r for r in read(annotations/'instances_train2017.json')['images']}
        write(run/'INPUT_CONTRACT.json',{'parent_data_sha256':sha(STUDY/'inputs/PARENT_DATA_CONTRACT.json'),
              'orders_sha256':sha(STUDY/'inputs/PARENT_IMAGE_ORDERS.json'),'download_domain':'images.cocodataset.org',
              'maximum_workers':8,'maximum_attempts_per_image':3,'maximum_seconds':10800,
              'resources':{'memory_available':psutil.virtual_memory().available,'disk_free':shutil.disk_usage(STUDY).free}})
        paths={'train':{},'val':{}};missing=[];errors=[]
        own=STUDY/'data/train_images';own.mkdir(parents=True,exist_ok=True)
        for split in ('train','val'):
            for iid in contract[split+'_ids']:
                expected=contract[split+'_image_hashes'][str(iid)]
                original=Path(f'D:/coco_wire/data/images/{split}2017/{iid:012d}.jpg')
                path=original if original.exists() else own/f'{iid:012d}.jpg'
                if path.exists():
                    if sha(path)!=expected: errors.append({'image_id':iid,'path':str(path),'error':'existing_SHA_mismatch'})
                    else: paths[split][str(iid)]=str(path.resolve())
                elif split=='train':missing.append(iid)
                else:errors.append({'image_id':iid,'error':'validation_image_missing'})
        write(run/'EXISTING_AUDIT.json',{'existing_train':len(paths['train']),'existing_val':len(paths['val']),
              'missing_train_ids':missing,'errors':errors})
        if errors: raise RuntimeError('existing input identity errors; see EXISTING_AUDIT.json')
        status(phase='download_missing_official_images',missing_train=len(missing),completed_downloads=0)
        def download(iid):
            if time.monotonic()-started>10800: raise RuntimeError('input preparation deadline')
            if shutil.disk_usage(STUDY).free<12*2**30: raise RuntimeError('disk reserve below 12 GiB')
            target=own/f'{iid:012d}.jpg'; temp=target.with_suffix('.jpg.partial')
            url=f'https://images.cocodataset.org/train2017/{iid:012d}.jpg'
            expected=contract['train_image_hashes'][str(iid)]
            attempts=[]
            for attempt in range(1,4):
                try:
                    req=urllib.request.Request(url,headers={'User-Agent':'ResearchInputVerification/1.0'})
                    with urllib.request.urlopen(req,timeout=35) as response, temp.open('wb') as output:
                        shutil.copyfileobj(response,output,1<<20)
                    if sha(temp)!=expected: raise RuntimeError('download SHA mismatch')
                    with Image.open(temp) as im:
                        row=annotation_images[iid]
                        if im.size!=(row['width'],row['height']): raise RuntimeError('image dimensions mismatch')
                        im.verify()
                    if target.exists(): raise RuntimeError('target unexpectedly appeared')
                    os.replace(temp,target)
                    return {'image_id':iid,'path':str(target.resolve()),'sha256':expected,'bytes':target.stat().st_size,
                            'url':url,'attempt':attempt,'previous_errors':attempts}
                except Exception as e: attempts.append(str(e))
            return {'image_id':iid,'url':url,'error':attempts}
        downloaded=0;total_bytes=0
        with (run/'DOWNLOAD_RECEIPTS.jsonl').open('x',encoding='utf-8') as receipt:
            with futures.ThreadPoolExecutor(max_workers=8) as pool:
                for row in pool.map(download,missing):
                    receipt.write(json.dumps(row)+'\n');receipt.flush()
                    if 'error' in row:errors.append(row)
                    else:
                        paths['train'][str(row['image_id'])]=row['path'];total_bytes+=row['bytes']
                    downloaded+=1
                    if total_bytes>8*2**30: raise RuntimeError('image download budget 8 GiB exceeded')
                    if downloaded%100==0 or downloaded==len(missing):
                        status(completed_downloads=downloaded,downloaded_bytes=total_bytes,download_errors=len(errors))
                        print('INPUT_PROGRESS',downloaded,len(missing),len(errors),flush=True)
        if errors:
            write(run/'ERRORS.json',errors);raise RuntimeError(f'{len(errors)} input downloads failed; IDs unchanged')
        write(STUDY/'inputs/IMAGE_PATHS.json',paths)
        summary={'status':'passed','train_images':len(paths['train']),'val_images':len(paths['val']),
                 'all_parent_image_hashes_exact':True,'newly_downloaded':len(missing),'downloaded_bytes':total_bytes,
                 'image_paths_sha256':sha(STUDY/'inputs/IMAGE_PATHS.json'),'training_steps':0}
        write(run/'SUMMARY.json',summary)
        status(execution_status='completed',phase='completed',finished_at=now(),exit_code=0)
        write(STUDY/'PREPARE_POINTER.json',{'run_id':run.name,'summary_sha256':sha(run/'SUMMARY.json')})
    except BaseException as e:
        (run/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
        status(execution_status='failed',phase='failed',finished_at=now(),exit_code=1,error=str(e));raise
    finally:
        if lock.exists() and read(lock).get('pid')==os.getpid():lock.unlink()

if __name__=='__main__': main()
