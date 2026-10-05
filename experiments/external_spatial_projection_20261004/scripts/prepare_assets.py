"""Server-only immutable official SAM source/weight acquisition, no model inference."""
import argparse,hashlib,json,os,subprocess,time,traceback,urllib.request
from pathlib import Path
from datetime import datetime,timezone

def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--out',required=True);args=a.parse_args()
 cfg=json.loads(Path(args.config).read_text());out=Path(args.out);root=Path(cfg['server_root'])
 if os.name=='nt' or str(root)!='/root/external_spatial_projection_20261004':raise RuntimeError('server only')
 started=time.monotonic();code=Path(cfg['sam_code']);weight=Path(cfg['sam_checkpoint'])
 def budget():
  if time.monotonic()-started>900:raise TimeoutError('900s asset preparation budget, preserve partial')
 try:
  if (out/'COMPLETE.json').exists() or (root/'ASSET_MANIFEST.json').exists():raise RuntimeError('Asset preparation already registered; inspect instead of repeat')
  code.parent.mkdir(parents=True,exist_ok=True);weight.parent.mkdir(parents=True,exist_ok=True)
  if code.exists():raise RuntimeError('Partial/source directory exists, do not overwrite')
  subprocess.run(['git','-c','http.version=HTTP/1.1','clone','--depth','1','https://github.com/facebookresearch/segment-anything.git',str(code)],check=True,timeout=180)
  commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
  if subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip():raise RuntimeError('Dirty official source')
  url='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth';part=weight.with_suffix('.pth.partial')
  if weight.exists() or part.exists():raise RuntimeError('Existing artifact retained; no overwrite')
  count=0;last=0;h=hashlib.sha256()
  with urllib.request.urlopen(url,timeout=45) as response,part.open('xb') as stream:
   expected=int(response.headers.get('Content-Length','0'));final_url=response.url
   for block in iter(lambda:response.read(1024**2),b''):
    budget();stream.write(block);h.update(block);count+=len(block)
    if time.monotonic()-last>10:
     progress={'bytes':count,'expected_bytes':expected,'elapsed_s':time.monotonic()-started};dump(out/'PROGRESS.json',progress);print(json.dumps(progress),flush=True);last=time.monotonic()
  if expected and count!=expected:raise RuntimeError('Incomplete checkpoint size')
  if count<300*1024**2:raise RuntimeError('Unexpected checkpoint size; do not deserialize')
  part.replace(weight);budget()
  files={p.relative_to(code).as_posix():digest(p) for p in sorted((code/'segment_anything').rglob('*.py'))}
  manifest={'source_repository':'https://github.com/facebookresearch/segment-anything.git','source_commit':commit,'source_path':str(code),'source_files_sha256':files,'checkpoint_url':url,'final_download_url':final_url,'checkpoint':str(weight),'checkpoint_bytes':count,'checkpoint_sha256':h.hexdigest(),'model_type':'vit_b','created_at':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.monotonic()-started,'downloaded_from_official_link':True,'upstream_published_sha256':'not supplied by README; local identity hash only','no_model_execution':True}
  dump(root/'ASSET_MANIFEST.json',manifest);dump(out/'ASSET_MANIFEST.json',manifest);dump(out/'COMPLETE.json',{'passed':True,'completed':True,'elapsed_s':time.monotonic()-started})
 except BaseException as e:
  dump(out/'FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
