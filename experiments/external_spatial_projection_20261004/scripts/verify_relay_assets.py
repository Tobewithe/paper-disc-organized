"""Verify unchanged official assets relayed to server; no model execution."""
import argparse,hashlib,json,os,zipfile,time,traceback
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args();cfg=json.loads(Path(a.config).read_text());r=Path(cfg['server_root']).resolve();out=Path(a.out);start=time.monotonic()
 if os.name=='nt' or str(r)!='/root/external_spatial_projection_20261004':raise RuntimeError('server only')
 try:
  src=r/'relay';meta=json.loads((src/'RELAY_MANIFEST.json').read_text());archive=src/'sam_source.zip';weight=src/'sam_vit_b_01ec64.pth';code=Path(cfg['sam_code']);target=Path(cfg['sam_checkpoint'])
  if (r/'ASSET_MANIFEST.json').exists() or target.exists():raise RuntimeError('Do not overwrite ready assets')
  assert meta['source_commit']=='dca509fe793f601edb92606367a655c15ac00fdf'
  assert digest(archive)==meta['source_archive_sha256'] and digest(weight)==meta['checkpoint_sha256'] and weight.stat().st_size==meta['checkpoint_bytes']==375042383
  if code.exists():
   assert code.resolve().is_relative_to(r)
   backup=r/'failed_assets'/'git_partial_http1';backup.parent.mkdir(exist_ok=True);assert not backup.exists();code.rename(backup)
  code.mkdir(parents=True)
  with zipfile.ZipFile(archive) as z:
   names=z.namelist();expected=set(meta['source_files_sha256'])|{'README.md','LICENSE'};assert len(names)==len(set(names)) and set(names)==expected
   for n in names:
    parts=PurePosixPath(n);assert not parts.is_absolute() and '..' not in parts.parts
    data=z.read(n)
    if n in meta['source_files_sha256']:assert hashlib.sha256(data).hexdigest()==meta['source_files_sha256'][n]
    dest=code.joinpath(*parts.parts);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
  target.parent.mkdir(exist_ok=True,parents=True);os.link(weight,target)
  result=dict(meta,source_path=str(code),checkpoint=str(target),model_type='vit_b',created_at=datetime.now(timezone.utc).isoformat(),no_model_execution=True,elapsed_s=time.monotonic()-start,upstream_published_sha256='not supplied by README; local identity only')
  dump(r/'ASSET_MANIFEST.json',result);dump(out/'ASSET_MANIFEST.json',result);dump(out/'COMPLETE.json',{'passed':True,'completed':True,'elapsed_s':time.monotonic()-start})
 except BaseException as e:dump(out/'FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
