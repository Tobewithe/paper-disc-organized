"""Desktop file relay only; no tensor/model imports or experiment computation."""
import concurrent.futures,hashlib,json,time,urllib.request,zipfile,sys
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
sys.stdout.reconfigure(encoding='utf-8')
r=Path(__file__).resolve().parents[1];dest=r/'_asset_relay';dest.mkdir(exist_ok=True)
commit='dca509fe793f601edb92606367a655c15ac00fdf';headers={'User-Agent':'research-asset-client'}
def fetch(url):
 with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=30) as f:return f.read()
tree=json.loads(fetch('https://api.github.com/repos/facebookresearch/segment-anything/git/trees/'+commit+'?recursive=1'))
assert not tree.get('truncated')
paths=[x['path'] for x in tree['tree'] if x['type']=='blob' and ((x['path'].startswith('segment_anything/') and x['path'].endswith('.py')) or x['path'] in ['LICENSE','README.md'])]
assert len(paths)>10
for name in paths:assert not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
source='https://raw.githubusercontent.com/facebookresearch/segment-anything/'+commit+'/'
def one(name):return name,fetch(source+name)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:contents=dict(pool.map(one,paths))
archive=dest/'sam_source.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
 for name,data in contents.items():z.writestr(name,data)
print('source_ready',len(contents),flush=True)
url='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth';weight=dest/'sam_vit_b_01ec64.pth';part=dest/'sam_vit_b_01ec64.pth.partial';n=0;h=hashlib.sha256();start=time.monotonic();last=0
with urllib.request.urlopen(url,timeout=45) as f,part.open('xb') as stream:
 expected=int(f.headers['Content-Length'])
 for b in iter(lambda:f.read(4*1024**2),b''):
  if time.monotonic()-start>600:raise TimeoutError('600s relay download budget')
  stream.write(b);n+=len(b);h.update(b)
  if time.monotonic()-last>15:print(json.dumps({'downloaded':n,'bytes':expected,'elapsed_s':time.monotonic()-start}),flush=True);last=time.monotonic()
assert n==expected==375042383
part.replace(weight)
manifest={'source_repository':'https://github.com/facebookresearch/segment-anything.git','source_commit':commit,'source_files_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in contents.items() if k.endswith('.py')},'checkpoint_url':url,'checkpoint_sha256':h.hexdigest(),'checkpoint_bytes':n,'source_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'downloaded_at':datetime.now(timezone.utc).isoformat(),'relay_reason':'Server git HTTP2/HTTP1.1 transport failures; files retrieved unchanged from official HTTPS endpoints; no model execution'}
(dest/'RELAY_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
p=r.parents[1];sys.path.insert(0,str(p/'experiments/prototype_guided_evidence_selection_20261003/scripts'));import server_transport as t
print(t.ssh('mkdir -p /root/external_spatial_projection_20261004/relay',30),flush=True)
t.put([str(archive),str(weight),str(dest/'RELAY_MANIFEST.json')],'/root/external_spatial_projection_20261004/relay/',timeout=600)
print('uploaded_assets',json.dumps(manifest),flush=True)
