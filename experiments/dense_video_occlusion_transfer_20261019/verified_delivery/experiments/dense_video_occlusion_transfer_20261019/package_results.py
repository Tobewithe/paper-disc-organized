"""Export traceable small artifacts; large feature caches remain on the server."""
import pathlib,json,tarfile,hashlib,shutil,datetime
R=pathlib.Path('/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion-v2')
S=R/'experiments/dense_video_occlusion_transfer_20261019'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 target=R/'result-delivery-v1.tar.gz';assert not target.exists()
 files=[]
 for p in S.rglob('*'):
  if p.is_file() and 'features' not in p.relative_to(S).parts and not p.name.endswith('.lock'):files.append(p)
 for f in ['scenes.json','ASSET_INSPECTION.json','assets_pointer.json','smoke_pointer.json','features_pointer.json','features_repair_pointer.json','train_pointer.json','evaluate_pointer.json','adjudicate_pointer.json','DEPENDENCY_RECEIPT.json','EXPORT_RECEIPT.json','VIDEO_RECEIPT.json','data/sequences.json','data/INPUT_FILES.json','real_videos/MANIFEST.json','assets/ASSET_MANIFEST.json','assets/PACKAGE_MANIFEST.json','annotations/occlusion_labels.json']:
  p=R/f
  if p.exists():files.append(p)
 for d in sorted((R/'synthetic_v2').iterdir()):
  files.append(d/'annotations.json')
  if int(d.name) in [0,1,8,15,22,29,48,55,60,65,66,71]:
   for i in [0,10,16,31]:files.extend([d/'rgb'/('%03d.png'%i),d/'labels'/('%03d.npz'%i)])
 manifest=[{'path':str(p.relative_to(R)),'sha256':sha(p),'size_bytes':p.stat().st_size} for p in sorted(set(files))]
 mp=R/'RESULT_FILES.json';mp.write_text(json.dumps(manifest,indent=2));files.append(mp)
 with tarfile.open(target,'w:gz',compresslevel=3) as t:
  for p in sorted(set(files)):t.add(p,arcname=str(p.relative_to(R)),recursive=False)
 receipt={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive':str(target),'sha256':sha(target),'size_bytes':target.stat().st_size,'files':len(manifest),'server_feature_caches':'retained, not exported','synthetic_frames':2304,'synthetic_observations':9216,'training':json.load(open(pathlib.Path(json.load(open(R/'train_pointer.json'))['path'])/'run.json')),'evaluation':json.load(open(pathlib.Path(json.load(open(R/'evaluate_pointer.json'))['path'])/'run.json'))}
 (R/'RESULT_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
