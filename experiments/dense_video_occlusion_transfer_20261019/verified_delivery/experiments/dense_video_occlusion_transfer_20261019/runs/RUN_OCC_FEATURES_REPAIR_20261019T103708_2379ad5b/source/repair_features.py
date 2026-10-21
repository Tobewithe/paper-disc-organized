"""Repair long-gap reactivation inputs only; preserve synthetic data and trained models."""
import pathlib,collections
import numpy as np
from .common import ROOT,Run,read,save,sha
from .features import Extractor,stream,video_frames,load_tracks,pack

def main():
 previous=read(ROOT/'features_pointer.json');p=pathlib.Path(previous['path']);catalog=read(p/'FEATURE_MANIFEST.json');seqmap={s['name']:s for s in read(ROOT/'data/sequences.json')['sequences']}
 with Run('features_repair',{'parent':previous,'change':'include every current observation before age filtering; no dropped long-gap reactivation','training_steps':0,'synthetic_changed':False,'checkpoints_changed':False}) as run:
  extractor=Extractor();dest=run.path/'features';dest.mkdir();repaired=[];checks=[]
  for i,e in enumerate(catalog):
   assert sha(ROOT/e['path'])==e['sha256']
   if e['kind']!='real':continue
   base=load_tracks(e['name']);d=np.load(ROOT/e['path']);keys=set(zip(d['frames'],d['ids']));missing=[(f['frame'],t['base_id']) for f in base for t in f['tracks'] if (f['frame'],t['base_id']) not in keys]
   if missing:
    name=e['name'];seq=seqmap[name];obs=collections.defaultdict(dict);scale=np.asarray([640/seq['width'],360/seq['height']]*2)
    for f in base:obs[f['frame']]={t['base_id']:{'box':(np.asarray(t['box'])*scale).tolist(),'score':t['score']} for t in f['tracks']}
    rows=stream(extractor,video_frames(ROOT/'real_videos'/(name+'.mp4'),seq['frames']),obs,640,360);new=pack(rows,dest/(name+'.npz'));new.update(kind='real',split=e['split'],name=name);catalog[i]=new;repaired.append({'sequence':name,'missing_before':len(missing),'old_sha256':e['sha256'],'new_sha256':new['sha256']});d=np.load(ROOT/new['path']);keys=set(zip(d['frames'],d['ids']))
   assert all((f['frame'],t['base_id']) in keys for f in base for t in f['tracks']);checks.append({'sequence':e['name'],'every_current_observation_covered':True});print('INPUT_COVERAGE',e['name'],len(missing),'repaired',flush=True)
  # All synthetic training / validation / test arrays remain byte-identical.
  old=read(p/'FEATURE_MANIFEST.json');assert [e for e in catalog if e['kind']=='synthetic']==[e for e in old if e['kind']=='synthetic']
  save(run.path/'FEATURE_MANIFEST.json',catalog);save(run.path/'SUMMARY.json',{'parent':previous,'repairs':repaired,'checks':checks,'synthetic_data_byte_identical':True,'trained_checkpoints_reused':True});save(ROOT/'features_pointer.json',{'run_id':run.id,'path':str(run.path)})
if __name__=='__main__':main()
