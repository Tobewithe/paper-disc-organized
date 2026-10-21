"""Frozen real-video transfer, existing references and mechanism ablations."""
import collections,gzip,json,pathlib,time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .common import ROOT,Run,read,save,sha
from .model import VisibilityHead,windows
from .features import load_tracks,overlap
from .recovery import recover
from .metrics import official_metrics,metric_data,compact,reappearance_events,iou_matrix

ARMS=['N','R','C','GEOM','FULL','NOW','NOCTX','SHUFFLE']

def write_gz(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(p,'wt') as f:json.dump(obj,f,separators=(',',':'))

def identity_geometry(frames):
 return [[(t['base_id'],t['det_id'],t['identity'],tuple(t['box']),t['score'],t.get('released_at')) for t in f['tracks']] for f in frames]

def geometry(frames):
 return [[(t['base_id'],t['det_id'],tuple(t['box']),t['score']) for t in f['tracks']] for f in frames]

@torch.inference_mode()
def infer(models,feature):
 x=torch.from_numpy(feature['x']).cuda();index=windows(x,feature['ids']).cuda()
 assert (index[:,-1]==torch.arange(len(x),device='cuda')).all() and (index<=torch.arange(len(x),device='cuda')[:,None]).all()
 out={}
 for arm,model in models.items():
  probabilities=[];visibility=[]
  for i in range(0,len(x),512):
   logit,value=model(x[index[i:i+512]],arm);probabilities.append(logit.softmax(-1).cpu().numpy());visibility.append(value.cpu().numpy())
  out[arm]={'visibility':np.concatenate(visibility),'prob':np.concatenate(probabilities)}
 return out

def physical_events(annotation,seq,outputs,gt):
 results=[];matches={};hist=collections.defaultdict(list)
 for frame in outputs:
  f=frame['frame'];tracks=frame['tracks'];g=gt[f];sim=iou_matrix(g[:,2:6],np.asarray([t['box'] for t in tracks]).reshape(-1,4))
  if sim.size:
   rr,cc=linear_sum_assignment(-sim)
   for r,c in zip(rr,cc):
    if sim[r,c]>=.5:
     gid=int(g[r,1]);pid=tracks[c]['identity'];matches[f,gid]=pid;hist[pid].append((f,gid))
 for e in annotation['events']:
  if e['sequence']!=seq or not e['eligible_partial_event']:continue
  observed=[]
  for label in e['visible_labels']:
   f=label['frame'];identity=matches.get((f,e['gt_id']))
   observed.append({**label,'predicted_id':identity,'matched':identity is not None})
  pre=[r['predicted_id'] for r in observed if r['phase']=='pre'];post=[r['predicted_id'] for r in observed if r['phase']=='post'];prior=collections.Counter(p for p in pre if p is not None).most_common(1)
  prior=prior[0][0] if prior else None
  same=sum(p==prior for p in post if p is not None) if prior is not None else 0
  strict=len(pre)>=3 and all(p is not None and p==pre[0] for p in pre[-3:]) and len(post)>=3 and all(p==pre[0] for p in post[:3])
  wrong=sum(p is not None and p!=prior and any(f<e['after'] and g!=e['gt_id'] for f,g in hist[p]) for p in post)
  results.append({'event_id':e['event_id'],'sequence':seq,'gt_id':e['gt_id'],'before':e['before'],'after':e['after'],'strict_3pre_3post_same_id':bool(strict),'post_labels':len(post),'post_detected':sum(p is not None for p in post),'same_id_post_observations':same,'wrong_reused_identity_post_observations':wrong,'pre_detected':sum(p is not None for p in pre),'matching':'all GT / all tracks one-to-one Hungarian IoU>=.5, prior frozen definition','scope':'AI-verified partial physical occlusion only; no verified complete occlusion','labels':observed})
 return results

def main():
 torch.set_num_threads(4);metrics=official_metrics(ROOT/'data');fp=pathlib.Path(read(ROOT/'features_pointer.json')['path']);tp=pathlib.Path(read(ROOT/'train_pointer.json')['path']);training=read(tp/'SUMMARY.json');catalog=read(fp/'FEATURE_MANIFEST.json')
 seqs=read(ROOT/'data/sequences.json')['sequences'];annotation=read(ROOT/'annotations/occlusion_labels.json')
 contract={'arms':ARMS,'sequences':[s['name'] for s in seqs],'heldout':'seen_holdout + confirmation; historically seen, not blind','train_run':str(tp),'feature_manifest_sha256':sha(fp/'FEATURE_MANIFEST.json'),'annotations_sha256':sha(ROOT/'annotations/occlusion_labels.json'),'recovery':'immutable C; only last5 gallery mean changes','threshold':.8,'margin':.03,'max_gap':60,'confirm':3,'buffer':4,'weight_floor':.05,'bootstrap':2000,'real_GT':'evaluation only'}
 with Run('evaluate',contract) as run:
  models={}
  for arm in ['FULL','NOW','NOCTX']:
   p=pathlib.Path(training[arm]['checkpoint']);assert sha(p)==training[arm]['sha256'];ck=torch.load(p,map_location='cpu',weights_only=False);model=VisibilityHead().cuda();model.load_state_dict(ck['state']);model.eval();models[arm]=model
  raw={arm:{} for arm in ARMS};table=[];checks=[];allphysical={a:[] for a in ARMS};predcatalog=[]
  for seq in seqs:
   start=time.perf_counter();name=seq['name'];entry=next(e for e in catalog if e.get('name')==name);assert sha(ROOT/entry['path'])==entry['sha256'];d=np.load(ROOT/entry['path']);predictions=infer(models,d)
   np.savez_compressed(run.path/(name+'_visibility.npz'),frames=d['frames'],ids=d['ids'],present=d['present'],**{a+'_'+k:v for a,p in predictions.items() for k,v in p.items()})
   predcatalog.append({'sequence':name,'sha256':sha(run.path/(name+'_visibility.npz'))})
   base=load_tracks(name);rec=read(ROOT/'data'/name/'detections.json');z=np.load(ROOT/'data'/name/'features64.npz')['z'];oldc=load_tracks(name,'confirm');reference=load_tracks(name,'reference')
   weights={a:{(int(f),int(k)):float(v) for f,k,v in zip(d['frames'],d['ids'],predictions[a]['visibility'])} for a in models}
   weights['GEOM']={};weights['SHUFFLE']={}
   for frame in base:
    tracks=sorted(frame['tracks'],key=lambda t:t['base_id']);f=frame['frame']
    for j,t in enumerate(tracks):
     k=t['base_id'];weights['GEOM'][f,k]=max(.05,1-overlap(t['box'],[tt['box'] for tt in tracks if tt['base_id']!=k]));weights['SHUFFLE'][f,k]=weights['FULL'][f,tracks[(j+1)%len(tracks)]['base_id']]
   replay,cevents=recover(base,rec,z);one,_=recover(base,rec,z,weights={(f['frame'],t['base_id']):1. for f in base for t in f['tracks']})
   assert identity_geometry(replay)==identity_geometry(oldc),(name,'C replay mismatch')
   assert replay==one,(name,'ones not exactly C')
   outputs={'N':base,'R':reference,'C':replay};eventsets={'N':[],'R':[],'C':cevents}
   for a in ['GEOM','FULL','NOW','NOCTX','SHUFFLE']:outputs[a],eventsets[a]=recover(base,rec,z,weights=weights[a]);assert geometry(outputs[a])==geometry(base)
   # Prefixes released before the cut must agree: no future identity confirmation.
   for a in ['C','GEOM','FULL','NOW','NOCTX','SHUFFLE']:
    for cut in [min(50,len(base)),min(100,len(base))]:
     partial,_=recover(base[:cut],rec[:cut],z,weights=None if a=='C' else weights[a]);assert partial[:-4]==outputs[a][:cut-4],(name,a,cut,'future access')
   checks.append({'sequence':name,'exact_C_replay':True,'ones_equal_C':True,'all_policy_geometry_equal':True,'prefix_causality':True,'identity_unique':True})
   del rec
   # Real GT loaded after all RGB inference and identity decisions exist.
   p=ROOT/'data'/name/'gt.txt';assert sha(p)==seq['gt_sha256'];gtrows=np.loadtxt(p,delimiter=',',ndmin=2);gtrows=gtrows[gtrows[:,6]==1];gtrows[:,4:6]+=gtrows[:,2:4];gt={f:gtrows[gtrows[:,0]==f] for f in range(1,seq['frames']+1)}
   for a,out in outputs.items():
    rs={k:m.eval_sequence(metric_data(gt,out)) for k,m in metrics.items()};raw[a][name]=rs;gaps=reappearance_events(gt,out);physical=physical_events(annotation,name,out,gt);allphysical[a]+=physical
    row={'arm':a,'sequence':name,'split':seq['decision_split'],**compact(rs),'confirmed':sum(e['type']=='confirmed' for e in eventsets[a]),'gap_proxy_eligible':gaps['eligible'],'gap_proxy_same_id':gaps['counts'].get('same_id',0),'gap_proxy_wrong_reuse':gaps['counts'].get('wrong_reuse',0)};table.append(row)
    dest=run.path/a/name;write_gz(dest/'tracks.json.gz',out);save(dest/'events.json',eventsets[a]);save(dest/'metric_details.json',rs);save(dest/'gap_proxy.json',gaps);save(dest/'physical_partial.json',physical)
    print('REAL_RESULT',row,flush=True)
   print('REAL_SEQUENCE_DONE',name,round(time.perf_counter()-start,2),flush=True)
  groups={'fit':[s['name'] for s in seqs if s['decision_split']=='fit'],'calibration':[s['name'] for s in seqs if s['decision_split']=='calibration'],'heldout11':[s['name'] for s in seqs if s['decision_split'] not in ['fit','calibration']]}
  assert len(groups['heldout11'])==11,groups
  combined={}
  for split,names in groups.items():
   combined[split]={a:compact({k:m.combine_sequences({s:raw[a][s][k] for s in names}) for k,m in metrics.items()}) for a in ARMS}
  names=groups['heldout11'];h=combined['heldout11'];rng=np.random.default_rng(20261019);samples=rng.integers(0,len(names),size=(2000,len(names)));boot={}
  for a in ARMS:
   ids=np.asarray([[raw[a][s]['Identity'][k] for k in ['IDTP','IDFP','IDFN']] for s in names]);total=ids[samples].sum(1);boot[a]=200*total[:,0]/np.maximum(1,2*total[:,0]+total[:,1]+total[:,2])
  comparisons={}
  for ref in ['N','R','C','GEOM','NOW','NOCTX','SHUFFLE']:
   comparisons[ref]={'delta_IDF1':h['FULL']['IDF1']-h[ref]['IDF1'],'paired_video_bootstrap_95':np.percentile(boot['FULL']-boot[ref],[2.5,97.5]).tolist(),'videos':len(names),'resamples':2000,'scope':'descriptive, historical nonblind small-sample'}
  pervideo={s:next(r['IDF1'] for r in table if r['sequence']==s and r['arm']=='FULL')-next(r['IDF1'] for r in table if r['sequence']==s and r['arm']=='C') for s in names}
  physical_summary={a:{'events':len(ev),'strict_success':sum(e['strict_3pre_3post_same_id'] for e in ev),'post_detected':sum(e['post_detected'] for e in ev),'post_labels':sum(e['post_labels'] for e in ev),'same_id_post_observations':sum(e['same_id_post_observations'] for e in ev),'verified_complete_occlusion_events':0} for a,ev in allphysical.items()}
  gates={'synthetic_macro_F1':training['FULL']['evaluation']['synthetic_test']['macro_F1']>=.65,'extra_C_magnitude':comparisons['C']['delta_IDF1']>=.5,'extra_C_CI':comparisons['C']['paired_video_bootstrap_95'][0]>0,'above_reference':comparisons['R']['paired_video_bootstrap_95'][0]>0,'per_video_guard':min(pervideo.values())>=-1,'HOTA_guard':h['FULL']['HOTA']-h['C']['HOTA']>=-.5}
  mechanism={'history_increment':comparisons['NOW'],'context_increment':comparisons['NOCTX'],'object_correspondence_increment':comparisons['SHUFFLE'],'simple_geometry_increment':comparisons['GEOM']}
  decision={'gates':gates,'scientific_status':'supported_in_scoped_pilot' if all(gates.values()) else 'utility_not_established','mechanism':mechanism,'next_action':'independent physical-occlusion confirmation' if all(gates.values()) else 'stop this fixed weighting configuration; prioritize directly supervised cross-gap identity discriminability (or domain gap if synthetic capacity fails)','started_next_experiment':False,'complete_physical_occlusion_claim':False}
  save(run.path/'SUMMARY.json',{'by_sequence':table,'combined':combined,'groups':groups,'comparisons':comparisons,'per_video_FULL_minus_C_IDF1':pervideo,'physical_partial':physical_summary,'engineering_checks':checks,'real_visibility_prediction_files':predcatalog});save(run.path/'DECISION.json',decision);save(run.path/'RAW_METRICS.json',raw);run.meta['scientific_status']=decision['scientific_status'];print('EVALUATION_DONE',h,comparisons,decision,flush=True)

if __name__=='__main__':main()
