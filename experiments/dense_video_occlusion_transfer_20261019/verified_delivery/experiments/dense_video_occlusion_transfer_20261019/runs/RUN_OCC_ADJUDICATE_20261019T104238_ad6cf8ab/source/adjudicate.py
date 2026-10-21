"""Correct supplementary physical-event matching to the original one-to-one contract."""
import pathlib,gzip,json
import numpy as np
from .common import ROOT,Run,read,save,sha
from .evaluate import physical_events,ARMS

def main():
 ep=pathlib.Path(read(ROOT/'evaluate_pointer.json')['path']);original=read(ep/'SUMMARY.json');annotation=read(ROOT/'annotations/occlusion_labels.json');seqs=read(ROOT/'data/sequences.json')['sequences'];names={e['sequence'] for e in annotation['events'] if e['eligible_partial_event']}
 with Run('adjudicate',{'evaluation_run':ep.name,'original_summary_sha256':sha(ep/'SUMMARY.json'),'change':'supplementary matching all-GT Hungarian, same as previous frozen physical-event contract','outputs_changed':False,'primary_metrics_changed':False,'training_steps':0}) as run:
  physical={a:[] for a in ARMS}
  for s in seqs:
   if s['name'] not in names:continue
   p=ROOT/'data'/s['name']/'gt.txt';assert sha(p)==s['gt_sha256'];a=np.loadtxt(p,delimiter=',',ndmin=2);a=a[a[:,6]==1];a[:,4:6]+=a[:,2:4];gt={f:a[a[:,0]==f] for f in range(1,s['frames']+1)}
   for arm in ARMS:
    with gzip.open(ep/arm/s['name']/'tracks.json.gz','rt') as f:out=json.load(f)
    results=physical_events(annotation,s['name'],out,gt);physical[arm]+=results;save(run.path/arm/(s['name']+'_physical.json'),results)
  summary={a:{'events':len(ev),'strict_success':sum(e['strict_3pre_3post_same_id'] for e in ev),'post_detected':sum(e['post_detected'] for e in ev),'post_labels':sum(e['post_labels'] for e in ev),'same_id_post_observations':sum(e['same_id_post_observations'] for e in ev),'wrong_reused_identity_post_observations':sum(e['wrong_reused_identity_post_observations'] for e in ev),'verified_complete_occlusion_events':0} for a,ev in physical.items()}
  final={**original,'physical_partial':summary,'supplementary_correction':{'source_run':run.id,'original_nearest_matching_superseded':True,'primary_metrics_and_decision_unchanged':True}}
  assert {k:v for k,v in final.items() if k not in ['physical_partial','supplementary_correction']}=={k:v for k,v in original.items() if k!='physical_partial'}
  save(run.path/'SUMMARY.json',final);save(run.path/'DECISION.json',read(ep/'DECISION.json'));print('PHYSICAL_FINAL',summary,flush=True)
if __name__=='__main__':main()
