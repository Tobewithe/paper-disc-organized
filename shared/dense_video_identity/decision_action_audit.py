from pathlib import Path
import math
import numpy as np
from .common import Run,read_json,write_json,setup_paths,load_gt,sha256
from .decision_evaluate import seqdata,publish,identity_stat
from .evaluate import official_metrics

def main(root):
    setup_paths(root);metric=official_metrics(root)['Identity'];ep=Path(read_json(root/'decision_evaluation_pointer.json')['path']);manifest=read_json(root/'input_manifest.json');allrows=[]
    with Run(root,'decision_action_alignment_audit',dict(protocol_sha=sha256(root/'ACTION_AUDIT_PROTOCOL.md'),split='calibration',thresholds=[.5,.7,.9,.95],limit=100,new_training_steps=0,operation='one actual calibration-triggered withdrawal at a time; official full-video IDF1')) as run:
        for s in manifest['sequences']:
            if s['decision_split']!='calibration':continue
            base,ev,rows,a=seqdata(root,s);probs=np.load(ep/'full'/s['name']/'prob_error.npy');gt=load_gt(root,s);ref,_=publish(base,ev,[],[],math.inf);refmetric=identity_stat(metric,gt,ref);idx={(r['frame'],r['base_id']):i for i,r in enumerate(rows)};candidates={}
            for threshold in [.5,.7,.9,.95]:
                _,events=publish(base,ev,rows,probs,threshold)
                for e in events:candidates[e['frame'],e['base_id']]=e
            if len(allrows)+len(candidates)>100:raise RuntimeError('Frozen action budget exceeded; no subset selection')
            for key,e in sorted(candidates.items()):
                i=idx[key];r=rows[i];p=np.zeros(len(rows),np.float32);p[i]=1;out,audit=publish(base,ev,rows,p,.5);assert len(audit)==1
                stat=identity_stat(metric,gt,out);allrows.append(dict(sequence=s['name'],**e,y=r['y'],anchor_gt=r['anchor_gt'],current_gt=r['current_gt'],native_C_IDF1=refmetric['IDF1'],counterfactual_IDF1=stat['IDF1'],delta_IDF1=stat['IDF1']-refmetric['IDF1']))
        counts={}
        for y,name in [(-1,'unknown'),(0,'correct_connection'),(1,'wrong_connection')]:
            v=[r['delta_IDF1'] for r in allrows if r['y']==y];counts[name]=dict(actions=len(v),positive=sum(d>1e-9 for d in v),zero=sum(abs(d)<=1e-9 for d in v),negative=sum(d< -1e-9 for d in v))
        c=counts['wrong_connection'];n=c['actions'];fraction=c['positive']/n if n else None
        if n>=3 and fraction>=.75:focus='protect correct links by improving cross-video risk discrimination'
        elif n>=3 and fraction<.5 and c['negative']>0:focus='align supervision with trajectory-level identity decision utility'
        else:focus='collect identifiable action-outcome labels before another model'
        decision=dict(status='posthoc_explanatory',counts=counts,next_priority=focus,no_new_model_or_threshold_selection=True,scope='single withdrawal effects; interactions not enumerated; cannot sum as deployable improvement')
        write_json(run.path/'SUMMARY.json',dict(rows=allrows,counts=counts));write_json(run.path/'DECISION.json',decision);write_json(root/'decision_action_audit_pointer.json',dict(run_id=run.id,path=str(run.path)));print('ACTION_AUDIT',decision,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
