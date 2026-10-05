"""One prespecified four-cell pilot, no validation feedback or tuning loops."""
import argparse
import csv
import json
import time
from pathlib import Path
from common import setup,atomic,progress


def main():
    ap=argparse.ArgumentParser()
    for name in ['protocol','output','features']:ap.add_argument('--'+name,required=True)
    args=ap.parse_args();p,out=setup(args.protocol,args.output)
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from portable_risk import PortableRisk
    from calibrator import export_hgb
    d=np.load(Path(args.features)/'features.npz');rows=json.loads((Path(args.features)/'instances.json').read_text())
    split=json.loads(Path(p['split']).read_text());fit=np.isin(d['keys'][:,0],split['fit_ids']);sel=np.isin(d['keys'][:,0],split['selection_ids'])
    assert len(split['fit_ids'])==1500 and len(split['selection_ids'])==500 and np.all(fit^sel)
    base=np.asarray([r['baseline_iou'] for r in rows]);target=np.asarray([r['target'] for r in rows])
    success=base>=.75;eligible=d['eligible'];y=d['labels'][:,:,0];k=y.shape[1]
    def metrics(chosen):
        effective=np.where(chosen>=0,d['labels'][np.arange(len(rows)),np.maximum(chosen,0),0],0.)
        cost=np.where(chosen>=0,d['labels'][np.arange(len(rows)),np.maximum(chosen,0),1],0.)
        after=base+effective;repair=(base<.75)&(after>=.75);damage=success&(after<.75)
        return dict(matched=int(sel.sum()),targets=int((sel&target).sum()),successes=int((sel&success).sum()),
            target_repairs=int((sel&target&repair).sum()),all_repairs=int((sel&repair).sum()),damages=int((sel&damage).sum()),
            net_repairs=int((sel&repair).sum()-(sel&damage).sum()),iou_gain_sum=float(effective[sel].sum()),
            mean_iou_gain_pp=float(100*effective[sel].mean()),target_mean_iou_gain_pp=float(100*effective[sel&target].mean()),
            target_mean_coverage_loss_pp=float(100*cost[sel&target].mean()))
    rcmc=PortableRisk(p['frozen_rcmc']);rcmc_score=rcmc.predict(d['shape'][:,0])
    chosen_rcmc=np.where(eligible[:,0]&(rcmc_score>0),0,-1)
    reference=metrics(chosen_rcmc);budget=reference['damages']
    assert reference['target_repairs']==52 and budget==47
    outcomes={'frozen_rcmc':reference};policies={};models={};choices={'frozen_rcmc':chosen_rcmc};search=[]
    for name in p['variants']:
        multi=name.startswith('multi');spatial=name.endswith('local')
        xx=d['shape']
        if multi:xx=np.concatenate([xx,d['tau'][:,:,None]],axis=2)
        if spatial:xx=np.concatenate([xx,d['local']],axis=2)
        trainx=xx[fit].reshape(-1,xx.shape[-1]) if multi else xx[fit,0]
        trainy=y[fit].ravel() if multi else y[fit,0]
        model=HistGradientBoostingRegressor(max_iter=p['trees'],**p['estimator'])
        weights=np.full(len(trainy),1/k) if multi else np.ones(len(trainy))
        t=time.perf_counter();model.fit(trainx,trainy,sample_weight=weights);elapsed=time.perf_counter()-t
        models[name]=export_hgb(model,out/(name+'.json'))|{'fit_seconds':elapsed,'features':xx.shape[-1],'action_rows':len(trainy)}
        scores=model.predict(xx.reshape(-1,xx.shape[-1])).reshape(len(rows),k) if multi else model.predict(xx[:,0])[:,None]
        scores=np.where(eligible if multi else eligible[:,:1],scores,-np.inf)
        maximum=scores.max(1);idx=scores.argmax(1)
        candidates=[]
        for cutoff in p['cutoff_grid']:
            chosen=np.where(maximum>cutoff,idx,-1);m=metrics(chosen)
            candidate=dict(variant=name,cutoff=cutoff,feasible=m['damages']<=budget,**m)
            candidates.append(candidate);search.append(candidate)
        feasible=[r for r in candidates if r['feasible']]
        selected=max(feasible,key=lambda r:(r['target_repairs'],-r['damages'],r['all_repairs'],r['iou_gain_sum'],r['cutoff']))
        policies[name]={'cutoff':selected['cutoff']};choices[name]=np.where(maximum>selected['cutoff'],idx,-1)
        outcomes[name]=metrics(choices[name]);progress(out,'fit_selected',variant=name,outcomes=outcomes[name])
        atomic(out/'partial.json',dict(outcomes=outcomes,policies=policies,models=models))
    def pareto(m,r):
        return (m['target_repairs']>=r['target_repairs'] and m['damages']<=r['damages'] and
                (m['target_repairs']>r['target_repairs'] or m['damages']<r['damages']))
    advance={name:all(pareto(outcomes[name],outcomes[ref]) for ref in ['single_shape','frozen_rcmc'])
             for name in p['variants'] if name!='single_shape'}
    # Direct comparison at a common zero cutoff is preserved through the search ledger;
    # selected policy results are development decisions, not confirmatory estimates.
    with (out/'selection_decisions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=None
        for i in np.flatnonzero(sel):
            line={z:rows[i][z] for z in ['image_id','annotation_id','candidate_index','baseline_iou','baseline_coverage','target']}
            for name,chosen in choices.items():
                j=int(chosen[i]);line['action_'+name]=j;line['iou_'+name]=float(base[i]+(y[i,j] if j>=0 else 0))
            if writer is None:writer=csv.DictWriter(f,fieldnames=list(line));writer.writeheader()
            writer.writerow(line)
    # Match the old direct refit at the same seed/settings where possible, without overwriting it.
    oracle={}
    for multi in [False,True]:
        yy=np.where(eligible if multi else eligible[:,:1],y if multi else y[:,:1],-np.inf)
        idx=yy.argmax(1);chosen=np.where(yy.max(1)>0,idx,-1)
        oracle['multi' if multi else 'single']=metrics(chosen)
    result=dict(outcomes=outcomes,policies=policies,models=models,damage_budget=budget,search=search,
        advance=advance,oracle_selection=oracle,fit_images=1500,selection_images=500,fit_rows=int(fit.sum()),
        selection_rows=int(sel.sum()),protocol=p['protocol_id'],scope=p['scope'],
        decision='freeze_promising_policies_for_confirmation' if any(advance.values()) else 'stop_this_pilot_no_retuning',
        limitations=['Selection outcomes choose policies; not independent evidence or AP.',
            'Same total tree/leaf ceiling, different action training rows and descriptors necessarily vary with action set.',
            'No GT features; label/evaluation GT is allowed; YOLO COCO pretraining includes train2017.',
            'Strong oracle headroom does not guarantee these learned features can exploit it.'])
    atomic(out/'SUMMARY.json',result);progress(out,'completed',decision=result['decision'],outcomes=outcomes)


if __name__=='__main__':main()
