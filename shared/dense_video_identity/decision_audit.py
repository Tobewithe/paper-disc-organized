"""Interpret frozen outputs; never picks another model, threshold, or test subset."""
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from .common import Run,read_json,write_json,sha256
from .evidence_probe import auc

def audit(root):
    ep=read_json(root/'decision_evaluation_pointer.json');runpath=Path(ep['path']);dp=Path(read_json(root/'decision_dataset_pointer.json')['path']);manifest=read_json(root/'input_manifest.json');summary=read_json(runpath/'SUMMARY.json');table=[];cases=[]
    with Run(root,'decision_frozen_output_audit',dict(evaluation=ep,operation='posthoc outcome attribution and event-balanced score AUC',new_training_steps=0,no_threshold_changes=True)) as run:
        for s in manifest['sequences']:
            if s['decision_split']=='fit':continue
            rows=read_json(dp/s['name']/'rows.json');idx={(r['frame'],r['base_id']):r for r in rows};known=[i for i,r in enumerate(rows) if r['y']>=0];counts=Counter(rows[i]['event'] for i in known);w=[1/counts[rows[i]['event']] for i in known]
            for arm in ['full','self','now','full_drop_comp','full_drop_time']:
                p=np.load(runpath/arm/s['name']/'prob_error.npy');ev=read_json(runpath/arm/s['name']/'events.json');c=Counter()
                for e in ev:
                    r=idx[e['frame'],e['base_id']];relation='unknown' if r['y']==-1 else 'wrong_connection' if r['y']==1 else 'correct_connection';c[relation]+=1
                    if arm=='full':cases.append(dict(e,sequence=s['name'],split=s['decision_split'],relation=relation,anchor_gt=r['anchor_gt'],current_gt=r['current_gt'],illustration_only=True))
                table.append(dict(sequence=s['name'],split=s['decision_split'],arm=arm,known_observations=len(known),unknown_observations=len(rows)-len(known),known_events=len(counts),auc=auc([rows[i]['y'] for i in known],p[known],w),rejections=dict(c),maximum_p=float(p.max()) if len(p) else None))
        pooled={}
        for split in ['calibration','seen_holdout','confirmation']:
            pooled[split]={}
            for arm in ['full','self','now']:
                ys=[];ps=[];weights=[];count=Counter()
                for s in manifest['sequences']:
                    if s['decision_split']!=split:continue
                    rows=read_json(dp/s['name']/'rows.json');p=np.load(runpath/arm/s['name']/'prob_error.npy');known=[i for i,r in enumerate(rows) if r['y']>=0];counts=Counter(rows[i]['event'] for i in known)
                    ys += [rows[i]['y'] for i in known];ps.extend(p[known]);weights += [1/counts[rows[i]['event']] for i in known]
                    for r in table:
                        if r['sequence']==s['name'] and r['arm']==arm:count.update(r['rejections'])
                pooled[split][arm]=dict(auc=auc(ys,ps,weights),known_observations=len(ys),rejections=dict(count))
        write_json(run.path/'SUMMARY.json',dict(by_sequence=table,pooled=pooled,cases=cases));write_json(root/'decision_audit_pointer.json',dict(run_id=run.id,path=str(run.path)));print('AUDIT',pooled,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();audit(a.root)
