"""Paired incremental effect; original S2 Gates for each arm remain authoritative."""
import argparse,json
from pathlib import Path
from verify_evaluation import read,jl,sha,statistics,STUDY


def compare(r,rep):
    contracts=[read(p/'EVAL_CONTRACT.json') for p in (r,rep)]
    assert [v['experiment_arm'] for v in contracts]==['R','REP']
    train=[STUDY/'runs'/v['training_run'] for v in contracts]
    c=[read(p/'EVAL_CONTRACT.json') for p in train]
    permitted={'experiment_arm','repulsion_weight','changes'}
    differences={k for k in c[0].keys()|c[1].keys() if c[0].get(k)!=c[1].get(k)}
    assert differences<=permitted and c[0]['repulsion_weight']==0 and c[1]['repulsion_weight']==.1,differences
    for name in ('IMAGE_ORDERS.json','TRAINING_JOIN.json'):assert sha(train[0]/name)==sha(train[1]/name)
    assert c[0]['model_state_initial_sha256']==c[1]['model_state_initial_sha256']
    rows=[list(jl(p/'FIXED_PAIR_EFFECTS.jsonl')) for p in (r,rep)]
    keyed=[{(v['image_id'],v['annotation_id']):v for v in rr} for rr in rows]
    assert keyed[0].keys()==keyed[1].keys() and len(rows[0])==2925
    delta=[]
    for key,a in keyed[0].items():
        b=keyed[1][key]
        for k in ('population','baseline_output_detection_id','baseline_iou'):assert a[k]==b[k]
        delta.append(dict(a,delta_iou=b['delta_iou']-a['delta_iou']))
    target=statistics(delta,True);overall=statistics(delta,False)
    summaries=[read(p/'SUMMARY.json') for p in (r,rep)];passed=[all(v['gates'].values()) for v in summaries]
    incremental=target['ci95'][0]>0 and passed[1]
    selected='REP' if passed[1] and (incremental or not passed[0]) else 'R' if passed[0] else None
    return dict(status='passed',R_run=r.name,REP_run=rep.name,changed_training_contract_fields=sorted(differences),
        REP_minus_R_target=target,REP_minus_R_all_bbox75=overall,REP_incremental_benefit_supported=incremental,
        R_absolute_gates=summaries[0]['gates'],REP_absolute_gates=summaries[1]['gates'],selected_candidate=selected,
        scope='exposed development data; causal comparison of the fixed .1 repulsion term at the same aligned GT supervision')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--r',type=Path,required=True);p.add_argument('--rep',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    value=compare(a.r,a.rep)
    with a.out.open('x',encoding='utf8') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(value),flush=True)
