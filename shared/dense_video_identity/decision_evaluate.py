from pathlib import Path
from copy import deepcopy
from collections import defaultdict
import math
import time
import numpy as np
import torch
from .common import Run,read_json,write_json,setup_paths,load_gt,sha256
from .evaluate import official_metrics,metric_data,compact
from .learn import load_cache,load_identity,embed_array
from .reversible_identity import recover,fasttrack
from .existing_baselines import run_sequence as existing
from .decision_model import load_model,predict,ARMS
from .reversible_adjudicate import bootstrap,idf


def publish(base,events,rows,prob,threshold):
    """C proposal ledger is immutable; learned decisions revoke only identities."""
    commits=defaultdict(list)
    for e in events:
        if e['type']=='confirmed':commits[e['frame']].append(e)
    scores=defaultdict(list)
    for r,p in zip(rows,prob):scores[r['frame']].append((r,float(p)))
    aliases={};revoked=set();buffer=[];out=[];audit=[]
    def emit(fr,at):
        used=set()
        for t in sorted(fr['tracks'],key=lambda t:(t['base_id'] in aliases,t['base_id'])):
            b=t['base_id'];identity=aliases.get(b,b)
            if identity in used:identity=b
            assert identity not in used;used.add(identity);t.update(identity=identity,status='reconnected' if identity!=b else 'native',released_at=at)
        out.append(fr)
    for fr in base:
        f=fr['frame']
        for e in commits[f]:aliases[e['base_id']]=e['old_identity']
        for r,p in scores[f]:
            b=r['base_id']
            if b not in revoked and b in aliases and p>=threshold:
                old=aliases.pop(b);revoked.add(b);audit.append(dict(frame=f,base_id=b,old_identity=old,p_error=p,type='learned_rejection',proposal_frame=r['proposal_frame']))
        buffer.append(deepcopy(fr))
        while buffer and buffer[0]['frame']<=f-4:emit(buffer.pop(0),f)
    last=base[-1]['frame'] if base else 0
    for fr in buffer:emit(fr,last)
    for a,b in zip(base,out):
        key=lambda t:(t['base_id'],t['det_id'],tuple(t['box']),t['score'])
        assert sorted(map(key,a['tracks']))==sorted(map(key,b['tracks']))
        assert all(0<=t['released_at']-b['frame']<=4 for t in b['tracks'])
    return out,audit


def seqdata(root,s):
    d=Path(read_json(root/'decision_dataset_pointer.json')['path'])/s['name']
    return read_json(d/'base.json'),read_json(d/'events.json'),read_json(d/'rows.json'),dict(np.load(d/'windows.npz'))


def identity_stat(metric,gt,out):
    v=metric.eval_sequence(metric_data(gt,out))
    return dict(IDF1=float(v['IDF1'])*100,IDTP=int(v['IDTP']),IDFP=int(v['IDFP']),IDFN=int(v['IDFN']))


def calibrate(root):
    setup_paths(root);torch.set_num_threads(4);metrics=official_metrics(root);ptr=read_json(root/'decision_model_pointer.json');manifest=read_json(root/'input_manifest.json');seqs=[s for s in manifest['sequences'] if s['decision_split']=='calibration'];thresholds=[.5,.7,.9,.95,.99]
    with Run(root,'decision_calibration',dict(model_pointer=ptr,thresholds=thresholds,selection='maximum combined IDF1 with per-video C delta >= -1 and aggregate >= C; highest threshold tie',model_protocol_sha=sha256(root/'METHOD_PROTOCOL.md'))) as run:
        table=[];choices={};cached={};checks=[]
        for s in seqs:
            base,ev,rows,a=seqdata(root,s);gt=load_gt(root,s);native,_=publish(base,ev,[],[],math.inf)
            expected=read_json(Path(s['v3_run'])/'confirm'/s['name']/'tracks.json')
            assert native==expected,s['name'];checks.append(dict(sequence=s['name'],disabled_exact_C=True))
            table.append(dict(sequence=s['name'],arm='C',**identity_stat(metrics['Identity'],gt,native)));cached[s['name']]=(base,ev,rows,a,gt)
        for arm in ARMS:
            model=load_model(ptr['models'][arm]['path']);probs={s['name']:predict(model,cached[s['name']][3],arm) for s in seqs}
            eligible=[]
            for threshold in thresholds:
                stats=[]
                for s in seqs:
                    base,ev,rows,a,gt=cached[s['name']];out,audit=publish(base,ev,rows,probs[s['name']],threshold);r=dict(sequence=s['name'],arm=arm,threshold=threshold,rejections=len(audit),**identity_stat(metrics['Identity'],gt,out));table.append(r);stats.append(r)
                c=[r for r in table if r['arm']=='C'];score=idf(stats);delta=score-idf(c);minimum=min(r['IDF1']-next(v['IDF1'] for v in c if v['sequence']==r['sequence']) for r in stats)
                if delta>=-1e-9 and minimum>=-1:eligible.append((score,threshold,delta,minimum))
            best=max(eligible) if eligible else None
            choices[arm]=dict(threshold=best[1] if best else None,IDF1=best[0] if best else idf([r for r in table if r['arm']=='C']),delta_C=best[2] if best else 0,status='calibrated' if best else 'no_usable_policy_disable_rejection')
        write_json(run.path/'SUMMARY.json',dict(table=table,choices=choices,checks=checks));write_json(root/'decision_calibration_pointer.json',dict(run_id=run.id,path=str(run.path),choices=choices,models=ptr['models']));print('CALIBRATION',choices,flush=True)


def evaluate(root):
    setup_paths(root);torch.set_num_threads(4);metrics=official_metrics(root);manifest=read_json(root/'input_manifest.json');cfg=read_json(root/'config.json');ptr=read_json(root/'decision_calibration_pointer.json');datasetptr=read_json(root/'decision_dataset_pointer.json');identity,ck=load_identity(root)
    models={k:load_model(v['path']) for k,v in ptr['models'].items()};old=['bytetrack','immediate','confirm','reversible','botsort','deepocsort','fasttrack'];new=['full','self','now','full_drop_comp','full_drop_time'];allarms=old+new;table=[];raw=defaultdict(lambda:defaultdict(dict));checks=[]
    with Run(root,'decision_full_evaluation',dict(arms=allarms,models=ptr,protocol_sha=sha256(root/'METHOD_PROTOCOL.md'),new_training_steps=0,GT='only official evaluator and posthoc attribution',latency_frames=4)) as run:
        for s in manifest['sequences']:
            if s['decision_split']=='fit':continue
            base,ev,rows,arrays=seqdata(root,s);rec,x=load_cache(root,s);z=embed_array(identity,ck,x);gt=load_gt(root,s);outputs={};eventsets={}
            for arm in old:
                if s.get('v3_run'):
                    path=Path(s['v3_run'])/arm/s['name'];out=read_json(path/'tracks.json');events=read_json(path/'events.json')
                elif arm=='bytetrack':
                    out=deepcopy(base);events=[]
                    for fr in out:
                        for t in fr['tracks']:t['released_at']=min(fr['frame']+4,len(rec))
                elif arm in ['immediate','confirm','reversible']:out,events=recover(base,rec,z,arm)
                elif arm=='fasttrack':out=fasttrack(rec,z,cfg,root);events=[]
                else:out=existing(arm,rec,z,cfg);events=[]
                outputs[arm]=out;eventsets[arm]=events
            parity,_=publish(base,ev,[],[],math.inf);assert parity==outputs['confirm'],s['name']
            for arm in new:
                key='full' if arm.startswith('full_drop') else arm;mode='self' if arm=='full_drop_comp' else 'now' if arm=='full_drop_time' else arm
                prob=predict(models[key],arrays,mode);threshold=ptr['choices'][key]['threshold'];threshold=math.inf if threshold is None else threshold
                out,events=publish(base,ev,rows,prob,threshold);outputs[arm]=out;eventsets[arm]=events
                d=run.path/arm/s['name'];d.mkdir(parents=True);np.save(d/'prob_error.npy',prob)
                cut=max(1,len(base)//2);indices=[i for i,r in enumerate(rows) if r['frame']<=cut];pout,_=publish(base[:cut],[e for e in ev if e['frame']<=cut],[rows[i] for i in indices],prob[indices],threshold)
                assert pout[:max(0,cut-4)]==out[:max(0,cut-4)],(s['name'],arm)
            for arm,out in outputs.items():
                dest=run.path/arm/s['name'];dest.mkdir(parents=True,exist_ok=True);write_json(dest/'tracks.json',out);write_json(dest/'events.json',eventsets[arm])
                rs={k:m.eval_sequence(metric_data(gt,out)) for k,m in metrics.items()};raw[s['decision_split']][arm][s['name']]=rs
                stats=dict(sequence=s['name'],split=s['decision_split'],arm=arm,frames=len(rec),rejections=sum(e['type']=='learned_rejection' for e in eventsets[arm]),**compact(rs));table.append(stats);write_json(dest/'metric_details.json',rs);print('EVAL',s['name'],arm,stats['IDF1'],flush=True)
            checks.append(dict(sequence=s['name'],geometry_count_preserved=True,disabled_exact_C=True,prefix_causal=True))
        combined={split:{arm:compact({k:m.combine_sequences({seq:rr[k] for seq,rr in data.items()}) for k,m in metrics.items()}) for arm,data in arms.items()} for split,arms in raw.items()}
        comparisons={split:{b:bootstrap([r for r in table if r['split']==split],'full',b) for b in ['bytetrack','botsort','confirm','self','now','full_drop_comp','full_drop_time']} for split in combined}
        cf=combined['confirmation'];cc=comparisons['confirmation'];names=[s['name'] for s in manifest['sequences'] if s['decision_split']=='confirmation'];idx={(r['sequence'],r['arm']):r for r in table};d=[idx[n,'full']['IDF1']-idx[n,'bytetrack']['IDF1'] for n in names]
        ho=comparisons['seen_holdout']['bytetrack'];hold=[r for r in table if r['split']=='seen_holdout' and r['arm']=='full'];hmin=min(r['IDF1']-idx[r['sequence'],'bytetrack']['IDF1'] for r in hold)
        gates=dict(native=cc['bytetrack']['delta_IDF1']>=1 and cc['bytetrack']['ci95'][0]>0,existing=cc['botsort']['delta_IDF1']>=.5 and cc['botsort']['ci95'][0]>0,increment=cc['confirm']['delta_IDF1']>=.5 and cc['confirm']['ci95'][0]>0,consistency=sum(v>0 for v in d)>=3 and min(d)>=-1,hota=cf['full']['HOTA']>=cf['bytetrack']['HOTA']-.5,seen_holdout=ho['delta_IDF1']>=0 and hmin>=-1)
        mechanism=all(cc[b]['delta_IDF1']>=.2 for b in ['self','now']) and all(cc[b]['delta_IDF1']>0 for b in ['full_drop_comp','full_drop_time'])
        decision=dict(status='limited_utility_supported' if all(gates.values()) else 'utility_not_supported_at_frozen_gate',gates=gates,mechanism_supported=mechanism,confirm_video_deltas=d,seen_holdout_minimum_delta=hmin,scope='four new duck sequences, one seed, not complete occlusion proof',next_action='independent scene replication' if all(gates.values()) else 'close this model configuration; adjudicate transfer and decision damage using fixed outputs')
        write_json(run.path/'SUMMARY.json',dict(by_sequence=table,combined=combined,comparisons=comparisons,checks=checks));write_json(run.path/'DECISION.json',decision);write_json(root/'decision_evaluation_pointer.json',dict(run_id=run.id,path=str(run.path)));print('DECISION',decision,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('mode',choices=['calibrate','evaluate']);a=p.parse_args();(calibrate if a.mode=='calibrate' else evaluate)(a.root)
