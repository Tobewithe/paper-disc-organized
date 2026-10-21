"""Frozen S0: causal competitor evidence, posthoc labels, video-cluster inference."""
from pathlib import Path
from collections import defaultdict, Counter
import copy
import platform
import sys
import numpy as np
from .common import Run, read_json, write_json, sha256, setup_paths, load_gt, unit
from .learn import load_cache, load_identity, embed_array, match_gt
from .reversible_identity import distance


def register(root):
    parent=Path(r'D:\thesis-video-v3-20261019')
    manifest=copy.deepcopy(read_json(parent/'input_manifest.json'))
    manifest['sequences']=[s for s in manifest['sequences'] if s.get('gt')]
    for s in manifest['sequences']:
        pointer=read_json(parent/f"reversible_{s['split']}_pointer.json")
        s['v3_run']=str(parent/pointer['path'])
        s['previous_split']=s['split'];s['split']='fit_source' if s['split']=='engineering' else 'seen_development'
    write_json(root/'input_manifest.json',manifest)
    write_json(root/'config.json',read_json(parent/'config.json'))
    write_json(root/'identity_pointer.json',read_json(parent/'identity_pointer.json'))
    with Run(root,'evidence_registration',dict(parent=str(parent),sequences=[s['name'] for s in manifest['sequences']],models_frozen=True)) as run:
        hashes=[]
        for s in manifest['sequences']:
            for field,expected in [('video',s['video_sha256']),('gt',s['gt_sha256'])]:
                p=Path(s[field]);actual=sha256(p);assert actual==expected,(s['name'],field)
                hashes.append(dict(sequence=s['name'],role=field,path=str(p),sha256=actual,bytes=p.stat().st_size))
            for p in [Path(s['cache_path'])/'features.npz',Path(s['cache_path'])/'detections.json',Path(s['v3_run'])/'bytetrack'/s['name']/'tracks.json',Path(s['v3_run'])/'confirm'/s['name']/'events.json']:
                hashes.append(dict(sequence=s['name'],role='frozen_input',path=str(p),sha256=sha256(p),bytes=p.stat().st_size))
        for key,p in manifest['models'].items():assert sha256(p)==manifest['model_sha256'][key]
        pointer=read_json(root/'identity_pointer.json');assert sha256(pointer['checkpoint'])==pointer['sha256']
        write_json(root/'INPUT_HASHES.json',hashes);write_json(run.path/'INPUT_HASHES.json',hashes)
        write_json(run.path/'SUMMARY.json',dict(sequences=len(manifest['sequences']),hashes_checked=len(hashes),identity_sha256=pointer['sha256'],new_training_steps=0))
        write_json(root/'ENVIRONMENT.json',dict(python=sys.version,host=platform.node(),executor=sys.executable))


def causal_features(base,records,z,events):
    """No GT or labels accepted here. Fixed anchors created from prefix only."""
    props={e['base_id']:e for e in events if e['type']=='proposed'}
    confs={e['base_id']:e for e in events if e['type']=='confirmed'}
    start=defaultdict(list)
    for b,e in confs.items():start[props[b]['frame']].append(b)
    state={};active={};rows=[]
    for fr,rec in zip(base,records):
        f=fr['frame'];obs={t['base_id']:t for t in fr['tracks']}
        feat={b:z[rec['feature_start']+t['det_id']] for b,t in obs.items()}
        for b in start[f]:
            p=props[b];old=p['old_base'];s=state[old]
            candidates=[]
            for alt,v in state.items():
                if alt in {old,b} or v['count']<3 or f-v['last']>60 or distance(obs[b]['box'],v['box'])>2:continue
                a=unit(np.mean(v['features'],axis=0));candidates.append((float(feat[b]@a),alt,a))
            candidates.sort(key=lambda t:(-t[0],t[1]))
            active[b]=dict(anchor=unit(np.mean(s['features'],axis=0)),old_base=old,proposal=f,confirmed=confs[b]['frame'],others=[a for _,_,a in candidates[:5]],trace=[],seen=0)
        for b,a in list(active.items()):
            if f-a['confirmed']>60:del active[b];continue
            if b not in obs or f<a['confirmed']:continue
            own=float(feat[b]@a['anchor'])
            visible=max((float(v@a['anchor']) for k,v in feat.items() if k!=b),default=-1.)
            historical=max((float(feat[b]@v) for v in a['others']),default=-1.)
            comp=max(visible,historical)-own
            a['trace'].append(comp);a['trace']=a['trace'][-5:];a['seen']+=1
            if a['seen']%3:continue
            rows.append(dict(frame=f,base_id=b,det_id=obs[b]['det_id'],old_base=a['old_base'],proposal_frame=a['proposal'],confirmed_frame=a['confirmed'],event=f"{b}:{a['proposal']}",S=1-own,C=comp,T=float(np.mean(a['trace'])),self_similarity=own,visible_competitor=visible,historical_competitor=historical,window_size=len(a['trace'])))
        for b,t in obs.items():
            s=state.setdefault(b,dict(count=0,features=[]));s.update(count=s['count']+1,last=f,box=t['box']);s['features'].append(feat[b].copy());s['features']=s['features'][-5:]
    return rows


def label_rows(rows,base,mapping):
    hist=defaultdict(list)
    for fr in base:
        for t in fr['tracks']:
            g=mapping[fr['frame']].get(t['det_id'])
            if g is not None:hist[t['base_id']].append((fr['frame'],int(g)))
    for row in rows:
        old=[g for f,g in hist[row['old_base']] if f<row['proposal_frame']][-3:]
        anchor=old[-1] if len(old)==3 and len(set(old))==1 else None
        current=mapping[row['frame']].get(row['det_id'])
        row.update(anchor_gt=anchor,current_gt=current,y=None if anchor is None or current is None else int(anchor!=current))
    return rows


def auc(y,score,w):
    y,score,w=np.asarray(y),np.asarray(score),np.asarray(w)
    if not np.any(y==0) or not np.any(y==1):return None
    idx=np.argsort(score,kind='stable');y=y[idx];score=score[idx];w=w[idx]
    acc=0.;neg=0.;i=0
    while i<len(y):
        j=i+1
        while j<len(y) and score[j]==score[i]:j+=1
        ww=w[i:j];yy=y[i:j];n=float(ww[yy==0].sum());p=float(ww[yy==1].sum());acc+=p*(neg+.5*n);neg+=n;i=j
    return acc/(w[y==0].sum()*w[y==1].sum())


def statistics(rows):
    known=[r for r in rows if r['y'] is not None];counts=Counter((r['sequence'],r['event']) for r in known)
    weights=[1/counts[r['sequence'],r['event']] for r in known]
    out=dict(observations=len(rows),known=len(known),unknown=len(rows)-len(known),events=len({(r['sequence'],r['event']) for r in rows}),correct_events=len({(r['sequence'],r['event']) for r in known if r['y']==0}),wrong_events=len({(r['sequence'],r['event']) for r in known if r['y']==1}),videos=len({r['sequence'] for r in known}),scores={})
    for k in ['S','C','T']:out['scores'][k]=auc([r['y'] for r in known],[r[k] for r in known],weights)
    return out


def probe(root):
    setup_paths(root);model,ck=load_identity(root);manifest=read_json(root/'input_manifest.json')
    with Run(root,'competitor_temporal_probe',dict(protocol_sha=sha256(root/'PROTOCOL.md'),inputs_sha=sha256(root/'INPUT_HASHES.json'),new_training_steps=0,GT='labels after feature construction',scores=['S','C','T'])) as run:
        rows=[];checks=[]
        for s in manifest['sequences']:
            rec,x=load_cache(root,s);z=embed_array(model,ck,x)
            base=read_json(Path(s['v3_run'])/'bytetrack'/s['name']/'tracks.json');events=read_json(Path(s['v3_run'])/'confirm'/s['name']/'events.json')
            rr=causal_features(base,rec,z,events)
            cut=max(1,len(base)//2);prefix_events=[e for e in events if e['frame']<=cut]
            pp=causal_features(base[:cut],rec[:cut],z,prefix_events)
            # Full log may contain proposals only confirmed after the cut; compare common confirmed events.
            common={e['base_id'] for e in prefix_events if e['type']=='confirmed'}
            assert pp==[r for r in rr if r['frame']<=cut and r['base_id'] in common]
            label_rows(rr,base,match_gt(rec,load_gt(root,s)))
            for r in rr:r.update(sequence=s['name'],split=s['split'])
            rows+=rr;checks.append(dict(sequence=s['name'],prefix_causal=True));print('PROBE_SEQUENCE',s['name'],statistics(rr),flush=True)
        by_sequence={s['name']:statistics([r for r in rows if r['sequence']==s['name']]) for s in manifest['sequences']}
        by_split={split:statistics([r for r in rows if r['split']==split]) for split in ['fit_source','seen_development']}
        dev=[r for r in rows if r['split']=='seen_development'];names=sorted({r['sequence'] for r in dev});rng=np.random.default_rng(20261019);deltas=defaultdict(list)
        for _ in range(2000):
            sample=[]
            for i,n in enumerate(rng.choice(names,len(names))):sample += [dict(r,sequence=f'{i}:{n}') for r in dev if r['sequence']==n]
            scores=statistics(sample)['scores']
            if all(v is not None for v in scores.values()):
                for a,b in [('C','S'),('T','S'),('T','C')]:deltas[a+'-'+b].append(scores[a]-scores[b])
        ci={k:np.percentile(v,[2.5,97.5]).tolist() for k,v in deltas.items() if v}
        v=by_split['seen_development'];sc=v['scores'];adequate=v['correct_events']>=10 and v['wrong_events']>=10 and v['videos']>=3
        supported=[k for k in ['C','T'] if sc[k] is not None and sc['S'] is not None and sc[k]>=.60 and sc[k]-sc['S']>=.03]
        decision=dict(status='eligible_for_frozen_learning_pilot' if adequate and supported else 'information_insufficient' if not adequate else 'information_gate_failed',sample_adequate=adequate,supported_scores=supported,time_increment_supported=sc['T'] is not None and sc['C'] is not None and sc['T']-sc['C']>=.02,ci95=ci,scope='seen-data information screening, not tracking utility or novelty',next_action='freeze matched learning pilot' if adequate and supported else 'prioritize discriminative identity representation data')
        write_json(run.path/'evidence_rows.json',rows);write_json(run.path/'SUMMARY.json',dict(by_split=by_split,by_sequence=by_sequence,ci95=ci,causality_checks=checks));write_json(run.path/'DECISION.json',decision)
        write_json(root/'evidence_probe_pointer.json',dict(run_id=run.id,path=str(run.path)));print('PROBE_DECISION',decision,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('mode',choices=['register','probe']);a=p.parse_args();(register if a.mode=='register' else probe)(a.root)
