"""CETD inputs are causal; GT labeling is a separate, explicit offline step."""
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from .common import Run, read_json, write_json, setup_paths, unit, load_gt, sha256
from .learn import load_cache, load_identity, embed_array, match_gt
from .reversible_identity import distance, recover
from .tracking import run_sequence


def windows(base,records,z,events):
    props={e['base_id']:e for e in events if e['type']=='proposed'}
    confs={e['base_id']:e for e in events if e['type']=='confirmed'}
    start=defaultdict(list)
    for b in confs:start[props[b]['frame']].append(b)
    state={};active={};rows=[];owns=[];rivals=[];masks=[]
    for fr,rec in zip(base,records):
        f=fr['frame'];obs={t['base_id']:t for t in fr['tracks']};feat={b:z[rec['feature_start']+t['det_id']] for b,t in obs.items()}
        for b in start[f]:
            p=props[b];old=p['old_base'];s=state[old];other=[]
            for alt,v in state.items():
                if alt in {old,b} or v['count']<3 or f-v['last']>60 or distance(obs[b]['box'],v['box'])>2:continue
                a=unit(np.mean(v['features'],axis=0));other.append((float(feat[b]@a),alt,a,v['box'],v['score']))
            other.sort(key=lambda x:(-x[0],x[1]))
            active[b]=dict(anchor=unit(np.mean(s['features'],axis=0)),box=s['box'],old_base=old,proposal=f,confirmed=confs[b]['frame'],others=other[:5],history=[],prev=None)
        for b,a in list(active.items()):
            if f-a['confirmed']>60:del active[b];continue
            if b not in obs or f<a['confirmed']:continue
            t=obs[b];q=feat[b];sim=float(q@a['anchor']);quality=rec['quality'][t['det_id']]
            own=np.array([sim,distance(t['box'],a['box']),min((f-a['proposal'])/60,1),t['score'],quality[1],quality[2],quality[3],float(q@a['prev']) if a['prev'] is not None else 1],np.float32)
            comp=np.zeros((6,7),np.float32);mask=np.zeros(6,np.float32)
            current=sorted([(float(v@a['anchor']),k,v) for k,v in feat.items() if k!=b],key=lambda x:(-x[0],x[1]))[:3]
            historical=sorted([(float(q@v),k,v,box,score) for _,k,v,box,score in a['others']],key=lambda x:(-x[0],x[1]))[:3]
            for j,(s,k,v) in enumerate(current):
                comp[j]=[s,float(q@v),s-sim,distance(t['box'],obs[k]['box']),obs[k]['score'],float(k==a['old_base']),0];mask[j]=1
            for j,(s,k,v,box,score) in enumerate(historical,3):
                comp[j]=[float(v@a['anchor']),s,s-sim,distance(t['box'],box),score,0,1];mask[j]=1
            own=np.clip(own,-4,4);comp=np.clip(comp,-4,4)
            a['history'].append((own,comp,mask));a['history']=a['history'][-5:];a['prev']=q.copy()
            h=[a['history'][0]]*(5-len(a['history']))+a['history']
            owns.append(np.stack([v[0] for v in h]));rivals.append(np.stack([v[1] for v in h]));masks.append(np.stack([v[2] for v in h]))
            rows.append(dict(frame=f,base_id=b,det_id=t['det_id'],old_base=a['old_base'],proposal_frame=a['proposal'],confirmed_frame=a['confirmed'],event=f"{b}:{a['proposal']}"))
        for b,t in obs.items():
            s=state.setdefault(b,dict(count=0,features=[]));s.update(count=s['count']+1,last=f,box=t['box'],score=t['score']);s['features'].append(feat[b].copy());s['features']=s['features'][-5:]
    return rows,dict(own=np.asarray(owns,np.float32).reshape(-1,5,8),rivals=np.asarray(rivals,np.float32).reshape(-1,5,6,7),mask=np.asarray(masks,np.float32).reshape(-1,5,6))


def label(rows,base,mapping):
    hist=defaultdict(list)
    for fr in base:
        for t in fr['tracks']:
            g=mapping[fr['frame']].get(t['det_id'])
            if g is not None:hist[t['base_id']].append((fr['frame'],int(g)))
    for r in rows:
        old=[g for f,g in hist[r['old_base']] if f<r['proposal_frame']][-3:];g=mapping[r['frame']].get(r['det_id']);anchor=old[-1] if len(old)==3 and len(set(old))==1 else None
        r.update(y=-1 if anchor is None or g is None else int(anchor!=g),anchor_gt=anchor,current_gt=g)


def prepare(root):
    setup_paths(root);model,ck=load_identity(root);cfg=read_json(root/'config.json');manifest=read_json(root/'input_manifest.json')
    with Run(root,'decision_dataset',dict(protocol_sha=sha256(root/'METHOD_PROTOCOL.md'),sequences=[s['name'] for s in manifest['sequences']],feature_shape=[5,8,6,7],GT='offline strict labels only')) as run:
        stats=[]
        for s in manifest['sequences']:
            rec,x=load_cache(root,s);z=embed_array(model,ck,x)
            if s.get('v3_run'):
                base=read_json(Path(s['v3_run'])/'bytetrack'/s['name']/'tracks.json');events=read_json(Path(s['v3_run'])/'confirm'/s['name']/'events.json')
            else:
                base,_=run_sequence('bytetrack',rec,z,cfg);_,events=recover(base,rec,z,'confirm')
            rows,arrays=windows(base,rec,z,events)
            cut=max(1,len(base)//2);ev=[e for e in events if e['frame']<=cut];pr,pa=windows(base[:cut],rec[:cut],z,ev);common={e['base_id'] for e in ev if e['type']=='confirmed'};indices=[i for i,r in enumerate(rows) if r['frame']<=cut and r['base_id'] in common]
            assert pr==[rows[i] for i in indices]
            for k,v in arrays.items():assert np.array_equal(pa[k],v[indices])
            label(rows,base,match_gt(rec,load_gt(root,s)))
            for r in rows:r.update(sequence=s['name'],split=s['decision_split'])
            dest=run.path/s['name'];dest.mkdir();np.savez_compressed(dest/'windows.npz',**arrays);write_json(dest/'rows.json',rows);write_json(dest/'base.json',base);write_json(dest/'events.json',events)
            counts=Counter(r['y'] for r in rows);stats.append(dict(sequence=s['name'],split=s['decision_split'],windows=len(rows),known_correct=counts[0],known_wrong=counts[1],unknown=counts[-1],events=len({r['event'] for r in rows}),causal_prefix=True));print('DATASET',stats[-1],flush=True)
        write_json(run.path/'SUMMARY.json',dict(sequences=stats));write_json(root/'decision_dataset_pointer.json',dict(run_id=run.id,path=str(run.path)))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();prepare(a.root)
