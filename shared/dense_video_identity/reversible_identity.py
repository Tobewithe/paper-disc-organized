"""Causal, bounded-latency identity transactions above immutable motion tracks."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import time
import numpy as np
import yaml
from .common import Run,read_json,write_json,setup_paths,unit,load_gt,sha256
from .learn import load_cache,load_identity,embed_array
from .tracking import run_sequence,frame_input
from .evaluate import official_metrics,metric_data,compact
from .existing_baselines import run_sequence as existing


def distance(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return float(np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)/max(np.linalg.norm(a[2:]-a[:2]),np.linalg.norm(b[2:]-b[:2]),1))


def recover(base,records,z,mode):
    assert mode in {'immediate','confirm','reversible'}
    states={};aliases={};attempted=set();pending={};active={};buffer=[];out=[];events=[]
    def emit(frame,at):
        used=set()
        for t in sorted(frame['tracks'],key=lambda t:(t['base_id'] in aliases,t['base_id'])):
            bid=t['base_id'];identity=aliases.get(bid,bid)
            if identity in used:identity=bid
            assert identity not in used
            used.add(identity);t['identity']=identity;t['status']='reconnected' if identity!=bid else 'native';t['released_at']=at
        out.append(frame)
    for fr,rec in zip(base,records):
        f=fr['frame'];present={t['base_id'] for t in fr['tracks']};obs={t['base_id']:t for t in fr['tracks']}
        feats={b:z[rec['feature_start']+t['det_id']] for b,t in obs.items()}
        # Previously committed identity links remain retractable without touching motion state.
        for bid,a in list(active.items()):
            if f-a['confirmed_at']>60:del active[bid];continue
            collision=any(b!=bid and aliases.get(b,b)==a['identity'] for b in present)
            sim=float(feats[bid]@a['anchor']) if bid in present else None
            a['bad']=a.get('bad',0)+1 if sim is not None and sim<.75 else 0
            if mode=='reversible' and (collision or a['bad']>=2):
                aliases.pop(bid,None);del active[bid]
                events.append(dict(frame=f,type='rollback',base_id=bid,old_identity=a['identity'],reason='identity_owner_reappeared' if collision else 'two_anchor_contradictions',similarity=sim))
        used={aliases.get(b,b) for b in present}
        new=[b for b in present if b not in states and b not in attempted]
        gallery={b:s for b,s in states.items() if b not in present and s['count']>=3 and 1<f-s['last']<=60 and aliases.get(b,b) not in used}
        pair={}
        for b in new:
            for old,s in gallery.items():
                if distance(obs[b]['box'],s['box'])<=2:
                    pair[b,old]=float(feats[b]@unit(np.mean(s['features'],axis=0)))
        proposals=[]
        for b in new:
            scores=sorted([(v,old) for (bb,old),v in pair.items() if bb==b],reverse=True)
            if not scores:continue
            score,old=scores[0];row=scores[1][0] if len(scores)>1 else -1
            col=max([v for (bb,oo),v in pair.items() if oo==old and bb!=b],default=-1)
            if score>=.8 and score-max(row,col)>=.03:proposals.append((score,b,old,score-max(row,col)))
        allocated=set()
        for score,b,old,margin in sorted(proposals,reverse=True):
            identity=aliases.get(old,old)
            if identity in allocated:continue
            allocated.add(identity);attempted.add(b)
            p=dict(old_base=old,identity=identity,anchor=unit(np.mean(gallery[old]['features'],axis=0)),first=f,count=0,sims=[],bad=0)
            pending[b]=p
            events.append(dict(frame=f,type='proposed',base_id=b,old_base=old,old_identity=identity,similarity=score,margin=margin))
        for bid,p in list(pending.items()):
            collision=any(b!=bid and aliases.get(b,b)==p['identity'] for b in present)
            if bid in present:
                sim=float(feats[bid]@p['anchor']);p['sims'].append(sim);p['count']+=1
                p['bad']=p['bad']+1 if sim<.75 else 0
            required=1 if mode=='immediate' else 3
            valid=p['count']>=required and min(p['sims'][-required:])>=.8
            if mode=='reversible' and (collision or p['bad']>=2):
                del pending[bid];events.append(dict(frame=f,type='proposal_rollback',base_id=bid,old_identity=p['identity'],reason='owner_present' if collision else 'anchor_contradiction'));continue
            if valid and not collision:
                aliases[bid]=p['identity'];active[bid]=dict(p,confirmed_at=f);del pending[bid]
                events.append(dict(frame=f,type='confirmed',base_id=bid,old_base=p['old_base'],old_identity=p['identity'],delay=f-p['first'],similarities=p['sims']))
            elif f-p['first']>=4:
                del pending[bid];events.append(dict(frame=f,type='expired',base_id=bid,old_identity=p['identity']))
        # Update raw-track memory only. Recovery never writes a different track's memory.
        for b,t in obs.items():
            if b not in states:states[b]=dict(count=0,features=[])
            s=states[b];s['count']+=1;s['last']=f;s['box']=t['box'];s['features'].append(feats[b].copy());s['features']=s['features'][-5:]
        buffer.append(deepcopy(fr))
        while buffer and buffer[0]['frame']<=f-4:emit(buffer.pop(0),f)
    last=base[-1]['frame'] if base else 0
    for fr in buffer:emit(fr,last)
    assert len(out)==len(base)
    for a,b in zip(base,out):
        key=lambda t:(t['base_id'],t['det_id'],tuple(t['box']),t['score'])
        assert sorted(map(key,a['tracks']))==sorted(map(key,b['tracks']))
        assert len({t['identity'] for t in b['tracks']})==len(b['tracks'])
    return out,events


def fasttrack(records,z,cfg,root):
    from ultralytics.trackers.fast_tracker import FASTTracker
    native=yaml.safe_load((root/'vendor/ultralytics_8_4_100/ultralytics/cfg/trackers/fasttrack.yaml').read_text(encoding='utf-8'))
    for k in ['track_high_thresh','track_low_thresh','new_track_thresh','track_buffer','match_thresh','fuse_score']:native[k]=cfg['tracker'][k]
    tracker=FASTTracker(SimpleNamespace(**native));out=[]
    for rec in records:
        boxes,feat=frame_input(rec,z);rows=tracker.update(boxes,img=None,feats=feat)
        out.append(dict(frame=rec['frame'],tracks=[dict(base_id=int(t[4]),det_id=int(t[-1]),box=rec['boxes'][int(t[-1])],score=float(rec['scores'][int(t[-1])]),identity=int(t[4]),status='tracked',write=True) for t in rows]))
    return out


def main(root,smoke=False,split=None):
    setup_paths(root);cfg=read_json(root/'config.json');manifest=read_json(root/'input_manifest.json');model,ck=load_identity(root);metrics=official_metrics(root)
    names=['pig_3','horse_4'] if smoke else None
    seqs=[s for s in manifest['sequences'] if s.get('gt') and (s['name'] in names if smoke else s['split']==split)]
    arms=['bytetrack','immediate','confirm','reversible','botsort','deepocsort','fasttrack']
    with Run(root,'reversible_smoke' if smoke else 'reversible_'+split,dict(arms=arms,sequences=[s['name'] for s in seqs],config=cfg,new_trainable_parameters=0,new_training_steps=0,max_output_latency_frames=4,GT='after inference only',model=read_json(root/'identity_pointer.json'))) as run:
        raw={m:{} for m in arms};table=[];checks=[]
        for s in seqs:
            rec,x=load_cache(root,s);z=embed_array(model,ck,x);rec=rec[:100] if smoke else rec
            base,_=run_sequence('bytetrack',rec,z,cfg)
            results={};event_sets={}
            for arm in arms:
                start=time.perf_counter()
                if arm=='bytetrack':out=deepcopy(base);events=[]
                elif arm in {'immediate','confirm','reversible'}:out,events=recover(base,rec,z,arm)
                elif arm=='fasttrack':out=fasttrack(rec,z,cfg,root);events=[]
                else:out=existing(arm,rec,z,cfg);events=[]
                if arm not in {'immediate','confirm','reversible'}:
                    for frame in out:
                        for t in frame['tracks']:t['released_at']=min(frame['frame']+4,len(rec))
                duration=time.perf_counter()-start;results[arm]=out;event_sets[arm]=events
                for f in out:assert len({t['identity'] for t in f['tracks']})==len(f['tracks'])
                if smoke and arm in {'immediate','confirm','reversible'}:
                    for cut in [20,50,80]:
                        partial,_=recover(base[:cut],rec[:cut],z,arm)
                        # A prefix only commits frames whose release time was already reached.
                        canonical=lambda value:json.dumps(value,sort_keys=True,default=lambda a:a.tolist())
                        assert canonical(partial[:cut-4])==canonical(out[:cut-4]),(arm,cut,'future access')
                    checks.append(dict(sequence=s['name'],arm=arm,prefix_causality=True,geometry_count_unchanged=True))
                dest=run.path/arm/s['name'];write_json(dest/'tracks.json',out);write_json(dest/'events.json',events)
                if not smoke:
                    gt=load_gt(root,s);rs={k:m.eval_sequence(metric_data(gt,out)) for k,m in metrics.items()};raw[arm][s['name']]=rs
                    stat=dict(sequence=s['name'],split=s['split'],arm=arm,frames=len(rec),seconds=duration,events=len(events),confirmed=sum(e['type']=='confirmed' for e in events),rollbacks=sum('rollback' in e['type'] for e in events),**compact(rs));table.append(stat)
                    write_json(dest/'metric_details.json',rs);print('V3_RESULT',stat,flush=True)
            # Native parity against independently generated repeated execution is an engineering invariant.
            again,_=run_sequence('bytetrack',rec,z,cfg)
            canonical=lambda value:json.dumps(value,sort_keys=True,default=lambda a:a.tolist())
            assert canonical(again)==canonical(base)
        combined={arm:compact({k:m.combine_sequences({s:r[k] for s,r in raw[arm].items()}) for k,m in metrics.items()}) for arm in arms} if not smoke else {}
        write_json(run.path/'SUMMARY.json',dict(by_sequence=table,combined=combined,engineering_checks=checks,smoke=smoke))
        write_json(run.path/'DECISION.json',dict(status='smoke_passed' if smoke else 'await_complete_frozen_adjudication',new_training=0))
        write_json(root/('reversible_smoke_pointer.json' if smoke else 'reversible_'+split+'_pointer.json'),dict(run_id=run.id,path=run.path.relative_to(root).as_posix()))
        print('V3_COMPLETE',combined,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--split',choices=['engineering','retrospective','independent']);a=p.parse_args();main(a.root,a.smoke,a.split)
