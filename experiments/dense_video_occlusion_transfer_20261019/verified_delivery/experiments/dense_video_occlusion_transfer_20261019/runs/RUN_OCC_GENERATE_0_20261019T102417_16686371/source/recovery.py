"""Original C recovery, with only gallery memory averaging changed."""
from copy import deepcopy
import numpy as np
def unit(x):
    x=np.asarray(x,np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)

def distance(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return float(np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)/max(np.linalg.norm(a[2:]-a[:2]),np.linalg.norm(b[2:]-b[:2]),1))


def recover(base,records,z,mode="confirm",weights=None):
    assert mode in {'immediate','confirm','reversible'}
    states={};aliases={};attempted=set();pending={};active={};buffer=[];out=[];events=[]
    def memory(s):
        if weights is None:return unit(np.mean(s["features"],axis=0))
        w=np.asarray(s["weights"],np.float32)
        if np.all(w==1):return unit(np.mean(s["features"],axis=0))
        return unit(np.sum(np.asarray(s["features"])*w[:,None],axis=0)/w.sum())
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
                    pair[b,old]=float(feats[b]@memory(s))
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
            p=dict(old_base=old,identity=identity,anchor=memory(gallery[old]),first=f,count=0,sims=[],bad=0)
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
            if b not in states:states[b]=dict(count=0,features=[],weights=[])
            s=states[b];s['count']+=1;s['last']=f;s['box']=t['box'];s['features'].append(feats[b].copy());s['features']=s['features'][-5:]
            s["weights"].append(1. if weights is None else max(.05,float(weights[f,b])));s["weights"]=s["weights"][-5:]
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

