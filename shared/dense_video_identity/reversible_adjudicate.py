"""Posthoc frozen-gate adjudication and GT attribution; never changes inference."""
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from .common import Run,read_json,write_json,load_gt,setup_paths,sha256,iou_matrix,now
from .learn import load_cache,match_gt

ARMS=['bytetrack','immediate','confirm','reversible','botsort','deepocsort','fasttrack']

def idf(rows):
    v=[sum(r[k] for r in rows) for k in ['IDTP','IDFP','IDFN']]
    return 200*v[0]/max(2*v[0]+v[1]+v[2],1)

def bootstrap(table,a,b):
    names=sorted({r['sequence'] for r in table});idx={(r['sequence'],r['arm']):r for r in table};rng=np.random.default_rng(20261019)
    delta=[]
    for _ in range(2000):
        selected=rng.choice(names,len(names));delta.append(idf([idx[n,a] for n in selected])-idf([idx[n,b] for n in selected]))
    return dict(delta_IDF1=idf([idx[n,a] for n in names])-idf([idx[n,b] for n in names]),ci95=np.percentile(delta,[2.5,97.5]).tolist(),video_clusters=len(names),replicates=2000)

def event_matches(gt,tracks):
    out={};hist=defaultdict(list)
    for frame in tracks:
        f=frame['frame'];g=gt[f];ts=frame['tracks'];sim=iou_matrix(g[:,2:6],[t['box'] for t in ts])
        if sim.size:
            rr,cc=linear_sum_assignment(-sim)
            for r,c in zip(rr,cc):
                if sim[r,c]>=.5:
                    gid=int(g[r,1]);tid=ts[c]['identity'];out[f,gid]=tid;hist[tid].append((f,gid))
    return out,hist

def partial_outcome(e,matches,hist):
    before=[p for p in e['visible_labels'] if p['phase']=='pre'];after=[p for p in e['visible_labels'] if p['phase']=='post']
    pre=[matches.get((p['frame'],e['gt_id'])) for p in before];post=[matches.get((p['frame'],e['gt_id'])) for p in after]
    known=[p for p in pre if p is not None];prior=Counter(known).most_common(1)[0][0] if known else None
    first3=post[:3];success=len(pre)==3 and prior is not None and all(p==prior for p in pre) and len(first3)==3 and all(p==prior for p in first3)
    correct=sum(p==prior for p in post if p is not None) if prior is not None else 0
    wrong=sum(p is not None and p!=prior and any(f<e['after'] and g!=e['gt_id'] for f,g in hist[p]) for p in post)
    return dict(event_id=e['event_id'],sequence=e['sequence'],gt_id=e['gt_id'],category=e['category'],full_occlusion=False,
                strict_three_before_after_success=success,prior_id=prior,pre=pre,post=post,post_frames=[p['frame'] for p in after],
                correct_visible_frames=correct,visible_frame_denominator=len(post),matched_visible_frames=sum(p is not None for p in post),wrong_reused_identity_frames=wrong,
                detection_condition_met=all(p is not None for p in pre+first3),
                reference_frame_delay=min([p['frame']-e['after'] for p,q in zip(after,post) if q==prior and prior is not None],default=None),
                delay_scope='relative to official relabelling frame, not physical reappearance latency')

def audit(seq,split,base,records,gt,outputs,events):
    mapping=match_gt(records,gt);history=defaultdict(list)
    for fr in base:
        for t in fr['tracks']:
            g=mapping[fr['frame']].get(t['det_id'])
            if g is not None:history[t['base_id']].append((fr['frame'],int(g)))
    def latest(b,f,max_age=None):
        vv=[(ff,g) for ff,g in history[b] if ff<=f and (max_age is None or f-ff<=max_age)]
        return vv[-1][1] if vv else None
    allrows=[];stats={}
    for arm in ['immediate','confirm','reversible']:
        proposals={};commits={};annotated=[];count=Counter();duration=Counter()
        for ev in events[arm]:
            b=ev['base_id'];f=ev['frame'];row=dict(ev,sequence=seq,split=split,arm=arm)
            if ev['type']=='proposed':
                proposals[b]=dict(ev,anchor_gt=latest(ev['old_base'],f-1));continue
            p=proposals.get(b,{})
            old=p.get('anchor_gt');new=latest(b,f,3)
            status='unknown' if old is None or new is None else 'same_GT' if old==new else 'different_GT'
            row.update(anchor_gt=old,current_gt=new,local_GT_relation=status,proposal_frame=p.get('frame'))
            if ev['type']=='confirmed':
                commits[b]=row;annotated.append(row);count['confirmed_'+status]+=1
            elif ev['type']=='rollback':
                count['committed_rollback_'+status]+=1;count['reason_'+ev['reason']]+=1;annotated.append(row)
            elif ev['type']=='proposal_rollback':count['pending_rollback_'+status]+=1;annotated.append(row)
        for fr in outputs[arm]:
            for t in fr['tracks']:
                b=t['base_id'];g=mapping[fr['frame']].get(t['det_id']);c=commits.get(b)
                if t['identity']!=b:
                    old=c.get('anchor_gt') if c else None
                    status='unknown' if old is None or g is None else 'correct' if old==g else 'wrong'
                    duration[status]+=1
        stats[arm]=dict(counts=dict(count),published_reconnected_observation_frames=dict(duration))
        allrows.extend(annotated)
    return stats,allrows

def main(root):
    setup_paths(root);labels=read_json(root/'annotations/occlusion_labels.json');freeze=read_json(root/'annotations/ANNOTATION_FREEZE.json')
    assert freeze['sha256']==sha256(root/'annotations/occlusion_labels.json') and not freeze['tracker_scores_read']
    seqs={s['name']:s for s in read_json(root/'input_manifest.json')['sequences']}
    contract=dict(protocol=sha256(root/'PROTOCOL.md'),annotation_freeze=freeze,classification='posthoc fixed detection-GT one-to-one Hungarian IoU>=0.5',
                  wrong_duration='published relabelled observation-frames with GT different from old raw-track last known GT before proposal; unknown separately',
                  partial_events='supplementary, AI-reviewed official geometry; not full-occlusion Gate',bootstraps=2000,seed=20261019)
    with Run(root,'reversible_adjudication',contract) as run:
        summaries={};comparisons={};audits=[];event_rows=[];partial=[];invariants=[]
        for split in ['engineering','retrospective','independent']:
            p=read_json(root/('reversible_'+split+'_pointer.json'));folder=root/p['path'];s=read_json(folder/'SUMMARY.json');summaries[split]=dict(run_id=p['run_id'],**s)
            comparisons[split]={b:bootstrap(s['by_sequence'],'reversible',b) for b in ['bytetrack','botsort','confirm','immediate']}
            names=sorted({r['sequence'] for r in s['by_sequence']})
            for name in names:
                seq=seqs[name];gt=load_gt(root,seq);records,_=load_cache(root,seq)
                outs={a:read_json(folder/a/name/'tracks.json') for a in ARMS};events={a:read_json(folder/a/name/'events.json') for a in ['immediate','confirm','reversible']}
                base=outs['bytetrack'];key=lambda t:(t['base_id'],t['det_id'],tuple(t['box']),t['score'])
                for arm in ['immediate','confirm','reversible']:
                    assert all(sorted(map(key,a['tracks']))==sorted(map(key,b['tracks'])) for a,b in zip(base,outs[arm]))
                    assert all(len({t['identity'] for t in fr['tracks']})==len(fr['tracks']) for fr in outs[arm])
                    assert all(0<=t['released_at']-fr['frame']<=4 for fr in outs[arm] for t in fr['tracks'])
                invariants.append(dict(sequence=name,frames=len(base),observations=sum(len(f['tracks']) for f in base),geometry_count_identity_unique_latency=True))
                stat,rows=audit(name,split,base,records,gt,outs,events);audits.append(dict(sequence=name,split=split,arms=stat));event_rows+=rows
                for arm,out in outs.items():
                    mapping,hist=event_matches(gt,out)
                    for e in labels['events']:
                        if e['sequence']==name and e['eligible_partial_event']:partial.append(dict(arm=arm,split=split,**partial_outcome(e,mapping,hist)))
        i=summaries['independent']['combined'];r=summaries['retrospective']['combined'];table=summaries['independent']['by_sequence'];ret=summaries['retrospective']['by_sequence']
        pair=lambda t,name,a:next(x for x in t if x['sequence']==name and x['arm']==a)['IDF1']
        names=sorted({t['sequence'] for t in table});rnames=sorted({t['sequence'] for t in ret})
        gates=dict(independent_vs_native=i['reversible']['IDF1']-i['bytetrack']['IDF1']>=1,
                   independent_vs_reference=i['reversible']['IDF1']-i['botsort']['IDF1']>=.5,
                   two_positive_intervals=all(comparisons['independent'][a]['ci95'][0]>0 for a in ['bytetrack','botsort']),
                   independent_two_of_three=sum(pair(table,n,'reversible')>pair(table,n,'bytetrack') for n in names)>=2,
                   independent_damage_guard=all(pair(table,n,'reversible')-pair(table,n,'bytetrack')>=-1 for n in names),
                   independent_HOTA_guard=i['reversible']['HOTA']-i['bytetrack']['HOTA']>=-.5,
                   retrospective_aggregate=r['reversible']['IDF1']>=r['bytetrack']['IDF1'],
                   retrospective_damage_guard=all(pair(ret,n,'reversible')-pair(ret,n,'bytetrack')>=-1 for n in rnames))
        mechanism=dict(independent_M_gt_C=i['reversible']['IDF1']>i['confirm']['IDF1'],independent_M_gt_I=i['reversible']['IDF1']>i['immediate']['IDF1'],
                       status='not_supported',full_occlusion_status='insufficient',full_occlusion_events=labels['complete_occlusion_events'])
        partial_summary={a:dict(events=sum(e['arm']==a for e in partial),success=sum(e['arm']==a and e['strict_three_before_after_success'] for e in partial),
            wrong_reuse_frames=sum(e['wrong_reused_identity_frames'] for e in partial if e['arm']==a),correct_frames=sum(e['correct_visible_frames'] for e in partial if e['arm']==a),
            visible_frame_denominator=sum(e['visible_frame_denominator'] for e in partial if e['arm']==a),matched_frames=sum(e['matched_visible_frames'] for e in partial if e['arm']==a)) for a in ARMS}
        totals={}
        for split in ['engineering','retrospective','independent']:
            totals[split]={}
            for a in ['immediate','confirm','reversible']:
                c=Counter();d=Counter()
                for row in audits:
                    if row['split']==split:c.update(row['arms'][a]['counts']);d.update(row['arms'][a]['published_reconnected_observation_frames'])
                totals[split][a]=dict(counts=dict(c),published_frames=dict(d))
        summary=dict(splits=summaries,comparisons=comparisons,gates=gates,stable_superiority=all(gates.values()),mechanism=mechanism,annotation_summary={k:v for k,v in labels.items() if k!='events'},
                     partial_events=partial_summary,audit_totals=totals,invariants=invariants)
        write_json(run.path/'SUMMARY.json',summary);write_json(run.path/'link_audit.json',event_rows);write_json(run.path/'audit_by_sequence.json',audits);write_json(run.path/'partial_events.json',partial)
        decision=dict(scientific_status='negative_fixed_rollback_configuration',stable_superiority=all(gates.values()),gates=gates,
                      decision='Close fixed rollback configuration. Keep positive identity-only reconnection evidence separate from unsupported temporal/rollback contributions.',
                      priority='Determine a rollback contradiction that distinguishes wrong identity from pose/view change; present anchor-cosine rollback is not supported.',
                      next_action='Use audited false-versus-correct rollback events to specify a discriminability test before any new learned model or threshold search.',started_new_training=False)
        write_json(run.path/'DECISION.json',decision);run.meta['scientific_status']=decision['scientific_status']
        write_json(root/'reversible_adjudication_pointer.json',dict(run_id=run.id,path=run.path.relative_to(root).as_posix()))
        meta=read_json(root/'study.json');meta.update(execution_status='comparisons_and_adjudication_completed',scientific_status=decision['scientific_status'],artifact_status='visualization_in_progress',updated_at=now());write_json(root/'study.json',meta)
        print('GATES',gates,flush=True);print('COMPARISONS',comparisons,flush=True);print('AUDITS',totals,flush=True);print('PARTIAL_EVENTS',partial_summary,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
