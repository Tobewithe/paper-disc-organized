"""Sparse, detection-conditioned identity evaluation, never a full MOT score."""
from pathlib import Path
from collections import Counter, defaultdict
from .common import Run, read_json, write_json, sha256


def score(observations, outputs, frames):
    pred = {(f['frame'], t['det_id']): t['identity'] for f in outputs for t in f['tracks']}
    labels = {(o['frame'], o['reference_id']): o for o in observations}
    refs = sorted({o['reference_id'] for o in observations})
    matched, unknown, missed = [], 0, 0
    for o in observations:
        if o.get('proposal_index') is None:
            unknown += 1; continue
        pid = pred.get((o['frame'], o['proposal_index']))
        if pid is None: missed += 1
        else: matched.append(dict(o, predicted_id=pid))
    pairs=[]; wrong_links=[]
    for f0, f1 in zip(frames[:-1], frames[1:]):
        p0={}; p1={}
        for r in refs:
            a,b=labels.get((f0,r)),labels.get((f1,r))
            if a is None or b is None:continue
            if a.get('proposal_index') is None or b.get('proposal_index') is None:
                pairs.append(dict(before=f0,after=f1,reference_id=r,outcome='label_unknown'));continue
            u=pred.get((f0,a['proposal_index'])); v=pred.get((f1,b['proposal_index']))
            if u is not None:p0[r]=u
            if v is not None:p1[r]=v
            outcome='output_missing' if u is None or v is None else 'same_id' if u==v else 'changed_id'
            pairs.append(dict(before=f0,after=f1,reference_id=r,prior_id=u,next_id=v,outcome=outcome))
        for a,u in p0.items():
            for b,v in p1.items():
                if a!=b and u==v:wrong_links.append(dict(before=f0,after=f1,prior_reference=a,next_reference=b,predicted_id=u))
    counts=Counter(p['outcome'] for p in pairs)
    eligible=len(pairs)-counts['label_unknown'];both=counts['same_id']+counts['changed_id']
    pid_refs=defaultdict(Counter)
    for o in matched:pid_refs[o['predicted_id']][o['reference_id']]+=1
    mixed={str(pid):dict(c) for pid,c in pid_refs.items() if len(c)>1}
    minority=sum(sum(c.values())-max(c.values()) for c in pid_refs.values())
    return dict(summary=dict(annotation_records=len(observations),usable_labels=len(observations)-unknown,
              unknown_labels=unknown,matched_labels=len(matched),missing_outputs=missed,
              pair_counts=dict(counts),eligible_pairs=eligible,both_observed_pairs=both,
              same_id_rate_all_eligible=counts['same_id']/eligible if eligible else None,
              same_id_rate_both_observed=counts['same_id']/both if both else None,
              adjacent_cross_identity_links=len(wrong_links),mixed_predicted_ids=len(mixed),minority_assignments=minority),
              pairs=pairs,wrong_links=wrong_links,mixed_ids=mixed,matched=matched)


def main(root):
    ann_path=root/'annotations/field_sparse_identity_v1.json';ann=read_json(ann_path)
    pointer=read_json(root/'conflict_demo_pointer.json');new=root/pointer['path']
    old=Path(read_json(root/'v1_reference.json')['demo'])
    arms={m:new/m for m in ['motion','all_appearance','margin_rule','net_benefit','correctness']}
    if (root/'trajectory_demo_pointer.json').exists():
        arms['trajectory']=root/read_json(root/'trajectory_demo_pointer.json')['path']/'trajectory'
    if (root/'protected_demo_pointer.json').exists():
        pp=root/read_json(root/'protected_demo_pointer.json')['path']
        arms.update({m:pp/m for m in ['extension','extension_trajectory']})
    arms.update({m:old/m for m in ['deepocsort','utility_recovery']})
    contract=dict(annotation_sha256=sha256(ann_path),scope=ann['evaluation_scope'],
                  human_gold_standard=False,fit_on_labels=False,model_run=pointer,
                  denominators='all adjacent labelled same-reference pairs; unknown excluded, missing outputs counted separately',
                  full_mot_score=False,physical_full_occlusion_verified=0)
    with Run(root,'sparse_field_evaluation',contract) as run:
        table=[]
        for seq in ann['sequences']:
            obs=[o for o in ann['observations'] if o['sequence']==seq['sequence']]
            for mode,path in arms.items():
                r=score(obs,read_json(path/seq['sequence']/'tracks.json'),seq['frames_reviewed'])
                write_json(run.path/mode/(seq['sequence']+'.json'),r)
                table.append(dict(arm=mode,sequence=seq['sequence'],role=seq['role'],**r['summary']))
        write_json(run.path/'SUMMARY.json',dict(by_sequence=table,scope=contract))
        write_json(root/'field_evaluation_pointer.json',dict(run_id=run.id,path=run.path.relative_to(root).as_posix()))
        for t in table:print('FIELD',t,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
