"""Freeze visually adjudicated event labels before reading tracker comparisons."""
from pathlib import Path
from collections import Counter
from .common import read_json,write_json,sha256,load_gt,Run,now

DECISIONS={
 'O002':('partial_occlusion','Overlapping ducks retain visible body patches; complete invisibility not established.'),
 'O005':('partial_occlusion','Female remains partly visible behind the drake; GT gap is longer than the visually obstructed interval.'),
 'O009':('partial_occlusion','Rear waterbird head remains above foreground goose throughout the gap.'),
 'O011':('partial_occlusion','Rear horse head/back or extremities remain visible through the overlap.'),
 'O013':('uncertain','Small target is not visually separable from foreground horse; official ID alone cannot certify hidden identity.'),
 'O022':('uncertain','Foal legs overlap adult legs and water; complete invisibility and precise identity continuity cannot be certified.'),
 'O024':('uncertain','Small blurred animals overlap; uncertain identity during group motion.'),
 'O030':('partial_occlusion','Gray animal has protruding ears/back before passing below foreground black animal; no certain full interval.'),
 'O031':('uncertain','Long group overlap with similar animals; cannot independently confirm full hidden trajectory.'),
 'O032':('partial_occlusion','High resolution f90 shows recumbent animal belly under standing pink pig; not completely hidden.'),
 'O033':('uncertain','Dense same-color group; endpoint GT geometry is insufficient to establish a fully hidden individual.'),
 'O035':('partial_occlusion','Rear back sliver/legs remain visible behind brown-white pig.'),
 'O040':('partial_occlusion','Small pig legs remain visible underneath foreground animal.'),
 'O043':('physical_occlusion_fullness_uncertain','Tree canopy clearly obstructs target. Faint body/branch patches preclude a reliable zero-visibility interval; reappears to right of canopy.'),
 'O046':('partial_occlusion','Rear smaller gray pig retains upper back visibility left/above dark pig.'),
 'O051':('physical_occlusion_fullness_uncertain','Shrub clearly obstructs zebra; compression and leaf/stripe patches prevent certifying complete invisibility. Original GT resumes through shrub.'),
}

def main(root):
    assert not (root/'annotations/occlusion_labels.json').exists()
    source=read_json(root/'annotations/initial_visual_review.json')
    candidates={e['event_id']:e for e in read_json(root/'annotations/review_manifest.json')['events']}
    dense={e['event_id']:e for e in read_json(root/'annotations/dense/manifest.json')}
    seq={s['name']:s for s in read_json(root/'input_manifest.json')['sequences']}
    with Run(root,'occlusion_label_freeze',dict(tracker_outputs_read=False,reviewer='Codex AI visual inspection; not independent human gold')) as run:
        events=[]
        for r in source['events']:
            c=candidates[r['id']];e=dict(c);e['initial_review']=r;e['dense_reviewed']=r['id'] in DECISIONS
            e['category'],e['review_note']=DECISIONS.get(r['id'],(r['category'],r['note']))
            e['full_occlusion_verified']=False;e['verified_hidden_interval']=None
            e['eligible_partial_event']=r['id'] in DECISIONS and e['category'] in {'partial_occlusion','physical_occlusion_fullness_uncertain'}
            e['identity_source']='official AnimalTrack identity plus raw-image AI review; not human reannotation'
            e['reviewed_pages']=dense.get(r['id'],{}).get('pages',[])
            e['visible_labels']=[]
            if e['eligible_partial_event']:
                gt=load_gt(root,seq[c['sequence']])
                for phase,frames in [('pre',range(c['before']-2,c['before']+1)),('post',range(c['after'],min(seq[c['sequence']]['frames']+1,c['after']+12)))]:
                    for f in frames:
                        row=gt[f][gt[f][:,1]==c['gt_id']]
                        if len(row):e['visible_labels'].append(dict(frame=f,gt_id=c['gt_id'],box=row[0,2:6].tolist(),phase=phase,geometry_source='official GT, visually reviewed boundary window'))
                e['reappearance_time']='Not exact: official first relabelled frame; target may have been partially visible earlier.'
            events.append(e)
        out=dict(schema_version=1,frozen_at=now(),reviewer='Codex AI visual review',independent_human_gold=False,tracker_outputs_read=False,
                 complete_occlusion_events=0,full_occlusion_gate='insufficient verified events; do not claim recovery efficacy',
                 candidate_events=len(events),dense_events=len(DECISIONS),dense_pages=sum(len(v['pages']) for v in dense.values()),
                 partial_event_count=sum(e['eligible_partial_event'] for e in events),visible_label_count=sum(len(e['visible_labels']) for e in events),
                 categories=dict(Counter(e['category'] for e in events)),events=events,
                 metric_scope='Full-video frozen gates unchanged. Partial event metrics are supplementary and do not replace full physical-occlusion gate. GT gap is not physical full occlusion.',
                 source_hashes={p:sha256(root/'annotations'/p) for p in ['initial_visual_review.json','review_manifest.json','dense/manifest.json','zoom/manifest.json']})
        p=root/'annotations/occlusion_labels.json';write_json(p,out)
        write_json(run.path/'SUMMARY.json',{k:v for k,v in out.items() if k!='events'})
        write_json(root/'annotations/ANNOTATION_FREEZE.json',dict(run_id=run.id,frozen_at=now(),sha256=sha256(p),tracker_scores_read=False))
        print({k:out[k] for k in ['candidate_events','dense_events','dense_pages','partial_event_count','visible_label_count','complete_occlusion_events','categories']},flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
