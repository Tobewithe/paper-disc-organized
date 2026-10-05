"""Independent standard-library audit of returned records; no model evaluation."""
from pathlib import Path
from collections import defaultdict,Counter
import json,math,hashlib,statistics

R=Path(__file__).resolve().parents[1]
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
finite=lambda x:isinstance(x,(int,float)) and math.isfinite(x)

def main():
    ids=load(R/'RUN_IDS.json');run=R/'runs'/ids['runs']['evaluation']
    rr=load(run/'run.json');done=load(run/'COMPLETE.json')
    assert rr['status']=='completed' and done['completed'] and done['passed']
    summary=load(run/'SUMMARY.json');audit=load(run/'AUDIT.json')
    rows=[json.loads(line) for line in (run/'PER_CANDIDATE.jsonl').read_text().splitlines() if line.strip()]
    planned=set(load(R/'CONFIRM_IDS.json')['confirm']);excluded=set(load(R/'EXCLUSION_IDS.json')['excluded_train2017_ids'])
    assert len(planned)==1536 and not planned&excluded
    seen={int(r['image_id']) for r in rows};empty=set(done['population']['no_positive_images'])
    assert seen|empty==planned and not seen&empty
    keys=('split','image_id','annotation_id','branch','raw_id','pyramid_level','target_gt_idx')
    assert len({tuple(r[k] for k in keys) for r in rows})==len(rows)==done['population']['candidates']
    predicates={'all':lambda r:True,'box_good_mask_bad':lambda r:bool(r['box_good_mask_bad']),
        'original_success':lambda r:bool(r['mask75_A']),'original_failure':lambda r:not r['mask75_A'],
        'retained_anyclass':lambda r:r['retained_anyclass'],'retained_exactclass':lambda r:r['retained_exactclass'],
        'retained_exactclass_correct':lambda r:r['retained_exactclass_correct']}
    for l in range(3):predicates['P'+str(l+3)]=lambda r,l=l:r['pyramid_level']==l
    for size in ('small','medium','large'):predicates[size]=lambda r,size=size:r['size_group']==size
    errors=[];checks=0
    def close(x,y):
        nonlocal checks
        checks+=1
        if x is None or y is None:assert x is None and y is None;return
        errors.append(abs(x-y));assert abs(x-y)<1e-12,(x,y)
    for name,predicate in predicates.items():
        subset=[r for r in rows if predicate(r)];groups=defaultdict(list)
        for r in subset:groups[r['image_id']].append(r)
        table=summary['tables'][name]
        assert len(subset)==table['candidates'] and len(groups)==table['images']
        for metric in ('iou','mask75','coverage','auc','fpr'):
            for arm in ('A','MIX'):
                values=[r[metric+'_'+arm] for r in subset if finite(r[metric+'_'+arm])]
                means=[statistics.mean([r[metric+'_'+arm] for r in g if finite(r[metric+'_'+arm])]) for g in groups.values() if any(finite(r[metric+'_'+arm]) for r in g)]
                close(statistics.mean(values) if values else None,table['candidate'][metric][arm])
                close(statistics.mean(means) if means else None,table['image_macro'][metric][arm])
        c=table['comparisons']['MIX_minus_A']
        repair=sum(r['mask75_MIX'] and not r['mask75_A'] for r in subset)
        damage=sum(r['mask75_A'] and not r['mask75_MIX'] for r in subset)
        assert c['crossings']==dict(repair=repair,damage=damage,net=repair-damage)
        diffs=[r['iou_MIX']-r['iou_A'] for r in subset]
        changes=c['continuous_iou_changes']
        assert changes['increased']==sum(d>0 for d in diffs)
        assert changes['decreased']==sum(d<0 for d in diffs)
        assert changes['increased_more_than_one_pp']==sum(d>.01 for d in diffs)
        assert changes['decreased_more_than_one_pp']==sum(d<-.01 for d in diffs)
    assert audit['passed'] and audit['baseline_direct_process_mask_pixel_difference_max']==0
    sm=R/'runs'/ids['runs']['smoke'];sm_audit=load(sm/'AUDIT.json')
    assert sm_audit['historical_smoke_reference_max_error']<=1e-12
    for name,row in load(R/'REUSED_CODE.json').items():assert sha(R/'scripts'/name)==row['sha256']
    sources=[json.loads(line) for line in (run/'SOURCE_ROWS.jsonl').read_text().splitlines() if line.strip()]
    assert len(sources)==len(rows)
    source_stats=dict(fallbacks=sum(r['source_match']['fallback'] for r in sources),
        score_below_001=sum(r['source_match']['source_score']<.001 for r in sources),
        source_box_iou_mean=statistics.mean(r['source_match']['box_iou'] for r in sources),
        source_box_iou_below_05=sum(r['source_match']['box_iou']<.5 for r in sources))
    result=dict(passed=True,standard_library_only=True,no_model_execution=True,checks=checks,
        max_mean_error=max(errors,default=0),candidate_count=len(rows),planned_images=len(planned),
        effective_images=len(seen),no_positive_images=len(empty),source_statistics=source_stats,
        same_frozen_dependencies=True,historical_smoke_max_error=sm_audit['historical_smoke_reference_max_error'],
        baseline_decode_pixel_difference=0,whole_cohort_preserved=True,unique_permanent_keys=True,
        source_hashes={p.name:sha(p) for p in (run/'SUMMARY.json',run/'PER_CANDIDATE.jsonl',run/'AUDIT.json',R/'CONFIRM_IDS.json')})
    dest=R/'POST_RUN_AUDIT.json'
    if dest.exists():raise RuntimeError('Post-run audit already exists; do not overwrite')
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
