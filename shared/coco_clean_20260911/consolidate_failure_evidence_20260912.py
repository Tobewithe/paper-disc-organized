"""Consolidate already observed failures; no new model inference or oracle fitting."""
import csv,hashlib,itertools,json
from pathlib import Path

root=Path(__file__).parent
data=root/'diagnostics/failure_decomposition_all_20260911'
read=lambda p:list(csv.DictReader(p.open(encoding='utf-8-sig')))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=read(data/'per_instance.csv')
assert len(rows)==36335 and len({r['annotation_id'] for r in rows})==len(rows)
metrics={'same_neighbor':'remove_same_neighbor_iou','all_neighbor':'remove_all_neighbor_iou','background':'remove_background_iou','all_fp':'remove_all_fp_iou','inside_fn':'fill_inside_box_iou','perfect_inside':'perfect_inside_box_iou'}
groups={}
for group in ['high','low','all']:
    selected=[r for r in rows if group=='all' or (float(r['ici'])>.5+1e-10)==(group=='high')]
    valid=[r for r in selected if int(r['valid_gt_pixels'])>0]
    matched=[r for r in valid if r['status']=='matched']
    failures=[r for r in matched if float(r['iou'])<.75]
    rescued={m:{r['annotation_id'] for r in failures if float(r[k])>=.75} for m,k in metrics.items()}
    singles={m:dict(n=len(ids),percent_matched_failures=100*len(ids)/len(failures),percent_valid_gt=100*len(ids)/len(valid)) for m,ids in rescued.items()}
    intersections={a+'&'+b:len(rescued[a]&rescued[b]) for a,b in itertools.combinations(metrics,2)}
    nonexclusive_union=set().union(*(rescued[m] for m in ['same_neighbor','background','inside_fn']))
    residual=[r for r in failures if r['annotation_id'] not in nonexclusive_union]
    mixed=[r for r in residual if float(r['perfect_inside_box_iou'])>=.75]
    boundary=int(sum(float(r['crop_ceiling'])<.75 for r in failures))
    groups[group]=dict(total_noncrowd=len(selected),valid_gt=len(valid),excluded_all_crowd_pixels=len(selected)-len(valid),matched=len(matched),unmatched=len(valid)-len(matched),matched_failures=len(failures),
        crop_support_inadequate=boundary,crop_support_adequate=len(failures)-boundary,
        opportunity_singles=singles,overlap_counts=intersections,
        union_same_or_background_or_fill=len(nonexclusive_union),not_rescued_by_any_of_these_three=len(residual),
        residual_with_adequate_crop=len(mixed),
        interpretation='Overlapping GT oracle opportunities at fixed bbox50 attribution, not causes or achievable task recall. Residual may require multiple edits or other-class neighbor correction; crop limit also applies. No exclusive error taxonomy implied.')
structure=root/'diagnostics/structure_main300_20260911/instances_same_iou.csv'
sr=read(structure);stages=['raw_geometry','argmax_class','score','nms','top300','nonempty','eval100']
stagecounts={}
for group in ['high','low']:
    cohort=[r for r in sr if (float(r['ici_same'])>.5+1e-10)==(group=='high')]
    available={s:sum(r[s+'_available75']=='True' for r in cohort) for s in stages}
    stagecounts[group]=dict(gt=len(cohort),available=available,first_missing_counts={s:sum(r[s+'_available75']!='True' and all(r[t+'_available75']=='True' for t in stages[:j]) for r in cohort) for j,s in enumerate(stages)},bbox75_but_maskfail=sum(r['bbox75']=='True' and r['segm75']!='True' for r in cohort),scope='300 dense-enriched reused val; nonexclusive bbox candidate existence, not full model or exclusive mask matching.')
result=dict(date='2026-09-12',purpose='Read-only synthesis of completed experiments; no new inference, train, or feature probe.',sources={str(p.relative_to(root)):sha(p) for p in [data/'per_instance.csv',data/'summary.csv',structure]},fullval_fixed_attribution=groups,structure300=stagecounts)
target=root/'diagnostics/FAILURE_EVIDENCE_SYNTHESIS_20260912.json';target.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(dict(path=str(target),high=groups['high'],stagecounts=stagecounts),ensure_ascii=False,indent=2))
