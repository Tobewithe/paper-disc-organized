"""Summarize official current-checkpoint assignment and fixed-source supervision."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
from candidate_lineage_probe import ROOT,read,need,write_json,sha,save_csv

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out
    rec=json.loads((out/'COMPLETE.json').read_text())
    for name,h in rec['hashes'].items():need(sha(out/name)==h,name)
    rows=read(out/'targets.csv');summaries=[]
    oracle={int(r['annotation_id']):float(r['original_coco_mask_iou']) for r in read(ROOT/'diagnostics/no_candidate317_20260912/original_coco_mask_iou.csv') if r['arm']=='free_coefficient'}
    for density in ['all','high','other']:
        base=[r for r in rows if density=='all' or (float(r['ici'])>.5+1e-10)==(density=='high')]
        for cohort in ['all_gt','official_mask75_failed','no_good_postconf_mask75','s015_oracle_recoverable']:
            rr=base
            if cohort=='official_mask75_failed':rr=[r for r in rr if r['official_mask75']=='False']
            if cohort=='no_good_postconf_mask75':rr=[r for r in rr if r['official_mask75']=='False' and r['no_good_postconf_mask75']=='True']
            if cohort=='s015_oracle_recoverable':rr=[r for r in rr if oracle.get(int(r['annotation_id']),0)>=.75]
            if not rr:continue
            matched=[r for r in rr if int(r['final_source'])>=0]
            summaries.append(dict(density=density,cohort=cohort,gt=len(rr),final_matched=len(matched),
                zero_positive=sum(int(r['positive_count'])==0 for r in rr),positive_count_mean=float(np.mean([int(r['positive_count']) for r in rr])),
                positive_count_median=float(np.median([int(r['positive_count']) for r in rr])),
                final_assignment=dict(Counter(r['final_source_assignment'] for r in rr)),
                final_zero_gradient=sum(float(r['final_source_mask_gradient'])==0 for r in matched),
                zero_mask_pixels160=sum(int(r['gt_mask_cells160'])==0 for r in rr),
                zero_foreground_inside_gt_crop160=sum(int(r['gt_mask_cells_in_gt_crop160'])==0 for r in rr),
                gt_mask_cells160_median=float(np.median([int(r['gt_mask_cells160']) for r in rr])),
                nonzero_gradient_per_positive=float(sum(int(r['positive_nonzero_gradient']) for r in rr)/max(sum(int(r['positive_count']) for r in rr),1))))
    # Same-GT winning vs suppressed mask-quality opportunities in S014.
    candidate_rows=[]
    for iid in json.loads((out/'protocol.json').read_text())['images']:
        with np.load(out/'images'/f'{iid}.npz') as z:data={k:z[k] for k in z.files}
        gi={int(a):j for j,a in enumerate(data['annotation_ids'])}
        prior=json.loads((ROOT/'diagnostics/candidate_lineage300_20260912/images'/f'{iid}.json').read_text())
        for r in prior['choices']:
            aid=int(r['annotation_id']);g=gi[aid];src=int(r['original_source']);alt=int(r['alternative_source'])
            def state(j):return 'not_positive' if not data['foreground'][j] else 'own_gt_positive' if data['target_gt_index'][j]==g else 'different_gt_positive'
            candidate_rows.append(dict(image_id=iid,annotation_id=aid,ici=r['ici'],original_source=src,alternative_source=alt,
                changed=src!=alt,original_assignment=state(src),alternative_assignment=state(alt),
                original_gradient=float(np.linalg.norm(data['coefficient_gradient'][src])),alternative_gradient=float(np.linalg.norm(data['coefficient_gradient'][alt]))))
    save_csv(out/'s014_alternative_supervision.csv',candidate_rows)
    write_json(out/'ANALYSIS.json',dict(groups=summaries,
        source_note='Mask loss gradient directly w.r.t. candidate coefficient, before shared-head Jacobian; zero direct gradient is not absence of all indirect learning through shared weights. Current deterministic checkpoint assignment, not historical training reconstruction.',
        script_sha256=sha(__file__),mask_recipe_note='Exact independent COCO masks with overlap_mask=False in official loss; differs from stock polygon/overlap preprocessing. Mask cells diagnostic not assertion of original training label corruption.'))
    write_json(out/'ANALYSIS_COMPLETE.json',dict(status='COMPLETE',script_sha256=sha(__file__),
        analysis_sha256=sha(out/'ANALYSIS.json'),alternative_sha256=sha(out/'s014_alternative_supervision.csv')))
    print(json.dumps(summaries),flush=True)

if __name__=='__main__':main()
