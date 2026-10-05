"""Descriptive same-GT ranking controls and high/low interaction; no new inference."""
import argparse,json
from pathlib import Path
import numpy as np
from candidate_lineage_probe import ROOT,read,need,save_csv,write_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out
    ids=json.loads((out/'protocol.json').read_text())['images']
    source=ROOT/'diagnostics/structure_main300_20260911/raw'
    records=[]
    for iid in ids:
        doc=json.loads((out/'images'/f'{iid}.json').read_text())
        with np.load(source/f'{iid}.npz') as z:raw={k:z[k] for k in z.files}
        with np.load(out/'images'/f'{iid}.npz') as z:probe={k:z[k] for k in z.files}
        col={int(j):k for k,j in enumerate(probe['source_ids'])}
        gi={int(j):k for k,j in enumerate(probe['annotation_ids'])}
        for choice in doc['choices']:
            old=int(choice['original_source']);alt=int(choice['alternative_source']);g=gi[choice['annotation_id']]
            scores=probe['source_scores'];classes=probe['source_classes'];parent=probe['first_suppressor']
            candidates=probe['source_ids']
            pool=candidates[(classes[candidates]==classes[old]) & (raw['bbox_iou'][g,candidates]>=.5)]
            q=probe['mask_iou'][g,[col[int(j)] for j in pool]]
            better=(q>probe['mask_iou'][g,col[old]]+1e-12)
            near=np.abs(raw['bbox_iou'][g,pool]-raw['bbox_iou'][g,old])<=.02
            corr=None
            if len(pool)>=3 and np.std(q)>1e-12 and np.std(scores[pool])>1e-12:
                corr=float(np.corrcoef(q,scores[pool])[0,1])
            records.append(dict(image_id=iid,annotation_id=choice['annotation_id'],ici=choice['ici'],
                original_source=old,alternative_source=alt,changed=old!=alt,
                alternative_score=float(scores[alt]),original_score=float(scores[old]),
                alternative_suppressed=int(parent[alt])>=0,original_suppressed_alternative=int(parent[alt])==old,
                alternative_lower_score=bool(scores[alt]<scores[old]),same_scale_level=(old<6400,old<8000)==(alt<6400,alt<8000),
                candidate_count=len(pool),within_gt_score_mask_iou_pearson=corr,
                same_box_quality_candidates=int(near.sum()),
                same_box_quality_better_lower_score=bool(np.any(near&better&(scores[pool]<scores[old]))),
                same_box_quality_best_gain=float(q[near].max()-probe['mask_iou'][g,col[old]]) if near.any() else 0.))
    save_csv(out/'ranking_controls.csv',records)
    summary=[]
    for name in ['all','high','low']:
        rr=[r for r in records if name=='all' or (r['ici']>.5+1e-10)==(name=='high')]
        changed=[r for r in rr if r['changed']]
        cs=[r['within_gt_score_mask_iou_pearson'] for r in rr if r['within_gt_score_mask_iou_pearson'] is not None]
        summary.append(dict(group=name,targets=len(rr),changed=len(changed),
            changed_lower_score=sum(r['alternative_lower_score'] for r in changed),
            changed_suppressed=sum(r['alternative_suppressed'] for r in changed),
            changed_suppressed_by_original=sum(r['original_suppressed_alternative'] for r in changed),
            changed_same_level=sum(r['same_scale_level'] for r in changed),
            near_box_better_lower_score=sum(r['same_box_quality_better_lower_score'] for r in rr),
            near_box_best_mask_gain_pp=100*float(np.mean([r['same_box_quality_best_gain'] for r in rr])),
            within_gt_pearson_n=len(cs),within_gt_pearson_mean=float(np.mean(cs)) if cs else None))
    spatial=read(out/'spatial.csv');base={int(r['target_annotation']):r for r in spatial if r['arm']=='original'}
    rng=np.random.default_rng(20260912);w=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
    ix={v:j for j,v in enumerate(ids)};interactions=[]
    for field in ['mask_iou','same_neighbor','coverage','background']:
        num=np.zeros((2,len(ids)));den=np.zeros_like(num)
        for r in spatial:
            if r['arm']!='coefficient_only':continue
            g=int(float(r['target_ici'])>.5+1e-10);j=ix[int(r['image_id'])]
            num[g,j]+=float(r[field])-float(base[int(r['target_annotation'])][field]);den[g,j]+=1
        bden=w@den.T;bnum=w@num.T;valid=np.all(bden>0,axis=1)
        values=bnum[valid]/bden[valid]
        interactions.append(dict(metric=field,high_minus_low_pp=100*float(num[1].sum()/den[1].sum()-num[0].sum()/den[0].sum()),
            ci95_pp=(100*np.quantile(values[:,1]-values[:,0],[.025,.975])).tolist(),
            note='Unadjusted oracle interaction; no class/size/box-quality composition adjustment.'))
    write_json(out/'RANKING_ANALYSIS.json',dict(groups=summary,coefficient_high_low_interactions=interactions,
        scope='Descriptive post hoc controls from existing raw caches, not method selection; box-IoU tolerance .02 chosen after main results. Ordinary GT independent targets, repeated correlated candidates; within-GT Pearson descriptive without significance claims.',
        script_sha256=sha(__file__)))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
