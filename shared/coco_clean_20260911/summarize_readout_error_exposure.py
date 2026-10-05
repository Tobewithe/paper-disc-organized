"""Post-hoc image-cluster intervals; no optimization or new model selection."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from readout_input_probe import write_json,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);args=ap.parse_args()
    with args.input.open(encoding='utf-8') as f:raw=list(csv.DictReader(f))
    rows=[]
    for r in raw:
        if int(r['valid_area'])<=0:continue
        area=float(r['valid_area']);exposure=float(r['same_neighbor_available'])/area
        p=float(r['neighbor_fp_rate']) if r['neighbor_fp_rate'] else 0.
        assert abs(exposure*p-float(r['false_positive_same_neighbor'])/area)<1e-10
        # Exact valid-domain IoU identity; ordinary COCO evaluation is separate.
        fp=sum(float(r[k]) for k in ['false_positive_same_neighbor','false_positive_other_class','false_positive_background'])/area
        coverage=float(r['own_coverage']);valid_iou=coverage/(1+fp)
        rows.append(dict(image_id=int(r['image_id']),high=r['density']=='high',
            neighbor_present=bool(r['neighbor_fp_rate']),exposure=exposure,neighbor_fpr=p,
            neighbor_fp_own=float(r['false_positive_same_neighbor'])/area,
            coverage=coverage,valid_iou=valid_iou))
    ids=sorted(set(r['image_id'] for r in rows));imap={v:k for k,v in enumerate(ids)}
    rng=np.random.default_rng(20260912);draws=rng.multinomial(len(ids),np.ones(len(ids))/len(ids),size=2000)
    result={}
    for metric in ['exposure','neighbor_fpr','neighbor_fp_own','coverage','valid_iou']:
        stats={}
        for group in ['high','other']:
            rr=[r for r in rows if r['high']==(group=='high') and (metric!='neighbor_fpr' or r['neighbor_present'])]
            count=np.zeros(len(ids));total=np.zeros(len(ids))
            for r in rr:count[imap[r['image_id']]]+=1;total[imap[r['image_id']]]+=r[metric]
            den=draws@count;boot=(draws@total)/den
            stats[group]=dict(targets=len(rr),mean=float(total.sum()/count.sum()),bootstrap=boot)
        delta=stats['high']['bootstrap']-stats['other']['bootstrap']
        result[metric]=dict(high={k:v for k,v in stats['high'].items() if k!='bootstrap'},
             other={k:v for k,v in stats['other'].items() if k!='bootstrap'},
             high_minus_other=float(stats['high']['mean']-stats['other']['mean']),
             ci95=np.quantile(delta,[.025,.975]).tolist())
    out=args.input.parent/'EXPOSURE_ANALYSIS.json'
    write_json(out,dict(metrics=result,bootstrap_images=len(ids),draws=2000,
       scope='Post-hoc unadjusted fixed-box matched targets. Conditional neighbor FPR excludes zero-exposure targets. '
       'Per-target means, image-cluster intervals, no multiple-comparison correction. '
       'Differences may reflect category/scale/box composition. Valid-domain IoU excludes crowd pixels and is not official AP/R75. '
       'Equal conditional-FPR means with broad intervals do not establish equivalence.',
       identity='valid_IoU=own_coverage/(1+sum(exposure_k*FPR_k)), k=same-neighbor, other-class, background; '
       'exposure_k=available crop pixels of k / valid ownGT area. Descriptive exact accounting, not a causal proof.',
       source_sha256=sha(args.input)))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
