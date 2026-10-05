"""S022 paired cluster summaries; do not infer historical optimizer conflict."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from analyze_native_labels import read,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out
    receipt=json.loads((out/'COMPLETE.json').read_text());assert receipt['status']=='COMPLETE'
    for name,h in receipt['hashes'].items():assert sha(out/name)==h,name
    rows=read(out/'targets.csv');protocol=json.loads((out/'protocol.json').read_text())
    assert len(rows)==len({r['annotation_id'] for r in rows})==len(protocol['target_ids'])
    ids=sorted({int(r['image_id']) for r in rows});index={v:i for i,v in enumerate(ids)}
    boot=np.random.default_rng(20260912).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
    def estimate(rr,fun):
        num=np.zeros(len(ids));den=np.zeros(len(ids))
        for r in rr:num[index[int(r['image_id'])]]+=float(fun(r));den[index[int(r['image_id'])]]+=1
        bd=boot@den;bn=boot@num;ok=bd>0
        return dict(n=len(rr),fraction=float(num.sum()/den.sum()) if len(rr) else None,ci95=np.quantile(bn[ok]/bd[ok],[.025,.975]).tolist() if ok.any() else None)
    summary=[]
    for density in ['high','other']:
        rr=[r for r in rows if r['density']==density];adj=[r for r in rr if int(r['same_class_adjacent_count'])>0]
        without_adj=[float(r['total_alignment_ratio'])-float(r['same_class_adjacent_alignment_ratio']) for r in rr]
        summary.append(dict(density=density,n=len(rr),aggregate_opposes=estimate(rr,lambda r:r['aggregate_opposes_target']=='True'),
            any_other_opposes=estimate(rr,lambda r:float(r['other_alignment_ratio'])<0),
            adjacent_opposes_given_present=estimate(adj,lambda r:float(r['same_class_adjacent_alignment_ratio'])<0),
            adjacent_is_decisive_for_total_reversal=sum(float(r['total_alignment_ratio'])<0 and v>=0 for r,v in zip(rr,without_adj)),
            groups={g:dict(present=sum(int(r[g+'_count'])>0 for r in rr),negative=sum(float(r[g+'_alignment_ratio'])<0 for r in rr),
                median=float(np.median([float(r[g+'_alignment_ratio']) for r in rr])),q10=float(np.quantile([float(r[g+'_alignment_ratio']) for r in rr],.1)),q90=float(np.quantile([float(r[g+'_alignment_ratio']) for r in rr],.9)))
                for g in ['self','same_gt_other_anchor','same_class_adjacent','same_class_nonadjacent','other_class']},
            stride={str(s):estimate([r for r in rr if int(r['stride'])==s],lambda r:r['aggregate_opposes_target']=='True') for s in [8,16,32]}))
    doc=dict(groups=summary,bootstrap='2,000 image-cluster resamples, target-weighted fractions, pointwise intervals; fixed selected failure cohort.',
        receipt_files_verified=len(receipt['hashes']),scope=protocol['limitations'],script_sha256=sha(__file__))
    (out/'GRADIENT_ANALYSIS.json').write_text(json.dumps(doc,indent=2),encoding='utf-8');print(json.dumps(doc,indent=2))

if __name__=='__main__':main()
