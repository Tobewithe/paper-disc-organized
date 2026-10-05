"""Descriptive category x COCO-area standardization of full-val diagnostics."""
import csv,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'diagnostics/failure_decomposition_all_20260911'

def main():
    with (ROOT/'per_instance.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    rows=[r for r in rows if int(r['valid_gt_pixels'])>0 and r['status']=='matched']
    cells={}
    for r in rows:
        area=float(r['coco_area']);size='small' if area<32**2 else ('medium' if area<96**2 else 'large')
        key=(int(r['category_id']),size);high=float(r['ici'])>.5+1e-10;cells.setdefault(key,{False:[],True:[]})[high].append(r)
    eligible={k:v for k,v in cells.items() if min(len(v[False]),len(v[True]))>=10}
    images=sorted({int(r['image_id']) for v in eligible.values() for group in v.values() for r in group});index={v:i for i,v in enumerate(images)}
    metrics=['iou','coverage','same_neighbor_ratio','background_ratio','failure75','neighbor_rescue75','background_rescue75']
    totals=np.array([len(v[False])+len(v[True]) for v in eligible.values()],dtype=float);weights=totals/totals.sum()
    sums=np.zeros((len(eligible),2,len(images),len(metrics)));counts=np.zeros((len(eligible),2,len(images)))
    cells_out=[]
    for c,(key,v) in enumerate(eligible.items()):
        cells_out.append(dict(category_id=key[0],area=key[1],low=len(v[False]),high=len(v[True]),weight=float(weights[c])))
        for high,group in v.items():
            for r in group:
                i=index[int(r['image_id'])];a=int(r['valid_gt_pixels']);failed=float(r['iou'])<.75
                vals=[float(r['iou']),float(r['coverage']),int(r['same_neighbor_pixels'])/a,int(r['background_pixels'])/a,float(failed),float(failed and float(r['remove_same_neighbor_iou'])>=.75),float(failed and float(r['remove_background_iou'])>=.75)]
                counts[c,int(high),i]+=1;sums[c,int(high),i]+=vals
    mean=sums.sum(2)/counts.sum(2)[:,:,None];estimate=(weights[:,None]*(mean[:,1]-mean[:,0])).sum(0)
    rng=np.random.default_rng(20260911);boot=[];skipped=0
    for _ in range(2000):
        multiplicity=np.bincount(rng.integers(len(images),size=len(images)),minlength=len(images));den=counts@multiplicity
        if np.any(den==0):skipped+=1;continue
        numerator=np.einsum('cgim,i->cgm',sums,multiplicity);means=numerator/den[:,:,None];boot.append((weights[:,None]*(means[:,1]-means[:,0])).sum(0))
    ci=np.quantile(boot,[.025,.975],axis=0)
    report=dict(design='Fixed bbox-matched valid instances; high ICI>.5 vs other instances. Category x COCO annotation-area strata require >=10 high and >=10 low. Common pooled stratum weights; paired image-cluster bootstrap, fixed selected strata. Descriptive adjustment, not causal; box quality/occlusion and selection remain unadjusted.',
        matched_total=len(rows),eligible_cells=len(eligible),retained_high=sum(len(v[True]) for v in eligible.values()),retained_low=sum(len(v[False]) for v in eligible.values()),images=len(images),bootstrap_valid=len(boot),bootstrap_omitted_empty_cell=skipped,
        contrasts=[dict(metric=k,high_minus_low_pp=float(100*estimate[j]),ci_low_pp=float(100*ci[0,j]),ci_high_pp=float(100*ci[1,j])) for j,k in enumerate(metrics)],cells=cells_out)
    (ROOT/'STANDARDIZED_CATEGORY_AREA.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cells'},indent=2))

if __name__=='__main__':main()
