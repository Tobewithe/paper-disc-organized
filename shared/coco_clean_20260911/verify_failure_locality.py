"""Independent local receipt, partition, sample-metric and paired-mean checks."""
import argparse,ast,csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2


def read(path):
    with path.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out
    hashes=0
    for receipt in ['COMPLETE.json','SUMMARY_COMPLETE.json']:
        r=json.loads((out/receipt).read_text());assert r['status']=='COMPLETE'
        for name,digest in r['hashes'].items():assert sha(out/name)==digest,name;hashes+=1
    protocol=json.loads((out/'protocol.json').read_text());script=Path(__file__).with_name('failure_locality_probe.py')
    assert sha(script)==protocol['script_sha256']
    summary_receipt=json.loads((out/'SUMMARY_COMPLETE.json').read_text())
    assert sha(Path(__file__).with_name('summarize_failure_locality.py'))==summary_receipt['source_sha256']
    # Extract only the numerical helpers: importing the execution pipeline is unnecessary.
    tree=ast.parse(script.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['auc','select_threshold','distances']]
    ns=dict(np=np,cv2=cv2);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(script),'exec'),ns)
    rng=np.random.default_rng(987);matherror=0.
    for _ in range(100):
        s=np.round(rng.normal(size=97),1);y=rng.random(97)>.6;p=s[y];n=s[~y]
        truth=float(((p[:,None]>n[None,:])+.5*(p[:,None]==n[None,:])).mean());matherror=max(matherror,abs(ns['auc'](p,n)-truth))
        thresholds=np.unique(np.r_[np.quantile(s,np.linspace(0,1,257)),np.nextafter(s.min(),-np.inf),s.max(),0.]);pred=s[None]>thresholds[:,None]
        iou=(pred&y).sum(1)/(y.sum()+(pred&~y).sum(1));best=np.flatnonzero(iou==iou.max());k=min(best,key=lambda j:(abs(thresholds[j]),thresholds[j]))
        tau,score=ns['select_threshold'](s,y);assert tau==thresholds[k];matherror=max(matherror,abs(score-iou[k]))
    mask=rng.random((13,17))>.6;inside,outside=ns['distances'](mask);padded=np.pad(mask,1);yy,xx=np.mgrid[:15,:19]
    for source,actual in [(~padded,inside),(padded,outside)]:
        points=np.argwhere(source);query=np.stack([yy[1:-1,1:-1],xx[1:-1,1:-1]],-1)
        truth=np.sqrt(((query[:,:,None,:]-points[None,None,:,:])**2).sum(-1)).min(-1)
        matherror=max(matherror,float(np.max(np.abs(actual-truth))))
    assert matherror<1e-6
    exact=read(out/'exact_error_maps.csv');lookup={(r['split'],r['annotation_id']):r for r in exact};error=0.;matched=0
    for r in exact:
        if r['status']!='matched':continue
        matched+=1;area=int(r['valid_gt_pixels']);tp=int(r['true_positive']);fp=int(r['false_positive']);fn=int(r['false_negative'])
        assert tp+fn==area and int(r['fn_inside_pixels'])+int(r['fn_outside_pixels'])==fn
        assert sum(int(r[k]) for k in ['same_neighbor_pixels','other_category_pixels','background_pixels'])==fp
        for region in ['same_neighbor','background','fn_inside']:assert sum(int(r[region+'_'+b+'_pixels']) for b in ['near','middle','far'])==int(r[region+'_pixels'])
        assert int(r['same_neighbor_core_pixels'])<=int(r['same_neighbor_pixels'])
        error=max(error,abs(tp/area-float(r['coverage'])),abs(tp/(area+fp)-float(r['iou'])))
    raw=read(out/'readouts.csv');group=defaultdict(list)
    for r in raw:
        group[(r['split'],r['annotation_id'],r['readout'])].append(r);n=int(r['eval_positives']);tp=float(r['coverage'])*n;fp=float(r['all_fp'])*n
        error=max(error,abs(tp-round(tp)),abs(fp-round(fp)),abs(tp/(n+fp)-float(r['sample_iou'])))
        assert float(r['same_neighbor'])+float(r['background'])<=float(r['all_fp'])+1e-12
        assert all(np.isfinite(float(r[k])) for k in ['coverage','sample_iou','same_neighbor','background','auc','train_iou'])
        if r['readout']=='original':assert float(r['threshold'])==0
    means=read(out/'target_readout_means.csv');means_by=defaultdict(dict)
    for r in means:
        rows=group[(r['split'],r['annotation_id'],r['arm'])];assert len(rows)==6
        for m in ['sample_iou','coverage','same_neighbor','background','all_fp','auc','train_iou']:error=max(error,abs(float(r[m])-np.mean([float(v[m]) for v in rows])))
        means_by[(r['split'],r['annotation_id'])][r['arm']]=r
    for key,arms in means_by.items():assert len(arms)==6;error=max(error,abs(float(arms['original']['auc'])-float(arms['threshold_oracle']['auc'])))
    contrasts=json.loads((out/'READOUT_ANALYSIS.json').read_text())['contrasts']
    expected_groups=defaultdict(set)
    for key,arms in means_by.items():
        base=lookup[key];high=float(base['ici'])>.5+1e-10;fail=float(base['iou'])<.75;adequate=float(base['crop_ceiling'])>=.75;area=float(base['coco_area'])
        gs=dict(all=True,high=high,low=not high,high_fail75=high and fail,low_fail75=not high and fail,
            high_small=high and area<32**2,high_medium=high and 32**2<=area<96**2,high_large=high and area>=96**2,
            high_fail75_mediumlarge=high and fail and area>=32**2,high_fail75_cropadequate=high and fail and adequate,high_fail75_cropblocked=high and fail and not adequate)
        expected_groups[key[0]].update(g for g,in_group in gs.items() if in_group)
    for split,groups in expected_groups.items():
        assert {r['group'] for r in contrasts if r['split']==split}==groups
        for g in groups:assert sum(r['split']==split and r['group']==g for r in contrasts)==42
    for r in contrasts:
        values=[]
        for key,arms in means_by.items():
            if key[0]!=r['split']:continue
            base=lookup[key];high=float(base['ici'])>.5+1e-10;fail=float(base['iou'])<.75;adequate=float(base['crop_ceiling'])>=.75
            area=float(base['coco_area'])
            groups=dict(all=True,high=high,low=not high,high_fail75=high and fail,low_fail75=not high and fail,
                high_small=high and area<32**2,high_medium=high and 32**2<=area<96**2,high_large=high and area>=96**2,
                high_fail75_mediumlarge=high and fail and area>=32**2,
                high_fail75_cropadequate=high and fail and adequate,high_fail75_cropblocked=high and fail and not adequate)
            if groups[r['group']]:values.append(float(arms[r['treatment']][r['metric']])-float(arms[r['control']][r['metric']]))
        assert len(values)==r['n'];error=max(error,abs(np.mean(values)*100-r['mean_pp']))
    assert error<1e-9
    result=dict(status='PASS',receipt_hashes_checked=hashes,matched_exact_rows=matched,readout_rows=len(raw),complete_target_arm_rows=len(means),contrast_means_checked=len(contrasts),
        numeric_identity_max_abs=error,numerical_helper_max_abs=matherror,limits='No independent re-inference; runtime stencil assertions and hashed cache manifests are execution evidence, not an independent historical dataset witness.')
    (out/'LOCAL_VERIFICATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
