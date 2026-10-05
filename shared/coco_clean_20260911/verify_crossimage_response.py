"""Read-only cross-checks of downloaded experiment artifacts and paired means."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out;checks=0
    for receipt in ['FIT_COMPLETE.json','EVALUATION_COMPLETE.json','TASK_COMPLETE.json']:
        data=json.loads((out/receipt).read_text());assert data['status']=='COMPLETE'
        for name,digest in data['hashes'].items():assert sha(out/name)==digest,(receipt,name);checks+=1
    selected=json.loads((out/'selection.json').read_text());fit=set(selected['fit']);cal=set(selected['calibration']);evaluation=set(selected['evaluation'])
    assert not fit&cal and not (fit|cal)&evaluation
    lock=json.loads((out/'LOCKED_SETTINGS.json').read_text());assert lock['status']=='LOCKED_BEFORE_EVALUATION'
    assert max(abs(v) for v in lock['confirmed_gap_pp'].values())<=.25 and lock['own_affine_pixel_xor']==0
    assert sha(out/'LOCKED_SETTINGS.json')==json.loads((out/'EVALUATION_COMPLETE.json').read_text())['lock_sha256']
    protocol=json.loads((out/'protocol.json').read_text());root=Path(__file__).parent
    for key,name in [('script_sha256','crossimage_response_experiment.py'),('decoder_sha256','crossimage_response_decoder.py')]:assert protocol[key]==sha(root/name)
    models=json.loads((out/'MODELS.json').read_text());maxfit=0.;maxres=0.
    for seed in range(3):
        path=out/f'fit_samples_s{seed}.npz'
        if not path.exists():continue
        with np.load(path) as q:
            joint=q['joint'].astype(float);sham=q['sham'].astype(float);y=q['y'].astype(float)
            assert set(map(int,q['image_id']))<=fit
            assert np.array_equal(joint[:,0],sham[:,0]) and np.array_equal(joint[:,2],sham[:,2])
            assert len(y)%128==0 and np.all(y.reshape(-1,128)[:,:64]==1) and np.all(y.reshape(-1,128)[:,64:]==-1)
            for kind,x in [('own',joint[:,:1]),('joint',joint),('sham',sham)]:
                model=models['own_models' if kind=='own' else 'models'][f'{kind}_s{seed}']
                mu=x.mean(0);std=x.std(0).clip(.01);z=(x-mu)/std
                w=np.linalg.solve(z.T@z/len(z)+.1*np.eye(x.shape[1]),z.T@(y-y.mean())/len(z))
                raw=w/std;intercept=y.mean()-mu@raw
                maxfit=max(maxfit,float(np.max(np.abs(raw-np.array(model['raw_weights'])[:x.shape[1]]))),abs(intercept-model['raw_intercept']))
    assert maxfit<1e-10,maxfit
    spatial=read(out/'evaluation_spatial.csv');gt=read(out/'gt_recovery.csv');pairs=read(out/'pair_recovery.csv');aggregate=read(out/'aggregate_summary.csv')
    fields={'spatial':['coverage','same_neighbor','neighbor','background','mask_iou','exclusive_same_neighbor','predicted_area_over_gt'],'task':['hit75'],'pair':['hit75']}
    data={'spatial':spatial,'task':gt,'pair':pairs};lookups={};summarylookup={}
    for r in spatial:
        t,n,b=float(r['coverage']),float(r['neighbor']),float(r['background'])
        assert 0<=t<=1 and n>=0 and b>=0
        maxres=max(maxres,abs(float(r['mask_iou'])-t/(1+n+b)),abs(float(r['predicted_area_over_gt'])-(t+n+b)))
    def val(r,m):return float(r[m]) if m!='hit75' else float(r[m]=='True')
    def key(r,kind):return int(r['target_annotation' if kind=='spatial' else 'annotation_id']) if kind!='pair' else (int(r['annotation_a']),int(r['annotation_b']))
    for domain in ['normal','expand20']:
        for kind,records in data.items():
            base=None
            for arm in ['initial']+list(lock['models']):
                rr={key(r,kind):r for r in records if r['domain']==domain and r['arm']==arm}
                assert len(rr)==len([r for r in records if r['domain']==domain and r['arm']==arm])
                if base is None:base=set(rr)
                assert set(rr)==base
                lookups[(domain,kind,arm)]=rr
    for r in aggregate:
        domain,kind,group,arm=(r[k] for k in ['domain','kind','group','arm']);keys=lookups[(domain,kind,'initial')]
        values=[];images=set()
        for k,q in keys.items():
            ici=float(q['target_ici' if kind=='spatial' else 'ici'])
            selectedgroup=group=='all' or (group=='high' and ici>.5+1e-10) or (group=='low' and ici<=.5+1e-10) or (group=='zero' and ici==0) or (group=='gt1' and ici>1)
            if not selectedgroup:continue
            active=[arm] if not arm.endswith('_mean') else [f'{arm[:-5]}_s{s}' for s in range(3)]
            values.append([np.mean([val(lookups[(domain,kind,s)][k],m) for s in active]) for m in fields[kind]])
            images.add(q['image_id'])
        assert len(values)==int(r['n']) and len(images)==int(r['images'])
        expected=np.mean(values,axis=0)
        for m,v in zip(fields[kind],expected):maxres=max(maxres,abs(float(r[m])-v))
        summarylookup[(domain,kind,group,arm)]=r
    paired=json.loads((out/'PAIRED_ANALYSIS.json').read_text())
    for r in paired['contrasts']:
        prefix=(r['domain'],r['kind'],r['group']);a=summarylookup[(*prefix,r['treatment'])];b=summarylookup[(*prefix,r['control'])];m=r['metric']
        maxres=max(maxres,abs(r['mean_pp']/100-(float(a[m])-float(b[m]))))
    assert maxres<1e-12,maxres
    write(out/'LOCAL_VERIFICATION.json',dict(status='PASS',scope='Downloaded receipts, seed sample provenance, ridge refit, identities, grouped and paired means; no GPU rerun and no endorsement of claims.',
        receipt_hash_entries=checks,fit_images=len(fit),calibration_images=len(cal),evaluation_images=len(evaluation),max_ridge_refit_abs=maxfit,max_arithmetic_abs=maxres,
        spatial_rows=len(spatial),gt_rows=len(gt),pair_rows=len(pairs),paired_contrasts=len(paired['contrasts']),gate=paired['gate'],source_sha256=sha(__file__)))
    print(json.dumps(dict(status='PASS',receipt_hashes=checks,max_ridge_refit=maxfit,max_arithmetic=maxres,gate=paired['gate'])))


if __name__=='__main__':main()
