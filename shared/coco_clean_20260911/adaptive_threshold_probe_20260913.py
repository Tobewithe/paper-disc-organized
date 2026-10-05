"""S084: image-disjoint, inference-visible adaptive threshold probe."""
import csv, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
ARM=ROOT/"diagnostics/decode_grid_threshold_probe_20260913/per_target_arm.csv"
S078=ROOT/"diagnostics/crop_support_failure_cross_20260913/per_target.csv"
CACHE=ROOT/"diagnostics/readout_input_scale1200_20260912/cache/images"
OUT=ROOT/"diagnostics/adaptive_threshold_probe_20260913"

def rows(p):
    with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def feat(z,j):
    c=z["coeff"][j].astype(np.float32)
    # sample_p is the frozen prototype response at 2048 inference-visible points.
    l=np.asarray(z["sample_p"][j] @ c, dtype=np.float32)
    q=np.quantile(l,[.05,.25,.5,.75,.95])
    b=z["boxes"][j]; H,W=map(int,z["input_shape"])
    return np.array([*q,l.mean(),l.std(),l.max(),(l>0).mean(),(l>.5).mean(),(b[2]-b[0])*(b[3]-b[1])/(W*H),float(z["detections"][j,4])],np.float64)
def main():
    a=rows(ARM); s={ (int(r['image_id']),int(r['annotation_id'])):r for r in rows(S078)}
    by={}
    for r in a:
        if r['arm'] not in ('input640_t0','input640_t0.5'):continue
        k=(int(r['image_id']),int(r['annotation_id'])); by.setdefault(k,{})[r['arm']]=r
    data=[]; zcache={}
    for k,v in by.items():
        if len(v)<2 or k not in s:continue
        sr=s[k]; p=CACHE/f"{k[0]}.npz"
        if not p.exists():continue
        if k[0] not in zcache: zcache[k[0]]=np.load(p)
        z=zcache[k[0]]; ids=z['annotation_ids'].astype(int); hit=np.where(ids==k[1])[0]
        if not len(hit):continue
        # sample_p rows are already aligned with annotation_ids.
        x=feat(z,int(hit[0])); d=float(v['input640_t0.5']['iou'])-float(v['input640_t0']['iou'])
        data.append((sr['split'],x,d,sr['state'],sr['group']))
    fit=[r for r in data if r[0]=='fit']; tr=[r for r in data if r[0]=='transfer']
    X=np.stack([r[1] for r in fit]); y=np.array([r[2] for r in fit]); mu=X.mean(0); sd=X.std(0); sd[sd<1e-8]=1
    Xn=(X-mu)/sd; A=np.c_[Xn,np.ones(len(Xn))]; lam=10.; w=np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),A.T@y)
    def score(rs):
        xx=(np.stack([r[1] for r in rs])-mu)/sd; return np.c_[xx,np.ones(len(rs))]@w
    OUT.mkdir(parents=True,exist_ok=True); out=[]
    for name,rs in [('fit',fit),('transfer',tr)]:
        pred=score(rs)
        for cut in [0.0,0.002,0.005]:
            vals=[]
            for q,r in zip(pred,rs): vals.append(float(r[2]) if q>cut else 0.0)
            out.append({'split':name,'policy':f'adaptive>{cut:g}','n':len(vals),'mean_delta_vs_t0':float(np.mean(vals))})
        out.append({'split':name,'policy':'fixed_t0.5','n':len(rs),'mean_delta_vs_t0':float(np.mean([r[2] for r in rs]))})
    for state in sorted(set(r[3] for r in tr)):
        rs=[r for r in tr if r[3]==state]; pred=score(rs)
        vals=[float(r[2]) if q>0.002 else 0.0 for q,r in zip(pred,rs)]
        out.append({'split':'transfer','policy':'adaptive>0.002','state':state,'n':len(vals),'mean_delta_vs_t0':float(np.mean(vals))})
    with (OUT/'SUMMARY.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=sorted({k for r in out for k in r}));wr.writeheader();wr.writerows(out)
    json.dump({'n_total':len(data),'n_fit':len(fit),'n_transfer':len(tr),'feature_count':X.shape[1],'summary':out},(OUT/'SUMMARY.json').open('w',encoding='utf-8'),indent=2)
    print(json.dumps({'n_total':len(data),'n_fit':len(fit),'n_transfer':len(tr),'summary':out},indent=2))
if __name__=='__main__':main()
