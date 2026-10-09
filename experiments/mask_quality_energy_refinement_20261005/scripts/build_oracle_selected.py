from __future__ import annotations
import argparse, gzip, json, math, sys, time, traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from ultralytics.utils import ops
sys.path.insert(0,str(Path(__file__).resolve().parent))
from online_runtime import load_asset, dump, resolve_runtime_config

LAMBDA=0.003

def solve(records, max_iter):
    n=len(records); dev=records[0]['p'].device
    delta=torch.zeros((n,32),dtype=torch.float64,device=dev,requires_grad=True)
    opt=torch.optim.LBFGS([delta],lr=1.0,max_iter=max_iter,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-12)
    def vg():
        total=torch.zeros((),dtype=torch.float64,device=dev); grad=torch.zeros_like(delta)
        for j,r in enumerate(records):
            z=r['p']@(r['c']+delta[j]); bce=(F.softplus(z)-r['y']*z).sum()/r['area']*r['gain']
            total=total+bce+LAMBDA*delta[j].square().sum()/2
            grad[j]=(((torch.sigmoid(z)-r['y'])/r['area']*r['gain'])@r['p'])+LAMBDA*delta[j]
        return total/n,grad/n
    def closure():
        opt.zero_grad(set_to_none=True)
        with torch.no_grad(): v,g=vg()
        delta.grad=g; return v
    opt.step(closure)
    with torch.no_grad(): v,g=vg()
    nit=int(opt.state[delta].get('n_iter',-1))
    return delta.detach().float().cpu(),float(v.cpu()),g.detach().norm(dim=1).cpu(),nit

def make_record(x,row,device):
    k=int(row['row_index']); box=x['target_boxes'][k].float().to(device); pfull=F.interpolate(x['proto'].float().to(device)[None],(640,640),mode='bilinear',align_corners=False)[0]
    support=ops.crop_mask(torch.ones((1,640,640),device=device),box[None])[0].bool()
    if not bool(support.any()): raise RuntimeError(f'empty support {x["image_id"]} {row["annotation_id"]}')
    p=pfull[:,support].T.contiguous().double(); y=(x['masks'].to(device)[support]==(int(x['owners'][k])+1)).double(); area=float(((box[2:]-box[:2])/640.).prod()*640.*640.)
    ident={q: int(row[q]) for q in ('image_id','annotation_id','raw_id','pyramid_level','target_gt_idx')}
    return {'p':p,'y':y,'c':x['c0'][k].float().to(device).double(),'area':area,'gain':float(x['segmentation_gain']),'ident':ident}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--out',required=True); ap.add_argument('--batch',type=int,default=4); ap.add_argument('--max-iter',type=int,default=60); ap.add_argument('--deadline',type=float,default=0.0)
    a=ap.parse_args(); cfg=resolve_runtime_config(json.loads(Path(a.config).read_text(encoding='utf-8-sig'))); manifest=json.loads(Path(a.manifest).read_text(encoding='utf-8-sig')); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.set_num_threads(4)
    rows=manifest['candidates']['failure']; deltas=[]; identities=[]; started=time.monotonic(); done=0
    byimg={}
    for r in rows: byimg.setdefault(int(r['image_id']),[]).append(r)
    for iid, rs in byimg.items():
        if a.deadline and time.time()>=a.deadline: raise TimeoutError('oracle deadline reached')
        x=load_asset(cfg,iid,verify=True); records=[make_record(x,r,device) for r in rs]
        for lo in range(0,len(records),a.batch):
            part=records[lo:lo+a.batch]; d,v,g,nit=solve(part,a.max_iter); deltas.append(d)
            for j,r in enumerate(part): identities.append({**r['ident'],'objective':v,'gradient_norm':float(g[j]),'iterations':nit,'exit_reason':'iteration_limit' if nit>=a.max_iter else 'LBFGS_termination'})
            done+=len(part)
        if len(byimg) and (len(identities)==done) and (len(byimg)%1==0):
            if len(identities)%256 < len(rs): print(json.dumps({'images_done':list(byimg).index(iid)+1,'images_total':len(byimg),'records':done,'elapsed_s':time.monotonic()-started}),flush=True)
    delta=torch.cat(deltas,0) if deltas else torch.empty((0,32))
    torch.save({'delta':delta,'identities':identities,'lambda':LAMBDA,'source':'QCR selected official one2one finite oracle','images':len(byimg)},out)
    summary={'completed':True,'records':done,'images':len(byimg),'delta_shape':list(delta.shape),'lambda':LAMBDA,'max_iter':a.max_iter,'device':str(device),'elapsed_s':time.monotonic()-started,'median_norm':float(delta.double().norm(dim=1).median()) if len(delta) else None,'max_gradient_norm':max((r['gradient_norm'] for r in identities),default=None)}
    dump(out.with_suffix('.SUMMARY.json'),summary); print(json.dumps(summary),flush=True)
if __name__=='__main__':
    try: main()
    except BaseException: traceback.print_exc(); raise
