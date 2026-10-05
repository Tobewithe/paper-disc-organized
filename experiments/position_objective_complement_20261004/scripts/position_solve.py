"""Solve the two missing position x objective cells on fixed caches."""
import argparse, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from ultralytics.utils import ops

LAMBDA = 0.003

def geom_rows(path):
    out=[]
    for r in torch.load(path,map_location='cpu',weights_only=False):
        out.append(dict(level=int(r['meta']['level']), image_id=int(r['meta']['image_id']),
            annotation_id=int(r['meta']['annotation_id']), raw_id=int(r['meta']['raw_id']),
            h=r['h'].double(), c=r['c'].double(), p=r['p'].float(), y=r['y'].float(),
            area=float(len(r['y'])/max(float(r['factor']),1e-12)), gain=1.0,
            box_iou=float(r['meta'].get('box_iou',float('nan')))))
    return out

def official_target_rows(geom, official_cache, ann_json):
    from pycocotools.coco import COCO
    coco=COCO(str(ann_json)); by_img={}
    for r in geom: by_img.setdefault(r['image_id'],[]).append(r)
    out=[]
    for iid, items in by_img.items():
        im=torch.load(official_cache/'images'/f'{iid:012d}.pt',map_location='cpu',weights_only=False)
        up=F.interpolate(im['proto'].float()[None],(640,640),mode='bilinear',align_corners=False)[0]
        ids=list(im['all_annotation_ids']); masks=im['masks'].float(); shape=tuple(im['original_shape'])
        (gainx,gainy),(padx,pady)=im['ratio_pad']; gainx=float(gainx); gainy=float(gainy); padx=float(padx); pady=float(pady)
        for r in items:
            ann=coco.anns[r['annotation_id']]; x,y,w,h=[float(v) for v in ann['bbox']]
            box=torch.tensor([x*gainx+padx,y*gainy+pady,(x+w)*gainx+padx,(y+h)*gainy+pady],dtype=torch.float32)
            support=ops.crop_mask(torch.ones((1,640,640)),box[None])[0].bool()
            oi=ids.index(r['annotation_id'])
            rr=dict(r); rr.update(p=up[:,support].T.float(),y=(masks[support]==oi+1).float(),
                area=float(((box[2:]-box[:2])/640).prod()*640*640),gain=float(im['segmentation_gain']))
            # verify the raw output is the same frozen candidate
            rr['h']=im['h'][r['raw_id']].double(); rr['c']=im['coeff'][r['raw_id']].double()
            out.append(rr)
    return out

class Bank:
    def __init__(self, rows, stats=None, bias=False, device='cuda'):
        self.rows=rows; self.n=len(rows); self.device=torch.device(device); self.bias=bias
        self.level=torch.tensor([r['level'] for r in rows],dtype=torch.long,device=self.device)
        self.c=torch.stack([r['c'] for r in rows]).double().to(self.device)
        self.gain=torch.tensor([r['gain'] for r in rows],dtype=torch.float64,device=self.device)
        self.area=torch.tensor([r['area'] for r in rows],dtype=torch.float64,device=self.device)
        hs=[]
        for r in rows:
            h=r['h'].double()
            if stats is not None:
                mu,sd=stats[r['level']]; h=(h-mu)/sd
            hs.append(torch.cat((h,torch.ones(1,dtype=torch.float64))))
        self.h=torch.stack(hs).to(self.device)
        offs=[0]
        for r in rows: offs.append(offs[-1]+len(r['y']))
        self.offs=offs; total=offs[-1]
        self.p=torch.empty((total,32),dtype=torch.float32,device=self.device); self.y=torch.empty((total,),dtype=torch.float32,device=self.device)
        for i,r in enumerate(rows): self.p[offs[i]:offs[i+1]].copy_(r['p']); self.y[offs[i]:offs[i+1]].copy_(r['y'])
    def value_grad(self,A,need_grad=True):
        total=torch.zeros((),dtype=torch.float64,device=self.device); bsum=total.clone(); regsum=total.clone(); grads=[torch.zeros_like(a) for a in A] if need_grad else None
        for i in range(self.n):
            d=A[self.level[i]] if self.bias else self.h[i]@A[self.level[i]]
            p=self.p[self.offs[i]:self.offs[i+1]].double(); y=self.y[self.offs[i]:self.offs[i+1]].double()
            z=p@ (self.c[i]+d)
            b=self.gain[i]*(F.softplus(z)-y*z).sum()/self.area[i]; reg=LAMBDA*d.square().sum()/2; total+=b+reg; bsum+=b; regsum+=reg
            if need_grad:
                gd=self.gain[i]*(torch.sigmoid(z)-y)@p/self.area[i]+LAMBDA*d
                if self.bias: grads[self.level[i]].add_(gd/self.n)
                else: grads[self.level[i]].add_(torch.outer(self.h[i],gd)/self.n)
        return total/self.n,bsum/self.n,regsum/self.n,grads

def stats(rows):
    return {l:(torch.stack([r['h'] for r in rows if r['level']==l]).double().mean(0),torch.stack([r['h'] for r in rows if r['level']==l]).double().std(0).clamp_min(1e-6)) for l in range(3)}

def solve(rows,bias=False,seg_name='',device='cuda'):
    st=None if bias else stats(rows); dim=32 if bias else 65
    bank=Bank(rows,st,bias=bias,device=device)
    A=[torch.zeros((dim,),dtype=torch.float64,device=bank.device,requires_grad=True) if bias else torch.zeros((dim,32),dtype=torch.float64,device=bank.device,requires_grad=True) for _ in range(3)]
    opt=torch.optim.LBFGS(A,lr=1.0,max_iter=60,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-12); trace=[]; start=time.monotonic()
    def closure():
        opt.zero_grad(); v,_,_,g=bank.value_grad(A,True)
        for a,gg in zip(A,g): a.grad=gg
        trace.append(float(v)); return v
    opt.step(closure); v,b,r,g=bank.value_grad(A,True)
    out={'objective':float(v),'bce':float(b),'regularizer':float(r),'stationarity_norm':max(float(x.norm()) for x in g),'iterations':int(opt.state[A[0]].get('n_iter',-1)),'trace':trace,'bias_only':bias,'stats':None if bias else {str(k):(v[0].tolist(),v[1].tolist()) for k,v in st.items()}}
    return [a.detach().cpu() for a in A],out,st

def eval_obj(rows,A,st,bias=False,device='cuda'):
    b=Bank(rows,st,bias=bias,device=device)
    with torch.no_grad(): v,bc,rg,_=b.value_grad([a.to(b.device) for a in A],False)
    return {'objective':float(v),'bce':float(bc),'regularizer':float(rg),'records':len(rows)}

def main(a):
    root=a.root; out=root/'stage1'; out.mkdir(parents=True,exist_ok=True)
    name=a.arm
    if name=='O-GB':
        fit=geom_rows(root/'official_native/fit.pt'); dev=geom_rows(root/'official_native/dev.pt'); bias=False
    elif name=='G-BIAS':
        fit=geom_rows(root/'geometry_cache/fit.pt'); dev=geom_rows(root/'geometry_cache/dev.pt'); bias=True
    elif name=='G-OFF':
        gf=geom_rows(root/'geometry_cache/fit.pt'); gd=geom_rows(root/'geometry_cache/dev.pt')
        fit=official_target_rows(gf,root/'official_cache',root/'data/annotations/instances_train2017.json'); del gf,gd; dev=None; bias=False
    else: raise ValueError(name)
    device=a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(json.dumps({'stage':'solve','arm':name,'fit':len(fit),'dev':None if dev is None else len(dev),'device':device}),flush=True)
    A,info,st=solve(fit,bias,name,device=device)
    if name=='G-OFF':
        gd=geom_rows(root/'geometry_cache/dev.pt'); dev=official_target_rows(gd,root/'official_cache',root/'data/annotations/instances_train2017.json'); del gd
    rec={'fit':info,'original_fit':eval_obj(fit,[torch.zeros_like(x) for x in A],st,bias,device=device),'dev':eval_obj(dev,A,st,bias,device=device),'A':[x.tolist() for x in A], 'stats':info['stats'],'arm':name}
    (out/f'{name}.json').write_text(json.dumps(rec),encoding='utf-8')
    print(json.dumps({'stage':'complete','arm':name,'out':str(out/f'{name}.json')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--arm',required=True,choices=['O-GB','G-OFF','G-BIAS']); p.add_argument('--device',default=None); main(p.parse_args())
