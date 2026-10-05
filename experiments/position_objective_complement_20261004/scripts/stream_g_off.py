"""Memory-bounded exact G-OFF solve: geometry raw ids, official ROI objective."""
import argparse,json,time,gc
from pathlib import Path
import torch, torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops
LAMBDA=.003; GAIN=9.83241

def meta_rows(path):
    raw=torch.load(path,map_location='cpu',weights_only=False); out=[]
    for r in raw:
        m=r['meta']; out.append({'image_id':int(m['image_id']),'annotation_id':int(m['annotation_id']),'raw_id':int(m['raw_id']),'level':int(m['level']),'h':r['h'].double(),'c':r['c'].double(),'box_iou':float(m.get('box_iou',float('nan')))})
    del raw; gc.collect(); return out
def stats(rows):
    return {l:(torch.stack([r['h'] for r in rows if r['level']==l]).mean(0),torch.stack([r['h'] for r in rows if r['level']==l]).std(0).clamp_min(1e-6)) for l in range(3)}
class Stream:
    def __init__(self,rows,cache,ann_json,stats,device):
        self.rows=rows; self.by={}
        for r in rows:self.by.setdefault(r['image_id'],[]).append(r)
        self.cache=cache; self.coco=COCO(str(ann_json)); self.st=stats; self.device=torch.device(device); self.n=len(rows)
        self.h=torch.stack([torch.cat(((r['h']-stats[r['level']][0])/stats[r['level']][1],torch.ones(1,dtype=torch.float64))) for r in rows])
    def _image(self,iid):
        im=torch.load(self.cache/'images'/f'{iid:012d}.pt',map_location='cpu',weights_only=False)
        up=F.interpolate(im['proto'].float()[None],(640,640),mode='bilinear',align_corners=False)[0].to(self.device)
        masks=im['masks'].float().to(self.device); ids=list(im['all_annotation_ids']); (gx,gy),(px,py)=im['ratio_pad']; gain=float(im['segmentation_gain'])
        dat=[]
        for r in self.by[iid]:
            ann=self.coco.anns[r['annotation_id']]; x,y,w,h=[float(v) for v in ann['bbox']]; box=torch.tensor([x*float(gx)+float(px),y*float(gy)+float(py),(x+w)*float(gx)+float(px),(y+h)*float(gy)+float(py)],device=self.device)
            support=ops.crop_mask(torch.ones((1,640,640),device=self.device),box[None])[0].bool(); oi=ids.index(r['annotation_id']); p=up[:,support].T.double(); yy=(masks[support]==oi+1).double(); area=float(((box[2:]-box[:2])/640).prod()*640*640)
            dat.append((r,p,yy,area,gain))
        return im,dat
    @torch.no_grad()
    def value_grad(self,A,need=True):
        total=torch.zeros((),dtype=torch.float64,device=self.device); bsum=total.clone(); rsum=total.clone(); grads=[torch.zeros_like(a) for a in A] if need else None
        for iid in self.by:
            im,dat=self._image(iid)
            for r,p,y,area,gain in dat:
                lev=r['level']
                hh=torch.cat(((r['h']-self.st[lev][0])/self.st[lev][1],torch.ones(1,dtype=torch.float64))).to(self.device)
                d=hh@A[lev]; z=p@(r['c'].to(self.device)+d); b=gain*(F.softplus(z)-y*z).sum()/area; reg=LAMBDA*d.square().sum()/2; total+=b+reg; bsum+=b; rsum+=reg
                if need:
                    gd=gain*(torch.sigmoid(z)-y)@p/area+LAMBDA*d; grads[lev].add_(torch.outer(hh,gd)/self.n)
            del im,dat
        return total/self.n,bsum/self.n,rsum/self.n,grads
def solve(rows,cache,ann,device):
    st=stats(rows); bank=Stream(rows,cache,ann,st,device); A=[torch.zeros((65,32),dtype=torch.float64,device=bank.device,requires_grad=True) for _ in range(3)]; trace=[]; start=time.monotonic(); opt=torch.optim.LBFGS(A,lr=1.,max_iter=25,line_search_fn='strong_wolfe',tolerance_grad=1e-7,tolerance_change=1e-11)
    def closure():
        opt.zero_grad(); v,_,_,g=bank.value_grad(A,True)
        for a,gg in zip(A,g):a.grad=gg
        trace.append(float(v)); print(json.dumps({'closure':len(trace),'objective':float(v),'elapsed_s':time.monotonic()-start}),flush=True); return v
    opt.step(closure); v,b,rg,g=bank.value_grad(A,True)
    z=[torch.zeros_like(x) for x in A]; v0,b0,r0,_=bank.value_grad(z,False)
    return {'fit':{'objective':float(v),'bce':float(b),'regularizer':float(rg),'stationarity_norm':max(float(x.norm()) for x in g),'iterations':int(opt.state[A[0]].get('n_iter',-1)),'trace':trace,'stats':{str(k):(v[0].tolist(),v[1].tolist()) for k,v in st.items()},'device':str(device)},'original_fit':{'objective':float(v0),'bce':float(b0),'regularizer':float(r0),'records':bank.n},'A':[x.detach().cpu().tolist() for x in A],'stats':{str(k):(v[0].tolist(),v[1].tolist()) for k,v in st.items()},'arm':'G-OFF','records':bank.n}
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--device',default='cuda'); a=ap.parse_args(); root=a.root; rows=meta_rows(root/'geometry_cache/fit.pt'); rec=solve(rows,root/'official_cache',root/'data/annotations/instances_train2017.json',a.device); (root/'stage1'/'G-OFF.json').write_text(json.dumps(rec),encoding='utf-8'); print(json.dumps({'stage':'complete','arm':'G-OFF'}),flush=True)
