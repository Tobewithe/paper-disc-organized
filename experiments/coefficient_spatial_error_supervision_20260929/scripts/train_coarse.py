import argparse,json,math
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from pycocotools.coco import COCO
class Net(nn.Module):
 def __init__(self):
  super().__init__(); self.enc=nn.Sequential(nn.Conv2d(32,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,16,3,padding=1),nn.SiLU(),nn.Flatten()); self.head=nn.Sequential(nn.Linear(64+16*16,256),nn.SiLU(),nn.Linear(256,16)); nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
 def forward(self,h,p): return self.head(torch.cat([h,self.enc(p)],1))
def load(p): return torch.load(p,map_location='cpu',weights_only=True)
def bounds(b): return max(0,min(639,math.floor(float(b[0])))),max(0,min(639,math.floor(float(b[1])))),max(1,min(640,math.ceil(float(b[2])))),max(1,min(640,math.ceil(float(b[3]))))
def prep(rows,coco_path):
 coco=COCO(str(coco_path)); H=[]; P=[]; T=[]; W=[]
 for r in rows:
  H.append(r['h'].float()); p=r['p_true'].float().reshape(32,4,4); P.append(p); T.append((p*r['delta'].float()[:,None,None]).sum(0))
  m=torch.from_numpy(coco.annToMask(coco.anns[int(r['annotation_id'])]).astype('float32'))[None,None]; m=F.interpolate(m,(640,640),mode='bilinear',align_corners=False)[0,0]; bd=(F.max_pool2d(m[None,None],5,1,2)-(-F.max_pool2d((-m)[None,None],5,1,2))).clamp(0,1)[0,0]; x1,y1,x2,y2=bounds(r['box']); w=F.interpolate(bd[y1:y2,x1:x2][None,None],(4,4),mode='bilinear',align_corners=False)[0,0]; w=(1+2*w); W.append(w/w.mean().clamp_min(1e-6))
 return torch.stack(H),torch.stack(P),torch.stack(T).flatten(1),torch.stack(W).flatten(1)
def score(net,d,weighted):
 h,p,t,w=d; net.eval(); out=[]
 with torch.no_grad():
  for i in range(0,len(h),512):
   sl=slice(i,min(i+512,len(h))); y=net(h[sl],p[sl]); ww=w[sl] if weighted else torch.ones_like(w[sl]); mse=((y-t[sl]).square()*ww).mean(1)/(t[sl].square().mean(1).clamp_min(.01)); cos=(y*t[sl]).sum(1)/((y.square().sum(1)+.01)*(t[sl].square().sum(1)+.01)).sqrt(); bw=w[sl]-1; n=bw.sum(1).clamp_min(1e-6); bc=(y*t[sl]*bw).sum(1)/(((y.square()*bw).sum(1)+.01*n)*((t[sl].square()*bw).sum(1)+.01*n)).sqrt(); out.append(torch.stack([mse,cos,bc],1).cpu())
 x=torch.cat(out); return {'mse':float(x[:,0].mean()),'cos':float(x[:,1].mean()),'boundary_cos':float(x[:,2].mean())}
def main(a):
 torch.manual_seed(a.seed); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); fit=load(a.fit); dev=load(a.dev); hm=torch.stack([r['h'].float() for r in fit]).mean(0); hs=torch.stack([r['h'].float() for r in fit]).std(0).clamp_min(.01); hp,pp,tp,wp=prep(fit,a.coco); hd,pd,td,wd=prep(dev,a.coco); hp=(hp-hm)/hs; hd=(hd-hm)/hs; pm=pp.mean((0,2,3),keepdim=True); ps=pp.std((0,2,3),keepdim=True).clamp_min(.01); pp=(pp-pm)/ps; pd=(pd-pm)/ps; fitd=tuple(x.cuda() for x in (hp,pp,tp,wp)); devd=tuple(x.cuda() for x in (hd,pd,td,wd)); init=Net().state_dict(); result={}
 for arm,weighted in [('uniform',False),('boundary_weighted',True)]:
  net=Net().cuda(); net.load_state_dict(init); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4); hist=[]
  for e in range(1,a.epochs+1):
   net.train(); order=torch.randperm(len(hp),device='cuda');
   for i in range(0,len(order),512):
    ix=order[i:i+512]; y=net(fitd[0][ix],fitd[1][ix]); t=fitd[2][ix]; w=fitd[3][ix] if weighted else torch.ones_like(fitd[3][ix]); mse=((y-t).square()*w).mean(1)/(t.square().mean(1).clamp_min(.01)); cos=(y*t).sum(1)/((y.square().sum(1)+.01)*(t.square().sum(1)+.01)).sqrt(); loss=(mse+.25*(1-cos)).mean(); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),10); opt.step()
   fs=score(net,fitd,weighted); ds=score(net,devd,weighted); hist.append({'epoch':e,'fit':fs,'dev':ds}); print(json.dumps({'arm':arm,'epoch':e,'dev':ds}),flush=True)
  result[arm]={'history':hist}
 (out/'RESULTS.json').write_text(json.dumps(result,indent=2)); (out/'COMPLETE.json').write_text(json.dumps({'status':'completed','fit':len(fit),'dev':len(dev),'resolution':'4x4','seed':a.seed}))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--fit',required=True); p.add_argument('--dev',required=True); p.add_argument('--coco',required=True); p.add_argument('--out',required=True); p.add_argument('--epochs',type=int,default=12); p.add_argument('--seed',type=int,default=0); main(p.parse_args())
