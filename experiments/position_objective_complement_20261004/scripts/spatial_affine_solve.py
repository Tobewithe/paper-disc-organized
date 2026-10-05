import argparse,json,time
from pathlib import Path
import torch
from position_solve import geom_rows,Bank,stats,eval_obj
def addq(rows):
 sizes=[(80,80),(40,40),(20,20)]; offs=[0,6400,8000]
 for r in rows:
  l=r['level']; w,h=sizes[l]; k=r['raw_id']-offs[l]; x=(k%w)/(w-1)*2-1; y=(k//w)/(h-1)*2-1; q=torch.tensor([1.,x,y,x*x,y*y,x*y],dtype=torch.float64); r['h']=torch.cat((r['h'].double(),q))
 return rows
def main(a):
 root=a.root; fit=addq(geom_rows(root/'geometry_cache/fit.pt')); dev=addq(geom_rows(root/'geometry_cache/dev.pt')); val=addq(geom_rows(root/'geometry_val/val.pt')); st=stats(fit); bank=Bank(fit,st,bias=False,device=a.device); A=[torch.zeros((71,32),dtype=torch.float64,device=a.device,requires_grad=True) for _ in range(3)]; trace=[]; start=time.monotonic(); opt=torch.optim.LBFGS(A,lr=1.,max_iter=40,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-12)
 def closure():
  opt.zero_grad(); v,_,_,g=bank.value_grad(A,True)
  for x,y in zip(A,g): x.grad=y
  trace.append(float(v)); return v
 opt.step(closure); v,b,r,g=bank.value_grad(A,True); rec={'arm':'SPATIAL_AFFINE','fit':{'objective':float(v),'bce':float(b),'regularizer':float(r),'stationarity_norm':max(float(x.norm()) for x in g),'iterations':int(opt.state[A[0]].get('n_iter',-1)),'trace':trace},'original_fit':eval_obj(fit,[torch.zeros_like(x) for x in A],st,False,a.device),'dev':eval_obj(dev,[x.detach() for x in A],st,False,a.device),'A':[x.detach().cpu().tolist() for x in A],'stats':{str(k):(v[0].tolist(),v[1].tolist()) for k,v in st.items()}}
 out=root/'spatial_affine'; out.mkdir(exist_ok=True); (out/'SOLVE.json').write_text(json.dumps(rec),encoding='utf-8'); torch.save({'A':[x.detach().cpu() for x in A],'stats':st},out/'PARAMS.pt'); print(json.dumps({'stage':'complete','fit':rec['fit'],'dev':rec['dev']}))
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--device',default='cuda'); a=ap.parse_args(); main(a)
