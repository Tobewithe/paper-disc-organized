import argparse,json,numpy as np,pandas as pd,torch
from pathlib import Path
from torch import nn

def main():
 p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--val',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 tr=pd.read_csv(a.train);va=pd.read_csv(a.val); feats=['score','global_rank','box_area_frac','box_aspect','center_x','center_y','mask_area_frac','mask_box_fill','mask_edge_frac','coef_norm','coef_std','mask_overlap_max','coef_cos_max','box_neighbor_count']
 clean=lambda d:d[feats].replace([np.inf,-np.inf],np.nan).fillna(0)
 mu=clean(tr).mean().to_numpy();sd=clean(tr).std().to_numpy()+1e-6
 X=clean(tr).to_numpy(); X=(X-mu)/sd; y=tr.target_mask_iou.to_numpy()
 model=nn.Sequential(nn.Linear(len(feats),64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1));opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
 rng=np.random.default_rng(0); groups=[g for _,g in tr.groupby(['image_id','annotation_id']) if len(g)>=2]
 for ep in range(250):
  losses=[]
  for g in groups:
   ii=g.index.to_numpy(); yy=g.target_mask_iou.to_numpy(); pairs=[]
   for i in range(len(ii)):
    for j in range(i+1,len(ii)):
     d=yy[i]-yy[j]
     if abs(d)>=.01:pairs.append((ii[i],ii[j],1 if d>0 else -1))
   if not pairs:continue
   if len(pairs)>40:pairs=[pairs[k] for k in rng.choice(len(pairs),40,replace=False)]
   ia=np.array([q[0] for q in pairs]);ib=np.array([q[1] for q in pairs]);sg=np.array([q[2] for q in pairs],dtype=np.float32)
   pa=model(torch.tensor(X[ia],dtype=torch.float32)).squeeze(1);pb=model(torch.tensor(X[ib],dtype=torch.float32)).squeeze(1)
   loss=nn.functional.softplus(-torch.tensor(sg)* (pa-pb)).mean();opt.zero_grad();loss.backward();opt.step();losses.append(float(loss))
  if ep%50==0: print(ep,float(np.mean(losses)))
 xv=(clean(va).to_numpy()-mu)/sd
 with torch.no_grad(): va=va.copy();va['pred_quality']=model(torch.tensor(xv,dtype=torch.float32)).numpy().ravel()
 grouped=[]
 for key,g in va.groupby(['image_id','annotation_id']):
  s=g.iloc[g.score.to_numpy().argmax()];q=g.iloc[g.pred_quality.to_numpy().argmax()];b=g.iloc[g.target_mask_iou.to_numpy().argmax()]
  grouped.append({'image_id':key[0],'annotation_id':key[1],'score_iou':float(s.target_mask_iou),'quality_iou':float(q.target_mask_iou),'oracle_iou':float(b.target_mask_iou),'score':float(s.score),'quality_pred':float(q.pred_quality),'neighbor':False})
 res=pd.DataFrame(grouped);rows=Path(a.val).with_name('rows.csv')
 if rows.exists(): rr=pd.read_csv(rows);res=res.merge(rr[['image_id','annotation_id','neighbor_count']],on=['image_id','annotation_id'],how='left');res['neighbor']=res.neighbor_count.fillna(0)>0
 def st(g):return {'n':len(g),'score_iou':float(g.score_iou.mean()),'quality_iou':float(g.quality_iou.mean()),'oracle_iou':float(g.oracle_iou.mean()),'gain':float((g.quality_iou-g.score_iou).mean()),'mask75_score':float((g.score_iou>=.75).mean()),'mask75_quality':float((g.quality_iou>=.75).mean())}
 rep={'features':feats,'train_rows':len(tr),'pair_groups':len(groups),'val_groups':len(res),'all':st(res),'neighbor':st(res[res.neighbor]),'nonneighbor':st(res[~res.neighbor])};(out/'REPORT.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2),encoding='utf-8');res.to_csv(out/'grouped.csv',index=False);torch.save({'state_dict':model.state_dict(),'features':feats,'mean':mu,'std':sd},out/'quality_head_pairwise.pt');print(json.dumps(rep,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
