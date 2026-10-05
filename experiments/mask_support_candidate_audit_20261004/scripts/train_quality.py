import argparse,json,numpy as np,pandas as pd,torch
from pathlib import Path
from torch import nn

def main():
 p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--val',required=True);p.add_argument('--out',required=True);a=p.parse_args(); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 tr=pd.read_csv(a.train); va=pd.read_csv(a.val)
 feats=['score','global_rank','box_area_frac','box_aspect','center_x','center_y','mask_area_frac','mask_box_fill','mask_edge_frac','coef_norm','coef_std','mask_overlap_max','coef_cos_max','box_neighbor_count']
 mu=tr[feats].replace([np.inf,-np.inf],np.nan).fillna(0).mean().to_numpy(); sd=tr[feats].replace([np.inf,-np.inf],np.nan).fillna(0).std().to_numpy()+1e-6
 X=torch.tensor(((tr[feats].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy()-mu)/sd),dtype=torch.float32); y=torch.tensor(tr.target_mask_iou.to_numpy(),dtype=torch.float32)[:,None]
 model=nn.Sequential(nn.Linear(len(feats),64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1),nn.Sigmoid()); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
 for epoch in range(300):
  opt.zero_grad(); pred=model(X); loss=nn.functional.mse_loss(pred,y); loss.backward(); opt.step()
 with torch.no_grad(): vp=model(torch.tensor(((va[feats].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy()-mu)/sd),dtype=torch.float32)).numpy().ravel()
 va=va.copy();va['pred_quality']=vp; va['neighbor']=False
 # join neighbor marker from rows if present
 rows=Path(a.val).with_name('rows.csv')
 if rows.exists():
  rr=pd.read_csv(rows); va=va.merge(rr[['image_id','annotation_id','neighbor_count']],on=['image_id','annotation_id'],how='left');va['neighbor']=va.neighbor_count.fillna(0)>0
 grouped=[]
 for key,g in va.groupby(['image_id','annotation_id']):
  s=g.iloc[g.score.to_numpy().argmax()]; q=g.iloc[g.pred_quality.to_numpy().argmax()]; b=g.iloc[g.target_mask_iou.to_numpy().argmax()]
  grouped.append({'image_id':key[0],'annotation_id':key[1],'neighbor':bool(g.neighbor.iloc[0]),'score_iou':float(s.target_mask_iou),'quality_iou':float(q.target_mask_iou),'oracle_iou':float(b.target_mask_iou),'score':float(s.score),'quality_pred':float(q.pred_quality),'n':len(g)})
 res=pd.DataFrame(grouped); res.to_csv(out/'grouped.csv',index=False)
 def stats(g): return {'n':len(g),'score_iou':float(g.score_iou.mean()),'quality_iou':float(g.quality_iou.mean()),'oracle_iou':float(g.oracle_iou.mean()),'quality_gain':float((g.quality_iou-g.score_iou).mean()),'oracle_gap':float((g.oracle_iou-g.score_iou).mean()),'mask75_score':float((g.score_iou>=.75).mean()),'mask75_quality':float((g.quality_iou>=.75).mean())}
 report={'features':feats,'train_rows':len(tr),'val_rows':len(va),'groups':len(res),'train_mse':float(loss.item()),'all':stats(res),'neighbor':stats(res[res.neighbor]) if res.neighbor.any() else None,'nonneighbor':stats(res[~res.neighbor]) if (~res.neighbor).any() else None}
 (out/'REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); torch.save({'state_dict':model.state_dict(),'features':feats,'mean':mu,'std':sd},out/'quality_head.pt'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

