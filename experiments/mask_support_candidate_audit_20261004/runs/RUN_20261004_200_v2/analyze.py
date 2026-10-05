import json, numpy as np, pandas as pd
from pathlib import Path
p=Path(r'D:\\coco_wire')
df=pd.read_csv(p/'rows.csv'); g=df[df.good_candidate_count>0].copy(); g['neighbor']=g.neighbor_count>0
rng=np.random.default_rng(20261004)
def ci(x, stat, B=2000):
 x=np.asarray(x,float); vals=[]
 for _ in range(B): vals.append(stat(x[rng.integers(0,len(x),len(x))]))
 return [float(np.mean(vals)),float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]
summary=json.loads((p/'SUMMARY.json').read_text())
out={'rows_all':len(df),'rows_good':len(g),'box_good_fraction':len(g)/len(df),'good_regret_mean_ci':ci(g.good_regret, np.mean),'score_good_iou_mean_ci':ci(g.score_good_iou,np.mean),'best_good_iou_mean_ci':ci(g.best_good_iou,np.mean),'good_regret_gt01':float((g.good_regret>.01).mean()),'good_regret_gt02':float((g.good_regret>.02).mean()),'mask75_score':float((g.score_good_iou>=.75).mean()),'mask75_best':float((g.best_good_iou>=.75).mean()),'mask75_repair_rate':float(((g.score_good_iou<.75)&(g.best_good_iou>=.75)).mean())}
for name,sub in [('neighbor',g[g.neighbor]),('nonneighbor',g[~g.neighbor])]:
 out[name+'_n']=len(sub); out[name+'_regret_ci']=ci(sub.good_regret,np.mean) if len(sub) else None; out[name+'_score_iou']=float(sub.score_good_iou.mean()) if len(sub) else None; out[name+'_best_iou']=float(sub.best_good_iou.mean()) if len(sub) else None; out[name+'_mask75_repair']=float(((sub.score_good_iou<.75)&(sub.best_good_iou>=.75)).mean()) if len(sub) else None
(p/'ANALYSIS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))

