import json, numpy as np, pandas as pd
from pathlib import Path
p=Path(r'D:\coco_wire'); df=pd.read_csv(p/'rows.csv'); g=df[df.good_candidate_count>0].copy(); g['neighbor']=g.neighbor_count>0
rng=np.random.default_rng(20261004)
def img_boot(sub,col='good_regret',B=3000):
 groups=[x[col].to_numpy(float) for _,x in sub.groupby('image_id')]; vals=[]
 for _ in range(B): vals.append(np.mean(np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])))
 return [float(np.mean(vals)),float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]
out={'rows':len(g),'images':int(g.image_id.nunique()),'good_regret_image_bootstrap_ci':img_boot(g), 'neighbor_image_bootstrap_ci':img_boot(g[g.neighbor]) if g.neighbor.any() else None,'nonneighbor_image_bootstrap_ci':img_boot(g[~g.neighbor]) if (~g.neighbor).any() else None}
# bootstrap difference paired at image level by resampling image IDs per stratum independently
out['neighbor_minus_nonneighbor']=float(g[g.neighbor].good_regret.mean()-g[~g.neighbor].good_regret.mean())
(p/'ANALYSIS_IMAGE_BOOT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))
