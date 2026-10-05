"""Describe fixed-cohort S018 objective/metric disagreements, no parameter tuning."""
import csv, hashlib, json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent/'diagnostics'
out=BASE/'official_loss_readout887_v2_20260912'
with open(out/'targets.csv',newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
assert len(rows)==887 and len({r['annotation_id'] for r in rows})==887
ids=sorted({int(r['image_id']) for r in rows});index={v:i for i,v in enumerate(ids)}
rng=np.random.default_rng(20260912)
weights=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
summary=[]
for density in ['high','other']:
    rr=[r for r in rows if r['density']==density]
    denom=np.zeros(len(ids));numer=np.zeros((len(ids),2))
    for r in rr:
        j=index[int(r['image_id'])];denom[j]+=1
        numer[j,0]+=float(r['official_BCE160_GTbox_coco_iou'])-float(r['original_coco_iou'])
        numer[j,1]+=float(r['previous_fullinput_oracle_coco_iou'])-float(r['official_BCE160_GTbox_coco_iou'])
    bden=weights@denom;valid=bden>0;bnum=weights@numer
    loss_decreased=[];metric_worse=[];opposite=[]
    for r in rr:
        lower=float(r['official_loss_after'])<float(r['official_loss_before'])-1e-6
        worse=float(r['official_BCE160_GTbox_coco_iou'])<float(r['original_coco_iou'])-1e-6
        if lower:loss_decreased.append(r)
        if worse:metric_worse.append(r)
        if lower and worse:opposite.append(r)
    summary.append(dict(density=density,n=len(rr),official_loss_decreased=len(loss_decreased),IoU_worsened=len(metric_worse),
        loss_decreased_but_IoU_worsened=len(opposite),disagreements_before_iteration_limit=sum(int(r['iterations'])<120 for r in opposite),
        solver_hit_iteration_limit=sum(int(r['iterations'])>=120 for r in rr),
        mean_loss_before=float(np.mean([float(r['official_loss_before']) for r in rr])),
        mean_loss_after=float(np.mean([float(r['official_loss_after']) for r in rr])),
        mean_old_oracle_loss=float(np.mean([float(r['previous_oracle_official_loss']) for r in rr])),
        official_gain_pp=100*float(numer[:,0].sum()/denom.sum()),official_gain_ci95_pp=(100*np.quantile(bnum[valid,0]/bden[valid],[.025,.975])).tolist(),
        old_oracle_minus_official_pp=100*float(numer[:,1].sum()/denom.sum()),old_oracle_minus_official_ci95_pp=(100*np.quantile(bnum[valid,1]/bden[valid],[.025,.975])).tolist()))
doc=dict(groups=summary,bootstrap='2000 image-cluster resamples, same weights for arms; conditional explored fixed-failure cohort.',
    interpretation='Finite optimization trajectory exposes objective/metric disagreement. Original and full-input oracle differ in mask grid, crop support, crowd domain, objective, regularization, and GT-IoU state selection; cannot attribute their gap to one factor.',
    csv_sha256=hashlib.sha256((out/'targets.csv').read_bytes()).hexdigest())
(out/'OBJECTIVE_METRIC_ANALYSIS.json').write_text(json.dumps(doc,indent=2),encoding='utf-8')
print(json.dumps(doc,indent=2))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
groups=json.loads((out/'ANALYSIS.json').read_text())['groups']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
labels=['Original','Official loss\ncoefficient oracle','Full-input\ncoefficient oracle']
colors=['#78889b','#d5973e','#27818d']
for ax,density,title in zip(axs,['high','other'],['High crowding (215 failed instances)','Other group (672 failed instances)']):
    rr=[r for r in groups if r['density']==density]
    vals=[r['mean_coco_iou'] for r in rr]
    ax.bar(np.arange(3),vals,color=colors,width=.62)
    for i,(v,r) in enumerate(zip(vals,rr)):
        ax.text(i,v+1.1,f'{v:.2f}',ha='center',weight='bold')
        ax.text(i,3,f"IoU >= .75\n{r['recovered75']}/{r['n']}",ha='center',va='bottom',fontsize=9,color='white')
    ax.set_xticks(np.arange(3),labels);ax.set_ylim(0,100);ax.set_title(title,fontsize=11)
    ax.spines[['top','right']].set_visible(False);ax.set_ylabel('Mean mask IoU (%)')
fig.suptitle('Fixed prototypes and prediction boxes; same-image GT-assisted fitting',fontsize=12)
fig.savefig(out/'OFFICIAL_LOSS_COMPARISON.png',dpi=180)
fig.savefig(out/'OFFICIAL_LOSS_COMPARISON.svg')
plt.close(fig)
