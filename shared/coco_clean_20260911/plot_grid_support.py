"""Plot completed fixed-cohort S019 readouts; no result-dependent exclusions."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
out=a.out;data=json.loads((out/'ANALYSIS.json').read_text());stats=json.loads((out/'FACTORIAL_ANALYSIS.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
arms=['original','A160_GT','B160_PRED','C640_GT','D640_PRED']
labels=['Original','160 / GT','160 / Pred','640 / GT','640 / Pred']
colors=['#8592a3','#e1b268','#b67c34','#69aebb','#287e8e']
fig,axs=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
for ax,density,title in zip(axs,['high','other'],['High crowding','Other group']):
    rows=[next(r for r in data['groups'] if r['density']==density and r['arm']==arm) for arm in arms]
    vals=[r['mean_coco_iou'] for r in rows]
    ax.bar(np.arange(5),vals,color=colors,width=.7)
    for i,(v,r) in enumerate(zip(vals,rows)):
        ax.text(i,v+1,f'{v:.2f}',ha='center',weight='bold',fontsize=10)
        ax.text(i,3,f"{r['recovered75']} / {r['n']}",ha='center',va='bottom',color='white',fontsize=8)
    ax.set_xticks(np.arange(5),labels,rotation=15);ax.set_ylim(0,100);ax.set_ylabel('Mean original-COCO mask IoU (%)')
    ax.set_title(f"{title}: {rows[0]['n']} fixed failed instances",fontsize=11);ax.spines[['top','right']].set_visible(False)
fig.suptitle('2 x 2 supervision probe: grid size / loss crop\nFrozen prototypes & prediction boxes; same-image GT-assisted coefficient fitting',fontsize=12)
fig.savefig(out/'GRID_SUPPORT_COMPARISON.png',dpi=180);fig.savefig(out/'GRID_SUPPORT_COMPARISON.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.5),constrained_layout=True)
names=['grid_GT','grid_PRED','support_160','support_640','both_D_minus_A']
text=['Higher grid, GT crop','Higher grid, prediction crop','Prediction crop, grid160','Prediction crop, grid640','Both changes (D - A)']
for density,color,offset in [('high','#287e8e',-.13),('other','#b67c34',.13)]:
    rows=[next(r for r in stats['contrasts'] if r['density']==density and r['contrast']==name and r['metric']=='coco_iou') for name in names]
    values=np.array([r['mean_pp'] for r in rows]);ci=np.array([r['ci95_pp'] for r in rows]);y=np.arange(len(names))+offset
    ax.errorbar(values,y,xerr=np.stack([values-ci[:,0],ci[:,1]-values]),fmt='o',color=color,capsize=3,label=density)
ax.axvline(0,color='#a8a8a8',lw=1);ax.set_yticks(np.arange(len(names)),text);ax.invert_yaxis();ax.set_xlabel('Paired mask IoU difference (percentage points)')
ax.set_title('Pointwise 95% image-cluster bootstrap intervals\nExploratory fixed failure cohort; no density-causal interpretation',fontsize=11)
ax.spines[['top','right']].set_visible(False);ax.legend()
fig.savefig(out/'GRID_SUPPORT_EFFECTS.png',dpi=180);fig.savefig(out/'GRID_SUPPORT_EFFECTS.svg');plt.close(fig)
