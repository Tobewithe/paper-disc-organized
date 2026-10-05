"""Plot observed development tradeoffs, explicitly not held-out efficacy."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
source=root/'diagnostics/relative_ownership_20260911'
out=root.parent.parent/'refine-logs/coco-structure/figures';out.mkdir(exist_ok=True)
with (source/'development_summary.csv').open() as f:rows=list(csv.DictReader(f))
high={r['arm']:r for r in rows if r['group']=='high'}
metrics=['same_neighbor','background','coverage','mask_iou']
labels=['Neighbor error','Background error','Target coverage','Mask IoU']
colors=['#356989','#B04B36','#7D6AA0']
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9,4.7))
x=np.arange(4)
for i,alpha in enumerate([.25,.5,1.]):
    v=high[f'relative_{alpha}']
    values=[100*(float(v[k])-float(high['initial'][k])) for k in metrics]
    bars=ax.bar(x+(i-1)*.24,values,width=.22,label=f'alpha={alpha}',color=colors[i])
    ax.bar_label(bars,fmt='%+.3f',padding=3,fontsize=8)
ax.axhline(0,color='#444',lw=.8)
ax.set(xticks=x,xticklabels=labels,ylabel='Change from original decoder (percentage points)',ylim=(-1.25,1.3),
       title='DEVELOPMENT ONLY: foreground count preserved, errors redistributed\n160 COCO train images; 426 matched high-ICI instances')
ax.legend(frameon=False,ncol=3,loc='upper left')
ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15)
fig.text(.03,.02,'All three nonzero settings lower mean Mask IoU. The predeclared selection rule chose alpha=0.\nForeground count is exact on the input grid; target coverage and original-image area are not constrained.',fontsize=9)
fig.tight_layout(rect=[0,.10,1,1])
for ext in ['png','svg','pdf']:fig.savefig(out/f'RELATIVE_OWNERSHIP_20260911.{ext}',dpi=180,bbox_inches='tight')
print(out/'RELATIVE_OWNERSHIP_20260911.png')
