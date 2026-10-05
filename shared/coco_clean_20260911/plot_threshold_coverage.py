"""Render the descriptive high-ICI threshold tradeoff; uses saved data only."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
source=root/'diagnostics/threshold_coverage_curve_20260911'
output=root.parent.parent/'refine-logs/coco-evaluation/figures'
output.mkdir(parents=True,exist_ok=True)
with (source/'curve.csv').open() as f:
    curve=[r for r in csv.DictReader(f) if r['group']=='high']
curve=sorted(curve,key=lambda r:float(r['coverage']))
analysis=json.loads((source/'DESCRIPTIVE_ANALYSIS.json').read_text())
rows=[r for r in analysis['results'] if r['group']=='high']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
colors={'bce_global':'#177e89','dice_global':'#d95d39'}
labels={'bce_global':'BCE residual','dice_global':'BCE + Dice residual'}
for ax,metric,title in zip(axes,['same_neighbor','background','mask_iou'],['Same-class neighbor error / GT','Background error / GT','Mean attributed mask IoU']):
    ax.plot([float(r['coverage'])*100 for r in curve],[float(r[metric])*100 for r in curve],color='#555555',linewidth=1.8,label='Original model: threshold sweep')
    for row in rows:
        x=row['learned_metrics_pct']['coverage'];y=row['learned_metrics_pct'][metric];control=row['interpolated_control_metrics_pct'][metric]
        ax.plot([x,x],[control,y],color=colors[row['family']],linestyle=':',linewidth=1.3)
        ax.scatter([x],[control],marker='o',facecolor='white',edgecolor=colors[row['family']],s=40,zorder=4)
        ax.scatter([x],[y],marker='D',color=colors[row['family']],s=42,zorder=5,label=labels[row['family']])
    ax.set_xlim(81,89);ax.set_xlabel('Mean target coverage (%)');ax.set_ylabel(title+' (%)');ax.grid(alpha=.18)
axes[0].set_ylim(8,16);axes[1].set_ylim(8,16);axes[2].set_ylim(70.2,72.2)
axes[0].legend(loc='upper left',fontsize=8)
fig.suptitle('COCO: 475 high-ICI matched instances in 600 evaluation images',fontsize=13)
fig.supxlabel('Descriptive evaluation-GT coverage matching; hollow dots are interpolated controls, not deployed predictions.',fontsize=9)
for extension in ['png','svg','pdf']:
    fig.savefig(output/f'THRESHOLD_COVERAGE_20260911.{extension}',dpi=180)
print(output/'THRESHOLD_COVERAGE_20260911.png')
