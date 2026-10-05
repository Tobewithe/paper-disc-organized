"""Plot fixed high-ICI paired effects with image-cluster pointwise intervals."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE/'diagnostics/crossimage_response_20260911'
DEST=HERE.parents[1]/'refine-logs/coco-structure/figures'


def main():
    obj=json.loads((OUT/'PAIRED_ANALYSIS.json').read_text());rows=obj['contrasts']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
    metrics=[('spatial','coverage','Own coverage'),('spatial','same_neighbor','Neighbor error'),('spatial','background','Background error'),
             ('spatial','mask_iou','Mask IoU'),('task','hit75','GT recall@75'),('pair','hit75','Pair recall@75')]
    for ax,domain,title in zip(axes,['normal','expand20'],['Normal crop','Expand 20% each side']):
        for treatment,control,color,offset,label in [('joint_mean','initial','#0072b2',-.10,'Joint vs original'),('joint_mean','sham_mean','#d55e00',.10,'Joint vs shuffled')]:
            values=[];low=[];high=[]
            for kind,metric,_ in metrics:
                r=next(r for r in rows if r['domain']==domain and r['group']=='high' and r['kind']==kind and r['metric']==metric and r['treatment']==treatment and r['control']==control)
                values.append(r['mean_pp']);low.append(r['mean_pp']-r['ci_low_pp']);high.append(r['ci_high_pp']-r['mean_pp'])
            ax.errorbar(values,np.arange(len(metrics))+offset,xerr=[low,high],fmt='o',capsize=3,color=color,label=label)
        ax.axvline(0,color='#777777',lw=1);ax.grid(axis='y',alpha=.15);ax.set_title(title);ax.set_xlabel('Difference (percentage points)')
        ax.set_yticks(np.arange(len(metrics)),[m[2] for m in metrics]);ax.invert_yaxis()
    fig.suptitle('Frozen cross-image response readout: observed-seed means, high ICI',fontsize=12)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    fig.subplots_adjust(left=.16,right=.98,top=.84,bottom=.22,wspace=.18)
    DEST.mkdir(exist_ok=True)
    for ext in ['png','svg','pdf']:fig.savefig(DEST/f'CROSSIMAGE_RESPONSE_20260911.{ext}',dpi=180)
    print(DEST/'CROSSIMAGE_RESPONSE_20260911.png')


if __name__=='__main__':main()
