"""Standalone publication-oriented figures from completed results only."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    r=json.loads((a.source/'RESULTS.json').read_text());d=json.loads(a.analysis.read_text())
    base=100*r['baseline']['metrics']['AP']
    fig,axs=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    names=['scalar_s0','coeff_s0','local4_s0','coeff_local4_s0'];labels=['Scalar','Coefficient','Local 4x4','Combined']
    vals=[100*r[k]['metrics']['AP']-base for k in names]
    bars=axs[0].bar(labels,vals,color=['#9ca3af','#60a5fa','#2dd4bf','#2563eb'])
    axs[0].bar_label(bars,fmt='%+.3f',padding=4,fontsize=10)
    axs[0].set(title='A. Trained components (seed 0)',ylabel='Mask AP gain over frozen YOLO (points)')
    axs[0].axhline(0,color='#555',lw=.8);axs[0].set_ylim(min(0,min(vals)-.15),max(vals)+.22)
    axs[0].tick_params(axis='x',labelsize=9)
    for mode,color,label in [('scalar','#777','Scalar'),('coeff_local4','#2563eb','Combined')]:
        y=np.array([100*r[f'{mode}_s{s}']['metrics']['AP'] for s in range(3)])
        axs[1].plot(range(3),y,'o-',label=label,color=color,lw=2)
    axs[1].axhline(base,color='#aaa',ls='--',label='Frozen YOLO')
    axs[1].set(title='B. Paired training seeds',xlabel='Seed',ylabel='Mask AP',xticks=[0,1,2]);axs[1].legend(frameon=False,fontsize=9)
    repair=[r[f'coeff_local4_s{s}']['repaired75'] for s in range(3)]
    damage=[r[f'coeff_local4_s{s}']['damaged75'] for s in range(3)]
    x=np.arange(3);axs[2].bar(x-.16,repair,.32,color='#2563eb',label='Repaired')
    axs[2].bar(x+.16,damage,.32,color='#f59e0b',label='Damaged')
    for i,(rr,dd) in enumerate(zip(repair,damage)):axs[2].text(i,max(rr,dd)+30,f'Net {rr-dd:+d}',ha='center',fontsize=10)
    axs[2].set(title='C. Combined correction at Mask75',xlabel='Seed',ylabel='GT instances',xticks=x)
    axs[2].set_ylim(0,max(repair+damage)*1.35);axs[2].legend(frameon=False,fontsize=9,ncol=2,loc='upper left')
    for ax in axs:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('COCO val2017: 5,000 images | 800 fit + 200 train-only selection | epoch 8',fontsize=12)
    for suffix in ('png','pdf','svg'):fig.savefig(a.out/f'component_seed_results.{suffix}',dpi=180)
    plt.close(fig)
    (a.out/'FIGURE_SCOPE.json').write_text(json.dumps(dict(source=str(a.source),analysis=str(a.analysis),
        note='Three seed replicates share one train split. No AP confidence intervals plotted. All strengths selected within train2017.'),indent=2))


if __name__=='__main__':main()
