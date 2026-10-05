"""Draw the completed preservation comparison and matched pipeline timing."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser()
    for k in ('source','timing','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    r=json.loads((a.source/'RESULTS.json').read_text());t=json.loads(a.timing.read_text())
    fig,axs=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    names=['local4_s0','local4_guard_s0','local4_safe_s0'];labels=['Plain (0.5)','Pixel guard (1.0)','Quality guard (1.0)']
    for n,label,color in zip(names,labels,['#64748b','#2563eb','#14b8a6']):
        axs[0].scatter(r[n]['damaged75'],r[n]['repaired75'],s=80,color=color)
        axs[0].annotate(label,(r[n]['damaged75'],r[n]['repaired75']),xytext=(6,7),textcoords='offset points',fontsize=9)
    xs=[r[n]['damaged75'] for n in names];ys=[r[n]['repaired75'] for n in names]
    axs[0].set(xlabel='Damaged GT (fewer is better)',ylabel='Repaired GT (more is better)',title='A. Repair / damage trade-off (seed 0)')
    axs[0].set_xlim(min(xs)-80,max(xs)+280);axs[0].set_ylim(min(ys)-80,max(ys)+100)
    for mode,label,color in [('scalar','Scalar','#94a3b8'),('local4','Local only','#14b8a6'),('coeff_local4','Combined','#2563eb')]:
        axs[1].plot(range(3),[100*r[f'{mode}_s{s}']['metrics']['AP'] for s in range(3)],'o-',label=label,color=color)
    axs[1].set(xlabel='Seed',ylabel='Mask AP',title='B. Local-only replication',xticks=[0,1,2]);axs[1].legend(frameon=False,fontsize=9)
    timing=t['summary'];keys=['baseline','scalar','local4','coeff_local4']
    vals=[timing[k]['total_ms']['mean'] for k in keys]
    bars=axs[2].bar(['Baseline','Scalar','Local','Combined'],vals,color=['#cbd5e1','#94a3b8','#14b8a6','#2563eb'])
    axs[2].bar_label(bars,fmt='%.1f',padding=4);axs[2].set_ylim(0,max(vals)*1.18)
    axs[2].set(ylabel='Milliseconds per image',title='C. Eager FP32 pipeline time')
    for ax in axs:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('COCO val2017 | AP: all 5,000 images | timing: 50 images x 3 repeats, RTX 4090',fontsize=12)
    for ext in ('png','pdf','svg'):fig.savefig(a.out/f'preservation_results.{ext}',dpi=180)
    plt.close(fig)
    (a.out/'COMPLETE.json').write_text(json.dumps({'figure':'preservation_results','timing_scope':t['note']}))


if __name__=='__main__':main()
