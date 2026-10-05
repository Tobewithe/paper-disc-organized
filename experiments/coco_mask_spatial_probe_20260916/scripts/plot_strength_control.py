"""Plot all declared strengths without selecting a validation winner."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    r=json.loads((a.source/'RESULTS.json').read_text())
    specs=[('local4_s0','Plain, 0.5','#275e9b','o'),
           ('local4_guard_half_s0','Pixel guard, 0.5','#12826e','o'),
           ('local4_guard_s0','Pixel guard, 1.0','#12826e','s'),
           ('local4_safe_half_s0','Quality guard, 0.5','#ad6722','o'),
           ('local4_safe_s0','Quality guard, 1.0','#ad6722','s')]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axs=plt.subplots(1,2,figsize=(11.8,4.7),layout='constrained')
    gain=[100*(r[k]['metrics']['AP']-r['baseline']['metrics']['AP']) for k,_,_,_ in specs]
    bars=axs[0].barh(np.arange(len(specs)),gain,color=[c for _,_,c,_ in specs],height=.62)
    axs[0].set_yticks(np.arange(len(specs)),[label for _,label,_,_ in specs]);axs[0].invert_yaxis()
    axs[0].axvline(0,color='#777777',lw=.7);axs[0].set_xlabel('Mask AP gain over frozen YOLO (points)')
    axs[0].bar_label(bars,labels=[f'{v:+.3f}' for v in gain],padding=4,fontsize=9)
    axs[0].set_xlim(min(0,min(gain)*1.2),max(gain)*1.28)
    axs[0].set_title('(a) Overall mask quality',loc='left',fontweight='bold')
    for k,label,color,marker in specs:
        v=r[k];axs[1].scatter(v['damaged75'],v['repaired75'],c=color,marker=marker,s=70,label=label)
        axs[1].annotate(f"{v['repaired75']} / {v['damaged75']}",(v['damaged75'],v['repaired75']),
                        xytext=(5,7),textcoords='offset points',fontsize=8,color=color)
    for obj,col in [('guard','#12826e'),('safe','#ad6722')]:
        vs=[r[f'local4_{obj}_half_s0'],r[f'local4_{obj}_s0']]
        axs[1].plot([v['damaged75'] for v in vs],[v['repaired75'] for v in vs],c=col,ls='--',alpha=.45)
    axs[1].set_xlabel('Newly failed GT at Mask75 (fewer is better)')
    axs[1].set_ylabel('Newly recovered GT at Mask75 (more is better)')
    axs[1].set_title('(b) Repair / damage tradeoff',loc='left',fontweight='bold')
    axs[1].margins(x=.22,y=.20);axs[1].legend(fontsize=8,loc='lower right',frameon=False)
    fig.suptitle('COCO val2017: frozen detector, same seed and checkpoint, explicit correction strength',fontsize=11)
    for ext in ('png','pdf','svg'):fig.savefig(a.out/f'strength_controls.{ext}',dpi=200)
    (a.out/'COMPLETE.json').write_text(json.dumps({'source':str(a.source),'variants':[k for k,_,_,_ in specs]}))


if __name__=='__main__':main()
