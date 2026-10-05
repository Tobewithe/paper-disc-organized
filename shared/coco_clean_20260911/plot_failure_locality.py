"""Plot diagnostic scope explicitly, separate from deployable task AP."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--figure',type=Path,required=True);a=ap.parse_args()
    data=json.loads((a.out/'LOCALITY_ANALYSIS.json').read_text());row=next(r for r in data['groups'] if r['split']=='val' and r['group']=='high_fail75')
    with (a.out/'readout_aggregate.csv').open() as f:agg=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');ax=axes[0];bottom=np.zeros(3)
    regions=['same_neighbor','background','fn_inside'];colors=['#a8cfeb','#477cbb','#164578']
    for band,color,label in zip(['near','middle','far'],colors,['Distance <= 0.02','0.02 < distance <= 0.05','Distance > 0.05']):
        values=np.array([row[r+'_pooled_distance_fractions'][band]*100 for r in regions]);ax.bar(np.arange(3),values,bottom=bottom,color=color,label=label);bottom+=values
    ax.set_xticks(range(3),['Same-class FP','Background FP','Inside-crop FN']);ax.set_ylim(0,100);ax.set_ylabel('Share of error pixels (%)');ax.set_title(f'Original high-ICI failures (n={row["n"]})');ax.legend(fontsize=8,loc='upper right')
    ax.text(.5,-.23,'Distance to target boundary / sqrt(GT area)\nPooled pixels; fixed normal crop; original resolution',ha='center',transform=ax.transAxes,fontsize=8)
    ax=axes[1];arms=['original','threshold_oracle','proto_oracle','local_oracle','coordinate_oracle','shuffle_oracle'];x=np.arange(6)
    for offset,group,color in [(-.18,'low','#9db9c6'),(.18,'high','#256b8e')]:
        selected=[next(r for r in agg if r['split']=='val' and r['group']==group and r['arm']==arm) for arm in arms]
        ax.bar(x+offset,[100*float(r['sample_iou']) for r in selected],width=.35,color=color,label=f'{group} ICI (n={selected[0]["n"]})')
    ax.set_xticks(x,['Original','Threshold','Global P','Local P','XY','Shuffled'],rotation=25,ha='right');ax.set_ylabel('Held-out sampled IoU (%)');ax.set_ylim(0,100);ax.set_title('GT-assisted readout opportunities');ax.legend(fontsize=8)
    ax.text(.5,-.23,'Conditional input-crop pixels; spatial cross-fitting\nGT oracle: not AP, not a deployable method',ha='center',transform=ax.transAxes,fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    a.figure.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg','pdf']:fig.savefig(a.figure.with_suffix('.'+ext),dpi=180)


if __name__=='__main__':main()
