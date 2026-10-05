"""Static scientific plots from saved S048/S049 tables only."""
import os,sys
from pathlib import Path
# Launching conda Python directly does not activate its native DLL search path.
conda_bin=Path(sys.prefix)/'Library/bin'
if os.name=='nt' and conda_bin.is_dir():
    os.environ['PATH']=str(conda_bin)+os.pathsep+os.environ.get('PATH','')
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent/'diagnostics'
a=json.loads((ROOT/'mask_geometry_failure_census_20260912/ANALYSIS.json').read_text())
b=json.loads((ROOT/'crowded_failure_branch_20260912/ANALYSIS.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(13,5))
groups=['low','middle','high'];colors=['#6baed6','#fdae6b','#e6550d','#969696','#74c476']
states=['box_good__mask_good','box_good__mask_bad','box_bad__mask_bad','no_bbox50_match','box_bad__mask_good']
labels=['Box good / mask good','Box good / mask bad','Box bad / mask bad','No bbox50 match','Box bad / mask good']
bottom=np.zeros(3)
for state,label,color in zip(states,labels,colors):
    v=[]
    for group in groups:
        r=next(x for x in a['groups'] if x['grouping']=='mask_density' and x['group']==group)
        v.append(100*r['states'].get(state,0)/r['gt'])
    axes[0].bar(groups,v,bottom=bottom,label=label,color=color);bottom+=v
axes[0].set_ylabel('Share of GT instances (%)');axes[0].set_title('S048: same-prediction failure classes\nGT boundary exposure at 4 input pixels')
axes[0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2)
names=['c1p0','c0p1','c1p1'];yy=np.arange(3)
for fill,offset,color,label in [('texture',-.1,'#1f77b4','Texture fill'),('local_color',.1,'#d95f02','Local color fill')]:
    mean=[];lo=[];hi=[]
    for name in names:
        q=next(r for r in b['contrasts'] if r['group']=='same_failure' and r['fill']==fill and r['combination']==name and r['contrast']=='neighbor_minus_control')['metrics']['mask_iou']
        mean.append(q['mean']*100);lo.append(q['ci95'][0]*100);hi.append(q['ci95'][1]*100)
    mean=np.array(mean);axes[1].errorbar(mean,yy+offset,xerr=[mean-lo,hi-mean],fmt='o',capsize=4,color=color,label=label)
axes[1].axvline(0,color='gray',linestyle='--');axes[1].set_yticks(yy,['Change coefficients','Change prototypes','Change both']);axes[1].invert_yaxis()
axes[1].set_xlabel('Mask IoU difference (percentage points)\nNeighbor edit minus distance-matched background edit')
axes[1].set_title('S049: 32 crowded mask failures, original box fixed\nMean and pointwise 95% image bootstrap CI');axes[1].legend(fontsize=8)
fig.tight_layout()
for ext in ['png','svg']:
    fig.savefig(ROOT/f'crowded_failure_branch_20260912/GEOMETRY_BRANCH_OVERVIEW.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
