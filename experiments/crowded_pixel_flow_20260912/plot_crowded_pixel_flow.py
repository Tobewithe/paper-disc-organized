"""S050 response-selected positive, control-damage and negative inspection cases."""
import os,sys
from pathlib import Path
if os.name=='nt':os.environ['PATH']=str(Path(sys.prefix)/'Library/bin')+os.pathsep+os.environ.get('PATH','')
import json
import numpy as np,pandas as pd
from PIL import Image
from pycocotools import mask as mu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent;OUT=BASE/'diagnostics/crowded_pixel_flow_20260912';SRC=BASE/'diagnostics/crowded_failure_branch_20260912'
records=pd.read_csv(OUT/'pixel_flows.csv');manifest={x['image_id']:x for x in json.loads((SRC/'manifest.json').read_text())['pairs']}
fig,axes=plt.subplots(3,3,figsize=(12,11))
ids=[86755,134886,60090]
for row,iid in enumerate(ids):
    pack=json.loads((OUT/'masks'/f'{iid}.json').read_text());regions={k:mu.decode(v).astype(bool) for k,v in pack['regions'].items()};own=regions['own_interior']|regions['own_boundary_neighbor']|regions['own_boundary_other'];neighbor=regions['chosen_neighbor'];base=mu.decode(pack['original']).astype(bool)
    yy,xx=np.nonzero(own);pad=max(12,int(max(xx.max()-xx.min(),yy.max()-yy.min())*.65));h,w=own.shape
    y0,y1=max(0,yy.min()-pad),min(h,yy.max()+pad+1);x0,x1=max(0,xx.min()-pad),min(w,xx.max()+pad+1)
    original=Image.open(BASE/'local_readout_runtime_20260912/data/images/val2017'/f'{iid:012}.jpg').convert('RGB');gain=min(640/h,640/w)
    for col,mode in enumerate(['original','neighbor','background_control']):
        if mode=='original':image=np.asarray(original);m=base;iou=manifest[iid]['original_mask_iou'];title='Original'
        else:
            r=next(x for x in pack['variants'] if x['mode']==mode and x['fill']=='texture' and x['seed']==0 and x['combination']=='c0p1');m=mu.decode(r['mask']).astype(bool)
            image=np.asarray(Image.open(SRC/'pairs'/str(iid)/f'{mode}_texture_0.png').convert('RGB'))
            iou=float(records[(records.image_id==iid)&(records['mode']==mode)&(records.fill=='texture')&(records.fill_seed==0)&(records.combination=='c0p1')].raw_iou.iloc[0]);title='Neighbor edit / change P' if mode=='neighbor' else 'Background edit / change P'
        rgb=image.astype(float)/255;colors=np.zeros_like(rgb);colors[m&own]=[.15,.85,.2];colors[m&~own]=[1,.15,.15];colors[own&~m]=[.15,.35,1];active=m|own;rgb[active]=.5*rgb[active]+.5*colors[active]
        ax=axes[row,col];ax.imshow(rgb[y0:y1,x0:x1],interpolation='nearest');ax.contour(own[y0:y1,x0:x1],levels=[.5],colors=['white'],linewidths=.75)
        if neighbor[y0:y1,x0:x1].any():ax.contour(neighbor[y0:y1,x0:x1],levels=[.5],colors=['orange'],linewidths=.7)
        ax.set_title(f'{title}\nMask IoU {iou*100:.2f}%',fontsize=10);ax.set_xticks([]);ax.set_yticks([])
        if col==0:ax.set_ylabel(f'Image {iid}, target {pack["target"]}\nBox IoU {manifest[iid]["original_box_iou"]*100:.1f}%',fontsize=9)
fig.suptitle('S050: selected inspection cases, not an unbiased effect estimate\nSame coefficient and original box; texture seed 0; green TP / red FP / blue FN',fontsize=13)
fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(OUT/'PIXEL_FLOW_CASES.png',dpi=170);fig.savefig(OUT/'PIXEL_FLOW_CASES.svg');plt.close(fig)
print('Saved case panel',ids)
