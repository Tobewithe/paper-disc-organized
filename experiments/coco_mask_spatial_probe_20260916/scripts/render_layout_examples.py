"""Median-effect examples, chosen for explanation rather than maximum gain."""
import argparse
import gzip
import json
from pathlib import Path
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser()
    for k in ['run','cohort','data','out']:ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(s) for s in (a.run/'instances.jsonl').read_text().splitlines()]
    eligible=[r for r in rows if r['group']=='local_only_repair' and r['metrics']['local']['iou']>=.75 and r['metrics']['scalar']['iou']<.75]
    chosen=[]
    for small in [True,False]:
        pool=[r for r in eligible if (r['area']<1024)==small]
        if pool:
            values=[r['metrics']['local']['iou']-r['metrics']['scalar']['iou'] for r in pool];median=np.median(values)
            chosen.append(min(pool,key=lambda r:(abs(r['metrics']['local']['iou']-r['metrics']['scalar']['iou']-median),r['annotation_id'])))
    ids={r['annotation_id'] for r in chosen}
    with gzip.open(a.cohort,'rt') as f:refs={r['annotation_id']:r for s in f if (r:=json.loads(s))['annotation_id'] in ids}
    coco=COCO(str(a.data/'annotations/instances_val2017.json'))
    fig,axs=plt.subplots(len(chosen),5,figsize=(14,3.5*len(chosen)),squeeze=False,layout='constrained')
    for row,r in enumerate(chosen):
        aid=r['annotation_id'];ann=coco.anns[aid];image=cv2.cvtColor(cv2.imread(str(a.data/'images/val2017'/coco.imgs[r['image_id']]['file_name'])),cv2.COLOR_BGR2RGB)
        gt=coco.annToMask(ann).astype(bool);h,w=gt.shape;x,y,bw,bh=ann['bbox'];pad=max(8,.3*max(bw,bh))
        x0=max(0,int(x-pad));x1=min(w,int(np.ceil(x+bw+pad)));y0=max(0,int(y-pad));y1=min(h,int(np.ceil(y+bh+pad)))
        axs[row,0].imshow(image[y0:y1,x0:x1]);axs[row,0].set_title(f"GT {aid}\n{'Small' if r['area']<1024 else 'Medium / large'}")
        for col,name in enumerate(['baseline','scalar','local'],1):
            mask=mu.decode(refs[aid]['reference'][name]).astype(bool) if name in refs[aid]['reference'] else np.zeros_like(gt)
            canvas=image.copy().astype(float);fp=mask&~gt;fn=~mask&gt
            canvas[fp]=.35*canvas[fp]+.65*np.array([238,60,65]);canvas[fn]=.35*canvas[fn]+.65*np.array([45,115,230])
            axs[row,col].imshow(canvas[y0:y1,x0:x1].astype(np.uint8))
            if gt[y0:y1,x0:x1].any():axs[row,col].contour(gt[y0:y1,x0:x1],levels=[.5],colors='lime',linewidths=.8)
            axs[row,col].set_title(f"{name.title()} IoU {100*r['metrics'][name]['iou']:.1f}%")
        field=4*np.array(r['layout']['grid']);v=max(abs(field).max(),.1)
        heat=axs[row,4].imshow(field,cmap='RdBu_r',vmin=-v,vmax=v);axs[row,4].set_title('Learned 4 x 4 logit correction')
        fig.colorbar(heat,ax=axs[row,4],shrink=.75)
        for ax in axs[row]:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Median local-minus-scalar gain within each size group | red: false positive; blue: missed target; green: GT')
    fig.savefig(a.out/'median_layout_examples.png',dpi=170);fig.savefig(a.out/'median_layout_examples.pdf')
    (a.out/'SELECTION.json').write_text(json.dumps({'rule':'median local-minus-scalar fixed-mask IoU within small/non-small local-only repair with same-reference local75 and scalar<75',
                                               'annotation_ids':sorted(ids)},indent=2));plt.close(fig)


if __name__=='__main__':main()
