"""Scientific S047 case panel from actual saved predictions; no generated pixels."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,json
from pathlib import Path
import cv2,numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--decode-only',action='store_true');a=parser.parse_args();out=a.out
    ids=[322194,214832]
    tags=['original_none_-1','neighbor_texture_0','neighbor_local_color_0','background_control_texture_0']
    if a.decode_only:
        import torch
        import torch.nn.functional as F
        from ultralytics.utils import ops
        for iid in ids:
            folder=out/'pairs'/str(iid)
            with np.load(folder/'original.npz') as n:shape=n['own'].shape
            with np.load(folder/'responses.npz') as n:responses={k:n[k] for k in n.files}
            decoded={}
            for tag in tags:
                z=F.interpolate(torch.tensor(responses[tag+'_logits160'],device='cuda')[None,None],(640,640),mode='bilinear',align_corners=False)[0,0]
                box=torch.tensor(responses['original_none_-1_box640'],device='cuda')[None]
                m=ops.crop_mask((z>0).byte()[None],box)
                decoded[tag]=(ops.scale_masks(m[:,None],shape)[0,0]>.5).cpu().numpy()
            np.savez_compressed(folder/'figure_masks.npz',**decoded)
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(14,8))
    for row,iid in enumerate(ids):
        folder=out/'pairs'/str(iid)
        with np.load(folder/'original.npz') as n:original={k:n[k] for k in n.files}
        with np.load(folder/'responses.npz') as n:responses={k:n[k] for k in n.files}
        with np.load(folder/'figure_masks.npz') as n:decoded={k:n[k] for k in n.files}
        own=original['own'];neighbor=original['neighbor'];shape=own.shape
        ys,xs=np.nonzero(own|neighbor);margin=20;y0,y1=max(0,ys.min()-margin),min(shape[0],ys.max()+margin+1);x0,x1=max(0,xs.min()-margin),min(shape[1],xs.max()+margin+1)
        for col,tag in enumerate(tags):
            path=out/'images'/f'{iid:012d}.jpg' if col==0 else folder/(tag+'.png')
            im=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
            m=decoded[tag]
            intersection=(m&own).sum();iou=intersection/max((m|own).sum(),1)
            axes[row,col].imshow(im[y0:y1,x0:x1]);axes[row,col].contour(own[y0:y1,x0:x1],levels=[.5],colors=['lime'],linewidths=1.2)
            axes[row,col].contour(neighbor[y0:y1,x0:x1],levels=[.5],colors=['orange'],linewidths=1.2)
            axes[row,col].contour(m[y0:y1,x0:x1],levels=[.5],colors=['cyan'],linewidths=1)
            axes[row,col].set_title(['Original','Neighbor: texture','Neighbor: local color','Background control'][col]+f'\nfixed-box IoU={iou*100:.2f}%')
            axes[row,col].axis('off')
        axes[row,0].text(0,-.05,f'COCO image {iid}',transform=axes[row,0].transAxes)
    fig.suptitle('S047 actual fixed-source predictions | green: target GT, orange: original neighbor, cyan: prediction\nTwo post-hoc illustrative cases; seed 0 shown, no implication of population recovery',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(out/'CASE_PANEL.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
