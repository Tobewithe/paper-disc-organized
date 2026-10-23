"""Given-asset QA and fixed learned-object examples; never relabel the asset as trained."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from runtime import dump, read, sha


def mesh(ax, vertices, faces, colors, elevation=30, azimuth=30):
    collection=Poly3DCollection(vertices[faces],facecolors=colors,edgecolors='#333333',linewidths=.08)
    ax.add_collection3d(collection)
    center=(vertices.min(0)+vertices.max(0))/2
    radius=max(np.ptp(vertices,axis=0).max()*.55,.3)
    ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
    ax.set_box_aspect((1,1,1));ax.view_init(elevation,azimuth);ax.set_axis_off()


def main(root, stage):
    asset_run=root/'runs'/read(root/'ASSET_POINTER.json')['run_id']
    data=np.load(asset_run/'export/asset.npz')
    faces=data['faces']
    palette=np.array(['#D55E00','#E69F00','#56B4E9','#0072B2','#009E73','#CC79A7','#8867AC','#7A6A53'])
    groups=data['part_weights'].argmax(1)
    colors=palette[groups[faces].max(1)]
    out=root/'presentation';out.mkdir(exist_ok=True)
    if stage=='asset':
        cfg=read(root/'CONFIG_V0.json')
        fig=plt.figure(figsize=(12,7.5),layout='constrained')
        for i,name in enumerate(cfg['poses']):
            ax=fig.add_subplot(2,3,i+1,projection='3d')
            mesh(ax,data['posed_vertices'][i],faces,colors)
            ax.set_title(name.replace('_',' '),fontsize=11)
        fig.suptitle('Given Pig rig and coarse body parts | NOT learned model results\nSix exported poses; identical part colors across views and poses',fontsize=12)
        fig.savefig(out/'01_given_asset_pose_QA.png',dpi=160)
        plt.close(fig)
    else:
        evaluation=root/'runs'/read(root/'EVALUATE_POINTER.json')['run_id']
        examples=np.load(evaluation/'FIXED_EXAMPLES.npz')
        selected=sorted(k[len('target_'):] for k in examples.files if k.startswith('target_'))
        for number,sample_id in enumerate(selected):
            fig=plt.figure(figsize=(10,3.5),layout='constrained')
            for i,prefix in enumerate(['target','free','structured']):
                ax=fig.add_subplot(1,3,i+1,projection='3d')
                mesh(ax,examples[prefix+'_'+sample_id],faces,colors)
                ax.set_title(prefix,fontsize=11)
            fig.suptitle('Fixed held-out synthetic object | '+sample_id+'\nTarget is synthetic geometry GT. These are object predictions, not real segmentation AP.',fontsize=10)
            fig.savefig(out/(f'object_{number:02d}_'+sample_id+'.png'),dpi=150)
            plt.close(fig)
    dump(out/'MANIFEST.json',{'stage':stage,'files':{p.name:sha(p) for p in out.glob('*.png')},'given_asset_is_not_learned':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['asset','learned'],required=True)
    a=p.parse_args();main(a.root.resolve(),a.stage)
