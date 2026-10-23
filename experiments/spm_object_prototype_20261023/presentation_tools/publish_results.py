"""Presentation-only rendering of frozen object evaluation and trained fields."""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from geometry import raster
from object_model import ObjectModel
from runtime import context, dump, read, sha


def draw(ax, vertices, faces, colors, reference, title):
    center=(reference.min(0)+reference.max(0))/2
    radius=max(np.ptp(reference,axis=0).max()*.58,.3)
    ax.add_collection3d(Poly3DCollection(vertices[faces],facecolors=colors,edgecolors='#333333',linewidths=.08))
    ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
    ax.set_box_aspect((1,1,1));ax.view_init(26,35);ax.set_axis_off();ax.set_title(title,fontsize=10)


def save(fig, run, name):
    fig.savefig(run/name,dpi=160,bbox_inches='tight');plt.close(fig)


@torch.no_grad()
def publish(root):
    torch.set_num_threads(2)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    with context(root,'presentation') as (run,meta,cfg):
        (run/'PRESENTATION_SOURCE.py').write_bytes(Path(__file__).read_bytes())
        evaluation=root/'runs'/read(root/'EVALUATE_POINTER.json')['run_id']
        training=root/'runs'/read(root/'TRAIN_POINTER.json')['run_id']
        folder=Path(read(root/'DATA_LOCATION.json')['directory'])
        manifest=read(folder/'MANIFEST.json'); rows={r['sample_id']:r for r in manifest['samples']}
        template={k:v.copy() for k,v in np.load(folder/'template.npz').items()}
        faces=template['faces']
        metrics=read(evaluation/'SUMMARY.json');decision=read(evaluation/'DECISION.json')
        examples=np.load(evaluation/'FIXED_EXAMPLES.npz')
        palette=np.array(['#D55E00','#E69F00','#56B4E9','#0072B2','#009E73','#CC79A7','#8867AC','#7A6A53'])
        colors=palette[template['part_weights'][faces].mean(1).argmax(1)]
        fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
        for arm,color in [('structured','#0072B2'),('free','#D55E00')]:
            records=read(training/(arm+'_TRAINING.json'))['losses']
            for ax,key in zip(axes,['loss','vertices','novel_view_points']):
                ax.plot([r['step'] for r in records],[r[key] for r in records],label=arm,color=color)
                ax.set(xlabel='Optimizer step',ylabel=key,yscale='log');ax.legend(fontsize=8)
        fig.suptitle('Equal-budget object learning | 4000 steps per arm\nTraining loss is not evidence of real segmentation gain.',fontsize=11)
        save(fig,run,'01_training_curves.png')

        fig,axes=plt.subplots(1,3,figsize=(12.5,3.8),layout='constrained')
        for ax,key,title in zip(axes,['vertex_error','novel_view_full_mask_iou','shape_drift'],
                                ['Complete-body vertex error (body lengths)','Different-view full projection IoU','Shape-code drift (RMS)']):
            values=[metrics['per_arm'][a][key] for a in ['free','structured']]
            ax.bar(['Free','Structured'],values,color=['#D55E00','#0072B2'])
            for i,value in enumerate(values):ax.text(i,value,f'{value:.4f}',ha='center',va='bottom')
            ax.set_title(title,fontsize=9);ax.set_ylim(0,max(values)*1.28+.002)
            if key=='vertex_error':
                ax.axhline(metrics['per_arm']['structured']['template_error'],ls='--',c='#555555',label='Static mean object')
                ax.set_ylim(0,max(metrics['per_arm']['structured']['template_error'],*values)*1.2);ax.legend(fontsize=8)
        fig.suptitle('288 occluded inputs | 8 held-out shape families | single given Pig asset\nKnown camera and synthetic geometry targets; not real Mask AP.',fontsize=11)
        save(fig,run,'02_held_out_object_ability.png')

        fig,ax=plt.subplots(figsize=(8.5,3.4),layout='constrained')
        ax.bar(range(24,32),np.array(metrics['family_values'])*100,color='#0072B2')
        ax.axhline(10,c='black',ls='--',label='Frozen relative reduction requirement: 10%')
        ax.axhline(0,c='gray',lw=.7);ax.legend(fontsize=8)
        ax.set(xlabel='Held-out synthetic shape family',ylabel='Vertex error reduction vs free (%)',
               title='Paired family results | positive favors structured')
        save(fig,run,'03_paired_family_results.png')

        fixed=sorted(k[len('target_'):] for k in examples.files if k.startswith('target_'))
        for number,sample_id in enumerate(fixed):
            reference=np.concatenate([examples[p+'_'+sample_id] for p in ['target','free','structured']])
            fig=plt.figure(figsize=(10.5,3.6),layout='constrained')
            for i,prefix in enumerate(['target','free','structured']):
                draw(fig.add_subplot(1,3,i+1,projection='3d'),examples[prefix+'_'+sample_id],faces,colors,reference,prefix)
            fig.suptitle('Fixed object '+sample_id+' | identical axes and scale\nTarget is synthetic geometry GT; cases were fixed by observation IDs before results.',fontsize=10)
            save(fig,run,f'04_fixed_object_{number:02d}.png')

        selected=[f'f{f:03d}_p{p}_v1_a0_o1' for f,p in [(24,0),(24,3),(31,4)]]
        fig,axes=plt.subplots(3,4,figsize=(10,7.4),layout='constrained')
        for i,sample_id in enumerate(selected):
            row=rows[sample_id]; data=np.load(folder/row['file'])
            axes[i,0].imshow(data['rgb']);axes[i,0].set_title('Occluded source: '+sample_id,fontsize=8)
            next_row=rows[sample_id.replace('_v1_','_v2_').replace('_o1','_o0')]
            matrix=torch.tensor(next_row['camera'])
            for j,prefix in enumerate(['target','free','structured'],1):
                projected=raster(torch.from_numpy(examples[prefix+'_'+sample_id]),torch.from_numpy(faces).long(),matrix,row['focal'],128)
                axes[i,j].imshow(projected['mask'],cmap='gray',vmin=0,vmax=1);axes[i,j].set_title(prefix+' at requested other view',fontsize=8)
        for ax in axes.flat:ax.axis('off')
        fig.suptitle('Partial observation -> complete object -> other-view silhouette\nNo target-view image enters the model; requested camera is supplied.',fontsize=11)
        save(fig,run,'05_occlusion_to_different_view.png')

        checkpoint=torch.load(training/'structured_last.pt',map_location='cpu',weights_only=False)
        assert sha(training/'structured_last.pt')==read(training/'structured_TRAINING.json')['checkpoint_sha256']
        model=ObjectModel(template,cfg,True);model.load_state_dict(checkpoint['model']);model.eval()
        source=selected[1]; row=rows[source]; data=np.load(folder/row['file'])
        rgb=torch.from_numpy(data['rgb'].copy()).permute(2,0,1)[None].float()/255
        output=model(rgb,torch.tensor(row['camera'])[None])
        # A training-only projection makes colors comparable; PCA colors carry no class/identity claim.
        train_shapes=[r['shape'] for r in manifest['samples'] if r['split']=='train' and r['pose']==0 and r['view']==0 and r['appearance']==0 and r['occluded']==0]
        identity=torch.eye(4)[:3][None,None].expand(1,model.weights.shape[1],-1,-1)
        train_features=[]
        for z in train_shapes: train_features.append(model.decode(torch.tensor(z)[None],identity)[1][0].numpy())
        train_features=np.concatenate(train_features)
        mean=train_features.mean(0); _,_,vt=np.linalg.svd(train_features-mean,full_matrices=False)
        projected=(train_features-mean)@vt[:3].T
        lo,hi=np.quantile(projected,[.05,.95],axis=0)
        features=output['features'][0].numpy()
        field_color=np.clip(((features-mean)@vt[:3].T-lo)/(hi-lo+1e-8),0,1)
        field_color=field_color[faces].mean(1)
        vertices=output['vertices'][0].numpy()
        fig=plt.figure(figsize=(9,3.8),layout='constrained')
        draw(fig.add_subplot(1,2,1,projection='3d'),vertices,faces,colors,vertices,'Given coarse-part labels on predicted object')
        draw(fig.add_subplot(1,2,2,projection='3d'),vertices,faces,field_color,vertices,'Learned 64D surface field (training-only PCA colors)')
        fig.suptitle('Predicted object and learned surface representation\nSynthetic part supervision; color is not instance identity or a semantic-mechanism proof.',fontsize=10)
        save(fig,run,'06_learned_surface_field.png')

        edit_vertices=[]
        for pose in torch.from_numpy(template['poses']):edit_vertices.append(model.decode(output['shape'],pose[None])[0][0].numpy())
        reference=np.concatenate(edit_vertices)
        fig=plt.figure(figsize=(12,6.5),layout='constrained')
        for i,(name,vertices) in enumerate(zip(cfg['poses'],edit_vertices)):
            draw(fig.add_subplot(2,3,i+1,projection='3d'),vertices,faces,colors,reference,name.replace('_',' '))
        fig.suptitle('One inferred shape, six explicitly requested pose controls\nPose matrices are supplied from the given rig; this is controllability visualization, not predicted-pose accuracy.',fontsize=11)
        save(fig,run,'07_controlled_pose_edit.png')

        # Export a genuinely predicted instance, with the full learned geometry rather than the asset mesh.
        predicted=output['vertices'][0].numpy()
        obj=run/'predicted_f024_prone_occluded.obj'
        with open(obj,'w',encoding='utf-8') as f:
            f.write('# Predicted by trained structured object model from occluded RGB and known camera.\n')
            for vertex in predicted:f.write('v '+' '.join(f'{x:.8f}' for x in vertex)+'\n')
            for face in faces:f.write('f '+' '.join(str(int(x)+1) for x in face)+'\n')
        dump(run/'SUMMARY.json',dict(figures=len(list(run.glob('*.png'))),fixed_sample_ids=fixed,
                                    source_training_run=training.name,source_evaluation_run=evaluation.name,
                                    source_evaluation_decision=decision['status'],new_optimizer_steps=0,
                                    selected_export_sample=source,export_is_model_prediction=True,
                                    figure_source_sha256=sha(Path(__file__)),real_segmentation_gain_computed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();publish(args.root.resolve())
