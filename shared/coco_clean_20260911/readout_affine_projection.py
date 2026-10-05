"""Decompose SAVED shared readout residuals into a scalar affine component.

Projection uses original/modified predictions, never labels. Coefficients are
estimated on fit-image training positions then applied to disjoint image set.
Labels only score the already fixed counterfactual predictions. Not a new method.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse,csv,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from rich_pixel_readout import GlobalHead,PixelHead,instance_features
from pixel_ownership_refiner import build_features
from readout_input_probe import sha,write_json
from readout_fit_transfer_diagnostic import norm,read_np,numbers,csv_save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--diagnostic',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--trained',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out;out.mkdir(exist_ok=False);start=time.monotonic()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    protocol=json.loads((a.diagnostic/'protocol.json').read_text());selection=protocol['selection']
    write_json(out/'protocol.json',dict(experiment='S028 frozen learned residual affine projection',images=selection,
      inputs='Same S02780fit/80transfer images, original2048nativeGTbox samples. Only predictions fit the affine projector; no labels in projection.',
      projection='For each learnedheadseed, scalar leastsquares delta(z)=a*z+b over first512positions in allselectedfit targets. '
      'Shared across allimages/targets. Coefficients then frozen and scored on both splits and disjointunusedpositions. '
      'Ordinary unweighted512points per target (equal target weighting). Label-definedsample support is inherited fromdiagnostic, '
      'so not proposed as deployable fitting pipeline.',
      analysis='Prediction-replay variance explained, signchanges and sampledBCE/Dice/IoU. '
      'No full-maskAP/Recall and no claim all spatialinfo absent. Limited affinefamily cannot explain nonlinear scorecalibration.',
      source_diagnostic=sha(a.diagnostic/'COMPLETE.json'),trained=sha(a.trained/'COMPLETE.json'),script=sha(Path(__file__)) ))
    n=torch.load(a.trained/'normalizer.pt',weights_only=True)
    normalization={k:{kk:vv.cuda() for kk,vv in v.items()} for k,v in n.items()}
    models=[]
    for seed in [0,1,2]:
        for mode in ['global','scalar','rich_neighbor']:
            model=(GlobalHead() if mode=='global' else PixelHead()).cuda()
            checkpoint=a.trained/f'{mode}_s{seed}/checkpoints/epoch015.pt'
            if sha(checkpoint)!=json.loads((a.trained/f'{mode}_s{seed}/COMPLETE.json').read_text())['final_sha256']:raise RuntimeError('Weights changed')
            model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['model']);models.append((mode,seed,model.eval()))
    saved=[]
    with torch.inference_mode():
        for split in ['fit','transfer']:
            for iid in selection[split]:
                item=read_np(a.cache/'images'/f'{iid}.npz');idx=item['prediction_indices']
                if not len(idx):continue
                c=torch.tensor(item['coeff'],device='cuda');h=torch.tensor(item['h'],device='cuda');boxes=torch.tensor(item['boxes'],device='cuda')
                det=torch.tensor(item['detections'],device='cuda');lv=torch.tensor(item['level'],device='cuda').long()
                xn=norm(instance_features(h,lv,boxes,tuple(item['input_shape'])),normalization['x']);hn=norm(h,normalization['h'])
                delta_c={s:model(xn) for m,s,model in models if m=='global'}
                for k,j in enumerate(idx):
                    p=torch.tensor(item['sample_p'][k],device='cuda');z0=p@c[j];pos=item['sample_positions'][k]
                    points=torch.tensor(np.stack([pos%640,pos//640],1),device='cuda').float()
                    sf=build_features(c@p.T,boxes,det,int(j),points,'neighbor');pn=norm(p,normalization['p'])
                    yy=item['sample_y'][k].astype(float);seen=set(map(int,pos[:512]));held=[]
                    for t in range(512,len(pos)):
                        if int(pos[t]) not in seen:held.append(t);seen.add(int(pos[t]))
                    zz={}
                    for mode,seed,model in models:
                        z=z0+(p@delta_c[seed][j] if mode=='global' else model(sf[None],pn[None],hn[j:j+1],mode)[0])
                        zz[mode,seed]=z.cpu().numpy().astype(float)
                    saved.append(dict(split=split,image_id=iid,annotation_id=int(item['annotation_ids'][k]),
                        z0=z0.cpu().numpy().astype(float),y=yy,held=held,factor=float(item['loss_factor'][k]),heads=zz))
    coefficients={};projection=[]
    for mode,seed,_ in models:
        rr=[r for r in saved if r['split']=='fit'];z=np.concatenate([r['z0'][:512] for r in rr])
        delta=np.concatenate([r['heads'][mode,seed][:512]-r['z0'][:512] for r in rr])
        vz=float(np.mean((z-z.mean())**2));a1=float(np.mean((z-z.mean())*(delta-delta.mean()))/vz)
        b=float(delta.mean()-a1*z.mean());coefficients[mode,seed]=(a1,b)
        projection.append(dict(mode=mode,seed=seed,delta_slope=a1,intercept=b,final_slope=1+a1,fit_targets=len(rr)))
    rows=[]
    for r in saved:
        for mode,seed,_ in models:
            a1,b=coefficients[mode,seed]
            for domain,idx in [('train512',slice(0,512)),('unused',r['held'])]:
                z=r['z0'][idx];y=r['y'][idx];new=r['heads'][mode,seed][idx]
                if not len(z):continue
                projected=(1+a1)*z+b;residual=new-z;unexplained=new-projected
                stats={}
                for name,val in [('original',z),('learned',new),('projected',projected)]:
                    vals=numbers(torch.tensor(val),torch.tensor(y),r['factor'])
                    stats.update({name+'_'+k:v for k,v in vals.items() if k!='n'})
                rows.append(dict(split=r['split'],image_id=r['image_id'],annotation_id=r['annotation_id'],mode=mode,seed=seed,domain=domain,
                    n=len(z),residual_energy=float(np.mean(residual**2)),unexplained_energy=float(np.mean(unexplained**2)),
                    learned_change=float(np.mean((new>0)!=(z>0))),projected_disagreement=float(np.mean((new>0)!=(projected>0))),**stats))
    csv_save(out/'metrics.csv',rows);summary=[]
    for split in ['fit','transfer']:
        for domain in ['train512','unused']:
            for mode in ['global','scalar','rich_neighbor']:
                rr=[r for r in rows if r['split']==split and r['domain']==domain and r['mode']==mode]
                # Equal targets and three seeds; explained energy is ratio of averages, may be negative on transfer.
                energy=np.mean([r['residual_energy'] for r in rr]);error=np.mean([r['unexplained_energy'] for r in rr])
                summary.append(dict(split=split,domain=domain,mode=mode,targets=len(rr)//3,
                    residual_energy_explained=float(1-error/energy),
                    **{k:float(np.mean([r[k] for r in rr])) for k in ['learned_change','projected_disagreement','original_objective','learned_objective',
                        'projected_objective','original_native_iou','learned_native_iou','projected_native_iou']}))
    write_json(out/'ANALYSIS.json',dict(projection=projection,summary=summary,seconds=time.monotonic()-start,
        scope='Prediction-onlyglobal affineprojection within label-defineddiagnostic support; noGT labels fit projector. '
        'Explained energy is1-E[(learned-projected)^2]/E[(learned-original)^2], averagedtargetsandseeds. '
        'A diagnostic approximation,not residual causal share; input-dependent projection leftout. '
        'No confidenceintervals or officialtaskmetric. Spatialdiscarding cannot be inferred from signagreement alone.'))
    print(json.dumps(dict(projection=projection,summary=summary)),flush=True)


if __name__=='__main__':main()
