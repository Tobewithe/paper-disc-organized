"""Actual Pig asset CPU-only forward/backward and ray/raster check; zero optimizer steps."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from geometry import camera, foreground_bar, raster, ray_reference, skin
from object_model import ObjectModel, objective
from prepare_data import shape_basis, shade
from runtime import context, dump, read, sha


def check(root):
    torch.set_num_threads(2)
    with context(root, 'actual_asset_cpu') as (run, meta, cfg):
        asset_run = root/'runs'/read(root/'ASSET_POINTER.json')['run_id']
        path = asset_run/'export/asset.npz'
        template = {k:v.copy() for k,v in np.load(path).items()}
        template['shape_basis']=shape_basis(template['canonical'],cfg['shape_dim'],cfg['seed']+44)
        tensors={k:torch.from_numpy(v) for k,v in template.items()}
        faces=tensors['faces'].long()
        cam=camera(140,50)
        z=torch.linspace(-.5,.5,cfg['shape_dim'])
        rest=tensors['canonical']+torch.einsum('d,dvc->vc',z,tensors['shape_basis'])
        pose=tensors['poses'][3]
        vertices=skin(rest[None],pose[None],tensors['weights'])[0]
        attributes={'canonical':tensors['canonical'],'parts':tensors['part_weights']}
        result=raster(vertices,faces,cam['matrix'],cam['focal'],128,attributes)
        bar_v,bar_f=foreground_bar(cam['matrix'])
        bar=raster(bar_v,bar_f,cam['matrix'],cam['focal'],128)
        covered=bar['mask']&result['mask']&(bar['depth']<result['depth'])
        clean=shade(result,vertices,faces,0,128)
        occluded=clean.clone()
        paint=bar['mask']&((~result['mask'])|(bar['depth']<result['depth']))
        occluded[paint]=torch.tensor([.21,.25,.28])
        for name,image in [('clean',clean),('occluded',occluded)]:
            Image.fromarray((image.numpy()*255).round().astype(np.uint8)).save(run/(name+'.png'))
        checks=[]
        for pi in range(6):
            posed=skin(rest[None],tensors['poses'][pi:pi+1],tensors['weights'])[0]
            for vi in range(3):
                c=camera(cfg['camera_azimuths_degrees'][vi],cfg['camera_elevations_degrees'][vi],focal=85)
                target=raster(posed,faces,c['matrix'],c['focal'],64)
                locations=np.random.default_rng(cfg['seed']+pi*3+vi).choice(64*64,128,replace=False)
                mask,depth,_,_=ray_reference(posed.numpy(),template['faces'],c['matrix'].numpy(),c['focal'],64,locations)
                assert np.array_equal(mask,target['mask'].numpy().ravel()[locations]),(pi,vi)
                err=float(np.abs(depth-target['depth'].numpy().ravel()[locations]).max())
                assert err<2e-5,(pi,vi,err)
                checks.append(dict(pose=pi,view=vi,samples=128,max_depth_error=err))
        batch=dict(rgb=torch.stack([clean,occluded]).permute(0,3,1,2),camera=cam['matrix'][None].expand(2,-1,-1),
                   shape=z[None].expand(2,-1),pose=pose[None].expand(2,-1,-1,-1),vertices=vertices[None].expand(2,-1,-1),
                   target_camera=camera(260,75)['matrix'][None].expand(2,-1,-1),focal=torch.tensor([170.,170.]),
                   canonical=result['attributes']['canonical'][None].expand(2,-1,-1,-1),
                   visible=torch.stack([result['mask'],result['mask']&~covered]),parts=result['attributes']['parts'].argmax(-1)[None].expand(2,-1,-1))
        counts,losses,gradients={},{},{}
        for arm in cfg['arms']:
            torch.manual_seed(cfg['seed'])
            model=ObjectModel(template,cfg,arm=='structured')
            output=model(batch['rgb'],batch['camera'])
            paired=model(batch['rgb'].flip(0),batch['camera'])
            loss,_=objective(output,paired,batch,tensors,cfg)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            gradients[arm]={n:float(p.grad.norm()) for n,p in model.named_parameters()}
            losses[arm]=float(loss.detach())
            counts[arm]=sum(p.numel() for p in model.parameters())
        assert counts['structured']==counts['free']
        dump(run/'SUMMARY.json',dict(status='passed',optimizer_steps=0,GPU_used=False,vertices=len(template['canonical']),
                                    bones=tensors['weights'].shape[1],matched_trainable_parameters=counts,
                                    finite_losses=losses,gradients=gradients,independent_ray_checks=checks,
                                    source_pixels=int(result['mask'].sum()),covered_pixels=int(covered.sum()),
                                    source_asset_export_run=asset_run.name,source_asset_npz_sha256=sha(path),
                                    environment={'torch':torch.__version__,'numpy':np.__version__,'cuda_build':torch.version.cuda},
                                    scientific_object_ability_not_evaluated=True))
        dump(root/'ASSET_LOCK.json',dict(asset_run_id=asset_run.name,asset_npz_sha256=sha(path),
                                       asset_report_sha256=sha(asset_run/'export/ASSET_REPORT.json'),
                                       actual_asset_check_run_id=run.name))
        print('ACTUAL_PIG_CPU_CHECK_PASSED',counts,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();check(a.root.resolve())
