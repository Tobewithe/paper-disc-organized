"""CPU forward/backward and projection checks. No optimizer, training or real data."""
import argparse
from pathlib import Path
import tempfile

import numpy as np
import torch

from geometry import camera, project, raster, skin
from object_model import ObjectModel, objective
from runtime import context, dump, read


def checks(root):
    torch.set_num_threads(2)
    torch.manual_seed(23102391)
    with context(root, 'engineering_cpu') as (run, meta, cfg):
        vertices = torch.tensor([[-.4,-.3,0.],[.4,-.3,0.],[0.,.4,0.]])
        faces = torch.tensor([[0,1,2]])
        cam = camera(20, 50)
        rendered = raster(vertices, faces, cam['matrix'], 28, 32, {'canonical':vertices})
        mask = rendered['mask']
        assert mask.any()
        points = rendered['attributes']['canonical'][mask]
        uv, depth = project(points, cam['matrix'], 28, 32)
        yy,xx=torch.where(mask)
        pixel_error=float((uv-torch.stack([xx+.5, yy+.5],1)).abs().max())
        assert pixel_error < 1e-4
        depth_error=float((depth-rendered['depth'][mask]).abs().max())
        assert depth_error < 1e-5
        identity = torch.eye(4)[:3][None,None].expand(2,2,-1,-1).clone()
        weights=torch.tensor([[1.,0.],[.5,.5],[0.,1.]])
        assert torch.allclose(skin(vertices[None].expand(2,-1,-1),identity,weights),vertices[None].expand(2,-1,-1))
        template=dict(canonical=vertices.numpy(),weights=weights.numpy(),part_weights=np.eye(8,dtype=np.float32)[:3],
                      shape_basis=np.zeros((cfg['shape_dim'],3,3),np.float32))
        counts={}; losses={}; gradients={}; initial=None
        for arm in cfg['arms']:
            torch.manual_seed(cfg['seed'])
            model=ObjectModel(template,cfg,arm=='structured')
            if initial is None: initial={k:v.clone() for k,v in model.state_dict().items()}
            else: model.load_state_dict(initial)
            counts[arm]=sum(p.numel() for p in model.parameters())
            batch=dict(rgb=torch.rand(2,3,128,128),camera=cam['matrix'][None].expand(2,-1,-1),
                       shape=torch.zeros(2,cfg['shape_dim']),pose=identity,vertices=vertices[None].expand(2,-1,-1),
                       target_camera=cam['matrix'][None].expand(2,-1,-1),focal=torch.tensor([170.,170.]),
                       canonical=torch.zeros(2,128,128,3),visible=torch.ones(2,128,128,dtype=torch.bool),
                       parts=torch.zeros(2,128,128,dtype=torch.long))
            result=model(batch['rgb'],batch['camera'])
            repeated=model(batch['rgb'],batch['camera'])
            assert torch.equal(result['vertices'],repeated['vertices'])
            loss,components=objective(result,repeated,batch,{k:torch.from_numpy(v) for k,v in template.items()},cfg)
            assert torch.isfinite(loss)
            loss.backward()
            missing=[n for n,p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
            assert not missing,(arm,missing)
            losses[arm]=float(loss.detach())
            gradients[arm]={n:float(p.grad.norm()) for n,p in model.named_parameters()}
            checkpoint=run/(arm+'_roundtrip.pt')
            torch.save(model.state_dict(),checkpoint)
            restored=ObjectModel(template,cfg,arm=='structured')
            restored.load_state_dict(torch.load(checkpoint,weights_only=True))
            assert torch.equal(result['vertices'],restored(batch['rgb'],batch['camera'])['vertices'])
        assert counts['structured']==counts['free']
        # Analytic ray-plane intersection supplies an independent reference to triangle depth.
        rotation=cam['matrix'][:,:3]; translation=cam['matrix'][:,3]
        origin=-rotation.T@translation
        rays=torch.stack([(xx+.5-16)/28,-(yy+.5-16)/28,torch.ones_like(xx)],1)@rotation
        distance=-origin[2]/rays[:,2]
        hits=origin[None]+distance[:,None]*rays
        ray_error=float((hits-points).abs().max())
        assert ray_error < 1e-5
        result=dict(status='passed',optimizer_steps=0,GPU_used=False,template='three_vertex_test_fixture_not_Pig_asset',
                    pixel_reprojection_max_error=pixel_error,depth_max_error=depth_error,
                    independent_ray_plane_max_error=ray_error,matched_parameters=counts,
                    finite_losses=losses,all_parameters_have_finite_gradient=True,
                    gradients=gradients,checkpoint_reload_exact=True,
                    actual_asset_and_real_data_validated=False)
        dump(run/'SUMMARY.json',result)
        print('CPU_PACKAGE_CHECK_PASSED',counts,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();checks(a.root.resolve())
