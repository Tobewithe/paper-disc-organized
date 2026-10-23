"""Export actual rig geometry, then render a finite grouped synthetic object dataset."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

from geometry import camera, foreground_bar, raster, ray_reference, skin
from runtime import context, dump, read, sha


def shape_basis(canonical, dim, seed):
    rng = np.random.default_rng(seed)
    basis = np.zeros((dim, len(canonical), 3), np.float32)
    for k in range(3):
        basis[k, :, k] = canonical[:, k]*.12
    for k in range(3, dim):
        centre = canonical[rng.integers(len(canonical))]
        envelope = np.exp(-np.sum((canonical-centre)**2, 1)/.07)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        basis[k] = envelope[:, None]*direction*.015
    return basis


def export(root):
    with context(root, 'asset') as (run, meta, cfg):
        cmd = [cfg['blender'], '-b', '--factory-startup', '-t', '2', '--python-exit-code', '1',
               '--python', str(root/'scripts/export_asset.py'), '--', '--root', str(root), '--output', str(run/'export')]
        dump(run/'COMMAND.json', cmd)
        with open(run/'BLENDER.log', 'w', encoding='utf-8') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = read(run/'export/ASSET_REPORT.json')
        dump(run/'SUMMARY.json', report)


def shade(labels, vertices, faces, style, size):
    device = vertices.device
    yy, xx = torch.meshgrid(torch.linspace(0, 1, size, device=device), torch.linspace(0, 1, size, device=device), indexing='ij')
    bg = torch.stack([.23+.08*xx, .25+.07*yy, .24+.05*xx], -1) if style == 0 else torch.stack([.40+.05*yy, .33+.05*xx, .24+.05*yy], -1)
    face_points = vertices[faces]
    normals = torch.linalg.cross(face_points[:, 1]-face_points[:, 0], face_points[:, 2]-face_points[:, 0])
    normals /= normals.norm(dim=1, keepdim=True).clamp_min(1e-8)
    light = vertices.new_tensor([-.4, -.3, 1.0] if style == 0 else [.6, .2, 1.0])
    light /= light.norm()
    illumination = .38+.62*(normals@light).abs()
    tri = labels['triangle'].clamp_min(0)
    xyz = labels['attributes']['canonical']
    pattern = 1+.035*torch.sin(xyz[..., 0]*63+xyz[..., 1]*37)*torch.cos(xyz[..., 2]*51)
    pigment = vertices.new_tensor([.82, .56, .51] if style == 0 else [.48, .36, .32])
    rgb = illumination[tri, None]*pigment*pattern[..., None]
    return torch.where(labels['mask'][..., None], rgb, bg).clamp(0, 1)


def generate(root, device):
    with context(root, 'data', authorization=True) as (run, meta, cfg):
        source = root/'runs'/read(root/'ASSET_POINTER.json')['run_id']/'export'
        lock = read(root/'ASSET_LOCK.json')
        assert sha(source/'asset.npz') == lock['asset_npz_sha256']
        asset = {k: v.copy() for k, v in np.load(source/'asset.npz').items()}
        dump(run/'INPUTS.json', {str(source/'asset.npz'): sha(source/'asset.npz')})
        basis = shape_basis(asset['canonical'], cfg['shape_dim'], cfg['seed']+44)
        folder = root/'data'/run.name
        folder.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(folder/'template.npz', **asset, shape_basis=basis)
        tensors = {k: torch.from_numpy(v).to(device) for k, v in asset.items()}
        faces = tensors['faces'].long()
        weights, canonical = tensors['weights'], tensors['canonical']
        fields = {'canonical': canonical, 'parts': tensors['part_weights']}
        rows, checks, ray_checks = [], [], []
        size = cfg['image_size']
        for family in range(cfg['families']):
            # Shape and appearance streams are independent; styles are shared across all families.
            rng = np.random.default_rng(cfg['seed']+100003*family)
            z = rng.uniform(-.9, .9, cfg['shape_dim']).astype(np.float32)
            rest = canonical + torch.from_numpy(np.einsum('d,dvc->vc', z, basis)).to(device)
            split = 'train' if family < cfg['train_families'] else 'dev'
            for pose, transforms in enumerate(tensors['poses']):
                vertices = skin(rest[None], transforms[None], weights)[0]
                for view in range(3):
                    cam = camera(cfg['camera_azimuths_degrees'][view], cfg['camera_elevations_degrees'][view])
                    matrix = cam['matrix'].to(device)
                    full = raster(vertices, faces, matrix, cam['focal'], size, fields)
                    assert int(full['mask'].sum()) > 100, ('empty_object', family, pose, view)
                    locations = np.random.default_rng(cfg['seed']+family*100+pose*3+view).choice(size*size,64,replace=False)
                    ray_mask, ray_depth, _, _ = ray_reference(vertices.cpu().numpy(),asset['faces'],matrix.cpu().numpy(),cam['focal'],size,locations)
                    assert np.array_equal(ray_mask,full['mask'].cpu().numpy().ravel()[locations]),('ray_mask',family,pose,view)
                    ray_error = float(np.abs(ray_depth-full['depth'].cpu().numpy().ravel()[locations]).max())
                    assert ray_error < 2e-5,('ray_depth',ray_error)
                    ray_checks.append(ray_error)
                    bar_v, bar_f = foreground_bar(matrix)
                    bar = raster(bar_v, bar_f, matrix, cam['focal'], size)
                    covered = bar['mask'] & full['mask'] & (bar['depth'] < full['depth'])
                    parts = full['attributes']['parts'].argmax(-1)
                    for style in range(cfg['appearances']):
                        clean_rgb = shade(full, vertices, faces, style, size)
                        for occluded in range(cfg['occlusion_states']):
                            rgb = clean_rgb.clone()
                            visible = full['mask'] & ~covered if occluded else full['mask']
                            if occluded:
                                paint = bar['mask'] & ((~full['mask']) | (bar['depth'] < full['depth']))
                                rgb[paint] = rgb.new_tensor([.21, .25, .28])
                            row_id = f'f{family:03d}_p{pose}_v{view}_a{style}_o{occluded}'
                            name = row_id+'.npz'
                            path = folder/name
                            rgb_np = (rgb.cpu().numpy()*255).round().astype(np.uint8)
                            np.savez_compressed(path, rgb=rgb_np, full=full['mask'].cpu().numpy(), visible=visible.cpu().numpy(),
                                                depth=full['depth'].cpu().numpy().astype(np.float16),
                                                canonical=full['attributes']['canonical'].cpu().numpy().astype(np.float16),
                                                parts=parts.cpu().numpy().astype(np.uint8),
                                                triangle=full['triangle'].cpu().numpy().astype(np.int16),
                                                barycentric=full['barycentric'].cpu().numpy().astype(np.float16))
                            rows.append(dict(sample_id=row_id, family=family, split=split, pose=pose, view=view,
                                             appearance=style, occluded=occluded, shape=z.tolist(), camera=cam['matrix'].tolist(),
                                             focal=cam['focal'], file=name, sha256=sha(path), visible_pixels=int(visible.sum()),
                                             full_pixels=int(full['mask'].sum()), covered_pixels=int(covered.sum()) if occluded else 0))
                            if family in [0, cfg['train_families']] and pose in [0, 3, 4] and view == 1:
                                Image.fromarray(rgb_np).save(run/(row_id+'.png'))
                    # Canonical->skin->camera depth/barycentric reconstruction, independent of RGB.
                    valid = full['mask']
                    chosen = faces[full['triangle'][valid]]
                    points = (vertices[chosen]*full['barycentric'][valid][:, :, None]).sum(1)
                    depth = (points@matrix[:, :3].T+matrix[:, 3])[:, 2]
                    error = float((depth-full['depth'][valid]).abs().max())
                    assert error < 2e-5
                    checks.append(error)
            print('FAMILY_COMPLETE', family, len(rows), flush=True)
        manifest = dict(config_sha256=sha(root/'CONFIG_V0.json'), samples=rows, template_sha256=sha(folder/'template.npz'),
                        coordinate_contract='normalized rest body length; native 128x128 pixel centers',
                        family_split='first 24 train, last 8 development; all observations stay together',
                        renderer=cfg['renderer'], asset_sha256=cfg['asset_sha256'])
        dump(folder/'MANIFEST.json', manifest)
        meta['rendered_images'] = len(rows)
        dump(run/'SUMMARY.json', dict(images=len(rows), train=sum(r['split']=='train' for r in rows),
                                     dev=sum(r['split']=='dev' for r in rows), families=cfg['families'],
                                     max_depth_barycentric_replay_error=max(checks), manifest_sha256=sha(folder/'MANIFEST.json'),
                                     independent_ray_samples=len(ray_checks)*64,independent_ray_max_depth_error=max(ray_checks),
                                     data_directory=str(folder), photorealism_claim=False))
        dump(root/'DATA_LOCATION.json', {'directory': str(folder), 'manifest_sha256': sha(folder/'MANIFEST.json')})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=['asset', 'render'], required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.stage == 'asset': export(a.root.resolve())
    else: generate(a.root.resolve(), a.device)
