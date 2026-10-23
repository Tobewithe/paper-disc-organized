"""Fixed-last-step object ability evaluation; no real segmentation/AP is computed."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import Objects
from geometry import raster, skin
from object_model import ObjectModel
from runtime import context, dump, read, sha


def paired_interval(values, rng, repetitions):
    values = np.asarray(values, np.float64)
    draws = values[rng.integers(0, len(values), (repetitions, len(values)))].mean(1)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


@torch.no_grad()
def evaluate(root, device):
    with context(root, 'evaluate', authorization=True) as (run, meta, cfg):
        location = read(root/'DATA_LOCATION.json')
        folder = Path(location['directory'])
        assert sha(folder/'MANIFEST.json') == location['manifest_sha256']
        dataset = Objects(folder, 'dev')
        train_set = Objects(folder, 'train')
        tr = root/'runs'/read(root/'TRAIN_POINTER.json')['run_id']
        template = {k: torch.from_numpy(v).to(device) for k, v in dataset.template.items()}
        faces = template['faces'].long()
        mean_shape = np.mean([r['shape'] for r in train_set.rows], axis=0)
        mean_rest = template['canonical']+torch.einsum('d,dvc->vc', torch.as_tensor(mean_shape, device=device, dtype=torch.float32), template['shape_basis'])
        prior = torch.stack([skin(mean_rest[None], pose[None], template['weights'])[0] for pose in template['poses']]).mean(0)
        rows = []
        predictions = {}
        for arm in cfg['arms']:
            trained = read(tr/(arm+'_TRAINING.json'))
            assert sha(tr/(arm+'_last.pt')) == trained['checkpoint_sha256']
            assert trained['optimizer_steps'] == cfg['training_steps_per_arm']
            checkpoint = torch.load(tr/(arm+'_last.pt'), map_location='cpu', weights_only=False)
            assert checkpoint['config_sha256'] == sha(root/'CONFIG_V0.json')
            assert checkpoint['source_data_manifest_sha256'] == location['manifest_sha256']
            model = ObjectModel(dataset.template, cfg, structured=arm == 'structured').to(device)
            model.load_state_dict(checkpoint['model']); model.eval()
            loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)
            for batch in loader:
                batch = {k:v.to(device) for k,v in batch.items()}
                result = model(batch['rgb'], batch['camera'])
                paired = model(batch['pair_rgb'], batch['camera'])
                wrong = model(batch['rgb'], batch['camera'], wrong_binding=True) if arm == 'structured' else None
                for j, index in enumerate(batch['index'].tolist()):
                    source = dataset.rows[index]
                    vertices = result['vertices'][j]
                    target_vertices = batch['vertices'][j]
                    cam = batch['target_camera'][j]
                    gt = raster(target_vertices, faces, cam, source['focal'], cfg['image_size'])
                    pred = raster(vertices, faces, cam, source['focal'], cfg['image_size'])
                    union = (gt['mask']|pred['mask']).sum()
                    iou = float((gt['mask']&pred['mask']).sum()/union.clamp_min(1))
                    row = dict(arm=arm, sample_id=source['sample_id'], family=source['family'], pose=source['pose'], view=source['view'],
                               occluded=source['occluded'], vertex_error=float((vertices-target_vertices).norm(dim=-1).mean()),
                               template_error=float((prior-target_vertices).norm(dim=-1).mean()), novel_view_full_mask_iou=iou,
                               shape_drift=float((result['shape'][j]-paired['shape'][j]).norm()/cfg['shape_dim']**.5),
                               uncertainty_radius=float(result['log_sigma'][j].exp()))
                    if wrong is not None:
                        changed = wrong['vertices'][j]
                        delta = changed-vertices
                        generator = torch.Generator(device=device).manual_seed(cfg['seed']+index)
                        noise = torch.randn(delta.shape, generator=generator, device=device)
                        noise *= delta.norm()/noise.norm().clamp_min(1e-8)
                        row['wrong_binding_error'] = float((changed-target_vertices).norm(dim=-1).mean())
                        row['matched_norm_noise_error'] = float((vertices+noise-target_vertices).norm(dim=-1).mean())
                        row['intervention_l2_norm'] = float(delta.norm())
                    rows.append(row)
                    if source['family'] in [cfg['train_families'], cfg['families']-1] and source['pose'] in [0,3,4] and source['view'] == 1 and source['appearance'] == 0:
                        predictions[arm+'_'+source['sample_id']] = vertices.cpu().numpy()
                        if arm == 'structured':
                            predictions['target_'+source['sample_id']] = target_vertices.cpu().numpy()
        families = sorted({r['family'] for r in rows})
        aggregate = {}
        for arm in cfg['arms']:
            selected = [r for r in rows if r['arm']==arm and r['occluded']==1]
            aggregate[arm] = {key:float(np.mean([r[key] for r in selected]))
                              for key in ['vertex_error', 'template_error', 'novel_view_full_mask_iou', 'shape_drift']}
        family_reductions = []
        for family in families:
            values = {a:np.mean([r['vertex_error'] for r in rows if r['arm']==a and r['family']==family and r['occluded']==1]) for a in cfg['arms']}
            family_reductions.append(float((values['free']-values['structured'])/max(values['free'],1e-8)))
        interval = paired_interval(family_reductions, np.random.default_rng(cfg['seed']), cfg['bootstrap_repetitions'])
        s, u, g = aggregate['structured'], aggregate['free'], cfg['gates']
        wrong = np.mean([r['wrong_binding_error']-r['vertex_error'] for r in rows if r['arm']=='structured' and r['occluded']==1])
        noise = np.mean([r['matched_norm_noise_error']-r['vertex_error'] for r in rows if r['arm']=='structured' and r['occluded']==1])
        conditions = dict(relative_vertex_error=float(np.mean(family_reductions)) >= g['relative_vertex_error_reduction_vs_free'],
                          paired_interval_lower_positive=interval[0] > 0,
                          absolute_vertex_error=s['vertex_error'] <= g['max_mean_vertex_error_body_lengths'],
                          novel_view_mask=s['novel_view_full_mask_iou'] >= g['min_novel_view_full_mask_iou'],
                          mask_vs_free=s['novel_view_full_mask_iou'] >= u['novel_view_full_mask_iou']-g['max_full_mask_iou_loss_vs_free'],
                          shape_stability=s['shape_drift'] <= g['max_shape_drift_clean_vs_occluded'],
                          wrong_binding_used=float(wrong) >= g['min_wrong_binding_error_increase'])
        passed = all(conditions.values())
        dump(run/'OBJECT_METRICS.json', rows)
        dump(run/'SUMMARY.json', dict(population='288 occluded inputs from 8 held-out single-asset shape families',
                                     per_arm=aggregate, paired_family_relative_reduction=float(np.mean(family_reductions)),
                                     family_bootstrap_95_interval=interval, family_values=family_reductions,
                                     wrong_binding_error_increase=float(wrong), matched_norm_noise_error_increase=float(noise),
                                     input_manifest_sha256=location['manifest_sha256'], actual_evaluated_rows=len(rows), real_data_used=False))
        dump(run/'DECISION.json', dict(status='v0_object_ability_pass' if passed else 'v0_object_ability_not_jointly_met',
                                      conditions=conditions, interpretation='single_asset_known_camera_object_model_only',
                                      mechanism_specific_over_equal_norm_noise=bool(wrong > noise),
                                      five_pp_goal_achieved=False, next_action='real_train_object_alignment_protocol' if passed else 'close_current_v0_parameterization_and_review_failure_before_new_training',
                                      automatic_followup_training=False))
        np.savez_compressed(run/'FIXED_EXAMPLES.npz', **predictions)
        dump(run/'INPUTS.json', {str(tr/(a+'_last.pt')):sha(tr/(a+'_last.pt')) for a in cfg['arms']})


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();torch.set_num_threads(4);evaluate(a.root.resolve(),a.device)
