"""Independent CPU acceptance, exact data pairing, and deployable model packaging."""
import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from runtime import context,dump,read,sha


def accept(root):
    torch.set_num_threads(2)
    with context(root,'acceptance') as (run,meta,cfg):
        (run/'ACCEPTANCE_SOURCE.py').write_bytes(Path(__file__).read_bytes())
        paths={stage:root/'runs'/read(root/(stage.upper()+'_POINTER.json'))['run_id'] for stage in ['data','train','evaluate']}
        artifact_checks=0
        for folder in paths.values():
            assert read(folder/'run.json')['execution_status']=='completed'
            for rel,record in read(folder/'ARTIFACT_MANIFEST.json').items():
                assert sha(folder/rel)==record['sha256'],str(folder/rel)
                artifact_checks+=1
        data_dir=Path(read(root/'DATA_LOCATION.json')['directory'])
        assert sha(data_dir/'MANIFEST.json')==read(root/'DATA_LOCATION.json')['manifest_sha256']
        manifest=read(data_dir/'MANIFEST.json');samples=manifest['samples']
        assert len(samples)==2304
        groups={}
        for row in samples:
            assert sha(data_dir/row['file'])==row['sha256']
            groups.setdefault((row['family'],row['pose'],row['view']),[]).append(row)
        assert len(groups)==576
        for key,items in groups.items():
            assert len(items)==4
            first=next(r for r in items if r['appearance']==0 and r['occluded']==0)
            with np.load(data_dir/first['file']) as q:
                reference={k:q[k].copy() for k in ['full','depth','canonical','parts','triangle','barycentric']}
            occluded=None
            for item in items:
                with np.load(data_dir/item['file']) as q:
                    assert all(np.array_equal(q[k],reference[k]) for k in reference),(key,item['sample_id'])
                    assert not np.any(q['visible']&~q['full'])
                    if item['occluded']==0:assert np.array_equal(q['visible'],q['full'])
                    elif occluded is None:occluded=q['visible'].copy()
                    else:assert np.array_equal(q['visible'],occluded)
        evaluation=paths['evaluate'];original=read(evaluation/'SUMMARY.json');decision=read(evaluation/'DECISION.json')
        rows=read(evaluation/'OBJECT_METRICS.json')
        assert len(rows)==1152
        keys={a:{r['sample_id'] for r in rows if r['arm']==a} for a in cfg['arms']}
        assert keys['structured']==keys['free'] and len(keys['free'])==576
        aggregate={};families=sorted({r['family'] for r in rows})
        for arm in cfg['arms']:
            selected=[r for r in rows if r['arm']==arm and r['occluded']==1]
            assert len(selected)==288
            aggregate[arm]={k:float(np.mean([r[k] for r in selected])) for k in ['vertex_error','template_error','novel_view_full_mask_iou','shape_drift']}
        family_values=[]
        for family in families:
            errors={a:np.mean([r['vertex_error'] for r in rows if r['arm']==a and r['family']==family and r['occluded']==1]) for a in cfg['arms']}
            family_values.append(float((errors['free']-errors['structured'])/max(errors['free'],1e-8)))
        rng=np.random.default_rng(cfg['seed'])
        draws=np.asarray(family_values)[rng.integers(0,8,(2000,8))].mean(1)
        interval=np.quantile(draws,[.025,.975]).tolist()
        errors=[abs(aggregate[a][k]-original['per_arm'][a][k]) for a in cfg['arms'] for k in aggregate[a]]
        errors.extend(abs(a-b) for a,b in zip(interval,original['family_bootstrap_95_interval']))
        assert max(errors)<1e-12
        s,u,g=aggregate['structured'],aggregate['free'],cfg['gates']
        wrong=float(np.mean([r['wrong_binding_error']-r['vertex_error'] for r in rows if r['arm']=='structured' and r['occluded']==1]))
        conditions=dict(relative_vertex_error=np.mean(family_values)>=g['relative_vertex_error_reduction_vs_free'],
                        paired_interval_lower_positive=interval[0]>0,absolute_vertex_error=s['vertex_error']<=g['max_mean_vertex_error_body_lengths'],
                        novel_view_mask=s['novel_view_full_mask_iou']>=g['min_novel_view_full_mask_iou'],
                        mask_vs_free=s['novel_view_full_mask_iou']>=u['novel_view_full_mask_iou']-g['max_full_mask_iou_loss_vs_free'],
                        shape_stability=s['shape_drift']<=g['max_shape_drift_clean_vs_occluded'],wrong_binding_used=wrong>=g['min_wrong_binding_error_increase'])
        conditions={k:bool(v) for k,v in conditions.items()}
        assert conditions==decision['conditions']
        artifacts=root/'artifacts';artifacts.mkdir(exist_ok=True)
        models={}
        for arm in cfg['arms']:
            trained=read(paths['train']/(arm+'_TRAINING.json'))
            assert trained['optimizer_steps']==4000 and trained['exposed_samples']==1728
            assert trained['min_exposure']>=37
            checkpoint=paths['train']/(arm+'_last.pt')
            assert sha(checkpoint)==trained['checkpoint_sha256']
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert saved['optimizer_steps']==4000 and saved['trainable_parameters']==453384
            assert all(not t.is_floating_point() or bool(torch.isfinite(t).all()) for t in saved['model'].values())
            shutil.copyfile(checkpoint,artifacts/checkpoint.name)
            models[arm]=dict(file=checkpoint.name,sha256=sha(checkpoint),optimizer_steps=4000,trainable_parameters=453384)
        with np.load(data_dir/'template.npz') as template:
            np.savez_compressed(artifacts/'template_inference.npz',**{k:template[k] for k in ['canonical','faces','weights','part_weights','shape_basis']})
        demo=next(r for r in samples if r['sample_id']=='f024_p3_v1_a0_o1')
        with np.load(data_dir/demo['file']) as sample:Image.fromarray(sample['rgb']).save(artifacts/'demo_occluded.png')
        dump(artifacts/'demo_camera.json',dict(matrix=demo['camera'],sample_id=demo['sample_id'],scope='known synthetic source camera; no shape or pose target'))
        dump(artifacts/'MODEL_MANIFEST.json',dict(models=models,inference_template=dict(file='template_inference.npz',sha256=sha(artifacts/'template_inference.npz')),
                                                source_template_sha256=sha(data_dir/'template.npz'),source_training_run=paths['train'].name,
                                                source_evaluation_run=evaluation.name,source_config_sha256=sha(root/'CONFIG_V0.json'),
                                                scope='single-given-asset known-camera image-to-object v0; no real segmentation claim',
                                                inference_inputs=['128x128 RGB','known source camera'],target_pose_input=False))
        result=dict(status='passed',GPU_used=False,optimizer_steps=0,completed_training_steps=8000,per_arm_steps=4000,
                    full_dataset_file_SHA_verified=2304,source_run_artifacts_SHA_verified=artifact_checks,
                    paired_geometry_groups_verified=576,independent_statistics_max_abs_error=max(errors),
                    decision_recomputed_exact=True,scientific_decision=decision['status'],per_arm=aggregate,
                    family_bootstrap_95_interval=interval,model_files=models,real_data_used=False)
        dump(run/'SUMMARY.json',result)
        print('ACCEPTANCE_PASSED',decision['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();accept(a.root.resolve())
