import argparse
from pathlib import Path
import tarfile

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib
import io

from common import dump, read, sha
from data import Store
from evaluate import predict
from task_context import task_context
from task_model import load, memory_module


def folder(root, stage):
    return root/'runs'/read(root/(stage.upper()+'_POINTER.json'))['run_id']


def independent_mask_ap(root, predictions):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(root/'DEV_COCO.json'))
        dt = gt.loadRes([{k: v for k, v in p.items() if k != 'bbox'} for p in predictions])
        evaluator = COCOeval(gt, dt, 'segm'); evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    return np.asarray(evaluator.stats)*100


def bundle(root, name, files):
    files = sorted(set(files)); path = root/name
    with tarfile.open(path, 'w:gz', compresslevel=1) as tar:
        for item in files:
            tar.add(item, arcname=item.relative_to(root).as_posix(), recursive=False)
    receipt = {'bundle': name, 'sha256': sha(path), 'size_bytes': path.stat().st_size,
               'files': {p.relative_to(root).as_posix(): {'sha256': sha(p), 'size_bytes': p.stat().st_size} for p in files}}
    dump(root/(name+'.receipt.json'), receipt); return receipt


def main(root):
    with task_context(root, 'acceptance', freeze='ACCEPTANCE_FREEZE.json') as (run, meta):
        evaluation = folder(root, 'evaluate_task'); assert read(evaluation/'run.json')['execution_status'] == 'completed'
        results = read(evaluation/'METRICS.json'); recomputed = {}
        for tag, expected in results.items():
            stats = independent_mask_ap(root, read(evaluation/'evaluation'/(tag+'_predictions.json')))
            error = float(np.abs(stats-np.asarray(expected['segm_stats'])).max()); assert error < 1e-10, (tag, error)
            recomputed[tag] = {'mask_ap': float(stats[0]), 'all_12_segm_stats_max_error': error}
        d = read(evaluation/'DECISION.json'); actual = recomputed['task_spm']['mask_ap']-recomputed['continued']['mask_ap']
        assert actual == d['measured_F_minus_continued_pp']
        assert (actual >= 5.) == d['five_pp_metric_target_met']
        data = read(root/'ALL_DATASET.json'); rows = data['rows']
        assert not data['test_included']
        real = [r for r in rows if r['domain'] == 'real']; assert len(real) == 1704
        assert sum(r['split'] == 'train' for r in real) == 1272
        assert sum(r['split'] == 'dev' for r in real) == 432
        train = {(r['domain'], r['group']) for r in rows if r['split'] == 'train'}
        dev = {(r['domain'], r['group']) for r in rows if r['split'] == 'dev'}; assert not train & dev
        warm, final = folder(root, 'prototype_warm'), folder(root, 'prototype_final')
        assert read(warm/'run.json')['optimizer_steps'] == 5000
        assert read(final/'run.json')['optimizer_steps'] == 15000
        memory = torch.load(final/'memory.pt', map_location='cpu', weights_only=False)
        assert memory['train_only'] and memory['prototype_total_steps'] == 20000
        assert memory['centers'].shape == (32, 128)
        norm_error = float((memory['centers'].norm(dim=1)-1).abs().max()); assert norm_error < 1e-5
        assert memory['mean'].shape == (512,) and memory['std'].shape == (512,)
        assert memory['upstream_checkpoint_sha256'] == sha(folder(root, 'baseline')/'best.pt')
        by_source = {r['source_id']: r for r in rows}
        for cluster in memory['clusters']:
            assert cluster['support_tokens'] >= 0
            for exemplar in cluster['exemplars']:
                source = by_source[exemplar['source_id']]; assert source['split'] == 'train'
                assert source['ordinal'] == exemplar['ordinal']
        weights = []; loading = {}
        store = Store(rows, max_cache=2); row = next(r for r in real if r['split'] == 'dev')
        batch = store.batch([row], 0, augment=False, device='cpu')
        for mode in ['baseline', 'continued', 'adapter', 'random_memory', 'task_spm']:
            path = folder(root, mode)/'best.pt'; model = load(path, device='cpu').eval()
            with torch.no_grad(): det, masks = predict(model, batch['img'])
            assert np.isfinite(det).all() and masks.shape[1:] == (640, 640)
            assert model.model[-1].nm == (32 if mode in ['baseline', 'continued'] else 64)
            module = memory_module(model)
            if module:
                assert module.projector.first.in_features == 512 and module.keys.shape == (32, 128)
            loading[mode] = {'checkpoint_sha256': sha(path), 'mask_bases': model.model[-1].nm,
                             'CPU_RGB_only_predictions': len(det), 'finite': True}
            weights.append(path); del model
        weights += [final/'memory.pt', final/'PROJECTOR_STATE.pt']
        receipt = {'evaluation_run_id': evaluation.name, 'independent_AP_recomputation': recomputed,
                   'memory_center_max_norm_error': norm_error, 'source_split_disjoint': True,
                   'train_only_exemplars': True, 'unused_prototype_slots': [c['prototype'] for c in memory['clusters'] if not c['support_tokens']],
                   'CPU_load_and_inference': loading,
                   'training_steps_in_acceptance_run': 0, 'scientific_goal_status': d['status'],
                   'local_transfer_verified': False}
        dump(run/'SUMMARY.json', receipt); dump(root/'SERVER_ACCEPTANCE_RECEIPT.json', receipt)
        dump(run/'DECISION.json', {'status': 'server_execution_and_model_loading_verified',
                                   'next_action': 'local_SHA_and_visual_acceptance',
                                   'five_pp_target_met': d['five_pp_metric_target_met']})
        meta.update(optimizer_steps=0, independent_AP_variants=len(recomputed), CPU_loaded_models=5)
        weight_receipt = bundle(root, 'SPM_V4_1_WEIGHTS.tar.gz', weights)
        print('WEIGHTS_BUNDLE', weight_receipt['size_bytes'], weight_receipt['sha256'], flush=True)
    # Include the now-completed acceptance Run and its artifact manifest.
    light = []
    for path in root.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in {'.json', '.md', '.py', '.log', '.svg', '.png', '.jpg', '.sbatch'}:
            continue
        rel = path.relative_to(root)
        if rel.parts[0] in ['cache', 'inputs', '__pycache__']: continue
        if rel.parts[0] == 'runs':
            if path.name.endswith('_predictions.json') and rel.parts[1] != folder(root, 'evaluate_task').name: continue
            # Failure and engineering logs are preserved; large arrays/weights are excluded by suffix.
            light.append(path)
        elif rel.parts[0] in ['scripts', 'render_plans'] or len(rel.parts) == 1:
            if 'scenes' in rel.parts: continue
            if path.name in ['REAL_DATASET.json', 'WARM_DATASET.json', 'ALL_DATASET.json']: continue
            if path.name.endswith('.receipt.json'): continue
            light.append(path)
    light += [root/'SPM_V4_1_WEIGHTS.tar.gz.receipt.json']
    receipt = bundle(root, 'SPM_V4_1_RESULTS_LIGHT.tar.gz', light)
    print('LIGHT_BUNDLE', receipt['size_bytes'], receipt['sha256'], len(light), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True); args = parser.parse_args()
    torch.set_num_threads(4); cv2.setNumThreads(1); main(Path(args.root))
