"""CPU independent official-COCO recalculation and frozen artifact acceptance."""
import argparse
import contextlib
import io
from pathlib import Path
import shutil
import sys

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from rt import context, dump, read, sha


def independent(gt_file, rows):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_file))
        detections = gt.loadRes([{k: v for k, v in p.items() if k != 'bbox'} for p in rows])
        evaluator = COCOeval(gt, detections, 'segm')
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    results = [float(x)*100 for x in evaluator.stats]
    t = np.flatnonzero(np.isclose(evaluator.params.iouThrs, .75))[0]
    success = {}
    for frame in evaluator.evalImgs:
        if frame is None or frame['aRng'] != [0, 1e10] or frame['maxDet'] != 100: continue
        for j, identity in enumerate(frame['gtIds']):
            if not frame['gtIgnore'][j]: success[int(identity)] = bool(frame['gtMatches'][t, j])
    return results, success


def execute(root):
    torch.set_num_threads(4)
    with context(root, 'acceptance') as (run, meta, cfg):
        source = root/'runs'/read(root/'FORMAL_POINTER.json')['run_id']
        actual = read(source/'run.json'); assert actual['execution_status'] == 'completed'
        assert actual['optimizer_steps'] == 8000
        frozen_post = read(root/'POSTPROCESS_SOURCE.json')
        for name, digest in frozen_post['files'].items():
            assert sha(root/name) == digest
            target = run/'POSTPROCESS_SOURCE'/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root/name, target)
        summary = read(source/'SUMMARY.json')
        cache = read(source/'CACHE_MANIFEST.json')
        verified = 0
        for item in cache['train']+cache['dev']:
            assert sha(item['file']) == item['sha256'], item['file']
            verified += 1
        for name, value in read(source/'ARTIFACT_MANIFEST.json').items():
            assert sha(source/name) == value['sha256'], name
        bank = read(root/'BANK_POINTER.json'); assert sha(bank['file']) == bank['bank_sha256']
        checks = {}
        for arm in cfg['arms']:
            receipt = read(source/(arm+'_TRAINING.json'))
            checkpoint = torch.load(source/(arm+'_last.pt'), map_location='cpu', weights_only=False)
            assert checkpoint['optimizer_steps'] == receipt['optimizer_steps'] == 4000
            assert sha(source/(arm+'_last.pt')) == receipt['checkpoint_sha256']
            assert checkpoint['bank_sha256'] == bank['bank_sha256']
            assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
            counts = np.load(source/(arm+'_exposures.npy'))
            assert counts.min() > 0 and int(counts.sum()) == 4000*64
            assert int(counts.min()) == receipt['min_exposure']
            checks[arm] = dict(steps=4000, sampler_objects=len(counts), exposures=int(counts.sum()),
                              min_exposure=int(counts.min()), checkpoint_sha256=receipt['checkpoint_sha256'])
        native = read(source/'native_predictions.json')
        gt = Path(cfg['parent_root'])/'DEV_COCO.json'
        tags = ['native', 'structured', 'scrambled_control', 'structured_off', 'structured_scrambled']
        errors, replays = [], {}
        original = None
        for tag in tags:
            rows = native if tag == 'native' else read(source/(tag+'_predictions.json'))
            assert len(rows) == len(native)
            for a, b in zip(native, rows):
                assert all(a[k] == b[k] for k in ('image_id', 'category_id', 'score', 'bbox'))
            metrics, success = independent(gt, rows)
            errors += np.abs(np.asarray(metrics)-np.asarray(summary['metrics'][tag]['stats'])).tolist()
            if original is None: original = success
            if tag != 'native':
                assert set(success) == set(original)
                damage = sum(ok and not success[g] for g, ok in original.items())
                repair = sum(not ok and success[g] for g, ok in original.items())
                assert damage == summary['damage'][tag]['success_to_failure']
                assert repair == summary['damage'][tag]['failure_to_success']
            replays[tag] = metrics
        error = max(errors); assert error < 1e-10
        result = dict(independent_acceptance_passed=True, source_run_id=source.name,
                      source_job_id=actual['slurm_job_id'], optimizer_steps=8000,
                      training_cache_objects_verified=verified, primary_artifacts_verified=len(read(source/'ARTIFACT_MANIFEST.json')),
                      official_COCO_statistics_verified=len(errors), maximum_metric_absolute_error_pp=error,
                      same_detection_box_class_score_order=True, checkpoint_checks=checks,
                      damage_and_repair_counts_exact=True, original_decision=read(source/'DECISION.json'),
                      real_test_used=False)
        dump(run/'SUMMARY.json', result); dump(run/'INDEPENDENT_METRICS.json', replays)
        print('INDEPENDENT_ACCEPTANCE', result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    a = p.parse_args(); execute(a.root.resolve())
