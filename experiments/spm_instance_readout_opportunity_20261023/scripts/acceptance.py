import argparse
from collections import defaultdict
import contextlib
import datetime as dt
import io
import os
from pathlib import Path
import sys
import traceback
import uuid

import numpy as np
from pycocotools import mask as cm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from readout_runtime import dump, read, sha


def recompute_ap(gt, predictions, kind):
    source = [{k: v for k, v in p.items() if k != ('bbox' if kind == 'segm' else 'segmentation')} for p in predictions]
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(gt)); detected = coco.loadRes(source)
        ev = COCOeval(coco, detected, kind)
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return (ev.stats*100).tolist()


def main(root):
    run = root/'runs'/('RUN_SPM_INSTANCE_ACCEPTANCE_'+uuid.uuid4().hex[:12]); run.mkdir(parents=True)
    meta = dict(study_id='STUDY_SPM_INSTANCE_READOUT_OPPORTUNITY_20261023', run_id=run.name,
                stage='independent_CPU_acceptance', execution_status='running', optimizer_steps=0,
                command=sys.argv, slurm_job_id=os.getenv('SLURM_JOB_ID'),
                started_at=dt.datetime.now(dt.timezone.utc).isoformat())
    dump(run/'run.json', meta)
    try:
        cfg = read(root/'CONFIG.json')
        source = root/'runs'/read(root/'OPPORTUNITY_POINTER.json')['run_id']
        assert read(source/'run.json')['execution_status'] == 'completed'
        contract = dict(protocol_sha256=sha(root/'PROTOCOL.md'), source_run_id=source.name,
                        acceptance_source_sha256=sha(__file__), source_manifest_sha256=sha(source/'ARTIFACT_MANIFEST.json'))
        dump(run/'EVAL_CONTRACT.json', contract)
        parent = Path(cfg['parent_root'])
        gt_path = parent/'DEV_COCO.json'
        gt_source = read(gt_path); anns = {a['id']: a for a in gt_source['annotations']}
        metrics = read(source/'METRICS.json')
        values = {}
        for name in ['native', 'T', 'U', 'R', 'T_wrong']:
            predictions = read(source/f'predictions_{name}.json')
            seg = recompute_ap(gt_path, predictions, 'segm')
            box = recompute_ap(gt_path, predictions, 'bbox')
            error = max(np.max(np.abs(np.asarray(seg)-metrics[name]['segm_stats'])),
                        np.max(np.abs(np.asarray(box)-metrics[name]['bbox_stats'])))
            assert error < 1e-9, (name, error)
            values[name] = dict(mask_ap=seg[0], all_24_stats_max_abs_error=float(error))
            print('INDEPENDENT_AP_PASS', name, seg[0], flush=True)
        original = read(parent/cfg['reference_predictions'])
        assert original == read(source/'predictions_native.json')
        identities = read(source/'OUTPUT_IDENTITIES.json')
        assert len(identities) == len(original)
        ledger = read(source/'OBJECT_LEDGER.json')
        summary = read(source/'OPPORTUNITY_SUMMARY.json')
        unique = set(); grouped = defaultdict(list); maximum = 0.
        for row in ledger:
            key = (row['image_id'], row['annotation_id'])
            assert key not in unique
            unique.add(key)
            source_identity = identities[row['coco_dt_id']-1]
            assert all(row[k] == source_identity[k] for k in source_identity)
            p = original[row['coco_dt_id']-1]; ann = anns[row['annotation_id']]
            assert p['image_id'] == ann['image_id'] == row['image_id'] and row['box_iou'] >= .75-1e-7
            native = float(cm.iou([p['segmentation']], [ann['segmentation']], [0])[0, 0])
            maximum = max(maximum, abs(native-row['native_mask_iou']))
            for name in ['T', 'U', 'R']:
                iou = float(cm.iou([row['oracle'][name]['segmentation']], [ann['segmentation']], [0])[0, 0])
                maximum = max(maximum, abs(iou-row['oracle'][name]['mask_iou']))
                if native < .75:
                    grouped[name].append((native, iou, row['oracle'][name]['prototype_available']))
        assert maximum < 1e-12
        independent = {}
        for name, records in grouped.items():
            n = len(records)
            gain = float(np.mean([b-a for a, b, _ in records]))
            repair = sum(b >= .75 for _, b, _ in records)/n
            support = sum(c for _, _, c in records)/n
            ref = summary[name]['failure']
            assert n == ref['objects'] and abs(gain-ref['mean_iou_delta']) < 1e-12
            assert repair == ref['mask75_fraction'] and support == ref['prototype_coverage']
            independent[name] = dict(objects=n, mean_iou_gain=gain, repair_fraction=repair, prototype_coverage=support)
        receipt = dict(source_run_id=source.name, all_24_AP_stats_per_variant=values,
                       official_matched_objects=len(ledger), object_IoU_max_abs_error=maximum,
                       independent_failure_summary=independent, native_predictions_exact=True,
                       scientific_decision=read(source/'DECISION.json'), GPU_tasks_in_this_run=0,
                       optimizer_steps=0, source_artifacts_unchanged=True)
        dump(run/'SUMMARY.json', receipt); dump(root/'SERVER_ACCEPTANCE_RECEIPT.json', receipt)
        meta.update(execution_status='completed', exit_code=0)
        dump(root/'ACCEPTANCE_POINTER.json', dict(run_id=run.name, slurm_job_id=meta['slurm_job_id']))
    except BaseException:
        (run/'ERROR.log').write_text(traceback.format_exc(), encoding='utf-8')
        meta.update(execution_status='failed', exit_code=1)
        raise
    finally:
        meta['ended_at'] = dt.datetime.now(dt.timezone.utc).isoformat(); dump(run/'run.json', meta)
        dump(run/'ARTIFACT_MANIFEST.json', {p.relative_to(run).as_posix(): dict(sha256=sha(p), size_bytes=p.stat().st_size)
                                           for p in run.rglob('*') if p.is_file() and p.name != 'ARTIFACT_MANIFEST.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    args = parser.parse_args(); main(Path(args.root))
