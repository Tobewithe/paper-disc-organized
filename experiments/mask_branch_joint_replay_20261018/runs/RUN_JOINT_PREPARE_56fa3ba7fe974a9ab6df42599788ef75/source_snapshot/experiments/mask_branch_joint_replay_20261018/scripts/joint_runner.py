"""Finite, explicitly invoked joint-adaptation stages. No daemon, scheduler or retries.

Usage: python joint_runner.py --config CONFIG.json prepare|native|smoke|train|eval
       train/eval additionally require --arm G_budget|C_native|PC_native.
Every invocation has its own immutable completed Run; partial failures remain.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import sys
import time
import traceback
import uuid

from joint_data import ROOT, read, write, sha, require, prepare, NativeBatches, batch_identity, tensor_sha

ARMS = ('G_budget', 'C_native', 'PC_native')
OPTIMIZER = {'lr': 1e-4, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 1e-4,
             'amsgrad': False, 'foreach': False, 'maximize': False, 'capturable': False,
             'differentiable': False, 'fused': False}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sources():
    files = [Path(__file__), Path(__file__).with_name('joint_data.py'), Path(__file__).with_name('joint_model.py')]
    files += [ROOT / 'experiments/mask_readout_mechanism_v2_20261012/scripts' / n for n in
              ('adapter_model.py', 'adapter_data.py', 'adapter_primitives.py', 's1_geometry.py', 'posttrain_run.py')]
    files += [ROOT / 'experiments/mask_readout_soft_iou_pilot_20261014/scripts/soft_iou_loss.py',
              ROOT / 'shared/evaluation/native_output_bridge.py', ROOT / 'shared/evaluation/native_source_pins.json',
              ROOT / 'shared/evaluation/posttrain_audit.py',
              ROOT / 'experiments/official_gt_failure_rebaseline_20261013/scripts/build_native_ledger.py']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


class Run:
    def __init__(self, config_path, stage, arm=None):
        self.config_path = Path(config_path).resolve()
        self.config = read(self.config_path)
        self.study = Path(self.config.get('study_root', Path(__file__).resolve().parents[1])).resolve()
        self.key = stage.upper() + ('_' + arm if arm else '')
        self.stage, self.arm = stage, arm
        self.path = self.study / 'runs' / ('RUN_JOINT_' + self.key + '_' + uuid.uuid4().hex)
        self.lock = self.study / '.JOINT_STAGE.lock'
        self.started = time.monotonic()
        self.budget = (self.config.get('training_hours', 8) * 3600 if stage == 'train'
                       else self.config.get(stage + '_hours', 8) * 3600)
        self.meta = {'run_id': self.path.name, 'stage': stage, 'arm': arm, 'started_at': now(),
                     'execution_status': 'running', 'scientific_status': 'not_adjudicated',
                     'pid': os.getpid(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                     'optimizer_steps': 0, 'optimization_started': False, 'config_sha256': sha(self.config_path)}

    def __enter__(self):
        self.study.mkdir(parents=True, exist_ok=True)
        try:
            with self.lock.open('x', encoding='utf-8') as f:
                json.dump({'run_id': self.path.name, 'pid': os.getpid(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID')}, f)
        except FileExistsError:
            raise RuntimeError('A stage lock exists; inspect its real process before any explicit recovery')
        try:
            pointers = read(self.study / 'POINTERS.json') if (self.study / 'POINTERS.json').exists() else {}
            require(self.key not in pointers, f'{self.key} already completed; no repeat run')
            if self.stage == 'train':
                for old in (self.study / 'runs').glob('*/run.json'):
                    record = read(old)
                    require(not (record.get('stage') == 'train' and record.get('arm') == self.arm and record.get('optimization_started')),
                            'this arm already consumed training budget; no automatic restart')
            self.path.mkdir(parents=True, exist_ok=False)
            frozen_path = Path(self.config['frozen_manifest'])
            frozen = read(frozen_path)
            require(frozen['config_sha256'] == self.meta['config_sha256'], 'config differs from frozen manifest')
            require(sha(self.config['protocol']) == frozen['protocol_sha256'], 'protocol differs from frozen manifest')
            for record in frozen['files']:
                source = (ROOT / record['path']).resolve()
                require(source.is_relative_to(ROOT.resolve()) and sha(source) == record['sha256'], 'frozen source/input changed: '+record['path'])
            self.meta.update(study_id=self.config['study_id'], code_commit=frozen['code_commit'],
                             protocol_sha256=frozen['protocol_sha256'], frozen_manifest_sha256=sha(frozen_path),
                             command=sys.argv, host=__import__('socket').gethostname())
            self.source_hashes = sources()
            write(self.path / 'SOURCE_MANIFEST.json', self.source_hashes)
            for rel in self.source_hashes:
                target = self.path / 'source_snapshot' / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / rel, target)
            shutil.copyfile(self.config_path, self.path / 'CONFIG_FROZEN.json')
            self.update(phase='initializing')
            return self
        except BaseException as error:
            if self.path.exists():
                (self.path/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
                self.update(execution_status='failed', scientific_status='engineering_failure', error=str(error), finished_at=now())
            self.lock.unlink(missing_ok=True)
            raise

    def pointer(self, key):
        record = read(self.study / 'POINTERS.json')[key]
        path = self.study / 'runs' / record['run_id']
        require(read(path / 'run.json')['execution_status'] == 'completed', f'{key} is not completed')
        require(sha(path / 'SUMMARY.json') == record['summary_sha256'], f'{key} summary changed')
        require(read(path / 'SOURCE_MANIFEST.json') == self.source_hashes, f'source changed since {key}')
        require(read(path / 'run.json')['config_sha256'] == self.meta['config_sha256'], f'configuration changed since {key}')
        return path

    def update(self, **values):
        self.meta.update(values)
        self.meta['updated_at'] = now()
        self.meta['elapsed_seconds'] = time.monotonic() - self.started
        write(self.path / 'run.json', self.meta, replace=True)
        write(self.study / 'EXECUTION_STATUS.json', self.meta, replace=True)

    def check(self):
        require(time.monotonic() - self.started <= self.budget, 'frozen stage time budget exceeded')
        if 'torch' in sys.modules:
            import torch
            if torch.cuda.is_initialized():
                used = max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
                require(used <= self.config.get('gpu_memory_limit_gib', 44) * 2**30, 'frozen GPU memory budget exceeded')

    def finish(self, summary, science='not_adjudicated'):
        require(sources() == self.source_hashes and sha(self.config_path) == self.meta['config_sha256'], 'sources/config changed during stage')
        self.check()
        write(self.path / 'SUMMARY.json', summary)
        self.update(execution_status='completed', scientific_status=science, finished_at=now(), phase='completed')
        manifest = {p.relative_to(self.path).as_posix(): sha(p) for p in self.path.rglob('*') if p.is_file()}
        write(self.path / 'ARTIFACT_MANIFEST.json', manifest)
        pointers = read(self.study / 'POINTERS.json') if (self.study / 'POINTERS.json').exists() else {}
        pointers[self.key] = {'run_id': self.path.name, 'summary_sha256': sha(self.path / 'SUMMARY.json')}
        write(self.study / 'POINTERS.json', pointers, replace=True)
        print('STAGE_COMPLETED', self.key, self.path.name, flush=True)

    def __exit__(self, kind, value, tb):
        if kind is not None:
            (self.path / 'ERROR.txt').write_text(''.join(traceback.format_exception(kind, value, tb)), encoding='utf-8')
            self.update(execution_status='failed', scientific_status='engineering_failure', error=str(value), finished_at=now())
        self.lock.unlink(missing_ok=True)


def environment(run):
    import torch
    from native_output_bridge import apply_runtime_policy, native_environment
    profile = read(run.config['profile'])
    apply_runtime_policy(profile['runtime_policy'])
    for key, value in run.config.get('expected_packages', {}).items():
        require(importlib.metadata.version(key) == value, f'package drift {key}')
    require(tuple(sys.version_info[:3]) == (3, 12, 3), 'frozen Python 3.12.3 required')
    torch.manual_seed(run.config.get('training_seed', 0))
    random.seed(run.config.get('training_seed', 0))
    require(torch.cuda.is_available(), 'CUDA GPU required')
    actual = native_environment()
    actual['gpu_name'] = torch.cuda.get_device_name(0)
    actual['gpu_total_bytes'] = torch.cuda.get_device_properties(0).total_memory
    write(run.path / 'ENVIRONMENT.json', actual)
    return actual


def branch(run, native, arm, checkpoint=None):
    import torch
    from joint_model import JointMaskBranch
    torch.manual_seed(run.config.get('training_seed', 0))
    model = JointMaskBranch(native.head, arm, run.config['adapter_initial'] if arm == 'G_budget' else None).cuda()
    if checkpoint:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        require(saved['arm'] == arm and saved['optimizer_steps'] == 15000 and saved['completed_epochs'] == 3, 'only complete final checkpoint may be evaluated')
        model.load_state_dict(saved['state_dict'], strict=True)
    return model


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(tensor.dtype).encode()); h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(run, name, model, optimizer, steps, epochs):
    import torch
    path = run.path / name
    require(not path.exists(), 'checkpoint overwrite forbidden')
    tmp = path.with_suffix('.pt.tmp')
    torch.save({'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'arm': model.arm, 'optimizer_steps': steps, 'completed_epochs': epochs,
                'seed': run.config.get('training_seed', 0), 'config_sha256': run.meta['config_sha256'],
                'source_manifest_sha256': sha(run.path / 'SOURCE_MANIFEST.json')}, tmp)
    os.replace(tmp, path)


def optimizer(model):
    import torch
    spec = dict(OPTIMIZER)
    spec['betas'] = tuple(spec['betas'])
    return torch.optim.AdamW(model.parameters(), **spec)


def decoded_rows(cache, native_rows, output, gate=9216):
    """Only segmentation changes; every final detection row remains present and in order."""
    from native_output_bridge import decode_native, encode_rle
    from pycocotools import mask as masks
    if not native_rows:
        require(len(output['current_coefficients']) == 0, 'empty native output gained rows')
        return []
    changed_cache = dict(cache, proto=output['current_proto'].detach())
    decoded = decode_native(changed_cache, output['current_coefficients'].detach())
    result = []
    for i, original in enumerate(native_rows):
        row = dict(original)
        if int(masks.area(original['segmentation'])) >= gate:
            row['segmentation'] = encode_rle(decoded[i])
        result.append(row)
    return result


class PredictionWriter:
    def __init__(self, path):
        self.stream = Path(path).open('x', encoding='utf-8')
        self.stream.write('[')
        self.count = 0
    def append(self, rows):
        for row in rows:
            if self.count:
                self.stream.write(',\n')
            self.stream.write(json.dumps(row, allow_nan=False))
            self.count += 1
        self.stream.flush()
    def finish(self):
        self.stream.write(']\n'); self.stream.close()


def evaluate_predictions(run, name, ids):
    from build_native_ledger import evaluate_ledger, load_results
    from faster_coco_eval import COCOeval_faster
    from pycocotools.coco import COCO
    path = run.path / (name + '_predictions.json')
    predictions = read(path)
    annotation = run.config['val_annotations']
    with (run.path / (name + '_cocoeval.log')).open('x', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        ledger, metrics = evaluate_ledger(annotation, predictions, ids, sha(path))
        gt = COCO(annotation)
        # Preserve accelerated evaluator evalImgs for the separately frozen paired bootstrap.
        native_metrics = {}
        for task in ('bbox', 'segm'):
            evaluator = COCOeval_faster(gt, load_results(gt, predictions), task)
            evaluator.params.imgIds = list(ids)
            evaluator.params.maxDets = [1, 10, 100]
            evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
            native_metrics[task] = 100 * float(evaluator.stats[0])
            require(abs(native_metrics[task] - metrics[task]['AP']) <= .02, 'COCO backend discrepancy exceeds .02 AP')
    with (run.path / (name + '_PER_GT.jsonl')).open('x', encoding='utf-8') as stream:
        for row in ledger:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    result = {'predictions_sha256': sha(path), 'outputs': len(predictions), 'images': len(ids),
              'metrics': metrics, 'faster_coco_eval_AP': native_metrics,
              'ordinary_gt': sum(not row['is_ignored'] for row in ledger),
              'mask75_success': sum(row['task_mask75'] == 'success' for row in ledger)}
    write(run.path / (name + '_METRICS.json'), result)
    return result, ledger


def native_stage(run):
    import torch
    from adapter_model import CoefficientAdapter
    from native_output_bridge import decode_native
    from joint_model import snapshot_state, assert_state_unchanged
    environment(run)
    prepared = run.pointer('PREPARE')
    native = NativeBatches(run.config, prepared, 'val', run.path / 'native_config')
    before = snapshot_state(native.model)
    old = CoefficientAdapter('B').cuda().eval()
    old.load_state_dict(torch.load(run.config['existing_G'], map_location='cpu', weights_only=True)['state_dict'], strict=True)
    writers = {name: PredictionWriter(run.path / (name + '_predictions.json')) for name in ('Ns', 'G_old_s')}
    identities, timings, output_map = [], [], []
    historical = read(run.config['historical_native_trace'])
    historical_images = {r['image_id']:r for r in historical['images']}
    geometry_checks = []
    with torch.no_grad():
        for index in range(len(native.plan)):
            run.check()
            caches, rows, features = native.batch(index, masks=True)
            for cache in caches:
                previous_cache = historical_images[cache['image_id']]
                for key in ('image_sha256','input_tensor_sha256','input_hw','original_hw','ratio_pad','batch_index','batch_position','head_feature_shapes'):
                    require(json.loads(json.dumps(cache[key])) == json.loads(json.dumps(previous_cache[key])), 'historical native geometry drift: '+key)
                geometry_checks.append({'image_id':cache['image_id'], 'historical_input_and_geometry_exact':True,
                                        'cross_device_P_c_bitwise_parity':'not_required; new native references define all method comparisons'})
            identities.append(batch_identity(caches, rows, features))
            torch.cuda.synchronize()
            method_started = time.perf_counter()
            for cache, native_rows in zip(caches, rows):
                for rank, raw_id in enumerate(cache['raw_ids']):
                    output_map.append({'image_id':cache['image_id'], 'output_rank':rank, 'raw_id':raw_id,
                                       'source_pred_json_index':writers['Ns'].count+rank,
                                       'source_batch_index':index, 'source_batch_position':cache['batch_position']})
                delta = old(cache, 'eval:0')
                output = {'current_proto': cache['proto'], 'current_coefficients': cache['c0'] + delta}
                writers['Ns'].append(native_rows)
                writers['G_old_s'].append(decoded_rows(cache, native_rows, output, run.config.get('area_gate', 9216)))
            torch.cuda.synchronize()
            timings.append({**native.last_timing, 'G_old_additional_seconds_including_write': time.perf_counter()-method_started})
            run.update(phase='native_and_existing_G_export', source_batches=index+1, images=sum(len(b['image_ids']) for b in native.plan[:index+1]), output_rows=writers['Ns'].count)
            print('NATIVE', index+1, len(native.plan), flush=True)
            del caches, rows, features, output, delta
            torch.cuda.empty_cache()
    for writer in writers.values(): writer.finish()
    write(run.path / 'NATIVE_IDENTITY.json', identities)
    write(run.path / 'BATCH_TIMINGS.json', timings)
    write(run.path / 'HISTORICAL_GEOMETRY_REPLAY.json', geometry_checks)
    prediction_sha = sha(run.path/'Ns_predictions.json')
    with (run.path/'OUTPUT_ID_MAP.jsonl').open('x',encoding='utf-8') as stream:
        for row in output_map:
            row['output_detection_id'] = prediction_sha+':'+str(row['source_pred_json_index'])
            stream.write(json.dumps(row)+'\n')
    write(run.path / 'FROZEN_BASE_CHECK.json', assert_state_unchanged(native.model, before))
    del native, old, before
    torch.cuda.empty_cache()
    ids = read(prepared / 'DATA_CONTRACT.json')['val_ids']
    ns, nledger = evaluate_predictions(run, 'Ns', ids)
    gs, gledger = evaluate_predictions(run, 'G_old_s', ids)
    require(ns['ordinary_gt'] == 36335, 'complete ordinary COCO val GT denominator changed')
    damaged, repaired = compare_ledgers(nledger, gledger)
    old_success = {row['annotation_id'] for row in
        (json.loads(line) for line in Path(run.config['historical_native_ledger']).read_text(encoding='utf-8').splitlines())
        if row['task_mask75']=='success'}
    new_success = {row['annotation_id'] for row in nledger if row['task_mask75']=='success'}
    write(run.path/'SUCCESS_POPULATION_MIGRATION.json', {'historical_success_count':len(old_success),
        'current_success_count':len(new_success), 'historical_only_ids':sorted(old_success-new_success),
        'current_only_ids':sorted(new_success-old_success), 'damage_denominator_policy':'current Ns exact full ledger'})
    write(run.path / 'G_old_s_DAMAGE.json', {'damaged_annotation_ids': damaged, 'repaired_annotation_ids': repaired,
          'native_success_denominator': ns['mask75_success'], 'damage_fraction': len(damaged)/ns['mask75_success']})
    migration = run.config['migration_guard']
    previous = {'bbox_AP':migration['historical_bbox_ap'], 'segm_AP':migration['historical_segm_ap'],
                'G_gain':migration['historical_G_segm_ap']-migration['historical_segm_ap']}
    delta = gs['metrics']['segm']['AP'] - ns['metrics']['segm']['AP']
    checks = {'native_box_AP': abs(ns['metrics']['bbox']['AP'] - previous['bbox_AP']) <= migration['max_absolute_ap_drift'],
              'native_mask_AP': abs(ns['metrics']['segm']['AP'] - previous['segm_AP']) <= migration['max_absolute_ap_drift'],
              'existing_G_gain': abs(delta - previous['G_gain']) <= migration['max_G_gain_drift']}
    summary = {'Ns': ns, 'G_old_s': gs, 'G_old_gain_AP_points': delta, 'migration_checks': checks,
               'migration_status': 'passed' if all(checks.values()) else 'failed',
               'native_gt_ledger_validity': 'passed', 'published_parity': 'passed' if (round(ns['metrics']['bbox']['AP'], 1), round(ns['metrics']['segm']['AP'], 1)) == (52.5,44.1) else 'failed',
               'all_output_identity_unchanged': True, 'additional_training_steps': 0}
    run.finish(summary, 'same_host_reference_verified' if all(checks.values()) else 'migration_guard_failed_stop_training')
    # Completed scientific guard failures remain completed evidence. main returns
    # nonzero only AFTER releasing the stage context; no manifest is rewritten.


def compare_ledgers(native, current):
    original = {r['annotation_id']: r for r in native}
    require(set(original) == {r['annotation_id'] for r in current}, 'GT ledger identity differs')
    damaged, repaired = [], []
    for row in current:
        old = original[row['annotation_id']]
        require(old['image_id'] == row['image_id'] and old['is_ignored'] == row['is_ignored'], 'GT identity drift')
        # Hash prefixes differ by method; COCO row index must not differ for the bbox match.
        def rank(value): return None if value is None else value.rsplit(':', 1)[1]
        require(old['task_box75'] == row['task_box75'] and rank(old['bbox75_match_output_id']) == rank(row['bbox75_match_output_id']), 'Box matched detection changed')
        if not old['is_ignored']:
            if old['task_mask75'] == 'success' and row['task_mask75'] != 'success': damaged.append(row['annotation_id'])
            if old['task_mask75'] == 'failure' and row['task_mask75'] == 'success': repaired.append(row['annotation_id'])
    return sorted(damaged), sorted(repaired)


def smoke_stage(run):
    import torch
    from pycocotools.coco import COCO
    from adapter_data import training_join, projected_targets
    from native_output_bridge import decode_native, encode_rle
    from joint_model import snapshot_state
    environment(run)
    prepared = run.pointer('PREPARE')
    migration = run.pointer('NATIVE')
    require(read(migration / 'SUMMARY.json')['migration_status'] == 'passed', 'migration prerequisite failed')
    train = NativeBatches(run.config, prepared, 'train', run.path / 'train_config')
    snapshot = snapshot_state(train.model)
    source_started = time.perf_counter()
    worst_batch = max(range(len(train.plan)), key=lambda index:(train.plan[index]['rect_shape'][0]*train.plan[index]['rect_shape'][1],-index))
    caches, rows, features = train.batch(worst_batch, masks=True)
    torch.cuda.synchronize()
    source_batch_seconds = time.perf_counter() - source_started
    coco = COCO(run.config['train_annotations'])
    receipts = {}
    models = {arm: branch(run, train, arm) for arm in ARMS}
    for arm, model in models.items():
        identity = []
        with torch.no_grad():
            initialized_outputs = model.forward_batch(caches,features,list(range(len(caches))),phase='eval:0')
        for i, (cache, native_rows) in enumerate(zip(caches, rows)):
            out = initialized_outputs[i]
            checked = {key:bool(torch.equal(out[a],out[b])) for key,a,b in (
                ('proto_equal','current_proto','native_proto'),('coefficients_equal','current_coefficients','native_coefficients'),
                ('logits_equal','logits','baseline_logits'))}
            require(all(checked.values()), 'initial complete source-batch P/c/logit identity failed')
            with torch.no_grad():
                if native_rows:
                    decoded = decode_native(dict(cache, proto=out['current_proto']), out['current_coefficients'])
                    require(all(encode_rle(decoded[j]) == row['segmentation'] for j, row in enumerate(native_rows)), 'initial branch RLE mismatch')
            identity.append({'image_id': cache['image_id'], **checked, 'RLE_exact': True})
        del initialized_outputs
        opt = optimizer(model)
        gradients = set()
        initial = state_hash(model)
        step_seconds = []
        for start in (0, 4):
            run.check()
            torch.cuda.synchronize()
            step_started = time.perf_counter()
            selection = list(range(start,start+4))
            opt.zero_grad(set_to_none=True)
            outs = model.forward_batch([caches[i] for i in selection], features, selection, phase='train:0')
            losses = []
            for i, out in zip(selection, outs):
                targets = projected_targets(coco, caches[i], training_join(rows[i], coco.imgToAnns.get(caches[i]['image_id'], [])))
                loss, _ = model.loss(out, targets); losses.append(loss)
            loss = torch.stack(losses).mean()
            require(bool(torch.isfinite(loss)), 'smoke loss nonfinite')
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    require(bool(torch.isfinite(parameter.grad).all()), 'smoke gradient nonfinite')
                    if bool(torch.count_nonzero(parameter.grad)): gradients.add(name)
            opt.step()
            torch.cuda.synchronize()
            step_seconds.append(time.perf_counter()-step_started)
            run.update(phase='engineering_smoke', optimizer_steps=run.meta['optimizer_steps']+1, optimization_started=True)
            del outs, losses, loss
        require(gradients and state_hash(model) != initial, 'smoke branch did not learn')
        represented_levels = {loc['level'] for c in caches[:8] for loc in c['raw_locations']}
        permitted_inactive = set()
        for name,p in model.named_parameters():
            for level in set(range(3))-represented_levels:
                if name.startswith(f'adapter.feature.{level}.') or name.startswith(f'coefficient_head.{level}.'):
                    permitted_inactive.add(name)
        require({name for name,p in model.named_parameters() if p.requires_grad} - gradients <= permitted_inactive,
                'smoke has unexplained trainable tensors without finite nonzero gradients')
        if arm == 'PC_native': require(any(n.startswith('proto.') for n in gradients), 'P detached: no Proto gradient')
        if arm != 'G_budget': require(any(n.startswith('coefficient_head.') for n in gradients), 'no coefficient gradient')
        model.verify_frozen_state(train.model, snapshot)
        estimate_seconds = (max(step_seconds) + source_batch_seconds/4) * 15000 / .8
        receipts[arm] = {'initial_replay': identity, 'optimizer_steps': 2, 'gradient_names': sorted(gradients),
                         'parameters': model.parameter_manifest(), 'frozen_base_verified': True,
                         'permitted_inactive_tensors_due_to_no_selected_native_raw_level':sorted(permitted_inactive-gradients),
                         'measured_step_seconds':step_seconds, 'source_batch_seconds':source_batch_seconds,
                         'train_source_batch_index':worst_batch, 'smoke_batch_selection':'maximum native rect pixels, lowest index tie-break',
                         'projected_formal_seconds_with_20percent_reserve':estimate_seconds,
                         'cost_method':'max of two synchronized microsteps + complete native source batch / 4; multiplied by 15000 / .8; conservative estimate, not guarantee'}
        write(run.path / (arm+'_SMOKE_RECEIPT.json'), receipts[arm])
        require(estimate_seconds <= run.config.get('training_hours',8)*3600, 'smoke forecast exceeds fixed 8h arm budget including 20% reserve')
        del opt
    del caches, rows, features, train, models, snapshot, coco
    torch.cuda.empty_cache()
    # A separate 16-image official val source batch tests inference-side geometry.
    val = NativeBatches(run.config, prepared, 'val', run.path / 'val_config')
    caches, rows, features = val.batch(0, masks=True)
    for arm in ARMS:
        model = branch(run, val, arm)
        with torch.no_grad():
            initialized_outputs = model.forward_batch(caches,features,list(range(len(caches))),phase='eval:0')
        for i, (cache, native_rows) in enumerate(zip(caches, rows)):
            out = initialized_outputs[i]
            require(torch.equal(out['current_proto'],out['native_proto']) and torch.equal(out['current_coefficients'],out['native_coefficients']), 'val initial tensor replay mismatch')
            with torch.no_grad():
                require(decoded_rows(cache, native_rows, out) == native_rows, 'val zero-delta gated output mismatch')
        del model, initialized_outputs
    require(torch.cuda.max_memory_reserved() <= run.config.get('gpu_memory_limit_gib',44)*2**30, 'smoke exceeds resource budget')
    write(run.path / 'ENGINEERING_RECEIPTS.json', receipts)
    run.finish({'status': 'passed', 'images': 32, 'native_forward_images': 32, 'optimizer_steps': 6,
                'steps_per_arm': 2, 'all_arms_exact_initial_RLE': True, 'all_arms_finite_gradients': True,
                'PC_proto_and_coefficient_gradients': True, 'max_cuda_allocated': torch.cuda.max_memory_allocated(),
                'max_cuda_reserved': torch.cuda.max_memory_reserved()}, 'engineering_only')


def train_stage(run):
    import torch
    from pycocotools.coco import COCO
    from adapter_data import training_join, projected_targets
    from joint_model import snapshot_state
    environment(run)
    prepared, smoke = run.pointer('PREPARE'), run.pointer('SMOKE')
    require(read(smoke / 'SUMMARY.json')['status'] == 'passed', 'engineering smoke prerequisite failed')
    require(read(run.pointer('NATIVE') / 'SUMMARY.json')['migration_status'] == 'passed', 'migration prerequisite failed')
    if run.arm != 'G_budget':
        reference = run.pointer('TRAIN_G_budget')
        expected_identities = read(reference / 'TRAINING_NATIVE_IDENTITY.json')
        expected_joins = read(reference / 'TRAINING_JOIN.json')
    native = NativeBatches(run.config, prepared, 'train', run.path / 'native_config')
    snapshot = snapshot_state(native.model)
    model = branch(run, native, run.arm)
    opt = optimizer(model)
    actual_opt = {k:v for k,v in opt.param_groups[0].items() if k != 'params'}
    write(run.path / 'ACTUAL_OPTIMIZER.json', actual_opt)
    require(json.loads(json.dumps(actual_opt)) == OPTIMIZER, 'effective optimizer differs from frozen choice')
    write(run.path / 'PARAMETER_MANIFEST.json', model.parameter_manifest())
    orders = read(prepared / 'IMAGE_ORDERS.json')
    shutil.copyfile(prepared / 'IMAGE_ORDERS.json', run.path / 'IMAGE_ORDERS.json')
    coco = COCO(run.config['train_annotations'])
    joins, identities, gradients = {}, {}, set()
    parameter_update_counts = {name:0 for name,p in model.named_parameters()}
    steps = images = empty_rois = no_matches = 0
    initial = state_hash(model)
    save_checkpoint(run, 'initial.pt', model, opt, 0, 0)
    with (run.path / 'training.jsonl').open('x', encoding='utf-8') as log, (run.path/'BATCH_TIMINGS.jsonl').open('x',encoding='utf-8') as timing_log:
        for epoch, sequence in enumerate(orders):
            for item in sequence:
                run.check()
                source_batch_started = time.perf_counter()
                index = item['source_batch_index']
                caches, rows, features = native.batch(index, masks=False)
                actual_identity = batch_identity(caches, rows, features)
                # Normalize tuples through JSON so receipts compare identically after reload.
                actual_identity = json.loads(json.dumps(actual_identity))
                if str(index) in identities:
                    require(identities[str(index)] == actual_identity, 'native tensors/geometry changed between epochs')
                else:
                    if run.arm != 'G_budget': require(expected_identities[str(index)] == actual_identity, 'native tensors/geometry differ across arms')
                    identities[str(index)] = actual_identity
                targets = []
                for cache, pred in zip(caches, rows):
                    iid = str(cache['image_id'])
                    join = training_join(pred, coco.imgToAnns.get(cache['image_id'], []))
                    normalized = {str(k): v for k,v in join.items()}
                    if iid in joins: require(joins[iid] == normalized, 'training join changed between epochs')
                    elif run.arm != 'G_budget': require(expected_joins[iid] == normalized, 'training join differs across arms')
                    joins[iid] = normalized
                    targets.append(projected_targets(coco, cache, join))
                if steps == 0:
                    for i, cache in enumerate(caches): model.verify_initial_replay(cache, features, batch_index=i)
                    run.update(optimization_started=True, phase='formal_training')
                for selection in item['microgroups']:
                    run.check()
                    opt.zero_grad(set_to_none=True)
                    outs = model.forward_batch([caches[i] for i in selection], features, selection, phase='train:'+str(epoch))
                    losses = []
                    for i, out in zip(selection, outs):
                        loss, accounting = model.loss(out, targets[i])
                        require(bool(torch.isfinite(loss)), 'training loss nonfinite')
                        losses.append(loss)
                        empty_rois += len(accounting['empty_roi_rows'])
                        no_matches += int(accounting['matched_supervised_instances'] == 0)
                        log.write(json.dumps({'epoch':epoch, 'step':steps+1, 'image_id':caches[i]['image_id'],
                                              'loss':float(loss.detach()), **accounting}, allow_nan=False)+'\n')
                    loss = torch.stack(losses).mean()
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            parameter_update_counts[name] += 1
                            require(bool(torch.isfinite(parameter.grad).all()), 'training gradient nonfinite')
                            if bool(torch.count_nonzero(parameter.grad)): gradients.add(name)
                    opt.step()
                    steps += 1; images += len(selection)
                    del outs, losses, loss
                    if steps % 25 == 0:
                        log.flush()
                        run.update(optimizer_steps=steps, images_seen=images, epoch=epoch+1)
                        print('TRAIN', run.arm, epoch+1, steps, images, flush=True)
                torch.cuda.synchronize()
                timing_log.write(json.dumps({'epoch':epoch, 'source_batch_index':index, 'ending_optimizer_step':steps,
                    'wall_seconds_including_input_targets_and_audit':time.perf_counter()-source_batch_started,
                    **native.last_timing, 'max_cuda_allocated':torch.cuda.max_memory_allocated(),
                    'max_cuda_reserved':torch.cuda.max_memory_reserved()})+'\n')
                timing_log.flush()
                del caches, rows, features, targets
                torch.cuda.empty_cache()
            model.verify_frozen_state(native.model, snapshot)
            save_checkpoint(run, 'last.pt' if epoch == 2 else f'epoch_{epoch+1}_unselected.pt', model, opt, steps, epoch+1)
    require(steps == 15000 and images == 60000 and len(joins) == 20000, 'training budget/coverage incomplete')
    require(state_hash(model) != initial and gradients, 'trained branch unchanged')
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'nonfinite checkpoint')
    for name,parameter in model.named_parameters():
        state = opt.state.get(parameter, {})
        actual_step = int(state['step']) if 'step' in state else 0
        require(actual_step == parameter_update_counts[name], 'parameter optimizer step does not equal its actual non-None gradient update count')
    write(run.path/'PARAMETER_UPDATE_COUNTS.json', {'global_optimizer_steps':steps, 'per_parameter':parameter_update_counts,
          'policy':'retain original AdamW grad=None skip semantics; counters measure actual eligible updates, no manufactured zero gradients'})
    write(run.path / 'TRAINING_JOIN.json', joins)
    write(run.path / 'TRAINING_NATIVE_IDENTITY.json', identities)
    frozen = model.verify_frozen_state(native.model, snapshot)
    write(run.path / 'CONFIG_AUDIT.json', {'status': 'passed', 'frozen_base': frozen,
          'optimizer_steps': steps, 'images_seen': images, 'same_cross_arm_native_inputs_and_GT_join': True,
          'final_checkpoint_sha256': sha(run.path / 'last.pt'), 'runtime_policy': read(run.config['profile'])['runtime_policy']})
    run.update(optimizer_steps=steps, images_seen=images)
    run.finish({'arm':run.arm, 'optimizer_steps':steps, 'images_seen':images, 'epochs':3,
                'initial_state_sha256':initial, 'final_state_sha256':state_hash(model),
                'checkpoint_sha256':sha(run.path / 'last.pt'), 'checkpoint_selection':'last_only',
                'parameters':model.parameter_manifest(), 'parameters_with_nonzero_gradient':sorted(gradients),
                'parameter_update_counts':parameter_update_counts,
                'empty_roi_instances':empty_rois, 'images_without_matched_supervision':no_matches,
                'max_cuda_allocated':torch.cuda.max_memory_allocated(), 'max_cuda_reserved':torch.cuda.max_memory_reserved()},
               'trained_not_evaluated')


def eval_stage(run):
    import torch
    from joint_model import snapshot_state
    environment(run)
    prepared, reference = run.pointer('PREPARE'), run.pointer('NATIVE')
    trained = run.pointer('TRAIN_' + run.arm)
    checkpoint = trained / 'last.pt'
    require(sha(checkpoint) == read(trained/'SUMMARY.json')['checkpoint_sha256'], 'final checkpoint changed')
    native = NativeBatches(run.config, prepared, 'val', run.path / 'native_config')
    snapshot = snapshot_state(native.model)
    model = branch(run, native, run.arm, checkpoint)
    expected = read(reference / 'NATIVE_IDENTITY.json')
    writer = PredictionWriter(run.path / (run.arm + '_predictions.json'))
    timings = []
    with torch.no_grad():
        for index in range(len(native.plan)):
            run.check()
            caches, rows, features = native.batch(index, masks=True)
            require(json.loads(json.dumps(batch_identity(caches, rows, features))) == expected[index], 'evaluation native replay differs from same-host reference')
            head_seconds = decode_seconds = 0.
            selection = list(range(len(caches)))
            torch.cuda.synchronize()
            method_started = time.perf_counter()
            # Inference has no intervening optimizer steps: evaluate the mask
            # head ONCE on its original full source batch, not four times.
            outs = model.forward_batch(caches, features, selection, phase='eval:0')
            torch.cuda.synchronize()
            head_seconds += time.perf_counter()-method_started
            for i, out in enumerate(outs):
                torch.cuda.synchronize()
                decode_started = time.perf_counter()
                converted = decoded_rows(caches[i], rows[i], out, run.config.get('area_gate',9216))
                torch.cuda.synchronize()
                decode_seconds += time.perf_counter()-decode_started
                writer.append(converted)
            del outs
            timings.append({**native.last_timing, 'additional_head_seconds':head_seconds,
                            'additional_decode_serialize_seconds':decode_seconds,
                            'method_seconds':native.last_timing['native_only_seconds']+head_seconds+decode_seconds})
            run.update(phase='full_evaluation_export', source_batches=index+1, output_rows=writer.count)
            print('EVAL', run.arm, index+1, len(native.plan), flush=True)
            del caches, rows, features
            torch.cuda.empty_cache()
    writer.finish()
    write(run.path / 'BATCH_TIMINGS.json', timings)
    import statistics
    eligible = [t for t in timings[1:] if t['images'] == 16]
    require(eligible, 'timing requires post-warmup complete native batches')
    native_median = statistics.median(t['native_only_seconds'] for t in eligible)
    method_median = statistics.median(t['method_seconds'] for t in eligible)
    timing = {'native_batch_median_seconds':native_median, 'method_batch_median_seconds':method_median,
              'method_over_native_ratio':method_median/native_median, 'eligible_batches':len(eligible),
              'guard_ratio_max':4., 'cost_guard_passed':method_median/native_median <= 4.,
              'scope':'same full batch16, first batch and last partial batch excluded; synchronized base + head + decoder/serializer; no disk writes, COCO stats or identity audits'}
    write(run.path / 'INFERENCE_COST.json', timing)
    write(run.path / 'FROZEN_BASE_CHECK.json', model.verify_frozen_state(native.model,snapshot))
    del model, native, snapshot
    torch.cuda.empty_cache()
    ids = read(prepared / 'DATA_CONTRACT.json')['val_ids']
    metrics, ledger = evaluate_predictions(run, run.arm, ids)
    native_ledger = [json.loads(line) for line in (reference/'Ns_PER_GT.jsonl').read_text().splitlines()]
    damaged, repaired = compare_ledgers(native_ledger, ledger)
    # Non-mask identity is a separate full-row check, not inferred from equal AP.
    original = read(reference / 'Ns_predictions.json')
    current = read(run.path / (run.arm+'_predictions.json'))
    require(len(original) == len(current), 'final output count changed')
    require(all({k:v for k,v in a.items() if k!='segmentation'} == {k:v for k,v in b.items() if k!='segmentation'} for a,b in zip(original,current)), 'non-mask output identity changed')
    denominator = sum(r['task_mask75']=='success' for r in native_ledger)
    damage = {'native_success_denominator':denominator, 'damaged_annotation_ids':damaged,
              'repaired_annotation_ids':repaired, 'damage_fraction':len(damaged)/denominator,
              'full_census':True, 'Box_identity_guard':'passed'}
    write(run.path / 'DAMAGE.json', damage)
    ns, gs = read(reference/'Ns_METRICS.json'), read(reference/'G_old_s_METRICS.json')
    run.finish({'arm':run.arm, **metrics, 'damage':damage,
                'gain_vs_Ns_AP_points':metrics['metrics']['segm']['AP']-ns['metrics']['segm']['AP'],
                'gain_vs_G_old_s_AP_points':metrics['metrics']['segm']['AP']-gs['metrics']['segm']['AP'],
                'full_row_identity_guard':'passed', 'checkpoint_sha256':sha(checkpoint),
                'additional_training_steps':0, 'inference_cost':timing,
                'final_gate':'pending_frozen_paired_bootstrap_and_cross_arm_adjudication'},
               'full_prediction_and_census_complete_gate_pending')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('stage', choices=('prepare','native','smoke','train','eval'))
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    require((args.stage in ('train','eval')) == bool(args.arm), '--arm is required only for train/eval')
    with Run(args.config, args.stage, args.arm) as run:
        if args.stage == 'prepare':
            run.finish(prepare(run.config, run.path), 'input_manifest_frozen')
        elif args.stage == 'native': native_stage(run)
        elif args.stage == 'smoke': smoke_stage(run)
        elif args.stage == 'train': train_stage(run)
        else: eval_stage(run)
    if args.stage == 'native' and read(run.path/'SUMMARY.json')['migration_status'] != 'passed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
