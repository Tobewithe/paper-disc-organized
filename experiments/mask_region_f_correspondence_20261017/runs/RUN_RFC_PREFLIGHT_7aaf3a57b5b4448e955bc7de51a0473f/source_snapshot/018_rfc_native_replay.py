"""Replay ORIGINAL complete rectangular batches; write only separate regional F sidecars."""
from __future__ import annotations
import argparse
import gc
import hashlib
import io
from pathlib import Path
import time
import torch

from rfc_runtime import *
from rfc_inputs import validate_replay, sample_region_features, make_sidecar, prepare_inputs
from rfc_contract import check_native_args
from adapter_data import FrozenCache
from native_output_bridge import make_trace_validator, decode_native, encode_rle, native_environment
from posttrain_audit import fingerprint_base_state


class OriginalBatches:
    """Select complete original batches AFTER the original dataset's rect grouping."""
    def __init__(self, loader, manifest, requested):
        self.dataset = loader.dataset
        self.groups = {}
        for row in manifest['images']:
            self.groups.setdefault(row['batch_index'], []).append(row)
        for group in self.groups.values():
            group.sort(key=lambda r:r['batch_position'])
            require([r['batch_position'] for r in group] == list(range(len(group))), 'bad original batch positions')
        ordered = [r for index in sorted(self.groups) for r in self.groups[index]]
        actual = [int(Path(p).stem) for p in self.dataset.im_files]
        require(actual == [r['image_id'] for r in ordered], 'native dataset regrouped/reordered images')
        self.offsets = {}
        start = 0
        for index in sorted(self.groups):
            self.offsets[index] = list(range(start, start+len(self.groups[index])))
            start += len(self.groups[index])
        self.selected = [k for k in sorted(self.groups) if any(r['image_id'] in requested for r in self.groups[k])]

    def __len__(self):
        return len(self.selected)

    def __iter__(self):
        for index in self.selected:
            yield self.dataset.collate_fn([self.dataset[i] for i in self.offsets[index]])


def export_split(run, data, split_name, wanted):
    from ultralytics import YOLO
    from ultralytics.models.yolo.segment.val import SegmentationValidator
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from pycocotools import mask as masks
    from ultralytics.data.dataset import YOLODataset
    trace = (CACHE if split_name == 'fit' else BASE)/'trace'
    manifest = read_json(trace/'TRACE_MANIFEST.json')
    desired = set(wanted)
    selected_batches = {r['batch_index'] for r in manifest['images'] if r['image_id'] in desired}
    context_ids = [r['image_id'] for r in manifest['images'] if r['batch_index'] in selected_batches]
    source = FrozenCache(trace, context_ids)
    old_args = read_json(trace/'NATIVE_ARGS.json')
    requested = {**data['profile']['native_args'], 'data':old_args['data']}
    out = run.path/('native_'+split_name)
    out.mkdir()
    records, checks = [], []
    parent = make_trace_validator(out, manifest['image_ids'], 'RFC_replay_only', 0,
                                 manifest['annotations_sha256'], requested, None, data['profile']['runtime_policy'])

    class Replay(parent):
        def get_dataloader(self, dataset_path, batch_size):
            full = super().get_dataloader(dataset_path, batch_size)
            return OriginalBatches(full, manifest, desired)

        def init_metrics(self, model):
            super().init_metrics(model)
            check_native_args(old_args, vars(self.args))
            require_no_drift(data['environment'], native_environment(), 'native replay runtime')

        def preprocess(self, batch):
            self.batch_i = self.dataloader.selected[self.batch_i]
            run.check_budget()
            return super().preprocess(batch)

        def postprocess(self, predictions):
            result = super().postprocess(predictions)
            self.region_dense = [x.detach() for x in predictions[1]['one2one']['feats']]
            run.check_budget()
            return result

        def update_metrics(self, preds, batch):
            expected = self.dataloader.groups[self.batch_i]
            require([int(Path(p).stem) for p in batch['im_file']] == [r['image_id'] for r in expected], 'batch members changed')
            for position, (cache, pred) in enumerate(zip(self.pending, preds)):
                image_path = Path(batch['im_file'][position]); iid = int(image_path.stem)
                original, rows = source.load(iid)
                cache.update(image_id=iid, input_hw=list(batch['img'].shape[2:]),
                             original_hw=[int(x) for x in batch['ori_shape'][position]],
                             ratio_pad=tuple(tuple(float(x) for x in pair) for pair in batch['ratio_pad'][position]),
                             batch_index=self.batch_i, batch_position=position,
                             image_sha256=sha256(image_path), prediction_indices=original['prediction_indices'],
                             input_tensor_sha256=hashlib.sha256(batch['img'][position].detach().cpu().contiguous().numpy().tobytes()).hexdigest())
                validate_replay(original, cache)
                require(cache['native_decode_hw'] == original['native_decode_hw'] == cache['input_hw'], 'decoder geometry drift')
                if rows:
                    pbatch = DetectionValidator._prepare_batch(self, position, batch)
                    self.pred_to_json(self.scale_preds(self._prepare_pred(pred), pbatch), pbatch)
                require(self.jdict == rows, 'native serializer differs from original complete output rows')
                decoded = decode_native(cache)
                require(all(encode_rle(decoded[j]) == row['segmentation'] for j,row in enumerate(rows)), 'zero-delta RLE replay mismatch')
                self.jdict = []
                self.actual_ids.append(iid)
                checks.append(dict(image_id=iid, rows=len(rows), batch_index=self.batch_i,
                                   tensor_identity_exact=True, native_output_exact=True, zero_delta_RLE_exact=True,
                                   source_cache_sha256=source.records[iid]['cache_sha256'],
                                   image_sha256=cache['image_sha256'], input_tensor_sha256=cache['input_tensor_sha256']))
                if iid in desired:
                    gate = torch.tensor([int(masks.area(r['segmentation'])) >= 9216 for r in rows],dtype=torch.bool)
                    selected = None if split_name == 'fit' else gate.nonzero().flatten().tolist()
                    regions = sample_region_features(cache, [x[position] for x in self.region_dense], selected)
                    scope = 'train_all_rows' if split_name == 'fit' else 'eval_gated_rows'
                    payload = make_sidecar(original, regions, source.records[iid]['cache_sha256'], manifest['predictions_sha256'], scope)
                    prepare_inputs(original, payload, source.records[iid]['cache_sha256'], manifest['predictions_sha256'],
                                   'train:0' if split_name == 'fit' else 'eval:0', None if split_name == 'fit' else gate)
                    stream = io.BytesIO(); torch.save(payload, stream)
                    require(run.F_existing+run.F_written+stream.tell() <= 48*GiB, 'regional F storage budget exceeded')
                    target = run.path/'regions'/split_name/f'{iid}.pt'; target.parent.mkdir(parents=True,exist_ok=True)
                    with target.open('xb') as f: f.write(stream.getbuffer())
                    run.F_written += target.stat().st_size
                    records.append(dict(image_id=iid,path=target.relative_to(run.path).as_posix(),sha256=sha256(target),
                                        bytes=target.stat().st_size,rows=len(rows),region_rows=len(rows) if selected is None else len(selected),
                                        source_cache_sha256=source.records[iid]['cache_sha256']))
                    del payload, regions, stream
                del original, rows, cache, decoded
                run.check_budget()
            self.pending=[]; self.region_dense=[]
            run.update(base_model_forward_images=run.meta['base_model_forward_images']+len(preds),
                       new_F_cache_exports=run.meta['new_F_cache_exports']+sum(r['image_id'] in desired for r in expected),
                       phase='regional_F_replay_'+split_name, completed_batches=len(checks))
            print('RFC_REPLAY',split_name,'batch',self.batch_i,'sidecars',len(records),'context',len(checks),flush=True)

        def gather_stats(self): pass
        def get_stats(self): return {}
        def finalize_metrics(self): pass
        def print_results(self): pass
        def eval_json(self, stats): raise RuntimeError('No scientific evaluator is part of F replay')

        def __call__(self, trainer=None, model=None):
            try:
                result = SegmentationValidator.__call__(self, trainer=trainer, model=model)
                require(sorted(self.actual_ids)==sorted(context_ids), 'replay context coverage incomplete')
                after=fingerprint_base_state(self.trace_model)
                require(after==self.base_before, 'base parameters/buffers/modes changed')
                write_new(out/'base_after.json',after)
                return result
            finally:
                if self.trace_head is not None:
                    if self.had_instance_topk: self.trace_head.get_topk_index=self.old_instance_topk
                    else: del self.trace_head.get_topk_index

    model=YOLO(str(WEIGHT))
    original_cache_labels=YOLODataset.cache_labels
    def reject_label_cache_write(*args,**kwargs):
        raise RuntimeError('Original dataset label cache missing/invalid; refusing a write into historical dataset')
    YOLODataset.cache_labels=reject_label_cache_write
    try:
        with torch.inference_mode():
            model.val(validator=Replay, **requested, project=str(out.resolve()), name='native', exist_ok=False)
    finally:
        YOLODataset.cache_labels=original_cache_labels
    require({r['image_id'] for r in records}==desired,'sidecar coverage mismatch')
    source.verify()
    write_new(run.path/(split_name+'_REPLAY_CHECKS.json'), checks)
    result=dict(split=split_name, source_trace=str(trace), source_manifest_sha256=sha256(trace/'TRACE_MANIFEST.json'),
                source_predictions_sha256=manifest['predictions_sha256'], wanted_ids=wanted,
                context_images=len(context_ids), records=records)
    del model, source
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--scope',choices=('smoke','full'),required=True)
    parser.add_argument('--smoke-deadline',type=float)
    args=parser.parse_args(); full=args.scope=='full'
    with Run('INPUT_FULL' if full else 'INPUT_SMOKE',CACHE,10800 if full else 1800,args.smoke_deadline) as run:
        data=inspect_inputs(); resources=initial_resources()
        run.F_existing=all_F_bytes(); run.F_written=0
        run.freeze(dict(scope=args.scope, environment=data['environment'], resources=resources,
                        design_sha256=sha256(STUDY/'DESIGN_SPEC.json'), data_split_sha256=sha256(CACHE/'SPLIT_CONTRACT.json'),
                        original_batches_unchanged=True, scientific_evaluation=False,
                        smoke_deadline=args.smoke_deadline))
        if full:
            for name in ('SMOKE_A','SMOKE_Q'):
                smoke=pointer(name)
                require(read_json(smoke/'SOURCE_MANIFEST.json')['files']==run.sources,'source changed after real smoke')
            require(not list((STUDY/'runs').glob('RUN_RFC_INPUT_FULL_*/SUMMARY.json')), 'full F export already completed')
        results={}
        results['fit']=export_split(run,data,'fit',data['split']['fit_ids'] if full else data['split']['fit_ids'][:8])
        if not full: results['dev']=export_split(run,data,'dev',data['split']['dev_ids'][:8])
        write_new(run.path/'REGION_MANIFEST.json',results)
        if full: require(sum(r['rows'] for r in results['fit']['records'])==218209,'fit output population changed')
        audit(run,data,tensor_identity_exact=True,zero_delta_RLE_exact=True,source_fingerprints_match=True,
              original_native_batch_members=True,base_state_unchanged=True)
        run.finish(dict(scope=args.scope, exported_images=sum(len(x['records']) for x in results.values()),
                        base_model_forward_images=run.meta['base_model_forward_images'], regional_F_bytes=all_F_bytes(),
                        tensor_identity_exact=True,zero_delta_RLE_exact=True,source_fingerprints_match=True),
                   scientific='input_replay_verified_not_method_effect',next_action='registered A/Q smoke' if not full else 'A then Q formal training')
    set_pointer('INPUT_FULL' if full else 'INPUT_SMOKE',run)

if __name__=='__main__': main()
