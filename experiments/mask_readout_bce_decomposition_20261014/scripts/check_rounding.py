"""Registered one-image engineering probe for the retained closure failure."""
import json
from pathlib import Path
import shutil
import sys

import run_bce_decomposition as base
from posttrain_run import RegisteredRun, read_json, write_new, require, sha256, tensor_state_hash
from adapter_data import FrozenCache, training_join, projected_targets
from adapter_model import loss_grids
from decomposition_statistics import partition


class RoundingRun(RegisteredRun):
    def __init__(self):
        parent=base.STUDY/'runs/RUN_BCE_DECOMP_FULL_1ce5436c83234cfaafa8f68f7c7465fa'
        codes=[*base.CODE_FILES, str(Path(__file__).relative_to(base.ROOT)).replace('\\','/'),
               (base.STUDY/'ROUNDING_CHECK_PROTOCOL.md').relative_to(base.ROOT).as_posix()]
        super().__init__('BCE_ROUNDING_CHECK','BCE_DECOMPOSITION',parent,codes,900)
        self.path=base.STUDY/'runs'/self.run_id; self.lock=base.STUDY/'.BCE_DECOMPOSITION.lock'
        self.meta.update(study_id='STUDY_MASK_READOUT_BCE_DECOMPOSITION_20261014',optimization_started=False,optimizer_steps=0)

    def freeze(self,contract):
        context=read_json(base.STUDY/'SOURCE_CONTEXT.json')
        self.sources={p:sha256(base.ROOT/p) for p in self.code_files}
        self.meta.update(code_commit=context['code_commit'],dirty_patch_sha256=sha256(base.STUDY/'SOURCE_DIRTY.patch'))
        write_new(self.path/'SOURCE_MANIFEST.json',{'files':self.sources,'code_commit':context['code_commit']})
        for i,p in enumerate(self.sources):
            target=self.path/'source_snapshot'/(f'{i:02d}_'+Path(p).name);target.parent.mkdir(exist_ok=True)
            shutil.copyfile(base.ROOT/p,target)
        shutil.copyfile(base.STUDY/'ROUNDING_CHECK_PROTOCOL.md',self.path/'PROTOCOL.md')
        write_new(self.path/'EVAL_CONTRACT.json',contract)
        self.update(plan_status='frozen',protocol_path=str(self.path/'PROTOCOL.md'),protocol_sha256=sha256(self.path/'PROTOCOL.md'),contract_sha256=sha256(self.path/'EVAL_CONTRACT.json'))


def main():
    import torch
    from torch.nn import functional as F
    from pycocotools.coco import COCO
    with RoundingRun() as run:
        data=base.inspect_inputs()
        contract={'scope':'retained_failure_numeric_probe','image_id':354533,'annotation_id':252163,'global_output_index':1209,
            'inputs_sha256':data['inputs'],'environment':data['environment'],'new_training_steps':0,'pointwise_BCE_dtype':'float32',
            'original_reduction':'float32','diagnostic_reduction':'float64','FP32_replay_tolerance':1e-7,'FP64_partition_closure_tolerance':1e-12}
        run.freeze(contract)
        source=FrozenCache(base.prior.CACHE_RUN/'trace',[354533]); coco=COCO(str(base.prior.GT))
        cache,original=source.load(354533); join=training_join(original,coco.imgToAnns[354533])
        expected={int(k):v for k,v in read_json(base.prior.B_RUN/'TRAINING_JOIN.json')['images']['354533']['output_rank_to_annotation_id'].items()}
        require(join==expected,'saved join changed')
        model=base.prior.checkpoint_model(); state=tensor_state_hash(model)
        with torch.inference_mode():
            logits,native,roi=loss_grids(cache,cache['c0'].cuda()+model(cache,'eval:0'))
            targets=projected_targets(coco,cache,join);pixels=roi.sum((1,2));valid=[i for i in targets if int(pixels[i])>0]
            y=torch.stack([targets[i].to(logits) for i in valid])
            a=F.binary_cross_entropy_with_logits(native[valid],y,reduction='none');b=F.binary_cross_entropy_with_logits(logits[valid],y,reduction='none')
            a32=(a*roi[valid]).sum((1,2))/pixels[valid];b32=(b*roi[valid]).sum((1,2))/pixels[valid]
            rank=cache['prediction_indices'].index(1209);k=valid.index(rank);require(join[rank]==252163,'wrong failure target')
            original_row=next(r for r in base.read_jsonl(base.PROTO_RUN/'PAIR_PROTO_AUDIT.jsonl') if r['image_id']==354533 and r['annotation_id']==252163)
            part=partition(native[valid][k:k+1],logits[valid][k:k+1],y[k:k+1],roi[valid][k:k+1],a[k:k+1],b[k:k+1])[0]
            a64=float((a[k].double()*roi[rank]).sum()/pixels[rank]);b64=float((b[k].double()*roi[rank]).sum()/pixels[rank])
            require(abs(float(a32[k])-original_row['zero_gt_bce'])<=1e-7 and abs(float(b32[k])-original_row['B_gt_bce'])<=1e-7,'FP32 replay changed')
            for arm,field in [('counts_zero','zero_proto'),('counts_B','B_proto')]:
                require(all(part[arm][q]==original_row[field][q] for q in ('tp','fp','fn','tn')),'integer counts changed')
            require(abs(part['totals']['zero_loss']-a64)<=1e-12 and abs(part['totals']['B_loss']-b64)<=1e-12,'FP64 component closure failed')
            require(abs(part['totals']['delta_loss']-(b64-a64))<=1e-12,'FP64 delta closure failed')
            summary={'image_id':354533,'annotation_id':252163,'zero_BCE_FP32':float(a32[k]),'B_BCE_FP32':float(b32[k]),
                'zero_BCE_FP64_sum':a64,'B_BCE_FP64_sum':b64,'delta_FP32':float(b32[k])-float(a32[k]),'delta_direct_FP64':b64-a64,
                'delta_eight_group_FP64':part['totals']['delta_loss'],'roundoff_delta':(float(b32[k])-b64)-(float(a32[k])-a64),
                'max_FP32_replay_error':max(abs(float(a32[k])-original_row['zero_gt_bce']),abs(float(b32[k])-original_row['B_gt_bce'])),
                'FP64_partition_delta_error':abs(part['totals']['delta_loss']-(b64-a64)),
                'integer_counts_exact':True,'source_failed_run':'RUN_BCE_DECOMP_FULL_1ce5436c83234cfaafa8f68f7c7465fa'}
        require(tensor_state_hash(model)==state and not any(p.requires_grad or p.grad is not None for p in model.parameters()),'model changed')
        source.verify()
        for p,h in data['inputs'].items():require(sha256(p)==h,'input changed')
        from native_output_bridge import native_environment
        require(native_environment()==data['environment'],'runtime changed')
        write_new(run.path/'FROZEN_MODEL_STATE.json',{'before':state,'after':tensor_state_hash(model),'new_optimizer_steps':0,'parameters_require_grad':False,'parameter_gradients_present':False,'model_eval':True})
        write_new(run.path/'CONFIG_AUDIT.json',{'status':'passed','new_optimizer_steps':0,'all_input_hashes_and_runtime_unchanged':True,'original_FP32_replay_and_counts_exact':True,'FP64_partition_closure_passed':True})
        run.finish(summary,scientific='engineering_roundoff_difference_identified',next_action='Freeze v2 accounting with separate exact pixel contributions and explicit original FP32 reduction roundoff; retain failed Run and rerun smoke/full')
        print('ROUNDING_CHECK_COMPLETED',run.run_id,json.dumps(summary),flush=True)


if __name__=='__main__':main()
