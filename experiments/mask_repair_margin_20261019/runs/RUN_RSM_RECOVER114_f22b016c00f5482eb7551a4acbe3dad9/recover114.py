"""One original native batch and no-GT scalar-normalization diagnosis."""
from pathlib import Path
import sys, hashlib, gc
STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'scripts'))
from common import *


def replay(run, profile, env):
    import torch
    from ultralytics import YOLO
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.models.yolo.segment.val import SegmentationValidator
    from adapter_data import FrozenCache
    from native_output_bridge import make_trace_validator, decode_native, encode_rle, native_environment
    from posttrain_audit import fingerprint_base_state
    manifest = read_json(BASE / 'trace/TRACE_MANIFEST.json')
    records = sorted(manifest['images'], key=lambda x: (x['batch_index'], x['batch_position']))
    group = [x for x in records if x['batch_index'] == 14]
    source = FrozenCache(BASE / 'trace', [x['image_id'] for x in group])
    out = run.path / 'regenerated';out.mkdir();(out / 'cache').mkdir()
    requested = {**profile['native_args'], 'data': read_json(BASE / 'trace/NATIVE_ARGS.json')['data']}
    parent = make_trace_validator(out, manifest['image_ids'], 'engineering_regeneration', 0,
             manifest['annotations_sha256'], requested, None, profile['runtime_policy'])
    checks, calls = [], []
    class OriginalBatch:
        def __init__(self, loader):
            self.dataset = loader.dataset
            require([int(Path(p).stem) for p in self.dataset.im_files] == [r['image_id'] for r in records], 'dataset order differs')
        def __len__(self): return 1
        def __iter__(self): yield self.dataset.collate_fn([self.dataset[i] for i in range(14*16, 15*16)])
    class Replay(parent):
        def get_dataloader(self, path, batch_size): return OriginalBatch(super().get_dataloader(path, batch_size))
        def preprocess(self, batch):
            self.batch_i = 14;resource_check(run)
            return super().preprocess(batch)
        def update_metrics(self, predictions, batch):
            require([int(Path(p).stem) for p in batch['im_file']] == [r['image_id'] for r in group], 'batch changed')
            for i, cache in enumerate(self.pending):
                iid = group[i]['image_id'];old, rows = source.load(iid);path = Path(batch['im_file'][i])
                cache.update(image_id=iid,input_hw=list(batch['img'].shape[2:]),original_hw=[int(x) for x in batch['ori_shape'][i]],
                  ratio_pad=tuple(tuple(float(x) for x in pair) for pair in batch['ratio_pad'][i]),batch_index=14,batch_position=i,
                  image_sha256=sha256(path),input_tensor_sha256=hashlib.sha256(batch['img'][i].detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                  prediction_indices=old['prediction_indices'])
                tensor_checks = {k: torch.equal(cache[k].cpu(), old[k].cpu()) for k in ['proto','c0','boxes_input','scores','classes']}
                tensor_checks['features'] = all(torch.equal(a,b) for a,b in zip(cache['features'],old['features'])) and len(cache['features']) == len(old['features'])
                fields=['image_id','input_hw','original_hw','ratio_pad','batch_index','batch_position','image_sha256','input_tensor_sha256','head_feature_shapes','feature_sources','raw_ids','raw_locations','output_ranks','prediction_indices','native_decode_hw']
                meta_checks={k:json.dumps(cache[k]) == json.dumps(old[k]) for k in fields}
                decoded=decode_native(cache);rle_exact=all(encode_rle(decoded[j])==row['segmentation'] for j,row in enumerate(rows))
                target=out/'cache'/f'{iid}.pt';torch.save(cache,target)
                checks.append({'image_id':iid,'tensor_checks':tensor_checks,'metadata_checks':meta_checks,'zero_delta_RLE_exact':rle_exact,
                  'rows':len(rows),'old_cache_sha256':source.records[iid]['cache_sha256'],'new_cache_sha256':sha256(target)})
                write_new(out/f'{iid}_IDENTITY.json',checks[-1])
                require(all(tensor_checks.values()) and all(meta_checks.values()) and rle_exact,'regenerated cache differs')
                self.actual_ids.append(iid)
            self.pending=[]
        def gather_stats(self): pass
        def get_stats(self): return {}
        def finalize_metrics(self): pass
        def print_results(self): pass
        def eval_json(self, stats): raise RuntimeError('No scientific evaluator authorized here')
        def __call__(self, trainer=None, model=None):
            try:
                value=SegmentationValidator.__call__(self,trainer=trainer,model=model)
                require(self.actual_ids==[r['image_id'] for r in group],'regeneration coverage')
                require(fingerprint_base_state(self.trace_model)==self.base_before,'base changed')
                return value
            finally:
                if self.trace_head is not None:
                    if self.had_instance_topk:self.trace_head.get_topk_index=self.old_instance_topk
                    else:del self.trace_head.get_topk_index
    weight=ROOT/'assets/models/coco_clean_20260911/yolo26m-seg.pt'
    require(sha256(weight)==profile['model_sha256'],'weight differs')
    model=YOLO(str(weight))
    handle=model.model.register_forward_pre_hook(lambda module, args:calls.append(list(args[0].shape)))
    old_cache=YOLODataset.cache_labels
    def deny_write(*args,**kwargs):raise RuntimeError('Refuse writing historical label cache')
    YOLODataset.cache_labels=deny_write
    try:
        with torch.inference_mode():model.val(validator=Replay,**requested,project=str(out),name='native',exist_ok=False)
    finally:
        YOLODataset.cache_labels=old_cache;handle.remove()
    require(native_environment()==env,'environment drift')
    source.verify()
    summary={'status':'passed','batch_index':14,'images':16,'input_hw':[416,672],'checks':checks,'model_forward_shapes':calls,'new_training_steps':0}
    write_new(run.path/'REGENERATED_IDENTITY.json',summary)
    run.update(base_model_forward_calls=len(calls),base_model_forward_shapes=calls,real_image_forward_batches=1,real_image_forward_images=16)
    del model,source;gc.collect();torch.cuda.empty_cache()
    return out/'cache/270883.pt'


def diagnose(run, cache_path):
    import numpy as np
    import torch
    from native_geometry import roi_for,lifted_basis
    from native_ops import Decoder
    from margin_math import recenter,draw
    cfg=read_json(STUDY/'CONFIG.json')
    old=next(r for r in lines(C_FULL/'PAIR_RESULTS.jsonl') if r['image_id']==270883 and r['annotation_id']==1665102)
    cache=torch.load(cache_path,map_location='cpu',weights_only=True);rank=old['rank'];d=Decoder(cache,rank)
    c=np.array(old['arms']['C']['coefficients_fp32'],dtype=np.float32)
    cr,first,z=d.evaluate(c,True);require(cr==old['arms']['C']['original_rle'],'C replay differs')
    roi=roi_for(cache,rank).numpy();B=lifted_basis(cache)[:,torch.from_numpy(roi)].T.contiguous().numpy()
    w,solver=recenter(B,c.astype(float),first[roi],cfg['solver'],lambda:resource_check(run))
    require(solver['converged'],'failed W did not reproduce')
    crms=float(np.sqrt(np.mean(z*z)));w=(w*(crms/d.rms(w)[0])).astype(np.float32);w=(w.astype(float)*(crms/d.rms(w)[0])).astype(np.float32)
    wr,wf,wz=d.evaluate(w,True);require(wr==cr and np.array_equal(first,wf),'W nominal mask differs')
    lib=np.load(FIT/'FIT_ERRORS.npz');vectors,labels=draw(lib['errors'],lib['mean'],lib['image_ids'],270883,rank,cfg)
    amplitude=crms*cfg['pressure_ratio'];base=vectors*(amplitude/d.rms(vectors))[:,None];tol=cfg['numerics']['rms_relative_tolerance']
    history=[];scales=np.ones(len(base))
    for iteration in range(9):
        q=(w.astype(float)[None]+base*scales[:,None]).astype(np.float32)
        actual=d.rms(q.astype(float)-w.astype(float));errors=actual/amplitude-1
        history.append({'iteration':iteration,'scales':scales.tolist(),'signed_relative_errors':errors.tolist()})
        if iteration<8:
            bad=np.abs(errors)>tol;scales[bad]*=amplitude/actual[bad]
    roots=[]
    for j in range(24):
        resource_check(run);evaluations=[]
        def evaluate(scale):
            coeff=(w.astype(float)+base[j]*scale).astype(np.float32)
            rms=float(d.rms(coeff.astype(float)-w.astype(float))[0]);error=rms/amplitude-1
            evaluations.append({'scale':float(scale),'signed_relative_error':float(error)})
            return error,coeff
        error,coeff=evaluate(1.)
        if abs(error)>tol:
            low=1-8*tol;high=1+8*tol;el,ql=evaluate(low);eh,qh=evaluate(high)
            if abs(el)<=tol:error,coeff=el,ql
            elif abs(eh)<=tol:error,coeff=eh,qh
            elif el*eh<0:
                for step in range(61):
                    mid=(low+high)/2;error,coeff=evaluate(mid)
                    if abs(error)<=tol or mid==low or mid==high:break
                    if error*el>0:low,el=mid,error
                    else:high,eh=mid,error
        _,_,qz=d.evaluate(coeff)
        actual_logit_diff_rms=float(np.sqrt(np.mean((qz-wz)**2)))
        exact_linear_rms=float(np.sqrt(np.mean((B@(coeff.astype(float)-w.astype(float)))**2)))
        roots.append({'direction':labels[j],'evaluations':evaluations,'matched':abs(error)<=tol,
          'coefficients':coeff.tolist(),'direction_rms_relative_error':error,
          'FP64_linear_rms_relative_error':exact_linear_rms/amplitude-1,
          'FP32_decoded_logit_difference_relative_error':actual_logit_diff_rms/amplitude-1})
    summary={'image_id':270883,'annotation_id':1665102,'rank':rank,'regenerated_cache_sha256':sha256(cache_path),
      'C_coefficients':c.tolist(),'W_coefficients':w.tolist(),'C_max_abs':float(abs(c).max()),'W_max_abs':float(abs(w).max()),
      'W_max_fp32_spacing':float(abs(np.spacing(w)).max()),'C_logit_rms':crms,'target_pressure_rms':amplitude,'tolerance':tol,
      'solver':solver,'old_fixed_point_history':history,'old_failed_directions':int((abs(np.array(history[-1]['signed_relative_errors']))>tol).sum()),
      'bounded_roots':roots,'all_direction_rms_roots_matched':all(x['matched'] for x in roots),
      'all_actual_decoded_logit_differences_matched':all(abs(x['FP32_decoded_logit_difference_relative_error'])<=tol for x in roots),
      'GT_scores_computed':0,'new_training_steps':0,'finite_coefficient_solves':1}
    write_new(run.path/'NUMERICS_114.json',summary)
    return summary


if __name__=='__main__':
    require(not list((STUDY/'runs').glob('RUN_RSM_RECOVER114_*')),'engineering diagnostic already exists')
    run=MarginRun('smoke',STUDY/'runs/RUN_RSM_FULL_f02b00986ed14331871c00f9564d6426')
    run.run_id=run.run_id.replace('SMOKE','RECOVER114');run.path=STUDY/'runs'/run.run_id
    run.meta.update(run_id=run.run_id,scope='engineering_recover114',helper_sha256=sha256(Path(__file__)))
    with run:
        inputs_check();profile,env=configure_native()
        run.freeze({'purpose':'regenerate_original_batch_and_diagnose_114_without_GT_scores','environment':env,
          'engineering_contract_sha256':sha256(STUDY/'NUMERICAL_RECOVERY_20261020.md'),'helper_sha256':sha256(Path(__file__)),
          'science_protocol_unchanged':True,'new_training_steps':0})
        shutil.copyfile(Path(__file__),run.path/'recover114.py');shutil.copyfile(STUDY/'NUMERICAL_RECOVERY_20261020.md',run.path/'ENGINEERING_CONTRACT.md')
        path=replay(run,profile,env);summary=diagnose(run,path);inputs_check()
        write_new(run.path/'CONFIG_AUDIT.json',{'status':'passed','engineering_only':True,'regenerated_native_exact':True,'GT_scores_computed':0,'scientific_gate_evaluated':False})
        run.finish({'regeneration':'exact','fixed_point_failed_directions':summary['old_failed_directions'],
          'bounded_scalar_matching':summary['all_direction_rms_roots_matched'],'actual_logit_difference_matching':summary['all_actual_decoded_logit_differences_matched'],
          'new_training_steps':0},scientific='ENGINEERING_ONLY',next_action='Review the original amplitude semantics and evidence before freezing any implementation correction')
