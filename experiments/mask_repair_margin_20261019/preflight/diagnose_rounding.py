"""Engineering-only reproduction of failed pressure normalization; no GT IoU is read."""
from pathlib import Path
import sys
STUDY=Path(__file__).resolve().parents[1];sys.path.insert(0,str(STUDY/'scripts'))
from common import *
import numpy as np
import torch
from adapter_data import FrozenCache
from native_geometry import lifted_basis,roi_for
from native_ops import Decoder
from margin_math import recenter,draw

cfg=read_json(STUDY/'CONFIG.json');profile,env=configure_native()
failed=STUDY/'runs/RUN_RSM_FULL_e4a8254fa12149b489b5bccfb54e4b7a'
completed=lines(failed/'PAIR_RESULTS.jsonl')
ordered=sorted(lines(C_FULL/'PAIR_RESULTS.jsonl'),key=lambda r:(r['image_id'],r['annotation_id']))
old=ordered[len(completed)]
run=MarginRun('smoke',failed)
run.run_id=run.run_id.replace('SMOKE','NUMERICS');run.path=STUDY/'runs'/run.run_id
run.meta.update(run_id=run.run_id,scope='engineering_numerics',command_wrapper_sha256=sha256(Path(__file__)))
with run:
    run.freeze({'scope':'rounding_diagnosis_no_GT_scores','target_image_id':old['image_id'],'rank':old['rank'],'prior_completed_pairs':len(completed),
                'protocol_unchanged':True,'new_training_steps':0,'environment':env})
    source=FrozenCache(BASE/'trace',[old['image_id']]);cache,_=source.load(old['image_id']);rank=old['rank']
    c=np.array(old['arms']['C']['coefficients_fp32'],dtype=np.float32);d=Decoder(cache,rank)
    _,first,z=d.evaluate(c);roi=roi_for(cache,rank).numpy();B=lifted_basis(cache)[:,torch.from_numpy(roi)].T.contiguous().numpy()
    w,solver=recenter(B,c.astype(float),first[roi],cfg['solver'],run.check_budget)
    crms=float(np.sqrt(np.mean(z*z)))
    w=(w*(crms/d.rms(w)[0])).astype(np.float32);w=(w.astype(float)*(crms/d.rms(w)[0])).astype(np.float32)
    library=np.load(FIT/'FIT_ERRORS.npz');v,labels=draw(library['errors'],library['mean'],library['image_ids'],old['image_id'],rank,cfg)
    amplitude=crms*cfg['pressure_ratio'];base=v*(amplitude/d.rms(v))[:,None]
    history=[];scale=np.ones(len(v))
    for iteration in range(9):
        coeff=(w.astype(float)[None]+base*scale[:,None]).astype(np.float32)
        actual=d.rms(coeff.astype(float)-w.astype(float));relative=np.abs(actual/amplitude-1)
        history.append({'corrections':iteration,'max_relative_error':float(relative.max()),'bad_directions':int((relative>cfg['numerics']['rms_relative_tolerance']).sum()),'actual_rms':actual.tolist(),'scales':scale.tolist()})
        if np.all(relative<=cfg['numerics']['rms_relative_tolerance']):break
        # Correct only directions still outside the exact frozen tolerance.
        bad=relative>cfg['numerics']['rms_relative_tolerance'];scale[bad]*=amplitude/actual[bad]
    summary={'target_image_id':old['image_id'],'annotation_id':old['annotation_id'],'rank':rank,'prior_completed_pairs':len(completed),
        'C_max_abs_coefficient':float(np.max(np.abs(c))),'W_max_abs_coefficient':float(np.max(np.abs(w))),
        'C_logit_rms':crms,'pressure_amplitude':amplitude,'solver':solver,'normalization_history':history,
        'correctable_with_original_target_and_tolerance':history[-1]['bad_directions']==0,'GT_scores_computed':0,'new_training_steps':0}
    write_new(run.path/'CONFIG_AUDIT.json',{'status':'passed','engineering_only':True,'protocol_unchanged':True,'GT_scores_computed':0})
    run.finish(summary,scientific='ENGINEERING_ONLY',next_action='If normalization is correctable, freeze an implementation-only correction and repeat smoke; never relax the original tolerance')
    print(json.dumps({k:summary[k] for k in ['target_image_id','C_max_abs_coefficient','W_max_abs_coefficient','pressure_amplitude','normalization_history','correctable_with_original_target_and_tolerance']}))
