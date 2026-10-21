"""Bounded no-GT engineering search along unchanged pressure directions."""
from pathlib import Path
import sys,hashlib,gc
STUDY=Path(__file__).resolve().parents[1];sys.path.insert(0,str(STUDY/'scripts'))
from common import *


def candidate_scales(center,delta,low,high,run):
    import numpy as np
    points=[np.array([low,high,1.],dtype=np.float64)]
    count=0
    for c,d in zip(center,delta):
        if d==0:continue
        left,right=sorted([np.float32(c+d*low),np.float32(c+d*high)])
        # Ordered integer keys make adjacent IEEE float32 states consecutive across zero.
        def ordered(x):
            b=int(np.asarray(x,dtype=np.float32).view(np.uint32));return (~b & 0xffffffff) if b&0x80000000 else b^0x80000000
        begin,end=ordered(left),ordered(right)
        count+=end-begin
        require(count<=2000000,'engineering state budget')
        if end==begin:continue
        keys=np.arange(begin,end+1,dtype=np.uint64).astype(np.uint32)
        bits=np.where((keys&0x80000000)!=0,keys^np.uint32(0x80000000),~keys).astype(np.uint32)
        values=bits.view(np.float32).astype(np.float64)
        breaks=((values[:-1]+values[1:])*.5-float(c))/float(d)
        points.append(breaks[(breaks>low)&(breaks<high)])
        resource_check(run)
    edges=np.unique(np.concatenate(points));mid=(edges[:-1]+edges[1:])*.5
    return np.unique(np.r_[1.,mid]),count


def execute(run,parent):
    import numpy as np
    import torch
    from native_ops import Decoder
    from native_geometry import roi_for,lifted_basis
    from margin_math import draw
    inputs_check();profile,env=configure_native();cfg=read_json(STUDY/'CONFIG.json');prior=read_json(parent/'NUMERICS_114.json')
    run.freeze({'purpose':'quantized_states114_fixed_target_no_GT','environment':env,'helper_sha256':sha256(Path(__file__)),
      'engineering_contract_sha256':sha256(STUDY/'NUMERICAL_RECOVERY_STATES_20261020.md'),'parent_numerics_sha256':sha256(parent/'NUMERICS_114.json'),
      'new_training_steps':0,'new_model_forward_calls':0,'new_coefficient_solves':0})
    shutil.copyfile(Path(__file__),run.path/'quantized_states114.py');shutil.copyfile(STUDY/'NUMERICAL_RECOVERY_STATES_20261020.md',run.path/'ENGINEERING_CONTRACT.md')
    cache=torch.load(parent/'regenerated/cache/270883.pt',map_location='cpu',weights_only=True);d=Decoder(cache,prior['rank'])
    c=np.asarray(prior['C_coefficients'],dtype=np.float32);w=np.asarray(prior['W_coefficients'],dtype=np.float32)
    roi=roi_for(cache,prior['rank']).numpy();B=lifted_basis(cache)[:,torch.from_numpy(roi)].T.contiguous().numpy();H=B.T@B/len(B)
    lib=np.load(FIT/'FIT_ERRORS.npz');vectors,labels=draw(lib['errors'],lib['mean'],lib['image_ids'],270883,prior['rank'],cfg)
    target=prior['target_pressure_rms'];tol=cfg['numerics']['rms_relative_tolerance'];norm=d.rms(vectors);deltas=vectors*(target/norm)[:,None]
    output=[]
    for name,center in [('C',c),('W',w)]:
        _,_,z0=d.evaluate(center)
        for i,delta in enumerate(deltas):
            resource_check(run);scales,breaks=candidate_scales(center.astype(float),delta,1-8*tol,1+8*tol,run)
            good=[]
            for start in range(0,len(scales),65536):
                q=(center.astype(float)[None]+scales[start:start+65536,None]*delta).astype(np.float32)
                diff=q.astype(float)-center.astype(float)
                actual=np.sqrt(np.maximum(0,np.einsum('ij,ij->i',diff@H,diff)))
                good.extend(scales[start:start+65536][np.abs(actual/target-1)<=tol].tolist())
            if good:
                selected=np.array(good)[np.linspace(0,len(good)-1,min(63,len(good)),dtype=int)].tolist()
            else:selected=[]
            selected=list(dict.fromkeys([1.]+selected));trials=[];accepted=None
            for scale in selected:
                coeff=(center.astype(float)+delta*scale).astype(np.float32)
                direction_error=float(d.rms(coeff.astype(float)-center.astype(float))[0]/target-1)
                _,_,z1=d.evaluate(coeff)
                decoded_error=float(np.sqrt(np.mean((z1-z0)**2))/target-1)
                trial={'scale':scale,'direction_relative_error':direction_error,'decoded_difference_relative_error':decoded_error}
                trials.append(trial)
                if abs(direction_error)<=tol and abs(decoded_error)<=tol:
                    accepted={'coefficients':coeff.tolist(),**trial};break
            row={'center':name,'direction_index':i,'direction':labels[i],'breakpoints':breaks,'states':len(scales),
                 'FP64_screened_states':len(good),'trials':trials,'accepted':accepted}
            output.append(row)
            with (run.path/'DIRECTION_RESULTS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            print(name,i,'states',len(scales),'matched',accepted is not None,flush=True)
    inputs_check()
    summary={'directions':len(output),'matched':sum(x['accepted'] is not None for x in output),
       'per_center':{k:{'matched':sum(x['accepted'] is not None for x in output if x['center']==k),'required':24} for k in ['C','W']},
       'all_original_targets_and_tolerances_met':all(x['accepted'] is not None for x in output),
       'new_training_steps':0,'new_model_forward_calls':0,'new_coefficient_solves':0,'GT_scores_computed':0}
    write_new(run.path/'CONFIG_AUDIT.json',{'status':'passed','protocol_and_config_unchanged':True,'original_scalar_interval_unchanged':True,'GT_scores_computed':0})
    run.finish(summary,scientific='ENGINEERING_ONLY',next_action='Freeze implementation only if all original amplitude requirements were constructed; otherwise close the attempted recovery')


if __name__=='__main__':
    require(not list((STUDY/'runs').glob('RUN_RSM_STATES114_*')),'diagnostic already registered')
    parent=STUDY/'runs/RUN_RSM_RECOVER114_f22b016c00f5482eb7551a4acbe3dad9'
    run=MarginRun('smoke',parent);run.run_id=run.run_id.replace('SMOKE','STATES114');run.path=STUDY/'runs'/run.run_id
    run.meta.update(run_id=run.run_id,scope='engineering_states114',helper_sha256=sha256(Path(__file__)))
    with run:execute(run,parent)
