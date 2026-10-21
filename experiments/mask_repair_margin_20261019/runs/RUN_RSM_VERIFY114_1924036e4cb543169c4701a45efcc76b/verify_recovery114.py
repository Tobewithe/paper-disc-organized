"""Independent full-shape native checks of retained engineering evidence, no new search."""
from pathlib import Path
import sys
STUDY=Path(__file__).resolve().parents[1];sys.path.insert(0,str(STUDY/'scripts'))
from common import *


def execute(run,recovered,states):
    import numpy as np
    import torch
    from ultralytics.utils import ops
    from adapter_data import FrozenCache
    from native_output_bridge import decode_native,encode_rle,native_environment
    from native_geometry import roi_for
    from margin_math import draw
    inputs_check();profile,env=configure_native();cfg=read_json(STUDY/'CONFIG.json');identity=read_json(recovered/'REGENERATED_IDENTITY.json')
    run.freeze({'purpose':'independent_verification_of_regeneration_and_saved_pressure_trials','environment':env,
      'helper_sha256':sha256(Path(__file__)),'recovered_run':recovered.name,'states_run':states.name,
      'new_training_steps':0,'new_coefficient_solves':0,'new_pressure_searches':0})
    shutil.copyfile(Path(__file__),run.path/'verify_recovery114.py')
    source=FrozenCache(BASE/'trace',[x['image_id'] for x in identity['checks']]);n_masks=0
    for record in identity['checks']:
        resource_check(run);iid=record['image_id'];old,rows=source.load(iid)
        file=recovered/'regenerated/cache'/f'{iid}.pt';require(sha256(file)==record['new_cache_sha256'],'new cache hash')
        fresh=torch.load(file,map_location='cpu',weights_only=True)
        for k in ['proto','c0','boxes_input','scores','classes']:require(torch.equal(old[k],fresh[k]),'native tensor identity')
        require(all(torch.equal(a,b) for a,b in zip(old['features'],fresh['features'])),'F identity')
        masks=decode_native(fresh)
        require(all(encode_rle(masks[j])==row['segmentation'] for j,row in enumerate(rows)),'regenerated zero-delta RLE')
        n_masks+=len(rows)
    source.verify()
    data=read_json(recovered/'NUMERICS_114.json');cache=torch.load(recovered/'regenerated/cache/270883.pt',map_location='cpu',weights_only=True)
    rank=data['rank'];roi=roi_for(cache,rank,'cuda');p=cache['proto'].cuda().float()
    def logits(coef):
        with torch.inference_mode():
            complete=cache['c0'].cuda().clone();complete[rank]=torch.tensor(coef,dtype=torch.float32,device='cuda')
            low=(complete@p.flatten(1))[rank].reshape(1,*p.shape[-2:])
            return ops.scale_masks(low[None],cache['native_decode_hw'])[0,0][roi].double().cpu().numpy()
    def rms(coef):return float(np.sqrt(np.mean(logits(coef)**2)))
    lib=np.load(FIT/'FIT_ERRORS.npz');vectors,labels=draw(lib['errors'],lib['mean'],lib['image_ids'],270883,rank,cfg)
    target=data['target_pressure_rms'];tol=cfg['numerics']['rms_relative_tolerance'];centers={x:np.array(data[x+'_coefficients'],dtype=np.float32) for x in ['C','W']}
    require(abs(rms(centers['C'])*cfg['pressure_ratio']/target-1)<1e-12,'target identity')
    norm=np.array([rms(v) for v in vectors]);delta=vectors*(target/norm)[:,None]
    records=lines(states/'DIRECTION_RESULTS.jsonl');counts={'C':0,'W':0};trials=0;linear_counts={'C':0,'W':0}
    for row in records:
        resource_check(run);center=centers[row['center']];i=row['direction_index'];z0=logits(center);satisfied=False;linear=False
        require(row['direction']==labels[i],'direction identity')
        for trial in row['trials']:
            coeff=(center.astype(float)+delta[i]*trial['scale']).astype(np.float32)
            err=rms(coeff.astype(float)-center.astype(float))/target-1
            physical=float(np.sqrt(np.mean((logits(coeff)-z0)**2))/target-1)
            require(abs(err-trial['direction_relative_error'])<1e-12 and abs(physical-trial['decoded_difference_relative_error'])<1e-12,'saved amplitude values differ')
            linear|=abs(err)<=tol;satisfied|=abs(err)<=tol and abs(physical)<=tol;trials+=1
        require(satisfied==(row['accepted'] is not None),'recorded acceptance differs')
        counts[row['center']]+=int(satisfied);linear_counts[row['center']]+=int(linear)
    expected=read_json(states/'SUMMARY.json')
    require(counts=={k:v['matched'] for k,v in expected['per_center'].items()},'acceptance counts')
    require(native_environment()==env,'environment drift');inputs_check()
    summary={'status':'passed','native_images_exact':16,'native_zero_delta_masks_exact':n_masks,'pressure_trials_recomputed':trials,
      'direction_counts':48,'original_direction_RMS_match_counts':linear_counts,'joint_actual_precision_match_counts':counts,
      'complete_repair_valid':counts=={'C':24,'W':24},'scientific_scores_computed':0,'new_training_steps':0,
      'new_model_forward_calls':0,'new_coefficient_solves':0,'new_pressure_searches':0}
    write_new(run.path/'INDEPENDENT_LAPTOP.json',summary)
    write_new(run.path/'CONFIG_AUDIT.json',{'status':'passed','protocol_and_config_unchanged':True,'old_cache_and_model_inputs_unchanged':True,'repair_valid':summary['complete_repair_valid']})
    run.finish(summary,scientific='ENGINEERING_VERIFICATION_ONLY',next_action='Report the exact cache replay and retained precision obstruction; do not launch smoke/full without a complete valid repair')


if __name__=='__main__':
    require(not list((STUDY/'runs').glob('RUN_RSM_VERIFY114_*')),'verification already exists')
    recovered=STUDY/'runs/RUN_RSM_RECOVER114_f22b016c00f5482eb7551a4acbe3dad9';states=STUDY/'runs/RUN_RSM_STATES114_02fb6e10ff3e40228f5b044467b245bc'
    run=MarginRun('verify',states);run.run_id=run.run_id.replace('VERIFY','VERIFY114');run.path=STUDY/'runs'/run.run_id
    run.meta.update(run_id=run.run_id,scope='verify_recovery114',helper_sha256=sha256(Path(__file__)))
    with run:execute(run,recovered,states)
