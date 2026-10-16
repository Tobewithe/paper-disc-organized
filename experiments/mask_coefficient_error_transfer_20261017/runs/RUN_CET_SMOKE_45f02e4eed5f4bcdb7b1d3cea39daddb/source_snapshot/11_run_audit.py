"""Only fit error statistics and replay saved coefficient tensors; never train."""
import argparse,json
from collections import defaultdict
import numpy as np
import torch
from common import *
from adapter_data import FrozenCache
from s1_geometry import iter_prediction_array
from native_output_bridge import decode_native,encode_rle
from math_ops import functional_rms,weighted_median,draw_directions,normalize,decode_selected,mask_measure
from audit_statistics import summarize


def fit():
    with Run('fit') as run:
        data=inspect_inputs();config=run.config
        run.freeze(data,dict(stage='fit_error_structure_only',heldout_error_rows_used=0))
        teachers=jsonlines(STUDY/'FIT_TEACHERS.jsonl')
        fit_ids=set(data['split']['fit_image_ids']);held=set(data['split']['heldout_image_ids'])
        require(len(teachers)==config['fit_pairs'] and all(t['image_id'] in fit_ids and t['image_id'] not in held for t in teachers),'fit population')
        groups=defaultdict(list)
        for t in teachers:groups[t['image_id']].append(t)
        require(len(groups)==config['fit_teacher_images'],'fit image count')
        source=FrozenCache(CACHE/'trace',sorted(groups))
        errors=[];c0s=[];cs=[];rms=[];weights=[];identities=[]
        with torch.inference_mode():
            for pos,iid in enumerate(sorted(groups),1):
                run.check_budget();cache,rows=source.load(iid)
                for t in sorted(groups[iid],key=lambda r:r['rank']):
                    rank=t['rank'];c0=cache['c0'][rank].numpy().astype(np.float64);c=np.array(t['coefficients_fp32'],dtype=np.float32).astype(np.float64)
                    require(cache['prediction_indices'][rank]==t['global_index'],'fit output identity')
                    e=c0-c;a=float(functional_rms(cache,rank,e[None])[0])
                    require(np.isfinite(a) and a>config['zero_rms_epsilon'],'fit functional amplitude')
                    errors.append(e);c0s.append(c0);cs.append(c);rms.append(a);weights.append(1/len(groups)/len(groups[iid]))
                    identities.append(dict(image_id=iid,rank=rank,annotation_id=t['annotation_id'],global_index=t['global_index'],output_detection_id=t['output_detection_id']))
                if pos%100==0 or pos==len(groups):
                    run.update(images_completed=pos,pairs_completed=len(errors));print('FIT',pos,len(groups),flush=True)
        errors=np.array(errors);weights=np.array(weights);rms=np.array(rms)
        mu=np.sum(errors*weights[:,None],axis=0)/weights.sum();xi=errors-mu
        cov=(xi*weights[:,None]).T@xi/weights.sum()
        a=weighted_median(rms,weights)
        require(np.linalg.norm(mu)>1e-10,'zero mean has no valid predefined MU pressure')
        np.savez(run.path/'FIT_ERRORS.npz',errors=errors,c0=np.array(c0s),teacher=np.array(cs),weights=weights,
            native_input_rms=rms,mean=mu,covariance=cov,image_ids=np.array([r['image_id'] for r in identities]))
        write_new(run.path/'FIT_IDENTITIES.json',identities)
        write_new(run.path/'FIT_MODEL.json',dict(amplitude=a,amplitude_definition='fit_image_weighted_median_native_input_crop_RMS',
            fit_pairs=len(errors),fit_images=len(groups),heldout_error_rows_used=0,library_sha256=sha256(run.path/'FIT_ERRORS.npz'),
            identities_sha256=sha256(run.path/'FIT_IDENTITIES.json'),mean=mu.tolist(),covariance=cov.tolist()))
        config_finish(run,source,dict(fit_only_estimation=True))
        run.finish(dict(fit_pairs=len(errors),fit_images=len(groups),amplitude=a,heldout_error_rows_used=0),scientific='fit_statistics_only',next_action='Frozen smoke; no training')


def evaluate(scope,fit_path):
    model=completed_fit(fit_path)
    require(read_json(fit_path/'INDEPENDENT_LAPTOP.json')['status']=='passed','fit independent check missing')
    if scope=='full':
        prior=list((STUDY/'runs').glob('RUN_CET_SMOKE_*'))
        require(len(prior)==1,'exactly one engineering smoke required')
        require(read_json(prior[0]/'run.json')['execution_status']=='completed','smoke incomplete')
        require(read_json(prior[0]/'INDEPENDENT_LAPTOP.json')['status']=='passed','smoke independent check')
        require(read_json(prior[0]/'EVAL_CONTRACT.json')['fit_run']==fit_path.name,'smoke fit mismatch')
        require(read_json(prior[0]/'SOURCE_MANIFEST.json')['files']==science_check()['files'],'smoke source mismatch')
    with Run(scope,fit_path) as run:
        data=inspect_inputs();config=run.config
        rows_all=jsonlines(STUDY/'HELDOUT_TEACHERS.jsonl')
        require(len(rows_all)==config['heldout_pairs'] and len({r['image_id'] for r in rows_all})==config['heldout_teacher_images'],'held population')
        ids=sorted({r['image_id'] for r in rows_all})
        if scope=='smoke':ids=ids[:2]
        groups=defaultdict(list)
        for t in rows_all:
            if t['image_id'] in ids:groups[t['image_id']].append(t)
        require(set(ids).issubset(data['split']['heldout_image_ids']),'recipient leakage')
        run.freeze(data,dict(stage=scope,fit_run=fit_path.name,fit_model_sha256=sha256(fit_path/'FIT_MODEL.json'),
            fit_library_sha256=sha256(fit_path/'FIT_ERRORS.npz'),amplitude=model['amplitude'],pressure_images=ids))
        library=np.load(fit_path/'FIT_ERRORS.npz');errors=library['errors'];mu=library['mean'];error_ids=library['image_ids']
        source=FrozenCache(CACHE/'trace',ids)
        mapping=read_json(OLD/'ROW_MAPPING.json')['original_global_indices']
        require(mapping==list(range(len(mapping))),'historic G must contain complete native index domain')
        wanted={t['global_index'] for values in groups.values() for t in values}
        old_g={j:r for j,r in enumerate(iter_prediction_array(OLD/'G_predictions.json')) if j in wanted}
        require(set(old_g)==wanted,'G output coverage')
        from pycocotools.coco import COCO
        coco=COCO(str(GT))
        results=[];all_coefficients=[];max_relative=0.;native_rows=0
        with (run.path/'PAIRS.jsonl').open('x',encoding='utf8') as out,torch.inference_mode():
            for pos,iid in enumerate(ids,1):
                run.check_budget();cache,base=source.load(iid)
                native=decode_native(cache)
                require(all(encode_rle(native[j])==r['segmentation'] for j,r in enumerate(base)),'full-image native RLE replay mismatch')
                native_rows+=len(base);del native
                for t in sorted(groups[iid],key=lambda r:r['rank']):
                    rank=t['rank'];original=base[rank];g=old_g[t['global_index']]
                    require(cache['prediction_indices'][rank]==t['global_index'] and t['cache_sha256']==source.records[iid]['cache_sha256'],'recipient identity')
                    require({k:v for k,v in g.items() if k!='segmentation'}=={k:v for k,v in original.items() if k!='segmentation'},'G nonmask identity')
                    c=np.array(t['coefficients_fp32'],dtype=np.float32);c0=cache['c0'][rank].numpy()
                    anchors=decode_selected(cache,rank,np.stack([c0,c]))
                    require(anchors[0]==original['segmentation'] and anchors[1]==t['oracle_rle'],'selected native/teacher decode differs')
                    gt=coco.annToRLE(coco.anns[t['annotation_id']]);gt=dict(size=gt['size'],counts=gt['counts'].decode('ascii'))
                    n=mask_measure(anchors[0],gt,anchors[0]);ct=mask_measure(anchors[1],gt,anchors[0]);gg=mask_measure(g['segmentation'],gt,anchors[0])
                    require(abs(n['iou']-t['baseline_iou'])<1e-12 and abs(ct['iou']-t['oracle_iou'])<1e-12,'anchor IoU drift')
                    vectors,labels=draw_directions(errors,mu,error_ids,iid,rank,config)
                    coefficients,actual,before=normalize(cache,rank,vectors,c,model['amplitude'],config)
                    predicted=decode_selected(cache,rank,coefficients)
                    max_relative=max(max_relative,float(np.max(np.abs(actual/model['amplitude']-1))))
                    trials=[]
                    for j,(label,rle) in enumerate(zip(labels,predicted)):
                        trials.append(dict(**label,**mask_measure(rle,gt,anchors[0]),rle=rle,
                            coefficient_index=len(all_coefficients)+j,functional_rms=float(actual[j]),raw_direction_rms=float(before[j])))
                    record=dict(image_id=iid,rank=rank,annotation_id=t['annotation_id'],global_index=t['global_index'],
                        output_detection_id=t['output_detection_id'],N=n['iou'],C=ct['iou'],G=gg['iou'],
                        native_rle=anchors[0],teacher_rle=anchors[1],G_rle=g['segmentation'],GT_rle=gt,trials=trials)
                    out.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');out.flush()
                    all_coefficients.extend(coefficients);results.append(record)
                if pos%10==0 or pos==len(ids):
                    run.update(images_completed=pos,pairs_completed=len(results),pressure_decodes=len(all_coefficients))
                    print(scope.upper(),pos,'/',len(ids),'pairs',len(results),flush=True)
        np.savez(run.path/'PRESSURE_COEFFICIENTS.npz',coefficients=np.array(all_coefficients,dtype=np.float32))
        summary,draws,order=summarize(results,config)
        np.savez(run.path/'BOOTSTRAP_INDICES.npz',indices=draws)
        write_new(run.path/'STATISTICAL_IMAGE_ORDER.json',order)
        summary.update(scope=scope,fit_amplitude=model['amplitude'],native_zero_delta_rows=native_rows,
            pressure_decodes=len(all_coefficients),max_functional_rms_relative_error=max_relative,
            G_reference_scope='historical_trained_on_all2000_not_independent_generalization',fit_model_sha256=sha256(fit_path/'FIT_MODEL.json'))
        if scope=='smoke':summary['decision']='ENGINEERING_SMOKE_ONLY';summary['prototype_tolerance_training_design_warranted']=False
        config_finish(run,source,dict(native_zero_delta_rows=native_rows,all_teacher_RLE_exact=True,
            heldout_errors_used_in_pressure_construction=False,functional_amplitude_matched=True,
            max_functional_rms_relative_error=max_relative,G_nonmask_identity=True))
        run.finish(summary,status=summary['decision'],scientific='engineering_only' if scope=='smoke' else 'internal_transfer_diagnostic',next_action='Independent verification; no automatic training')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','smoke','full']);p.add_argument('--fit-run',type=Path)
    args=p.parse_args()
    if args.stage=='fit':fit()
    else:evaluate(args.stage,args.fit_run.resolve())
