"""Execute fixed-mask re-centering and matched pressure; zero network training."""
from common import *
import argparse, gc, time


def execute(run):
    import numpy as np
    import torch
    from adapter_data import FrozenCache
    from native_geometry import lifted_basis, roi_for
    from native_output_bridge import decode_native, encode_rle, native_environment
    from native_ops import Decoder, measure, matched_coefficients
    from margin_math import recenter, geometric_margin, draw
    from margin_stats import analyze
    from verify import verify
    scope=run.meta['scope'];cfg=read_json(STUDY/'CONFIG.json');prep=read_json(STUDY/'PREPARATION.json')
    inputs_check();profile,env=configure_native();require(env==prep['environment'],'runtime drift')
    controls=lines((C_FULL if scope=='full' else C_SMOKE)/'PAIR_RESULTS.jsonl')
    controls=sorted(controls,key=lambda r:(r['image_id'],r['annotation_id']))
    history={r['output_detection_id']:r for r in lines(OLD/'PAIRED_MASKS.jsonl')}
    ids=sorted({r['image_id'] for r in controls});source=FrozenCache(BASE/'trace',ids)
    library=np.load(FIT/'FIT_ERRORS.npz')
    require(not set(ids)&set(library['image_ids'].tolist()),'fit donor leakage')
    run.freeze({'environment':env,'profile':profile,'native_args':prep['native_args'],'candidate_policy':'same_final_Box75_detection',
                'scope':'GT_assisted_fixed_detection_stability_not_AP','source_C_run':C_FULL.name if scope=='full' else C_SMOKE.name,
                'new_training_steps':0,'source_pressure_library':FIT.name,'G_budget':cfg['G_budget'],
                'native_pipeline':'full_Nx32_GEMM_then_selected_native_scale_gt0_crop_scale_byte','config':cfg})
    results=[];zero_count=0;full_checks=0;processing=0.
    with (run.path/'PAIR_RESULTS.jsonl').open('x',encoding='utf8') as stream,torch.inference_mode():
        for image_number,image_id in enumerate(ids):
            resource_check(run);started=time.monotonic()
            cache,native_rows=source.load(image_id)
            native_full=decode_native(cache).cpu()
            for index,pred in enumerate(native_rows):
                require(encode_rle(native_full[index])==pred['segmentation'],'native zero-delta RLE mismatch')
                zero_count+=1
            del native_full
            basis=lifted_basis(cache)
            for old in [r for r in controls if r['image_id']==image_id]:
                resource_check(run);rank=old['rank'];oid=old['output_detection_id'];c=np.array(old['arms']['C']['coefficients_fp32'],dtype=np.float32)
                require(cache['prediction_indices'][rank]==old['global_index'],'final detection rank changed')
                require(old['native_prediction']==native_rows[rank] and old['cache_sha256']==source.records[image_id]['cache_sha256'],'source C identity differs')
                require(cache['c0'][rank].tolist()==old['c0'] and old['arms']['C']['solver']['solver_status']=='converged','source C/c0 invalid')
                decoder=Decoder(cache,rank);roi=roi_for(cache,rank).numpy()
                B=basis[:,torch.from_numpy(roi)].T.contiguous().numpy()
                gt=old['GT_rle']
                nr,_,_=decoder.evaluate(cache['c0'][rank].numpy(),True);full_checks+=1
                require(nr==native_rows[rank]['segmentation'],'selected N mismatch')
                cr,cf,cz=decoder.evaluate(c,True);full_checks+=1
                require(cr==old['arms']['C']['original_rle'],'C final replay mismatch')
                require(encode_rle(torch.from_numpy(cf).byte())==old['arms']['C']['input_rle'],'C first threshold replay mismatch')
                labels=cf[roi];rms_C=float(np.sqrt(np.mean(cz**2)))
                require(rms_C>cfg['numerics']['zero_rms_epsilon'],'zero C RMS')
                proxy_diff=np.abs(B@c.astype(float)-cz)
                proxy_valid=bool(np.all(proxy_diff<=cfg['numerics']['proxy_atol']+cfg['numerics']['proxy_rtol']*np.abs(cz)))
                candidate,solver=recenter(B,c.astype(float),labels,cfg['solver'],lambda:resource_check(run))
                margin_C=geometric_margin(B,c.astype(float),labels,cfg['solver'])
                accepted=False;reasons=[];candidate_rms=None;candidate_rle=None;candidate_margin=None
                if solver['converged']:
                    candidate=(candidate*(rms_C/decoder.rms(candidate)[0])).astype(np.float32)
                    candidate=(candidate.astype(float)*(rms_C/decoder.rms(candidate)[0])).astype(np.float32)
                    candidate_rle,first,z=decoder.evaluate(candidate,scope=='smoke');full_checks+=int(scope=='smoke')
                    candidate_rms=float(np.sqrt(np.mean(z**2)))
                    candidate_margin=geometric_margin(B,candidate.astype(float),labels,cfg['solver'])
                    err=np.abs(B@candidate.astype(float)-z)
                    proxy_valid &= bool(np.all(err<=cfg['numerics']['proxy_atol']+cfg['numerics']['proxy_rtol']*np.abs(z)))
                    if not np.array_equal(cf,first):reasons.append('first_mask_not_exact')
                    if candidate_rle!=cr:reasons.append('final_mask_not_exact')
                    if abs(candidate_rms/rms_C-1)>cfg['numerics']['rms_relative_tolerance']:reasons.append('center_scale_mismatch')
                    if candidate_margin is None or margin_C is None or candidate_margin<margin_C-cfg['solver']['full_feasibility_tol']:reasons.append('margin_not_preserved')
                    accepted=not reasons
                else:reasons.append(solver['status'])
                w=candidate.copy() if accepted else c.copy()
                wr,wf,wz=decoder.evaluate(w,scope=='smoke');full_checks+=int(scope=='smoke')
                require(wr==cr and np.array_equal(wf,cf),'accepted/reverted center changed nominal mask')
                rms_W=float(np.sqrt(np.mean(wz**2)))
                require(abs(rms_W/rms_C-1)<=cfg['numerics']['rms_relative_tolerance'],'nominal scale differs')
                margin_W=candidate_margin if accepted else margin_C
                vectors,pressure_labels=draw(library['errors'],library['mean'],library['image_ids'],image_id,rank,cfg)
                amplitude=cfg['pressure_ratio']*rms_C
                cp,camp,vector_rms,cnorm=matched_coefficients(decoder,c,vectors,amplitude,cfg['numerics']['rms_relative_tolerance'])
                wp,wamp,_,wnorm=matched_coefficients(decoder,w,vectors,amplitude,cfg['numerics']['rms_relative_tolerance'])
                pressure=[]
                for index,label in enumerate(pressure_labels):
                    check=scope=='smoke' or index==0
                    cpr,_,_=decoder.evaluate(cp[index],check);wpr,_,_=decoder.evaluate(wp[index],check)
                    full_checks+=2*int(check)
                    pressure.append({**label,'vector':vectors[index].tolist(),'pre_normalization_rms':float(vector_rms[index]),
                        'C_coefficients':cp[index].tolist(),'W_coefficients':wp[index].tolist(),'C_actual_rms':float(camp[index]),'W_actual_rms':float(wamp[index]),
                        'C_normalization_scale':cnorm['scales'][index],'W_normalization_scale':wnorm['scales'][index],
                        'C':measure(cpr,gt),'W':measure(wpr,gt)})
                oldg=None
                if oid in history:
                    h=history[oid];require(h['native_prediction']==native_rows[rank] and h['GT_rle']==gt,'old G identity differs')
                    oldg=measure(h['gated_rle'],gt)
                row={k:old[k] for k in ['image_id','annotation_id','output_detection_id','rank','global_index','raw_id','cache_sha256']}
                row.update(GT_rle=gt,c0=cache['c0'][rank].tolist(),C_coefficients=c.tolist(),W_coefficients=w.tolist(),
                    candidate_coefficients=np.asarray(candidate).tolist(),candidate_rle=candidate_rle,candidate_rms=candidate_rms,
                    solver=solver,accepted=bool(accepted),fallback_reasons=reasons,proxy_valid=proxy_valid,
                    proxy_max_abs_error=float(proxy_diff.max()),margin_C=margin_C,margin_W=margin_W,
                    candidate_margin=candidate_margin,C_logit_rms=rms_C,W_logit_rms=rms_W,pressure_amplitude=amplitude,
                    normalization_C=cnorm['history'],normalization_W=wnorm['history'],
                    endpoints={'N':measure(nr,gt),'C':measure(cr,gt),'W':measure(wr,gt),'G_old':oldg,'G_budget':None},pressure=pressure)
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();results.append(row)
                print('PAIR',len(results),'/',len(controls),'accepted',accepted,'solver',solver['status'],flush=True)
                del decoder,B,row,pressure,cf,cz,wf,wz,cp,wp
                gc.collect();torch.cuda.empty_cache()
            processing+=time.monotonic()-started
            run.update(phase='margin_and_pressure',completed_pairs=len(results),completed_images=image_number+1,
                       accepted_centers=sum(r['accepted'] for r in results),zero_delta_exact_rows=zero_count,
                       full_native_decoder_checks=full_checks,finite_coefficient_solves=len(results),processing_seconds=processing)
            del basis,cache,native_rows;gc.collect();torch.cuda.empty_cache()
    require(len(results)==len(controls),'incomplete panel')
    source.verify();inputs_check();require(native_environment()==env,'runtime changed')
    summary=analyze(results,cfg,scope,run.path/'BOOTSTRAP_INDICES.npz')
    summary.update(processing_seconds=processing,zero_delta_exact_rows=zero_count,full_native_decoder_checks=full_checks,
                   projected_full_seconds=processing*256/len(results)*1.3+180 if scope=='smoke' else None,
                   max_cuda_allocated=torch.cuda.max_memory_allocated())
    write_new(run.path/'STATISTICS.json',summary)
    independent=verify(run.path,replay=True);write_new(run.path/'INDEPENDENT_LAPTOP.json',independent)
    require(independent['status']=='passed','independent check failed')
    if scope=='smoke':
        require(all(r['proxy_valid'] for r in results),'smoke proxy invalid')
        require(summary['projected_full_seconds']<=cfg['budget']['full_seconds'],'smoke exceeds full budget')
    audit={'status':'passed','native_and_C_RLE_exact':True,'nominal_W_equals_C':True,'pressure_RMS_matched':True,
           'inputs_and_sources_unchanged':True,'source_environment_exact':True,'network_training_steps':0,
           'base_model_forward_calls':0,'adapter_forward_calls':0,'finite_coefficient_solves':len(results),
           'G_budget_comparison':'unavailable_not_substituted','solver_numerical_valid':summary['numerical_valid']}
    write_new(run.path/'CONFIG_AUDIT.json',audit)
    next_action='Execute frozen full panel once' if scope=='smoke' else ('Recover the G_budget checkpoint and complete matched reference before any stability-training design' if summary['decision']=='STABILITY_DESIGN_SUPPORTED_REFERENCE_PENDING' else 'Do not train this margin objective; retain measured evidence and restore G_budget final-weight backup for an independent-confirmation design')
    run.finish(summary,status='passed' if scope=='smoke' else 'supported' if summary['decision']=='STABILITY_DESIGN_SUPPORTED_REFERENCE_PENDING' else 'not_passed',scientific=summary['decision'],next_action=next_action)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['smoke','full'],required=True);parser.add_argument('--parent-run');args=parser.parse_args()
    if args.scope=='full':
        require(args.parent_run and Path(args.parent_run).name==args.parent_run,'smoke id required')
        parent=STUDY/'runs'/args.parent_run
        require(read_json(parent/'run.json')['execution_status']=='completed','smoke incomplete')
        require(read_json(parent/'INDEPENDENT_LAPTOP.json')['status']=='passed','smoke verification incomplete')
    else:parent=C_SMOKE
    require(not list((STUDY/'runs').glob('RUN_RSM_'+args.scope.upper()+'_*')),'Run already exists; explicit engineering retry required')
    run=MarginRun(args.scope,parent);study=read_json(STUDY/'study.json')
    try:
        with run:
            study.update(execution_status='running',active_run_id=run.run_id,current_phase=args.scope,artifact_status='partial')
            save(STUDY/'study.json',study);execute(run)
    finally:
        if (run.path/'run.json').exists():
            m=read_json(run.path/'run.json');study.update(execution_status=m['execution_status'],scientific_status=m['scientific_status'],active_run_id=None,
               latest_run_id=run.run_id,artifact_status=m['artifact_status'],current_phase='full_ready' if args.scope=='smoke' and m['execution_status']=='completed' else 'awaiting_transfer' if m['execution_status']=='completed' else 'engineering_error_retained')
            save(STUDY/'study.json',study)


if __name__=='__main__':main()
