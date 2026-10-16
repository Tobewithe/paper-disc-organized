"""Frozen AMT S2: single panel, native replay, complete U/G/control outputs."""
import argparse
import subprocess
import time
from itertools import zip_longest
from s2_common import *


def prepare():
    import psutil
    require(not (STUDY/'EVALUATION_CONTRACT.json').exists(),'evaluation already frozen; do not redraw')
    require(not (STUDY/'S2_DAMAGE_PANEL.json').exists(),'panel exists; do not redraw')
    require(not list(ROOT.glob('experiments/*/.*.lock')),'existing experiment lock')
    require(not list(ROOT.glob('experiments/*/.AUTONOMOUS_OWNER.guard')),'existing owner')
    others=[p.pid for p in psutil.process_iter(['name']) if p.pid!=os.getpid() and (p.info['name'] or '').lower().startswith('python')]
    require(not others,'other Python jobs exist: '+str(others))
    require(psutil.virtual_memory().available>=4*1024**3 and shutil.disk_usage(ROOT).free>=8*1024**3,'resources unavailable')
    data=inputs()
    tests=[Path(__file__).with_name('test_s2.py'),amt.PARENT/'scripts/test_s2_evaluation.py',
        ROOT/'experiments/mask_readout_large_mask_gate_20261015/scripts/test_area_gate.py']
    for path in tests:
        with (STUDY/('S2_'+path.stem+'.log')).open('x',encoding='utf8') as log:
            p=subprocess.run([sys.executable,'-B',str(path)],stdout=log,stderr=subprocess.STDOUT,timeout=180)
        require(p.returncode==0,'evaluation test failed: '+path.name)
    baseline=read_jsonl(BASE/'ledger/PER_GT_FAILURE.jsonl')
    old=read_jsonl(OLD/'FIXED_PAIR_EFFECTS.jsonl')
    require(len(old)==2925 and sum(r['population']=='T' for r in old)==655,'fixed pair denominator drift')
    require(bootstrap(image_effects(old))['mean']==.010024491543988425,'original G identity drift')
    require(sum(population(r)=='S' for r in baseline)==22957,'success population drift')
    require(not [p for p in (STUDY/'runs').glob('*/DAMAGE_PANEL.json') if read_json(p)['checkpoint_sha256']==CHECKPOINT_SHA],
        'prior checkpoint panel exists')
    contract=dict(stage='S2_AMT',primary='G',descriptive='U',control='original_soft_IoU_G',
        created_at=now(),checkpoint_sha256=CHECKPOINT_SHA,checkpoint_path=str(CHECKPOINT),
        source_training_run=TRAIN.name,source_baseline_run=BASE.name,control_development_run=OLD.name,control_full_run=S3.name,
        original_protocol_sha256=sha256(STUDY/'PROTOCOL.md'),execution_protocol_sha256=sha256(STUDY/'EVALUATION_EXECUTION.md'),
        source_freeze_sha256=sha256(STUDY/'S2_SOURCE_FREEZE.json'),
        inputs_sha256=data['inputs_sha256'],environment=data['environment'],external_evaluator_sources=data['external_evaluator_sources'],
        native_profile=data['profile'],dev_ids=data['split']['dev_ids'],
        denominators=dict(dev_images=512,target_pairs=655,all_bbox75_pairs=2925,target_images=257,all_bbox75_images=508,success_population=22957,damage_sample=1024),
        native_area_min=9216,area_source='native_original_image_RLE',evaluation_phase='eval:0',
        geometry='each_original_cached_P_c0_F_input_hw; no regrouping',
        fixed_boxes_classes_scores_order=True,same_full_image_coefficient_matrix=True,no_GT_in_forward=True,
        candidate_selection='last_checkpoint_G_only; U_descriptive_only',
        matching='pinned_pycocotools_full_JSON_with_bbox; bbox_and_segm_independent; all_GT_and_outputs_per_image',
        maxDets=[1,10,100],useCats=1,
        bootstrap=dict(resamples=2000,seed=1729,quantile='linear',CI=[.025,.975],unit='IoU_fraction',weighting='within_image_then_equal_images'),
        gates=dict(target_delta_min=.005,target_ci_lower_strict_min=0.,all_bbox75_delta_min=-.001,
            damage_one_sided95_upper_max=.01,new_G_minus_original_G_ci_lower_strict_min=0.),
        damage_design='one_system_128bit_seed_SRSWOR1024_after_contract_before_any_method_output',
        union_image_count='determined_once_by_panel; no preset_old1314',
        smoke_images='sorted_union_first8',new_training_steps=0,base_model_forward_calls=0,
        budgets=dict(smoke_seconds=600,full_seconds=3600,independent_smoke_seconds=600,independent_full_seconds=1800,peak_RAM_GiB=6,run_artifacts_GiB=4),
        automatic_S3=False,automatic_S4=False,scientific_scope='reused_COCO_development_not_independent_confirmation')
    write_new(STUDY/'EVALUATION_CONTRACT.json',contract)
    panel=select_damage_panel(baseline,sha256(BASE/'ledger/PER_GT_FAILURE.jsonl'),CHECKPOINT_SHA,
        sha256(STUDY/'EVALUATION_CONTRACT.json'),1024)
    write_new(STUDY/'S2_DAMAGE_PANEL.json',panel)
    write_new(STUDY/'S2_PANEL_RECEIPT.json',dict(at=now(),panel_sha256=sha256(STUDY/'S2_DAMAGE_PANEL.json'),
        contract_sha256=sha256(STUDY/'EVALUATION_CONTRACT.json'),python=sys.version,seed_source='secrets.randbits(128)',
        new_method_outputs_seen=False,draws=1,population=22957,n=1024))
    ids=sorted(set(contract['dev_ids'])|set(panel['image_ids']))
    write_new(STUDY/'S2_EVALUATION_IMAGES.json',ids)
    write_new(STUDY/'S2_PREPARATION.json',dict(status='passed',at=now(),inputs=len(data['inputs_sha256']),
        source_files=len(CODE),images=len(ids),damage_images=len(panel['image_ids']),available_RAM=psutil.virtual_memory().available,
        available_disk=shutil.disk_usage(ROOT).free,tests=[p.name for p in tests]))
    print('S2_PREPARATION_PASSED',len(ids),'images',flush=True)


def export(run,source,template,coco):
    import torch
    from adapter_model import CoefficientAdapter
    from native_output_bridge import decode_native,encode_rle
    model=CoefficientAdapter('B').cuda()
    checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    require(checkpoint['arm']=='B' and checkpoint['completed_epochs']==3,'wrong final state')
    model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
    initial_hash=tensor_state_hash(model)
    require(initial_hash==read_json(TRAIN/'SUMMARY.json')['final_state_sha256'],'loaded model does not equal final training state')
    del checkpoint
    byimage=defaultdict(list)
    for row in template:byimage[row['image_id']].append(row)
    eff={k:[] for k in ('U','G')};indices=[];gate_count=changed_count=zero_count=0
    paths={k:run.path/(k+'_predictions.json') for k in ('NATIVE','U','G')}
    with contextlib.ExitStack() as stack,torch.inference_mode():
        streams={k:stack.enter_context(p.open('x',encoding='utf8')) for k,p in paths.items()}
        gate=stack.enter_context((run.path/'GATE_ROWS.jsonl').open('x',encoding='utf8'))
        for f in streams.values():f.write('[')
        comma=False
        for n,image_id in enumerate(source.records,1):
            run.check_budget()
            cache,original=source.load(image_id)
            zero=decode_native(cache)
            for j,row in enumerate(original):require(encode_rle(zero[j])==row['segmentation'],'zero-delta native replay differs')
            zero_count+=len(original);del zero
            delta=model(cache,'eval:0')
            require(torch.isfinite(delta).all().item(),'nonfinite delta')
            decoded=decode_native(cache,cache['c0'].to(delta)+delta)
            u=[{**row,'segmentation':encode_rle(decoded[j])} for j,row in enumerate(original)]
            compare_outputs(original,u)
            g=[]
            for j,(native,updated) in enumerate(zip(original,u)):
                mixed,choice=mix_row(native,updated);g.append(mixed)
                choice.update(image_id=image_id,original_global_index=cache['prediction_indices'][j])
                gate.write(json.dumps(choice)+'\n')
                gate_count+=int(choice['apply_delta']);changed_count+=int(choice['mask_changed'])
            for old in byimage[image_id]:
                pos=cache['prediction_indices'].index(int(old['baseline_output_detection_id'].rsplit(':',1)[1]))
                actual=full_rle_iou(original[pos]['segmentation'],coco.annToRLE(coco.anns[old['annotation_id']]))
                require(abs(actual-old['baseline_iou'])<1e-12,'fixed pair native IoU differs')
            for arm,rows in [('U',u),('G',g)]:eff[arm].extend(effects(rows,cache['prediction_indices'],byimage[image_id],coco))
            for j in range(len(original)):
                for arm,rows in [('NATIVE',original),('U',u),('G',g)]:
                    f=streams[arm]
                    if comma:f.write(',')
                    f.write(json.dumps(rows[j],separators=(',',':'),allow_nan=False))
                comma=True
            indices.extend(cache['prediction_indices'])
            del cache,original,delta,decoded,u,g
            if n%50==0 or n==len(source.records):
                for f in streams.values():f.flush()
                run.update(phase='export_U_G_native_zero_replay',completed_images=n,total_images=len(source.records))
                print('AMT_S2_EXPORT',n,'/',len(source.records),flush=True)
                gc.collect();torch.cuda.empty_cache()
        for f in streams.values():f.write(']\n')
    require(indices==sorted(set(indices)),'original row order changed')
    require(tensor_state_hash(model)==initial_hash,'model state changed')
    del model;torch.cuda.empty_cache()
    for arm,rows in eff.items():jsonl(run.path/(arm+'_FIXED_PAIR_EFFECTS.jsonl'),rows)
    write_new(run.path/'ROW_MAPPING.json',dict(source_prediction_sha256=source.manifest['predictions_sha256'],original_global_indices=indices))
    stats=dict(images=len(source.records),outputs=len(indices),zero_delta_RLE_exact=zero_count,gate_on=gate_count,G_masks_changed=changed_count,
        checkpoint_state_sha256=initial_hash,adapter_forward_calls=len(source.records),base_model_forward_calls=0,new_training_steps=0)
    write_new(run.path/'EXPORT.json',stats)
    return eff,indices,stats


def extract_control(run,indices):
    wanted=set(indices);written=[];count=0;total=0
    with (run.path/'CONTROL_predictions.json').open('x',encoding='utf8') as f:
        f.write('[')
        for i,(native,control) in enumerate(zip_longest(iter_prediction_array(NATIVE),iter_prediction_array(CONTROL))):
            require(native is not None and control is not None,'control output denominator differs')
            total+=1
            if i not in wanted:continue
            compare_outputs([native],[control]);written.append(i)
            if count:f.write(',')
            f.write(json.dumps(control,separators=(',',':'),allow_nan=False));count+=1
            if count%10000==0:run.check_budget()
        f.write(']\n')
    require(written==indices and total==539128,'full control row coverage differs')


def execute(scope,smoke_run):
    with EvalRun(scope) as run:
        data=inputs();contract,panel=check_contract(data)
        if scope=='full':
            sm=STUDY/'runs'/smoke_run
            require(read_json(sm/'DECISION.json')['status']=='smoke_passed','smoke not passed')
            require(read_json(sm/'INDEPENDENT_LAPTOP.json')['status']=='passed','independent smoke missing')
            require(sha256(sm/'EVAL_CONTRACT.json')==sha256(STUDY/'EVALUATION_CONTRACT.json'),'smoke contract differs')
        run.freeze(contract)
        ids=read_json(STUDY/'S2_EVALUATION_IMAGES.json')
        require(ids==sorted(set(contract['dev_ids'])|set(panel['image_ids'])),'image union drift')
        if scope=='smoke':ids=ids[:8]
        write_new(run.path/'EVALUATION_IMAGES.json',ids)
        baseline=read_jsonl(BASE/'ledger/PER_GT_FAILURE.jsonl')
        template=[r for r in read_jsonl(OLD/'FIXED_PAIR_EFFECTS.jsonl') if r['image_id'] in set(ids)]
        from adapter_data import FrozenCache
        from pycocotools.coco import COCO
        source=FrozenCache(BASE/'trace',ids);coco=COCO(str(GT))
        eff,indices,stats=export(run,source,template,coco)
        extract_control(run,indices)
        control_rows=read_json(run.path/'CONTROL_predictions.json')
        eff['CONTROL']=effects(control_rows,indices,template,coco)
        for old,new in zip(template,eff['CONTROL']):
            require(abs(old['method_iou']-new['method_iou'])<1e-12,'old G dev result replay drift')
        jsonl(run.path/'CONTROL_FIXED_PAIR_EFFECTS.jsonl',eff['CONTROL'])
        del control_rows,coco;gc.collect()
        if scope=='smoke':
            with (run.path/'NATIVE_ledger.log').open('x',encoding='utf8') as log,contextlib.redirect_stdout(log):
                native,_=evaluate_ledger(GT,read_json(run.path/'NATIVE_predictions.json'),ids,sha256(run.path/'NATIVE_predictions.json'))
            validate_ledger(native,baseline,ids,indices,source.manifest['predictions_sha256'],native=True)
            jsonl(run.path/'NATIVE_GT_LEDGER.jsonl',native)
            write_new(run.path/'NATIVE_SMOKE_PARITY.json',dict(status='passed',images=len(ids),GT=len(native)))
        results={}
        for arm in ('U','G','CONTROL'):
            run.check_budget();run.update(phase='COCO_full_image_matching',arm=arm)
            path=run.path/(arm+'_predictions.json')
            with (run.path/(arm+'_ledger.log')).open('x',encoding='utf8') as log,contextlib.redirect_stdout(log):
                ledger,ap=evaluate_ledger(GT,read_json(path),ids,sha256(path))
            wanted=validate_ledger(ledger,baseline,ids,indices,source.manifest['predictions_sha256'])
            jsonl(run.path/(arm+'_GT_LEDGER.jsonl'),ledger)
            if arm=='U':jsonl(run.path/'BASELINE_SUBSET_GT_LEDGER.jsonl',wanted.values())
            if scope=='full':
                require(len(eff[arm])==2925 and sum(r['population']=='T' for r in eff[arm])==655,'fixed denominator drift')
                results[arm]=result(eff[arm],ledger,panel,baseline)
                require(results[arm]['target']['images']==257 and results[arm]['all_bbox75']['images']==508,'image denominator drift')
                results[arm].update(prediction_sha256=sha256(path),secondary_AP_on_union=ap)
                write_new(run.path/(arm+'_RESULTS.json'),results[arm])
            del ledger;gc.collect()
        if scope=='full':
            new=image_effects(eff['G']);old=image_effects(eff['CONTROL'])
            contrast=bootstrap({i:new[i]-old[i] for i in new})
            status,gates=decide(results['G'],contrast)
            summary=dict(G=results['G'],U=results['U'],CONTROL=results['CONTROL'],G_minus_CONTROL=contrast,gates=gates,export=stats,
                scientific_scope='COCO_development; not independent confirmation; union AP not S3 full5000')
        else:
            status='smoke_passed';summary=dict(export=stats,fixed_pairs=len(template),scientific_scope='engineering_only')
        source.verify();del source;gc.collect()
        after=inputs();check_contract(after)
        require(after['inputs_sha256']==data['inputs_sha256'] and after['environment']==data['environment'],'post evaluation drift')
        import psutil
        memory=psutil.Process().memory_info();peak=getattr(memory,'peak_wset',memory.rss)
        require(peak<=6*1024**3,'RAM budget exceeded')
        require(sum(p.stat().st_size for p in run.path.rglob('*') if p.is_file())<=4*1024**3,'artifact budget exceeded')
        run.check_budget()
        write_new(run.path/'CONFIG_AUDIT.json',dict(status='passed',checkpoint_sha256=CHECKPOINT_SHA,
            contract_sha256=sha256(run.path/'EVAL_CONTRACT.json'),damage_panel_sha256=sha256(run.path/'DAMAGE_PANEL.json'),peak_rss_bytes=peak,
            checks=dict(original_training_freeze_and_audits=True,source_inputs_runtime_before_after=True,
                all_c0_RLE_exact=True,unchanged_P_c0_F_geometry=True,output_identity_order_nonmask_fields_unchanged=True,
                G_uses_native_area_only=True,complete_same_image_outputs_and_GT=True,bbox_matching_identity_unchanged=True,
                segm_independently_rematched=True,old_G_fixed_pair_exact_replay=True,one_fresh_bound_panel=True,
                original_dev_pair_denominators=True,original_G_comparison=True,no_training_or_base_model_forward=True)))
        next_action=('Same checkpoint full5000 S3; requires next authorization' if status=='passed' else
            'Close AMT-v1; prioritize predictability of repair conditions with existing fixed labels; no retraining or threshold scans')
        if scope=='smoke':next_action='Independent smoke verification then authorized full S2'
        run.finish(summary,status=status,scientific='S2_development_'+status if scope=='full' else 'engineering_only',next_action=next_action)
        print('AMT_S2_DECISION',status,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--scope',choices=('smoke','full'));p.add_argument('--smoke-run',default='')
    a=p.parse_args()
    if a.prepare:prepare()
    elif a.scope:execute(a.scope,a.smoke_run)
    else:p.error('choose prepare or scope')
