"""Registered, finite, zero-training audit on the authorized laptop."""
import os, sys, json, time, uuid, hashlib, datetime, socket, shutil, traceback
from pathlib import Path

STUDY=Path(__file__).resolve().parents[1];ROOT=STUDY.parents[1]
RAY=ROOT/'experiments/mask_adapter_ray_audit_20261016/runs/RUN_ADAPTER_RAY_FULL_362251e449e3493396e69c56562af96b'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    q.write_text(json.dumps(v,ensure_ascii=True,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)
def require(ok,msg):
    if not ok:raise AssertionError(msg)

def setup():
    require(socket.gethostname().upper()=='YANGLEI','wrong device')
    old=read(RAY/'EVAL_CONTRACT.json')['environment']['execution_env']
    for k,v in old.items():
        if v is None:os.environ.pop(k,None)
        else:os.environ[k]=v
    os.environ['OPENBLAS_NUM_THREADS']='1'
    sys.path.insert(0,str(ROOT/'_site'))
    for p in ['experiments/mask_readout_mechanism_v2_20261012/scripts','shared/evaluation',
              'experiments/mask_repair_preservation_20261016/scripts','experiments/mask_repair_margin_20261019/scripts']:
        sys.path.append(str(ROOT/p))

BASE=ROOT/'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170'
S1=ROOT/'experiments/mask_readout_mechanism_v2_20261012/runs/RUN_S1_ORACLE_f35a1399ac244f74aeebf94e4e843949'
CBASE=ROOT/'experiments/mask_repair_preservation_20261016/runs'
CFULL=CBASE/'RUN_REPAIR_KEEP_FULL_68b1d8211b30456abb172db046963e01'
CSMOKE=CBASE/'RUN_REPAIR_KEEP_SMOKE_1473cf76889b4163b962d2e7086995b7'

def check_freeze():
    frozen=read(STUDY/'FREEZE_MANIFEST.json')
    for p,digest in frozen['files'].items():require(sha(ROOT/p)==digest,'frozen input/source changed: '+p)
    return frozen

def resources(start,limit):
    import psutil,torch
    require(time.monotonic()-start<limit,'whole-stage time budget')
    require(psutil.virtual_memory().available>=1<<30,'available RAM reserve')
    require(psutil.Process().memory_info().rss<=6<<30,'RSS budget')
    require(shutil.disk_usage(ROOT).free>=12<<30,'disk reserve')
    require(torch.cuda.memory_allocated()<=3<<30,'CUDA budget')

def prepare():
    import importlib.metadata as md
    from posttrain_run import configure_native
    from test_controls import test
    require(not (STUDY/'FREEZE_MANIFEST.json').exists(),'already frozen')
    test_result=test();require(test_result['status']=='passed','math controls')
    profile,environment=configure_native()
    require(environment==read(RAY/'EVAL_CONTRACT.json')['environment'],'native environment differs')
    full=lines(CFULL/'PAIR_RESULTS.jsonl');smoke=lines(CSMOKE/'PAIR_RESULTS.jsonl')
    panel=read(S1/'TARGET_PANEL.json')['targets']
    require(len(full)==256 and len(smoke)==4,'panel counts')
    require({(r['output_detection_id'],r['annotation_id']) for r in full}=={(r['output_detection_id'],r['annotation_id']) for r in panel},'panel identity')
    require(not {r['image_id'] for r in full}&{r['image_id'] for r in smoke},'smoke overlaps')
    require(sum(r['arms']['C']['original_iou']>=.75 for r in full)==159,'C witnesses')
    ids={r['image_id'] for r in full+smoke};trace=read(BASE/'trace/TRACE_MANIFEST.json')
    files=[STUDY/'PROTOCOL.md',STUDY/'study.initial.json',S1/'TARGET_PANEL.json',S1/'ORACLE_RESULTS.jsonl',
           CFULL/'PAIR_RESULTS.jsonl',CSMOKE/'PAIR_RESULTS.jsonl',RAY/'EVAL_CONTRACT.json',BASE/'trace/TRACE_MANIFEST.json',
           Path(trace['predictions_path']),ROOT/'paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json',
           ROOT/'assets/models/coco_clean_20260911/yolo26m-seg.pt']
    files+=sorted((STUDY/'scripts').glob('*.py'))
    deps=['experiments/mask_readout_mechanism_v2_20261012/scripts/'+n for n in ['posttrain_run.py','adapter_data.py','s1_geometry.py']]
    deps+=['shared/evaluation/'+n for n in ['native_output_bridge.py','native_source_pins.json','posttrain_audit.py']]
    deps+=['experiments/mask_repair_preservation_20261016/scripts/native_geometry.py','experiments/mask_repair_margin_20261019/scripts/native_ops.py']
    files += [ROOT/p for p in deps]
    records=[r for r in trace['images'] if r['image_id'] in ids]
    require(len(records)==len(ids),'cache images')
    for r in records:
        cache=BASE/'trace'/r['cache'];require(sha(cache)==r['cache_sha256'],'cache hash');files.append(cache)
    require(sha(ROOT/'assets/models/coco_clean_20260911/yolo26m-seg.pt')==profile['model_sha256'],'weight')
    frozen={'frozen_at':now(),'authoring_commit':'f70c9a0a','files':{p.relative_to(ROOT).as_posix():sha(p) for p in files},
            'environment':environment,'profile':profile,'native_args':trace['native_args'],
            'solver_versions':{k:md.version(k) for k in ['scipy','numpy','sympy','psutil']},
            'main_pairs':256,'main_images':243,'source_C_witnesses':159,'training_steps':0,'test_controls':test_result}
    write(STUDY/'FREEZE_MANIFEST.json',frozen)
    study=read(STUDY/'study.json');study.update(plan_status='frozen',execution_status='not_started',artifact_status='frozen',
             frozen_at=frozen['frozen_at'],protocol_sha256=sha(STUDY/'PROTOCOL.md'))
    write(STUDY/'study.json',study);print('FROZEN',len(files),flush=True)

def summary(rows):
    groups={'all':rows,'S1_not_found':[r for r in rows if r.get('S1_iou') is not None and r['S1_iou']<.75],
            'C_not_found':[r for r in rows if r['C_iou']<.75],'C_found':[r for r in rows if r['C_iou']>=.75]}
    out={}
    from collections import Counter
    for name,group in groups.items():
        valid=[r for r in group if 'GT' in r]
        out[name]={'pairs':len(group),'processed_scientific_rows':len(valid),
                   'GT_linear_status':dict(Counter(r['GT']['linear_status'] for r in valid)),
                   'GT_margin_status':dict(Counter(r['GT']['margin_status'] for r in valid)),
                   'C_mask_margin_status':dict(Counter(r['C_pattern']['margin_status'] for r in valid)),
                   'new_GT_LP_Mask75_witnesses':sum(r.get('GT_native',{}).get('iou',0)>=.75 for r in valid),
                   'engineering_unresolved':len(group)-len(valid)}
    return {'groups':out,'training_steps':0,'task_level_IoU75_unattainability_certified':False,
            'scope':'exact projected GT labels and existing C first-threshold pattern, not best-IoU bound',
            'scientific_status':'COMPLETED_CERTIFICATE_AUDIT_PENDING_RESEARCH_ADJUDICATION'}

def execute(scope):
    import numpy as np,torch
    from adapter_data import FrozenCache
    from posttrain_run import configure_native
    from native_geometry import lifted_basis,roi_for
    from native_output_bridge import decode_native,encode_rle
    from native_ops import Decoder,measure
    from pycocotools import mask as masks
    from linear_audit import audit
    from certificates import exact_row,verify
    frozen=check_freeze();profile,environment=configure_native()
    require(environment==frozen['environment'],'environment drift')
    if scope=='full':
        previous=read(STUDY/'SMOKE_POINTER.json')
        require(read(STUDY/'runs'/previous['run_id']/'run.json')['execution_status']=='completed','smoke required')
        require(not list((STUDY/'runs').glob('RUN_PLF_FULL_*')),'full already attempted; inspect original')
    run=STUDY/'runs'/('RUN_PLF_'+scope.upper()+'_'+uuid.uuid4().hex);run.mkdir(parents=True)
    meta={'study_id':'STUDY_PROTOTYPE_LINEAR_FEASIBILITY_20261020','run_id':run.name,'scope':scope,
          'execution_status':'running','started_at':now(),'pid':os.getpid(),'host':socket.gethostname(),
          'training_steps':0,'protocol_sha256':sha(STUDY/'PROTOCOL.md'),'freeze_sha256':sha(STUDY/'FREEZE_MANIFEST.json'),
          'command':sys.argv,'authoring_commit':'f70c9a0a','processed_pairs':0}
    start=time.monotonic();output=[];all_replays=0
    write(run/'run.json',meta);write(STUDY/'EXECUTION_STATUS.json',meta)
    shutil.copyfile(STUDY/'FREEZE_MANIFEST.json',run/'EVAL_CONTRACT.json')
    shutil.copyfile(STUDY/'PROTOCOL.md',run/'PROTOCOL.md')
    for p in (STUDY/'scripts').glob('*.py'):
        target=run/'source_snapshot'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target)
    study=read(STUDY/'study.json');study.update(execution_status='running',active_run_id=run.name,current_phase=scope);write(STUDY/'study.json',study)
    try:
        controls=sorted(lines((CSMOKE if scope=='smoke' else CFULL)/'PAIR_RESULTS.jsonl'),key=lambda r:(r['image_id'],r['annotation_id']))
        source=FrozenCache(BASE/'trace',sorted({r['image_id'] for r in controls}))
        s1={(r['output_detection_id'],r['annotation_id']):r for r in lines(S1/'ORACLE_RESULTS.jsonl')}
        with (run/'PAIR_RESULTS.jsonl').open('x',encoding='utf-8') as stream:
            for image_id in source.image_ids:
                resources(start,1800 if scope=='smoke' else 43200)
                cache,native_rows=source.load(image_id);proto=cache['proto'].cpu().numpy()
                original=decode_native(cache).cpu()
                for i,p in enumerate(native_rows):require(encode_rle(original[i])==p['segmentation'],'zero-delta native RLE differs')
                all_replays+=len(native_rows);del original
                basis=lifted_basis(cache).numpy()
                # Conservative absolute bound on 4-term FP64 dyadic interpolation.
                basis_error=float(np.max(np.abs(proto)))*np.finfo(np.float64).eps*16
                for old in [r for r in controls if r['image_id']==image_id]:
                    resources(start,1800 if scope=='smoke' else 43200)
                    rank=old['rank'];require(cache['prediction_indices'][rank]==old['global_index'],'output row changed')
                    require(old['native_prediction']==native_rows[rank] and source.records[image_id]['cache_sha256']==old['cache_sha256'],'cache identity changed')
                    decoder=Decoder(cache,rank);cr,first,_=decoder.evaluate(old['arms']['C']['coefficients_fp32'],True)
                    require(cr==old['arms']['C']['original_rle'],'source C original replay')
                    require(encode_rle(torch.from_numpy(first).byte())==old['arms']['C']['input_rle'],'source C first threshold replay')
                    roi=roi_for(cache,rank).numpy();pixels=np.argwhere(roi);B=basis[:,roi].T.copy()
                    # Independent exact row checks bind the floating search map to P.
                    for ix in np.unique(np.linspace(0,len(B)-1,min(len(B),5),dtype=int)):
                        row=np.array([float(v) for v in exact_row(proto,*pixels[ix])])
                        require(np.all(np.abs(row-B[ix])<=basis_error),'bilinear operator mismatch')
                    key=(old['output_detection_id'],old['annotation_id'])
                    row={k:old[k] for k in ['image_id','annotation_id','output_detection_id','rank','global_index','raw_id','cache_sha256']}
                    row.update(N_iou=old['arms']['N']['original_iou'],C_iou=old['arms']['C']['original_iou'],
                               G_old_iou=old['arms']['R_G']['original_iou'],S1_iou=s1[key]['oracle_mask_iou'] if key in s1 else None,
                               roi_pixels=len(B),basis_error_bound=basis_error,GT_rle=old['GT_rle'])
                    try:
                        target=masks.decode(old['target_grid_rle']).astype(bool)
                        row['GT_pixels_outside_fixed_box']=int(np.count_nonzero(target&~roi))
                        for name,labels in [('GT',target[roi]),('C_pattern',first[roi])]:
                            result=audit(B,labels,pixels,proto,basis_error)
                            for cert in [result['farkas'],result['margin_certificate']]:
                                if cert:verify(proto,cert)
                            if result['coefficients_float64'] is not None:
                                coeff=np.asarray(result['coefficients_float64'],dtype=np.float32)
                                rle,actual_first,_=decoder.evaluate(coeff,True)
                                row[name+'_native']={**measure(rle,old['GT_rle']),
                                    'input_label_mismatches':int(np.count_nonzero(actual_first[roi]!=labels)),
                                    'coefficients_fp32':coeff.tolist()}
                                if result['linear_status'].startswith('VERIFIED_') and row[name+'_native']['input_label_mismatches']:
                                    result['native_status']='REAL_NATIVE_GAP'
                                else:result['native_status']='REPLAYED'
                            row[name]=result
                    except Exception as e:
                        row['engineering_error']=repr(e);row['traceback']=traceback.format_exc()
                        if scope=='smoke':raise
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();output.append(row)
                    meta.update(processed_pairs=len(output),updated_at=now(),elapsed_seconds=time.monotonic()-start)
                    write(run/'run.json',meta);write(STUDY/'EXECUTION_STATUS.json',meta)
                    print('PAIR',len(output),row['image_id'],row.get('GT',{}).get('linear_status','UNRESOLVED'),flush=True)
                    del decoder,B
                del basis,cache;torch.cuda.empty_cache()
        check_freeze();require(configure_native()[1]==environment,'postrun environment drift')
        write(run/'SUMMARY.json',summary(output))
        write(run/'CONFIG_AUDIT.json',{'status':'passed','sources_inputs_unchanged':True,'environment_equal':True,
            'native_zero_delta_detection_rows':all_replays,'source_C_replay_pairs':len(output),'training_steps':0,
            'solver_status_not_used_as_certificate':True,'G_budget':'unavailable_not_substituted'})
        write(run/'INDEPENDENT_CERTIFICATES.json',verify_saved(run))
        meta.update(execution_status='completed',finished_at=now(),exit_code=0)
        if scope=='smoke':write(STUDY/'SMOKE_POINTER.json',{'run_id':run.name})
    except BaseException as e:
        (run/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
        meta.update(execution_status='failed',error=repr(e),finished_at=now(),exit_code=1);raise
    finally:
        write(run/'run.json',meta);write(STUDY/'EXECUTION_STATUS.json',meta)
        study=read(STUDY/'study.json');study.update(execution_status=meta['execution_status'],latest_run_id=run.name,
             active_run_id=None,scientific_status='pending_adjudication' if scope=='full' and meta['execution_status']=='completed' else 'hypothesis_only',
             current_phase='awaiting_full' if scope=='smoke' and meta['execution_status']=='completed' else 'awaiting_report' if meta['execution_status']=='completed' else 'error_retained')
        write(STUDY/'study.json',study)

def verify_saved(run):
    # No LP/SOCP code called: reconstruct support rows from immutable native P.
    import torch
    from certificates import verify
    manifest=read(BASE/'trace/TRACE_MANIFEST.json');records={r['image_id']:r for r in manifest['images']}
    total=0;last=None;proto=None
    for row in lines(run/'PAIR_RESULTS.jsonl'):
        if last!=row['image_id']:
            record=records[row['image_id']];p=BASE/'trace'/record['cache']
            require(sha(p)==row['cache_sha256'],'certificate source cache differs')
            proto=torch.load(p,map_location='cpu',weights_only=True)['proto'].numpy();last=row['image_id']
        for name in ['GT','C_pattern']:
            if name not in row:continue
            for key in ['farkas','margin_certificate']:
                c=row[name].get(key)
                if c:verify(proto,c);total+=1
    return {'status':'passed','certificates_exactly_checked':total,'optimizer_calls':0,'verifier':'certificates.py from raw P and rational 4x geometry'}

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','smoke','full','pipeline','verify']);p.add_argument('--run');args=p.parse_args()
    setup()
    if args.stage=='prepare':prepare()
    elif args.stage=='verify':print(json.dumps(verify_saved(STUDY/'runs'/args.run)))
    elif args.stage=='pipeline':
        lock=STUDY/'.PLF.lock'
        with lock.open('x') as f:f.write(str(os.getpid()))
        try:
            with (STUDY/'pipeline.log').open('x',encoding='utf-8',buffering=1) as log:
                sys.stdout=sys.stderr=log;execute('smoke');execute('full')
        finally:lock.unlink(missing_ok=True)
    else:execute(args.stage)

if __name__=='__main__':main()
