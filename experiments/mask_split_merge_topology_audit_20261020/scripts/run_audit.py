"""Finite CPU audit of existing frozen masks; no new model execution."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json,sys,time,uuid,hashlib,datetime,socket,shutil,traceback
from pathlib import Path
STUDY=Path(__file__).resolve().parents[1];ROOT=STUDY.parents[1]
sys.path.insert(0,str(ROOT/'_site'))
import numpy as np
from pycocotools import mask as masks
from topology import Context,signature,tests

C=ROOT/'experiments/mask_repair_preservation_20261016/runs/RUN_REPAIR_KEEP_FULL_68b1d8211b30456abb172db046963e01'
S1=ROOT/'experiments/mask_readout_mechanism_v2_20261012/runs/RUN_S1_ORACLE_f35a1399ac244f74aeebf94e4e843949'
RSM=ROOT/'experiments/mask_repair_margin_completion_20261020/runs/RUN_RSMC_FULL_c130c3b5270744fabdb50df0b0326c95'
GT=ROOT/'assets/datasets/coco_native_s0_20261013/annotations/instances_val2017.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(v,ensure_ascii=True,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def require(ok,msg):
    if not ok:raise AssertionError(msg)
def decode(rle):return masks.decode(rle).astype(bool)
def identity(r):return (r['image_id'],r['annotation_id'],r['output_detection_id'])

def sources():
    return [STUDY/'PROTOCOL.md',*sorted((STUDY/'scripts').glob('*.py')),C/'PAIR_RESULTS.jsonl',C/'run.json',
            S1/'ORACLE_RESULTS.jsonl',S1/'TARGET_PANEL.json',S1/'run.json',RSM/'PAIR_RESULTS.jsonl',
            RSM/'run.json',RSM/'CONFIG_AUDIT.json',GT]
def freeze():
    import importlib.metadata as md
    require(socket.gethostname().upper()=='YANGLEI','wrong executor')
    require(not (STUDY/'FREEZE_MANIFEST.json').exists(),'already frozen')
    result=tests();require(result['status']=='passed','synthetic controls')
    for p in [C,S1,RSM]:require(read(p/'run.json')['execution_status']=='completed','source incomplete')
    manifest={'frozen_at':now(),'files':{p.relative_to(ROOT).as_posix():sha(p) for p in sources()},
              'test_controls':result,'packages':{p:md.version(p) for p in ['numpy','scipy','pycocotools','psutil']},
              'native_masks':'existing original 8.4.100 FP32 native RLE; no new inference/decoding',
              'whole_panel':256,'pressure_records_expected':6144,'training_steps':0}
    write(STUDY/'FREEZE_MANIFEST.json',manifest)
    s=read(STUDY/'study.json');s.update(plan_status='frozen',execution_status='not_started');write(STUDY/'study.json',s)
    print('FROZEN',sha(STUDY/'FREEZE_MANIFEST.json'),flush=True)
def check():
    f=read(STUDY/'FREEZE_MANIFEST.json')
    for p,d in f['files'].items():require(sha(ROOT/p)==d,'frozen drift: '+p)
    return f

def aggregate(rows,pressures):
    names=['N','S1','C','G_old']
    endpoint={}
    fields=['iou','split_mass','bridge_mass','split_event','bridge_event','material_split','material_bridge',
            'coverage','missed_gt_components','within_instance_false_joins','unmatched_prediction_islands']
    ids=sorted({r['image_id'] for r in rows});rng=np.random.default_rng(1729)
    indexes=rng.integers(0,len(ids),size=(2000,len(ids)))
    def image_values(items,func):
        return np.array([np.mean([func(r) for r in items if r['image_id']==i]) for i in ids])
    def stats(a):
        good=np.isfinite(a)
        draws=np.nanmean(a[indexes],axis=1)
        return {'image_macro_mean':float(np.nanmean(a)), 'ci95':np.quantile(draws[np.isfinite(draws)],[.025,.975]).tolist(),
                'images':int(good.sum())}
    for name in names:
        endpoint[name]={field:stats(image_values(rows,lambda r:r['endpoints'][name]['8'][field])) for field in fields}
        endpoint[name]['counts']={k:sum(bool(r['endpoints'][name]['8'][k]) for r in rows) for k in ['split_event','bridge_event','material_split','material_bridge','bridge_eligible']}
        endpoint[name]['material_split_or_bridge']=sum(r['endpoints'][name]['8']['material_split'] or r['endpoints'][name]['8']['material_bridge'] for r in rows)
    comparisons={}
    for ref in ['N','G_old']:
        comparisons['C_minus_'+ref]={field:stats(image_values(rows,lambda r:r['endpoints']['C']['8'][field]-r['endpoints'][ref]['8'][field])) for field in ['iou','split_mass','bridge_mass']}
    groups={}
    for name,sub in [('all',rows),('S1_not_found',[r for r in rows if r['S1_iou']<.75]),
                     ('C_not_found',[r for r in rows if r['C_iou']<.75]),('C_found',[r for r in rows if r['C_iou']>=.75])]:
        groups[name]={'pairs':len(sub),'GT_multipart_8':sum(r['endpoints']['C']['8']['gt_components']>1 for r in sub)}
        for arm in names:
            groups[name][arm]={k:sum(bool(r['endpoints'][arm]['8'][k]) for r in sub) for k in ['split_event','bridge_event','material_split','material_bridge']}
            groups[name][arm]['material_split_or_bridge']=sum(r['endpoints'][arm]['8']['material_split'] or r['endpoints'][arm]['8']['material_bridge'] for r in sub)
    by_pair={}
    for p in pressures:by_pair.setdefault(identity(p),[]).append(p)
    pp=[]
    for r in rows:
        p=by_pair[identity(r)];valid=[v for v in p if v['precision_valid']]
        small=[v for v in valid if v['changed_fraction']<=.01]
        pp.append({'image_id':r['image_id'],'valid':len(valid),
                   'topology_flip_rate_valid':np.mean([v['topology_signature_changed'] for v in valid]) if valid else float('nan'),
                   'split_or_bridge_flip_rate_valid':np.mean([v['split_or_bridge_changed'] for v in valid]) if valid else float('nan'),
                   'material_flip_rate_valid':np.mean([v['material_split_or_bridge_changed'] for v in valid]) if valid else float('nan'),
                   'IoU_delta_valid':np.mean([v['iou_delta'] for v in valid]) if valid else float('nan'),
                   'small_valid_directions':len(small),'small_topology_flips':sum(v['topology_signature_changed'] for v in small)})
    pressure_summary={'total':len(pressures),'valid':sum(v['precision_valid'] for v in pressures),
          'invalid':sum(not v['precision_valid'] for v in pressures),
          'small_valid_directions':sum(v['precision_valid'] and v['changed_fraction']<=.01 for v in pressures),
          'small_valid_topology_flips':sum(v['precision_valid'] and v['changed_fraction']<=.01 and v['topology_signature_changed'] for v in pressures),
          'small_valid_split_or_bridge_flips':sum(v['precision_valid'] and v['changed_fraction']<=.01 and v['split_or_bridge_changed'] for v in pressures),
          'small_valid_material_flips':sum(v['precision_valid'] and v['changed_fraction']<=.01 and v['material_split_or_bridge_changed'] for v in pressures)}
    for key in ['topology_flip_rate_valid','split_or_bridge_flip_rate_valid','material_flip_rate_valid','IoU_delta_valid']:
        pressure_summary[key]=stats(image_values(pp,lambda r:r[key]))
    return {'pairs':len(rows),'images':len(ids),'endpoint':endpoint,'comparisons':comparisons,'groups':groups,
        'pressure':pressure_summary,'training_steps':0,'new_model_forward_calls':0,'new_coefficient_solves':0,
        'scientific_scope':'descriptive_topology_and_stability_audit_not_global_coupling_causal_proof',
        'globally_optimal_fixed_P_topology_proven':False}

def run():
    import psutil
    from pycocotools.coco import COCO
    frozen=check();require(not list((STUDY/'runs').glob('RUN_STA_FULL_*')),'already attempted')
    path=STUDY/'runs'/('RUN_STA_FULL_'+uuid.uuid4().hex);path.mkdir(parents=True)
    meta={'run_id':path.name,'study_id':read(STUDY/'study.json')['study_id'],'execution_status':'running',
          'started_at':now(),'pid':os.getpid(),'host':socket.gethostname(),'training_steps':0,
          'freeze_sha256':sha(STUDY/'FREEZE_MANIFEST.json'),'protocol_sha256':sha(STUDY/'PROTOCOL.md'),
          'command':sys.argv,'processed_pairs':0,'processed_pressures':0}
    write(path/'run.json',meta);write(STUDY/'EXECUTION_STATUS.json',meta)
    shutil.copyfile(STUDY/'FREEZE_MANIFEST.json',path/'EVAL_CONTRACT.json');shutil.copyfile(STUDY/'PROTOCOL.md',path/'PROTOCOL.md')
    for f in (STUDY/'scripts').glob('*.py'):
        target=path/'source_snapshot'/f.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(f,target)
    s=read(STUDY/'study.json');s.update(execution_status='running',active_run_id=path.name);write(STUDY/'study.json',s)
    start=time.monotonic();pairs=[];pressure_rows=[];iou_checks=0
    try:
        old=sorted(lines(C/'PAIR_RESULTS.jsonl'),key=lambda r:(r['image_id'],r['annotation_id']))
        s1={identity(r):r for r in lines(S1/'ORACLE_RESULTS.jsonl')};rsm={identity(r):r for r in lines(RSM/'PAIR_RESULTS.jsonl')}
        require(len(old)==len(s1)==len(rsm)==256,'panel sizes')
        require({identity(r) for r in old}==set(s1)==set(rsm),'panel identity')
        coco=COCO(str(GT));cache_image=None;neighbors=None
        with (path/'PAIR_RESULTS.jsonl').open('x') as out,(path/'PRESSURE_RESULTS.jsonl').open('x') as pout:
            for row in old:
                require(time.monotonic()-start<7200,'time budget');require(psutil.Process().memory_info().rss<3<<30,'RSS budget')
                require(shutil.disk_usage(ROOT).free>=12<<30,'disk reserve')
                key=identity(row);a=s1[key];r=rsm[key]
                require(row['GT_rle']==r['GT_rle'],'GT source mismatch')
                require(row['arms']['C']['original_rle']==r['endpoints']['C']['rle'],'C source mismatch')
                gt=decode(row['GT_rle'])
                ann=coco.anns[row['annotation_id']];require(np.array_equal(coco.annToMask(ann).astype(bool),gt),'COCO target differs')
                if cache_image!=row['image_id']:
                    anns=coco.loadAnns(coco.getAnnIds(imgIds=[row['image_id']]))
                    neighbors=[(t['id'],coco.annToMask(t).astype(bool)) for t in anns if not t.get('iscrowd',0) and not t.get('ignore',0)]
                    cache_image=row['image_id']
                own_neighbors=[(i,m) for i,m in neighbors if i!=row['annotation_id']]
                context={str(k):Context(gt,own_neighbors,k) for k in [8,4]}
                masks_by_name={'N':row['arms']['N']['original_rle'],'C':row['arms']['C']['original_rle'],
                               'G_old':row['arms']['R_G']['original_rle'],'S1':a['oracle_rle']}
                expected={'N':row['arms']['N']['original_iou'],'C':row['arms']['C']['original_iou'],
                          'G_old':row['arms']['R_G']['original_iou'],'S1':a['oracle_mask_iou']}
                record={k:row[k] for k in ['image_id','annotation_id','output_detection_id','rank','global_index','raw_id','cache_sha256']}
                record.update(S1_iou=expected['S1'],C_iou=expected['C'],crowd_annotations=sum(bool(t.get('iscrowd',0)) for t in anns),endpoints={})
                binary={}
                for name,rle in masks_by_name.items():
                    binary[name]=decode(rle);record['endpoints'][name]={k:c.measure(binary[name]) for k,c in context.items()}
                    require(abs(record['endpoints'][name]['8']['iou']-expected[name])<=1e-12,'source IoU mismatch');iou_checks+=1
                target_input=decode(row['target_grid_rle']);input_context=Context(target_input,connectivity=8)
                record['input_stage']={name:input_context.measure(decode(row['arms'][name]['input_rle'])) for name in ['N','C']}
                base=record['endpoints']['C']['8'];base_mask=binary['C']
                for pi,p in enumerate(r['pressure']):
                    changed=decode(p['C']['rle']);measurement=context['8'].measure(changed)
                    require(abs(measurement['iou']-p['C']['iou'])<=1e-12,'pressure IoU mismatch');iou_checks+=1
                    different=changed!=base_mask;union=int((changed|base_mask).sum())
                    valid=all(abs(p[k]/r['pressure_amplitude']-1)<=5e-5 for k in ['C_actual_rms','C_decoded_difference_rms'])
                    data={k:record[k] for k in ['image_id','annotation_id','output_detection_id']}
                    data.update(direction=pi,arm=p['arm'],precision_valid=valid,metrics=measurement,
                         iou_delta=measurement['iou']-base['iou'],changed_fraction=int(different.sum())/union if union else 0.,
                         old_errors_fixed=int((different&(changed==gt)).sum()),old_correct_damaged=int((different&(changed!=gt)).sum()),
                         topology_signature_changed=signature(base)!=signature(measurement),
                         split_or_bridge_changed=(base['split_event'],tuple(base['bridge_neighbor_ids']))!=(measurement['split_event'],tuple(measurement['bridge_neighbor_ids'])),
                         material_split_or_bridge_changed=(base['material_split'],base['material_bridge'])!=(measurement['material_split'],measurement['material_bridge']))
                    pout.write(json.dumps(data,allow_nan=False)+'\n');pressure_rows.append(data)
                out.write(json.dumps(record,allow_nan=False)+'\n');out.flush();pout.flush();pairs.append(record)
                meta.update(processed_pairs=len(pairs),processed_pressures=len(pressure_rows),elapsed_seconds=time.monotonic()-start,updated_at=now())
                write(path/'run.json',meta);write(STUDY/'EXECUTION_STATUS.json',meta);print('PAIR',len(pairs),flush=True)
        require(len(pressure_rows)==6144,'pressure count');check()
        write(path/'SUMMARY.json',aggregate(pairs,pressure_rows))
        write(path/'CONFIG_AUDIT.json',{'status':'passed','frozen_sources_and_inputs_unchanged':True,'source_IoU_checks':iou_checks,
            'all_panel_identities_equal':True,'C_RSM_endpoints_exact':True,'native_masks_regenerated':False,
            'training_steps':0,'new_model_forward_calls':0,'new_coefficient_solves':0,'packages':frozen['packages'],
            'connectivity_main':8,'connectivity_endpoint_sensitivity':4,'minimum_component_area':None,'morphology':None})
        meta.update(execution_status='completed',finished_at=now(),exit_code=0)
    except BaseException as e:
        (path/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8');meta.update(execution_status='failed',error=repr(e),finished_at=now(),exit_code=1);raise
    finally:
        write(path/'run.json',meta);write(STUDY/'EXECUTION_STATUS.json',meta)
        s=read(STUDY/'study.json');s.update(execution_status=meta['execution_status'],active_run_id=None,latest_run_id=path.name,
            scientific_status='descriptive_audit_pending_adjudication' if meta['execution_status']=='completed' else 'error_retained',artifact_status='complete' if meta['execution_status']=='completed' else 'partial')
        write(STUDY/'study.json',s)

if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='run':run()
    elif sys.argv[1]=='tests':print(json.dumps(tests()))
