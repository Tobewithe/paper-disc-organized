"""Exact full annotations/images, synthetic controls, then execution freeze."""
from lcf_runtime import *
import copy,zipfile,importlib.metadata

def main():
    require(socket.gethostname().upper()=='YANGLEI','wrong executor')
    require(not (STUDY/'FREEZE_MANIFEST.json').exists(),'already frozen')
    preparation=STUDY/'runs'/('RUN_LCF_PREPARE_'+uuid.uuid4().hex);preparation.mkdir(parents=True)
    meta={'run_id':preparation.name,'stage':'prepare','execution_status':'running','started_at':now(),'pid':os.getpid(),'optimizer_steps':0}
    write(preparation/'run.json',meta)
    try:
        config=read(STUDY/'CONFIG.json');parent=read(STUDY/'inputs/PARENT_DATA_CONTRACT.json')
        annotation=Path(config['train_annotations']);annotation.parent.mkdir(parents=True,exist_ok=True)
        if not annotation.exists():
            with zipfile.ZipFile(ROOT/'LCF_ANNOTATIONS.zip') as z:
                require(z.namelist()==['instances_train2017.json'],'unexpected annotation bundle');z.extractall(annotation.parent)
        require(sha(annotation)==parent['inputs']['train_annotations']['sha256'],'full annotation SHA differs')
        ann=read(annotation);require(len(ann['images'])==118287 and len(ann['annotations'])==860001,'not full official train annotations')
        del ann
        require(sha(config['val_annotations'])==parent['inputs']['val_annotations']['sha256'],'val annotation SHA')
        prior=ROOT/'experiments/mask_sampling_readout_20261020'
        inventory=read(prior/'inputs/REMOTE_INVENTORY.json');paths={'train':{},'val':{}};external=[]
        for split in ('train','val'):
            for i in parent[split+'_ids']:
                existing=inventory['existing_paths'][split].get(str(i))
                p=Path(existing) if existing else prior/'data/train_images'/f'{i:012d}.jpg'
                require(p.exists(),'missing '+str(p));expected=parent[split+'_image_hashes'][str(i)]
                require(sha(p)==expected,'image SHA '+str(i));paths[split][str(i)]=str(p)
                external.append({'path':str(p),'sha256':expected})
            print('IMAGES',split,len(paths[split]),flush=True)
        write(STUDY/'inputs/IMAGE_PATHS.json',paths)
        contract=copy.deepcopy(parent)
        from native_stream import historical_val_plan
        contract['val_batches']=historical_val_plan(read(config['historical_native_trace']))
        required=('weights','profile','existing_G','train_annotations','val_annotations','historical_native_trace')
        contract['inputs']={k:{'path':config[k],'sha256':sha(config[k])} for k in required}
        for k in required:external.append(contract['inputs'][k])
        write(STUDY/'inputs/DATA_CONTRACT.json',contract)
        shutil.copyfile(STUDY/'inputs/PARENT_IMAGE_ORDERS.json',STUDY/'inputs/IMAGE_ORDERS.json')
        for split in ('train','val'):
            (STUDY/'inputs'/(split+'.txt')).write_text(''.join(paths[split][str(i)]+'\n' for i in parent[split+'_ids']),encoding='utf-8')
        from test_components import run
        import torch
        torch.set_num_threads(1)
        tests=run(config['model'],config['loss']);require(tests['status']=='passed','controls failed')
        write(STUDY/'preflight/COMPONENT_TESTS.json',tests)
        profile,env=environment();write(preparation/'ENVIRONMENT.json',env)
        require(sha(config['weights'])==profile['model_sha256'],'weight differs')
        own=[STUDY/'PROTOCOL.md',STUDY/'CONFIG.json',*sorted((STUDY/'scripts').glob('*.py'))]
        own += [STUDY/'inputs'/p for p in ['DATA_CONTRACT.json','IMAGE_ORDERS.json','IMAGE_PATHS.json','train.txt','val.txt']]
        own += [ROOT/'experiments/mask_readout_mechanism_v2_20261012/scripts'/p for p in ['adapter_model.py','adapter_data.py','s1_geometry.py','posttrain_run.py']]
        own += [ROOT/'shared/evaluation'/p for p in ['native_output_bridge.py','native_source_pins.json','posttrain_audit.py']]
        own += [ROOT/'experiments/mask_repair_preservation_20261016/scripts/native_geometry.py',
                ROOT/'experiments/official_gt_failure_rebaseline_20261013/scripts/build_native_ledger.py',
                ROOT/'experiments/mask_gate_s3r_20261015/scripts/coco_bootstrap.py',
                ROOT/'experiments/mask_split_merge_topology_audit_20261020/scripts/topology.py',
                ROOT/'experiments/mask_readout_mechanism_v2_20261012/scripts/adapter_primitives.py']
        for key in ['existing_G_predictions','historical_native_ledger']:external.append({'path':config[key],'sha256':sha(config[key])})
        frozen={'frozen_at':now(),'protocol_sha256':sha(STUDY/'PROTOCOL.md'),'config_sha256':sha(STUDY/'CONFIG.json'),
                'code_commit':'f70c9a0a','files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in own],
                'input_files':external,'environment':env,'parameters':tests['parameters'],
                'source_full_training_annotations_SHA_verified':True,'formal_training_started':False,
                'solver_versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','pycocotools']}}
        write(STUDY/'FREEZE_MANIFEST.json',frozen)
        meta.update(execution_status='completed',finished_at=now(),exit_code=0,freeze_sha256=sha(STUDY/'FREEZE_MANIFEST.json'))
        study=read(STUDY/'study.json');study.update(plan_status='frozen',execution_status='not_started',current_phase='smoke_ready',latest_run_id=preparation.name);write(STUDY/'study.json',study)
        print('FROZEN',tests['parameters'],flush=True)
    except BaseException as e:
        (preparation/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
        meta.update(execution_status='failed',finished_at=now(),error=repr(e),exit_code=1);raise
    finally:write(preparation/'run.json',meta)

if __name__=='__main__':main()
