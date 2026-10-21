"""CPU-only isolated execution port; frozen scientific modules remain byte-identical."""
import copy, hashlib, json, os, shutil, sys
from pathlib import Path

ROOT = Path('/srv/gpu-lab/users/a25023104/projects/lcf-server-20261020')
OLD = Path('/srv/gpu-lab/users/a25023104/projects/paper-disc-organized')
PARENT = ROOT/'experiments/mask_local_connectivity_factorial_20261020'
STUDY = ROOT/'experiments/mask_local_connectivity_server_20261020'
DATA = Path('/srv/gpu-lab/users/a25023104/datasets/coco2017')

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p, v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def require(ok,msg):
    if not ok: raise RuntimeError(msg)

def main():
    STUDY.mkdir(exist_ok=True); scripts=STUDY/'scripts';scripts.mkdir(exist_ok=True)
    vendor=ROOT/'_site'; vendor.mkdir(exist_ok=True)
    site=OLD/'.venv/lib/python3.12/site-packages'
    for p in site.glob('ultralytics*'):
        link=vendor/p.name
        if not link.exists(): link.symlink_to(p,target_is_directory=p.is_dir())
    source=read(PARENT/'FREEZE_MANIFEST.json')
    scientific=['lcf_model.py','lcf_loss.py','lcf_targets.py','connectivity_loss.py','native_stream.py','test_components.py','lcf_evaluation.py','lcf_statistics.py']
    science_receipts={}
    for name in scientific:
        p=PARENT/'scripts'/name
        expected=next(r['sha256'] for r in source['files'] if r['path']==p.relative_to(ROOT).as_posix())
        require(sha(p)==expected,'parent source drift '+name)
        shutil.copyfile(p,scripts/name);science_receipts[name]=expected
    profile=read(OLD/'paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json')
    laptop=read(STUDY/'LAPTOP_PROFILE.json') if (STUDY/'LAPTOP_PROFILE.json').exists() else read(ROOT/'paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json')
    require(profile['native_args']==laptop['native_args'],'native args changed')
    require(profile['runtime_policy']==laptop['runtime_policy'],'runtime policy changed')
    write(STUDY/'LAPTOP_PROFILE.json',laptop)
    profile['packages']['torch']='2.5.1+cu124'
    profile['packages']['torchvision']='0.20.1+cu124'
    write(ROOT/'paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json',profile)
    baseline=OLD/'experiments/mask_branch_joint_replay_20261018/runs/RUN_JOINT_NATIVE_299731c6c64d4cc0acf3b0915401e1e6'
    baseline_summary=read(baseline/'SUMMARY.json')
    require(read(baseline/'run.json')['execution_status']=='completed','old server native incomplete')
    require(abs(baseline_summary['Ns']['metrics']['segm']['AP']-44.01417484774351)<=.02,'native AP migration')
    require(abs((baseline_summary['G_old_s']['metrics']['segm']['AP']-baseline_summary['Ns']['metrics']['segm']['AP'])-.14157289505649)<=.02,'G gain migration')
    for method in ['Ns','G_old_s']:
        require(sha(baseline/(method+'_predictions.json'))==baseline_summary[method]['predictions_sha256'],'baseline predictions SHA')
    cfg=read(PARENT/'CONFIG.json'); original=copy.deepcopy(cfg)
    for key,val in list(cfg.items()):
        if isinstance(val,str) and val.startswith('D:/coco_wire/baseline_native_20261013/'):
            cfg[key]=str(OLD/val.split('baseline_native_20261013/',1)[1])
    cfg.update(study_id='STUDY_MASK_LOCAL_CONNECTIVITY_SERVER_20261020',study_root=str(STUDY),
               profile=str(ROOT/'paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json'),
               train_annotations=str(DATA/'annotations/instances_train2017.json'),val_annotations=str(DATA/'annotations/instances_val2017.json'),
               existing_G_predictions=str(baseline/'G_old_s_predictions.json'),historical_native_ledger=str(baseline/'Ns_PER_GT.jsonl'))
    trace=read(OLD/'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170/trace/TRACE_MANIFEST.json')
    cfg['historical_native_trace']=str(STUDY/'inputs/SERVER_REFERENCE_TRACE.json')
    trace['predictions_path']=str(baseline/'Ns_predictions.json')
    trace['server_reference_run']=str(baseline)
    write(STUDY/'inputs/SERVER_REFERENCE_TRACE.json',trace)
    cfg['authorization']={'user':'migrate four arms to server; freeze; smoke; Slurm; up to four GPUs','no_schedules':True,'no_goal':True,'no_auto_commit_or_push':True}
    # Only paths, execution identity and authorization metadata differ.
    science_keys=['arms','training_seed','fit_images','epochs','source_batch','effective_batch','steps_per_arm','optimizer','hours','area_gate','runtime_policy','policy','bootstrap','gates','model','loss','topology_gates','protocol_version']
    require(all(cfg[k]==original[k] for k in science_keys),'scientific configuration drift')
    write(STUDY/'CONFIG.json',cfg)
    parent=read(PARENT/'inputs/PARENT_DATA_CONTRACT.json'); contract=copy.deepcopy(parent)
    sys.path.insert(0,str(scripts))
    from native_stream import historical_val_plan
    contract['val_batches']=historical_val_plan(trace)
    contract['inputs']={k:{'path':cfg[k],'sha256':sha(cfg[k])} for k in ['weights','profile','existing_G','train_annotations','val_annotations','historical_native_trace']}
    require(contract['inputs']['train_annotations']['sha256']=='610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d','train annotation')
    require(contract['inputs']['val_annotations']['sha256']=='e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f','val annotation')
    external=[]
    for split in ['train','val']:
        paths=[DATA/(split+'2017')/f'{i:012d}.jpg' for i in contract[split+'_ids']]
        for i,p in zip(contract[split+'_ids'],paths):
            require(sha(p)==contract[split+'_image_hashes'][str(i)],f'image SHA {p}')
            external.append({'path':str(p),'sha256':contract[split+'_image_hashes'][str(i)]})
        (STUDY/'inputs'/(split+'.txt')).write_text(''.join(str(p)+'\n' for p in paths),encoding='utf-8')
        print('HASHED',split,len(paths),flush=True)
    write(STUDY/'inputs/DATA_CONTRACT.json',contract)
    shutil.copyfile(PARENT/'inputs/PARENT_IMAGE_ORDERS.json',STUDY/'inputs/IMAGE_ORDERS.json')
    external+=list(contract['inputs'].values())
    external+=[{'path':str(baseline/n),'sha256':sha(baseline/n)} for n in ['Ns_predictions.json','G_old_s_predictions.json','Ns_PER_GT.jsonl','SUMMARY.json','ENVIRONMENT.json']]
    write(STUDY/'SCIENTIFIC_CONFIGURATION_COMPARISON.json',{'status':'passed','equal_fields':science_keys,'byte_identical_scientific_sources':science_receipts,'native_args_exact':True,'runtime_policy_exact':True,'scientific_configuration_unchanged':True,'execution_differences':['Linux/A6000','private scipy overlay','independent per-arm process','44GiB device resource guard','server native identity population'],'old_server_baseline':baseline_summary})
    # The execution runtime is separate; no original frozen file is edited.
    text=(PARENT/'scripts/lcf_runtime.py').read_text(encoding='utf-8')
    text=text.replace('STUDY=Path(__file__).resolve().parents[1]\nROOT=STUDY.parents[1]',"BASE=Path(__file__).resolve().parents[1]\nROOT=BASE.parents[1]\nARM=os.environ.get('LCF_ARM')\nSTUDY=BASE/'arms'/ARM if ARM else BASE")
    text=text.replace("frozen=read(STUDY/'FREEZE_MANIFEST.json')","frozen=read(BASE/'SERVER_FREEZE_MANIFEST.json')")
    text=text.replace("sha(STUDY/'PROTOCOL.md')","sha(BASE/'MIGRATION_PROTOCOL.md')")
    text=text.replace("sha(STUDY/'FREEZE_MANIFEST.json')","sha(BASE/'SERVER_FREEZE_MANIFEST.json')")
    text=text.replace("'RUN_LCF_'+stage.upper()", "'RUN_LCFS_'+str(ARM or 'ALL')+'_'+stage.upper()")
    text=text.replace("<=8*2**30,'RSS exceeds 8 GiB'","<=32*2**30,'RSS exceeds 32 GiB'")
    text=text.replace(">=.75*2**30",">=2*2**30").replace("<=6*2**30,'CUDA allocated exceeds 6 GiB'","<=44*2**30,'CUDA allocated exceeds 44 GiB'").replace("<=6*2**30,'CUDA reserved exceeds 6 GiB'","<=44*2**30,'CUDA reserved exceeds 44 GiB'").replace(">=12*2**30",">=50*2**30")
    text=text.replace("'host':socket.gethostname()", "'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'slurm_procid':os.environ.get('SLURM_PROCID'),'arm':ARM,'retry_of':'RUN_LCF_TRAIN_ced92f98a1aa4d8898fe5a24c0b13a39','restart_from_initialization':True,'host':socket.gethostname()")
    (scripts/'lcf_runtime.py').write_text(text,encoding='utf-8')
    text=(PARENT/'scripts/lcf_runner.py').read_text(encoding='utf-8')
    # Import-safe training implementation remains identical except periodic non-selection saving.
    anchor="                    if counts['D']%20==0:"
    require(anchor in text,'runner portability anchor')
    text=text.replace(anchor,"                    if min(counts.values())%20==0:")
    anchor="                    del caches,rows,features,targets,loss"
    text=text.replace(anchor,"                    if min(counts.values())%500==0:\n                        for arm in ARMS:save_model(run.path/arm/('recovery_step_%05d.pt'%counts[arm]),models[arm],opts[arm],counts[arm],epoch)\n"+anchor)
    (scripts/'lcf_runner.py').write_text(text,encoding='utf-8')
    for arm in ['A','B','C','D']:
        folder=STUDY/'arms'/arm;folder.mkdir(parents=True,exist_ok=True);(folder/'runs').mkdir(exist_ok=True)
        for name in ['CONFIG.json','inputs']:
            link=folder/name
            if not link.exists():link.symlink_to(STUDY/name,target_is_directory=name=='inputs')
        write(folder/'study.json',{'study_id':cfg['study_id'],'arm':arm,'plan_status':'frozen','execution_status':'not_started','scientific_status':'hypothesis_only'})
    # Freeze after copying wrappers and protocol, but before any GPU smoke or formal update.
    frozen_paths=[p for p in STUDY.rglob('*') if p.is_file() and not any(x in p.parts for x in ['arms','runs','__pycache__']) and p.name not in ['SERVER_FREEZE_MANIFEST.json']]
    frozen_paths += [ROOT/r['path'] for r in source['files'] if (ROOT/r['path']).is_file() and not r['path'].startswith('experiments/mask_local_connectivity_factorial_20261020/')]
    write(STUDY/'SERVER_FREEZE_MANIFEST.json',{'frozen_at':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'code_commit':'f70c9a0a+frozen_migration_sources','protocol_sha256':sha(STUDY/'MIGRATION_PROTOCOL.md'),'config_sha256':sha(STUDY/'CONFIG.json'),'parent_freeze_sha256':sha(PARENT/'FREEZE_MANIFEST.json'),'files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in sorted(set(frozen_paths))],'input_files':external,'science_module_hashes':science_receipts})
    print('SERVER_FROZEN',sha(STUDY/'SERVER_FREEZE_MANIFEST.json'),flush=True)

if __name__=='__main__': main()
