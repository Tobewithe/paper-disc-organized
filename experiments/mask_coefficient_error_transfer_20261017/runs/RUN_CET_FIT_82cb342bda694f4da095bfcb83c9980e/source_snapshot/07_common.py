"""CET identities and immutable inputs. No model construction or training."""
from pathlib import Path
import json, os, shutil, sys

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'experiments/mask_readout_mechanism_v2_20261012'
CACHE = PARENT / 'runs/RUN_S2_CACHE_FULL_61ce0f1ca0e6479ca6704fb2fcceabec'
CLD = ROOT / 'experiments/mask_direction_learnability_20261016'
TEACHER = ROOT / 'experiments/mask_attainable_teacher_20261016/runs/RUN_AMT_TEACHER_FULL_24a4d184c8e54f45bf4327b0ab4710ca'
OLD = ROOT / 'experiments/mask_soft_fit_audit_20261016/runs/RUN_SOFT_FIT_FULL_14971e618ba8498a9fa8efcd19d21770'
GT = CACHE / 'dataset/training_annotations.json'
for p in (PARENT/'scripts', ROOT/'shared/evaluation'):
    sys.path.append(str(p))
from posttrain_run import RegisteredRun, configure_native, read_json, write_new, sha256, require, now


def jsonlines(p):
    with Path(p).open(encoding='utf8') as f:
        return [json.loads(l) for l in f if l.strip()]


def science_check():
    record = read_json(STUDY/'EXECUTION_FREEZE.json')
    for rel, digest in record['files'].items():
        require(sha256(ROOT/rel) == digest, 'frozen source differs: '+rel)
    return record


def inspect_inputs():
    science_check()
    manifest = read_json(STUDY/'INPUT_MANIFEST.json')
    for item in manifest['files']:
        require(sha256(ROOT/item['path']) == item['sha256'], 'source input differs: '+item['path'])
    profile, environment = configure_native()
    inherited = read_json(CLD/'INHERITED_CONTRACT.json')
    require(environment == inherited['source_environment'], 'runtime differs from frozen source cache')
    trace = read_json(CACHE/'trace/TRACE_MANIFEST.json')
    require(trace['native_args'] == inherited['actual_cached_native_args'], 'native args drift')
    require(sha256(ROOT/'assets/models/coco_clean_20260911/yolo26m-seg.pt') == profile['model_sha256'], 'base weights drift')
    require(sha256(GT) == manifest['annotations_sha256'], 'GT drift')
    require(sha256(Path(trace['predictions_path'])) == trace['predictions_sha256'], 'native predictions drift')
    require(sha256(OLD/'G_predictions.json') == manifest['g_predictions_sha256'], 'G predictions drift')
    require(sha256(Path(manifest['original_annotation_path'])) == manifest['original_annotation_sha256'], 'original annotation drift')
    split = read_json(STUDY/'SPLIT.json')
    require(len(split['fit_image_ids']) == 1600 and len(split['heldout_image_ids']) == 400, 'split size')
    require(not set(split['fit_image_ids']) & set(split['heldout_image_ids']), 'split leakage')
    return dict(profile=profile,environment=environment,split=split,trace=trace)


class Run(RegisteredRun):
    def __init__(self, stage, parent='none'):
        self.config = read_json(STUDY/'CONFIG.json')
        freeze = science_check()
        super().__init__('CET_'+stage.upper(), 'CET', parent, list(freeze['files']), self.config[stage+'_budget_seconds'])
        self.path = STUDY/'runs'/self.run_id
        self.lock = STUDY/'.CET.lock'
        self.meta.update(study_id=self.config['study_id'], optimizer_steps=0, model_forward_calls=0, oracle_optimization_calls=0)

    def __enter__(self):
        import psutil
        others = [p.info for p in psutil.process_iter(['pid','name','cmdline']) if p.pid != os.getpid() and 'python' in (p.info['name'] or '').lower()]
        require(not others, 'other Python tasks: '+str(others))
        own_guard=STUDY/'.AUTONOMOUS_OWNER.guard'
        if own_guard.exists():require(read_json(own_guard)['controller_pid']==os.getppid(),'foreign controller owns this Study')
        locks = list(ROOT.glob('experiments/*/.*.lock')) + [p for p in ROOT.glob('experiments/*/.AUTONOMOUS_OWNER.guard') if p!=own_guard]
        require(not locks, 'unresolved execution locks: '+str(locks))
        self.resources = dict(available_ram_bytes=psutil.virtual_memory().available, free_disk_bytes=shutil.disk_usage(ROOT).free, other_python_processes=others)
        require(self.resources['available_ram_bytes']>=4*1024**3 and self.resources['free_disk_bytes']>=8*1024**3,'resources')
        return super().__enter__()

    def freeze(self, data, extra):
        f = science_check()
        self.sources = f['files']
        for index, rel in enumerate(self.sources):
            target = self.path/'source_snapshot'/f'{index:02d}_{Path(rel).name}'
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT/rel,target)
        for name in ('PROTOCOL.md','CONFIG.json','SPLIT.json','INPUT_MANIFEST.json','EXECUTION_CONTEXT.json'):
            shutil.copyfile(STUDY/name,self.path/name)
        write_new(self.path/'SOURCE_MANIFEST.json',f)
        write_new(self.path/'RESOURCES.json',self.resources)
        write_new(self.path/'EVAL_CONTRACT.json',dict(environment=data['environment'],native_args=data['trace']['native_args'],
            model_sha256=data['profile']['model_sha256'],candidate_policy='fixed_output_detection',
            scope='GT_assisted_internal_holdout_stress_not_deployable_method',**extra))
        context=read_json(STUDY/'EXECUTION_CONTEXT.json')
        self.update(plan_status='frozen',code_commit=context['code_commit'],dirty_patch_sha256=context['tracked_diff_sha256'],
            protocol_sha256=sha256(STUDY/'PROTOCOL.md'),input_manifest_sha256=sha256(STUDY/'INPUT_MANIFEST.json'),
            contract_sha256=sha256(self.path/'EVAL_CONTRACT.json'),execution_freeze_sha256=sha256(STUDY/'EXECUTION_FREEZE.json'))


def config_finish(run, source, extra=None):
    source.verify()
    inspect_inputs()
    write_new(run.path/'CONFIG_AUDIT.json',dict(status='passed',inputs_and_sources_unchanged=True,new_optimizer_steps=0,
        base_model_forward_calls=0,adapter_forward_calls=0,oracle_optimization_calls=0,**(extra or {})))


def completed_fit(p):
    p=Path(p).resolve()
    require(p.parent == (STUDY/'runs').resolve(),'foreign fit path')
    require(read_json(p/'run.json')['execution_status']=='completed','incomplete fit')
    require(read_json(p/'CONFIG_AUDIT.json')['status']=='passed','fit audit')
    require(read_json(p/'SOURCE_MANIFEST.json')['files']==science_check()['files'],'fit source version')
    model=read_json(p/'FIT_MODEL.json')
    require(sha256(p/'FIT_ERRORS.npz')==model['library_sha256'],'fit library changed')
    require(model['heldout_error_rows_used']==0,'fit leakage')
    return model
