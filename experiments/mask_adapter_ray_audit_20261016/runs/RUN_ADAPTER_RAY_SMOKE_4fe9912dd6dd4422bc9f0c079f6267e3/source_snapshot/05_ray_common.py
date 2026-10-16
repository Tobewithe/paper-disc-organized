"""Own Study only; immutable dependencies and registered execution."""
from pathlib import Path
import json, os, shutil, socket, sys

ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'experiments/mask_adapter_ray_audit_20261016'
PREVIOUS=ROOT/'experiments/mask_repair_preservation_20261016'
BASE=ROOT/'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170'
S1=ROOT/'experiments/mask_readout_mechanism_v2_20261012/runs/RUN_S1_ORACLE_f35a1399ac244f74aeebf94e4e843949'
TRAIN=ROOT/'experiments/mask_readout_soft_iou_pilot_20261014/runs/RUN_SOFT_IOU_TRAIN_53afd0bb8aba428482159f5e426878eb'
AUDIT=ROOT/'experiments/mask_opportunity_audit_20261015/runs/RUN_OPPORTUNITY_FULL_18ab59a063b0489d81f541d45a907bfd'
C_FULL=PREVIOUS/'runs/RUN_REPAIR_KEEP_FULL_68b1d8211b30456abb172db046963e01'
C_SMOKE=PREVIOUS/'runs/RUN_REPAIR_KEEP_SMOKE_1473cf76889b4163b962d2e7086995b7'
S3=ROOT/'experiments/mask_gate_s3r_20261015/runs/RUN_S3_FULL_670fffd010664a648e096572fc7c476d'
GT=ROOT/'assets/datasets/coco_native_s0_20261013/annotations/instances_val2017.json'
sys.path.insert(0,str(ROOT/'experiments/mask_readout_mechanism_v2_20261012/scripts'))
sys.path.insert(0,str(PREVIOUS/'scripts'))
from posttrain_run import RegisteredRun,configure_native,read_json,write_new,sha256,require,now,save,tensor_state_hash
from posttrain_audit import read_jsonl

OWN=['PROTOCOL.md','CONFIG.json','INPUT_EXPECTATIONS.json','SOURCE_CONTEXT.json','SOURCE_DIRTY.patch',
     'scripts/ray_common.py','scripts/ray_geometry.py','scripts/analysis.py','scripts/prepare.py',
     'scripts/run_audit.py','scripts/verify_ray.py','scripts/test_ray.py','scripts/launch.ps1']
CODE=['experiments/mask_adapter_ray_audit_20261016/'+p for p in OWN]
CODE+=['experiments/mask_readout_mechanism_v2_20261012/scripts/'+p for p in ('posttrain_run.py','adapter_model.py','adapter_data.py','adapter_primitives.py','s1_geometry.py')]
CODE+=['shared/evaluation/'+p for p in ('posttrain_audit.py','native_output_bridge.py','native_source_pins.json')]
CODE+=['experiments/mask_repair_preservation_20261016/scripts/native_geometry.py','paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json']


def inspect_inputs():
    cfg=read_json(STUDY/'CONFIG.json'); expected=read_json(STUDY/'INPUT_EXPECTATIONS.json')
    require(socket.gethostname().upper()==cfg['executor']['host'],'unauthorized host')
    require(str(ROOT).lower()==cfg['executor']['root'].lower(),'wrong root')
    require(Path(sys.executable).resolve()==Path(cfg['executor']['python']).resolve(),'wrong interpreter')
    for rel,digest in expected['files'].items():require(sha256(ROOT/rel)==digest,'input changed: '+rel)
    profile,environment=configure_native()
    require(environment==read_json(AUDIT/'EVAL_CONTRACT.json')['environment'],'native runtime differs')
    for source in (BASE,S1,TRAIN,AUDIT,C_FULL,C_SMOKE,S3):require(read_json(source/'run.json')['execution_status']=='completed','incomplete source')
    return cfg,expected,profile,environment


def ownership_check():
    import psutil
    require(not list(ROOT.glob('experiments/*/.*.lock')),'existing lock; leave unchanged')
    require(not list(ROOT.glob('experiments/*/.AUTONOMOUS_OWNER.guard')),'existing execution owner')
    others=[p.info for p in psutil.process_iter(['pid','name','cmdline']) if p.info['pid']!=os.getpid() and 'python' in (p.info['name'] or '').lower()]
    require(not others,'other Python process present: '+str(others))


def resources(run):
    import psutil
    cfg=read_json(STUDY/'CONFIG.json')['budget'];run.check_budget()
    require(psutil.Process().memory_info().rss<cfg['rss_bytes'],'RSS limit')
    require(psutil.virtual_memory().available>=cfg['min_free_ram_bytes'],'RAM reserve')
    require(shutil.disk_usage(ROOT).free>=cfg['min_free_disk_bytes'],'disk reserve')
    require(sum(p.stat().st_size for p in run.path.rglob('*') if p.is_file())<=cfg['output_bytes'],'output budget')


class RayRun(RegisteredRun):
    def __init__(self,scope,parent):
        cfg=read_json(STUDY/'CONFIG.json')
        super().__init__('ADAPTER_RAY_'+scope.upper(),'DIRECTION_AMPLITUDE',parent,CODE,cfg['budget'][scope+'_seconds'],protocol='PROTOCOL.md')
        self.path=STUDY/'runs'/self.run_id;self.lock=STUDY/'.DIRECTION_AMPLITUDE.lock'
        self.meta.update(study_id='STUDY_MASK_ADAPTER_RAY_AUDIT_20261016',scope=scope,new_training_steps=0,base_model_forward_calls=0,
                         adapter_forward_calls=0,baseline_run_id=BASE.name,source_S1_run_id=S1.name,checkpoint_run_id=TRAIN.name)
    def __enter__(self):
        ownership_check()
        return super().__enter__()
    def freeze(self,contract):
        frozen=read_json(STUDY/'FREEZE_MANIFEST.json');self.sources={p:sha256(ROOT/p) for p in CODE}
        require(self.sources==frozen['files'],'code/protocol drift')
        require(sha256(STUDY/'PREPARATION.json')==frozen['preparation_sha256'],'preparation drift')
        origin=read_json(STUDY/'SOURCE_CONTEXT.json')
        self.meta.update(code_commit=origin['code_commit'],dirty_patch_sha256=sha256(STUDY/'SOURCE_DIRTY.patch'))
        write_new(self.path/'SOURCE_MANIFEST.json',frozen);mapping={}
        for i,rel in enumerate(CODE):
            dest=self.path/'source_snapshot'/(f'{i:02d}_'+Path(rel).name);dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT/rel,dest);mapping[rel]=dest.relative_to(self.path).as_posix()
        write_new(self.path/'SOURCE_SNAPSHOT_MAP.json',mapping)
        for name in ('PROTOCOL.md','CONFIG.json','INPUT_EXPECTATIONS.json','PREPARATION.json'):shutil.copyfile(STUDY/name,self.path/name)
        write_new(self.path/'EVAL_CONTRACT.json',contract)
        self.update(plan_status='frozen',protocol_path=str(self.path/'PROTOCOL.md'),protocol_sha256=sha256(self.path/'PROTOCOL.md'),
                    input_manifest_sha256=sha256(STUDY/'INPUT_EXPECTATIONS.json'),contract_sha256=sha256(self.path/'EVAL_CONTRACT.json'))
