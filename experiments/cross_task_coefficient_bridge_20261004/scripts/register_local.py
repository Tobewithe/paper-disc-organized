"""One-time local records and standard-library orchestration source generation."""
from pathlib import Path
import hashlib,json,uuid
root=Path(__file__).resolve().parents[1]
old=root.parent/'mask_supervision_support_20261004'
def put(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (root/'RUN_IDS.json').exists()
stages=('prepare','smoke','train_T','train_M','train_R','evaluation')
ids={'study_id':'STUDY_'+uuid.uuid4().hex,'runs':{s:'RUN_'+uuid.uuid4().hex for s in stages}}
put(root/'RUN_IDS.json',ids)
put(root/'study.json',{'study_id':ids['study_id'],'title':'跨任务隐藏表示到原生系数分支的短预算对照','kind':'method','status':'registered','protocol':'PROTOCOL.md'})
cfg=json.loads((old/'RUN_CONFIG.json').read_text(encoding='utf-8-sig'))
for k in ('support_dir','dense_topk','dense_topk2','original_topk2','original_loss_weight','method_checkpoint'):cfg.pop(k,None)
remote='/root/'+root.name
cfg.update(server_root=remote,split=remote+'/SPLIT.json',arms=['A','N','T','M','R','TW'],
           feature_cache=remote+'/features/FEATURES.pt',dense_root='/root/mask_supervision_support_20261004',
           smoke_run=remote+'/runs/'+ids['runs']['smoke'],prepare_run=remote+'/runs/'+ids['runs']['prepare'],
           checkpoints={a:remote+'/runs/'+ids['runs']['train_'+a]+'/final.pt' for a in ('T','M','R')},
           bridge_lr=.0001,train_max_seconds=1800,smoke_max_seconds=300,
           scope='Frozen detection hidden representations bridge into jointly trained native coefficient branch',
           main_metric='dev image macro original-image MaskIoU T vs A/N/M',
           standardization_std_min=1e-6,feature_sources={'T':'cls256','M':'wrong_cls256','R':'box64'})
put(root/'RUN_CONFIG.json',cfg)
put(root/'REUSED_CODE.json',{p.name:{'source':str(old/'scripts'/p.name),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    for p in (root/'scripts').glob('*.py') if (old/'scripts'/p.name).exists()})
# Transport and collection are pure file operations, kept hidden by the established SSH worker.
control=(root.parents[1]/'_maintenance/workbench_alignment/mask_supervision_support_20261004/transport_control.py').read_text(encoding='utf-8')
control=control.replace('mask_supervision_support_20261004',root.name).replace('pipeline_support.py','pipeline_bridge.py')
control=control.replace("'REPORT.md','RUN_CONFIG.json'","'REPORT.md','RUN_CONFIG.json'")
(root/'scripts/transport_control.py').write_text(control,encoding='utf-8')
collect=(old/'scripts/collect_support.py').read_text(encoding='utf-8')
collect=collect.replace('mask_supervision_support_20261004',root.name).replace('("smoke", "prepare", "train_D", "evaluation")',repr(stages))
(root/'scripts/collect_bridge.py').write_text(collect,encoding='utf-8')
print(json.dumps(ids))
