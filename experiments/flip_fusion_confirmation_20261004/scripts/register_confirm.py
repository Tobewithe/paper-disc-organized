"""Desktop metadata and source files only. Never execute a model."""
from pathlib import Path
import hashlib,json,subprocess,tempfile,uuid

P=Path(r'C:\Dpan\codexproject\paper-disc-organized')
R=P/'experiments/flip_fusion_confirmation_20261004'
OLD=P/'experiments/flip_response_projection_20261004'
CLI=Path(r'C:\Dpan\codexproject\Aggregation Workbench\researchctl.ps1')

def record(name,value):
    dest=R/name
    if dest.exists(): raise RuntimeError('Preserve existing registration: '+str(dest))
    text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',suffix=dest.suffix,delete=False) as f:
        f.write(text); temp=Path(f.name)
    try:
        args=['pwsh','-NoProfile','-File',str(CLI),'--project','paper-disc','record',dest.relative_to(P).as_posix(),'--input',str(temp),'--expected-revision','new']
        done=subprocess.run(args,capture_output=True,text=True,encoding='utf-8',errors='replace',creationflags=0x08000000)
        if done.returncode: raise RuntimeError(done.stdout+done.stderr)
        print(name,hashlib.sha256(dest.read_bytes()).hexdigest())
    finally: temp.unlink(missing_ok=True)

def main():
    R.mkdir(parents=True,exist_ok=True);(R/'scripts').mkdir(exist_ok=True)
    ids=dict(study_id='STUDY_'+uuid.uuid4().hex,runs={k:'RUN_'+uuid.uuid4().hex for k in ('smoke','evaluation')})
    record('study.json',dict(study_id=ids['study_id'],title='冻结整图翻转融合的COCO方法开发外确认',kind='diagnostic',status='registered',execution_status='not_started',protocol='PROTOCOL.md'))
    record('RUN_IDS.json',ids)
    remote='/root/flip_fusion_confirmation_20261004'
    cfg=json.loads((OLD/'RUN_CONFIG.json').read_text(encoding='utf-8-sig'))
    cfg.update(server_root=remote,images=remote+'/data/images',confirm_ids=remote+'/CONFIRM_IDS.json',
        prior_results='/root/flip_response_projection_20261004/runs/RUN_eb9eeafabad047c7b26557958811b80c/PER_CANDIDATE.jsonl',
        original_screen_root='/root/prototype_readout_fast_screen_20261003',
        planned_images=1536,smoke_max_seconds=240,evaluation_max_seconds=1800,max_seconds=1800,
        alpha=.5,bootstrap=5000,bootstrap_seed=20261004,
        smoke_run=remote+'/runs/'+ids['runs']['smoke'],
        primary_group='box_good_mask_bad',primary_comparison='MIX_minus_A',
        target_practical_effect=.005,overall_noninferiority_margin=.001,
        main_metric='confirmation image-macro original-image MaskIoU on prespecified A box-good mask-bad candidates',
        scope='No learned parameters; method-development holdout from COCO train2017. Official pretraining and earlier full-COCO project training have seen train2017; not globally unseen data. TAL-conditioned localization confirmation, not complete output AP.',
        output_reference=dict(max_det=300,conf=.001,branch='one2one',purpose='secondary retained-candidate annotation only'),
        frozen_dependencies=['flip_source.py','online_runtime.py','evaluation_metrics.py','legacy_prepare_cache.py'])
    for k in ('projection_lambda','planned_images_old','split','screening_effect_all','screening_effect_target'):
        cfg.pop(k,None)
    record('RUN_CONFIG.json',cfg)
    sources={name:OLD/'scripts'/name for name in ('flip_source.py','online_runtime.py','evaluation_metrics.py')}
    sources['legacy_prepare_cache.py']=P/'experiments/prototype_readout_fast_screen_20261003/scripts/legacy_prepare_cache.py'
    reused={}
    for name,src in sources.items():
        dst=R/'scripts'/name
        if dst.exists():raise RuntimeError('Existing dependency '+name)
        dst.write_bytes(src.read_bytes())
        reused[name]=dict(source=src.relative_to(P).as_posix(),sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),unchanged=True)
    record('REUSED_CODE.json',reused)

if __name__=='__main__':main()
