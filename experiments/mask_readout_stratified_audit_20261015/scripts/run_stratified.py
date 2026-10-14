"""Registered CPU-only reanalysis on existing measurements and frozen cache metadata."""
import argparse
from collections import Counter, defaultdict
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT/'experiments/mask_readout_stratified_audit_20261015'
PARENT = ROOT/'experiments/mask_readout_benefit_audit_20261014/runs/RUN_BENEFIT_AUDIT_FULL_44b006cece39486aa502ca64398166d4'
EVAL = ROOT/'experiments/mask_readout_soft_iou_pilot_20261014/runs/RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0'
TRACE = ROOT/'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170/trace'
sys.path.insert(0, str(ROOT/'experiments/mask_readout_mechanism_v2_20261012/scripts'))
from posttrain_run import RegisteredRun, require, sha256, read_json, write_new
from stratified_stats import analyze, decide, area_group, crowd_group, neighbors, STAGES

CODE = ['experiments/mask_readout_stratified_audit_20261015/'+p for p in
        ('PROTOCOL.md','INPUT_EXPECTATIONS.json','SOURCE_CONTEXT.json','SOURCE_DIRTY.patch',
         'scripts/run_stratified.py','scripts/stratified_stats.py','scripts/test_stratified.py','scripts/launch_stratified.ps1')]
CODE += ['experiments/mask_readout_mechanism_v2_20261012/scripts/posttrain_run.py','shared/evaluation/posttrain_audit.py']


def lines(path):
    with path.open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def inputs():
    manifest = read_json(STUDY/'INPUT_EXPECTATIONS.json')
    for kind in ('files','frozen_sources'):
        for relative, expected in manifest[kind].items():
            require(sha256(ROOT/relative) == expected, 'source/input drift: '+relative)
    require(read_json(PARENT/'run.json')['execution_status'] == 'completed', 'parent incomplete')
    require(read_json(PARENT/'CONFIG_AUDIT.json')['status'] == 'passed', 'parent config failed')
    require(read_json(EVAL/'DECISION.json')['status'] == 'failed', 'S2 status changed')
    artifact_map = {a['relative_path']:a['sha256'] for a in read_json(PARENT/'ARTIFACT_MANIFEST.json')['artifacts']}
    require(sha256(PARENT/'PAIR_MEASUREMENTS.jsonl') == artifact_map['PAIR_MEASUREMENTS.jsonl'], 'old measurements changed')
    environment = dict(python=sys.version, platform=platform.platform(),
                       packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','torch','psutil')},
                       device='CPU', arithmetic='float64 box/statistics; source stage values untouched')
    return manifest, environment


class StrataRun(RegisteredRun):
    def __init__(self, scope):
        super().__init__('STRATIFIED_'+scope.upper(), 'STRATIFIED_AUDIT', PARENT, CODE, 300 if scope=='smoke' else 1200)
        self.path = STUDY/'runs'/self.run_id
        self.lock = STUDY/'.STRATIFIED_AUDIT.lock'
        self.meta.update(study_id='STUDY_MASK_READOUT_STRATIFIED_AUDIT_20261015', scope=scope,
                         new_training_steps=0, model_forward_calls=0, decoder_calls=0,
                         baseline_run_id='RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170')

    def __enter__(self):
        require(not list(ROOT.glob('experiments/*/.*.lock')), 'unresolved study lock exists')
        return super().__enter__()

    def freeze(self, contract):
        context = read_json(STUDY/'SOURCE_CONTEXT.json')
        self.sources = {r:sha256(ROOT/r) for r in CODE}
        self.meta.update(code_commit=context['code_commit'], dirty_patch_sha256=sha256(STUDY/'SOURCE_DIRTY.patch'))
        write_new(self.path/'SOURCE_MANIFEST.json', dict(files=self.sources, code_commit=context['code_commit']))
        for relative in self.sources:
            target = self.path/'source_snapshot'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/relative,target)
        shutil.copyfile(STUDY/'PROTOCOL.md', self.path/'PROTOCOL.md')
        write_new(self.path/'EVAL_CONTRACT.json',contract)
        self.update(plan_status='frozen',protocol_path=str(self.path/'PROTOCOL.md'),
                    protocol_sha256=sha256(self.path/'PROTOCOL.md'),contract_sha256=sha256(self.path/'EVAL_CONTRACT.json'),
                    input_manifest_sha256=sha256(STUDY/'INPUT_EXPECTATIONS.json'))


def prepare():
    rows = lines(PARENT/'PAIR_MEASUREMENTS.jsonl')
    old_list, new_list = lines(EVAL/'BASELINE_SUBSET_GT_LEDGER.jsonl'), lines(EVAL/'B_GT_LEDGER.jsonl')
    key = lambda r:(r['image_id'],r['annotation_id'])
    old,new = {key(r):r for r in old_list},{key(r):r for r in new_list}
    require(len(old)==len(old_list) and len(new)==len(new_list) and old.keys()==new.keys(),'ledger keys differ/duplicate')
    panel = read_json(EVAL/'DAMAGE_PANEL.json')
    require({key(r) for r in rows if r['role']=='success_segm75'} == {key(r) for r in panel['sample']},'panel changed')
    require(len(rows)==len({(r['role'],*key(r)) for r in rows})==3949, 'measurement count/duplicates')
    require(Counter(r['role'] for r in rows)=={'dev_bbox75':2925,'success_segm75':1024},'role counts differ')
    mapping = read_json(EVAL/'B_ROW_MAPPING.json')
    def native_id(oid):
        return None if oid is None else mapping['source_prediction_sha256']+':'+str(mapping['original_global_indices'][int(oid.rsplit(':',1)[1])])
    for r in rows:
        a,b=old[key(r)],new[key(r)]
        require(not a['is_ignored'] and not b['is_ignored'], 'ignored GT included')
        require(a['gt_area']==b['gt_area'] and a['gt_category_id']==b['gt_category_id'],'GT area/class changed')
        identity = 'bbox75_match_output_id' if r['role']=='dev_bbox75' else 'segm75_match_output_id'
        require(r['output_id']==a[identity], 'fixed output identity changed')
        require(native_id(b['bbox75_match_output_id'])==a['bbox75_match_output_id'],'bbox match changed')
        require(r['task_damaged']==(a['task_mask75']=='success' and b['task_mask75']=='failure'),'task damage mismatch')
        if r['role']=='success_segm75':
            require(a['task_mask75']=='success' and r['stages']['binary_original']['before']['iou']>=.75,'invalid success row')
        if r['population']=='T':
            require(a['task_mask75']=='failure' and r['stages']['binary_original']['before']['iou']<.75,'invalid target row')
        for stage in STAGES[1:]:
            for side in ('before','after'):
                v=r['stages'][stage][side]
                union=v['tp']+v['fp']+v['fn']
                require(abs(v['iou']-(v['tp']/union if union else 1.))<1e-12,'integer IoU mismatch')
        r.update(gt_area=a['gt_area'],gt_category_id=a['gt_category_id'],gt_area_group=area_group(a['gt_area']),
                 native_foreground_area=r['signals']['native_foreground_pixels'],
                 native_area_group=area_group(r['signals']['native_foreground_pixels']),
                 task_repaired=a['task_mask75']=='failure' and b['task_mask75']=='success',
                 baseline_task=a['task_mask75'],method_task=b['task_mask75'])
    require(sum(r['role']=='dev_bbox75' and r['population']=='T' for r in rows)==655,'T denominator changed')
    require(sum(r['task_damaged'] for r in rows if r['role']=='success_segm75')==53,'damage count changed')
    return rows,panel


def execute(scope, smoke_id):
    with StrataRun(scope) as run:
        expected,environment = inputs()
        parent_contract = read_json(PARENT/'EVAL_CONTRACT.json')
        ids = parent_contract['image_ids'] if scope=='full' else parent_contract['image_ids'][:8]
        rows,panel=prepare()
        selected=set(ids)
        rows=[r for r in rows if r['image_id'] in selected]
        trace=read_json(TRACE/'TRACE_MANIFEST.json')
        records={r['image_id']:r for r in trace['images'] if r['image_id'] in selected}
        require(set(records)==selected,'missing cache metadata')
        contract=dict(scope=scope, image_ids=ids, dev_ids=parent_contract['dev_ids'], success_image_ids=panel['image_ids'],
                      source_run=PARENT.name, source_evaluation=EVAL.name, checkpoint_sha256=parent_contract['checkpoint_sha256'],
                      native_profile_sha256=parent_contract['native_profile_sha256'], original_S2_gate_status='failed_unchanged',
                      new_training_steps=0, model_forward_calls=0, decoder_calls=0,
                      interpretation='exploratory_reanalysis_not_confirmation', environment=environment,
                      bootstrap=dict(resamples=2000,seed=1729,contrast_quantiles=[.05/16,1-.05/16]),
                      area_boundaries=[1024,9216], crowd_boundaries=[.1,.5], input_hashes=expected,
                      source_inference_contract_sha256=sha256(EVAL/'EVAL_CONTRACT.json'),
                      immutable_cache={str(i):records[i]['cache_sha256'] for i in ids})
        if scope=='full':
            smoke=STUDY/'runs'/smoke_id
            require(read_json(smoke/'run.json')['execution_status']=='completed','full requires successful smoke')
            old_contract=read_json(smoke/'EVAL_CONTRACT.json')
            require(old_contract['environment']==environment and old_contract['input_hashes']==expected,'smoke environment/input drift')
            require(read_json(smoke/'run.json')['protocol_sha256']==sha256(STUDY/'PROTOCOL.md'),'smoke protocol changed')
            old_source=read_json(smoke/'SOURCE_MANIFEST.json')['files']
            require(all(sha256(ROOT/r)==v for r,v in old_source.items()),'smoke sources changed')
        run.freeze(contract)
        run.update(smoke_run_id=smoke_id or None, phase='metadata_and_frozen_table_join')
        import torch
        import psutil
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        require(not torch.cuda.is_initialized(),'CUDA was initialized in CPU-only analysis')
        process=psutil.Process()
        by_image=defaultdict(list)
        for row in rows:
            by_image[row['image_id']].append(row)
        maximum_rss=0
        same_error=0.
        metadata_path=run.path/'BOX_METADATA.jsonl'
        with metadata_path.open('x',encoding='utf8') as stream:
            for n,image_id in enumerate(ids,1):
                record=records[image_id]
                path=(TRACE/record['cache']).resolve()
                require(path.is_relative_to(TRACE.resolve()),'cache escaped trace')
                require(sha256(path)==record['cache_sha256'],'cache changed before read')
                cache=torch.load(path,map_location='cpu',weights_only=True)
                require(cache['image_id']==image_id and cache['prediction_indices']==record['prediction_indices'],'cache identity mismatch')
                require(cache['input_hw']==record['input_hw'],'native geometry mismatch')
                selected_rows=by_image[image_id]
                ranks=sorted({r['output_rank'] for r in selected_rows})
                boxes,classes,raw=cache['boxes_input'].tolist(),cache['classes'].tolist(),cache['raw_ids']
                signals=neighbors(boxes,classes,raw,ranks)
                for row in selected_rows:
                    rank=row['output_rank']
                    require(raw[rank]==row['raw_id'] and cache['input_hw']==row['input_hw'],'cached row alignment mismatch')
                    oid=trace['predictions_sha256']+':'+str(cache['prediction_indices'][rank])
                    require(row['output_id']==oid,'output rank mapping mismatch')
                    old=row['signals']['same_class_neighbor_iou']
                    same_error=max(same_error,abs(old-signals[rank]['same_class_neighbor_iou']))
                    require(same_error<=1e-12,'same-class crowd recomputation mismatch')
                    row.update(signals[rank])
                    row['all_crowd_group']=crowd_group(row['all_class_neighbor_iou'])
                    row['same_crowd_group']=crowd_group(old)
                stream.write(json.dumps(dict(image_id=image_id,boxes_input=boxes,classes=classes,raw_ids=raw,
                    prediction_indices=cache['prediction_indices'],input_hw=cache['input_hw'],cache_sha256=record['cache_sha256']))+'\n')
                del cache
                maximum_rss=max(maximum_rss,process.memory_info().rss)
                require(maximum_rss<=2*1024**3,'RSS budget exceeded')
                run.check_budget()
                if n%200==0 or n==len(ids):
                    print('METADATA',n,len(ids),'RSS',maximum_rss,flush=True)
        with (run.path/'STRATIFIED_PAIRS.jsonl').open('x',encoding='utf8') as stream:
            for row in rows:
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        print('COMPUTING_FROZEN_TABLES',flush=True)
        result=analyze(rows,parent_contract['dev_ids'],panel['image_ids'],scientific=scope=='full')
        decision=decide(result) if scope=='full' else dict(status='smoke_passed',direction=None,next_action='Run authorized full frozen analysis')
        maximum_rss=max(maximum_rss,process.memory_info().rss)
        require(maximum_rss<=2*1024**3,'RSS budget exceeded after statistics')
        run.check_budget()
        for image_id,record in records.items():
            require(sha256(TRACE/record['cache'])==record['cache_sha256'],'cache changed after analysis')
        again,again_environment=inputs()
        require(again==expected and again_environment==environment,'final inputs/environment changed')
        require(not torch.cuda.is_initialized(),'unexpected CUDA initialization')
        write_new(run.path/'CONFIG_AUDIT.json',dict(status='passed',
            source_measurement_sha256=sha256(PARENT/'PAIR_MEASUREMENTS.jsonl'),
            checks=dict(inputs_and_sources_before_after=True,all_selected_cache_hashes_before_after=True,
                        stage_measurements_reused_unchanged=True,strict_GT_and_output_join=True,
                        original_cohorts_and_S2_gates_preserved=True,no_model_or_decoder_execution=True,
                        no_training_no_cuda=True,smoke_full_config_equal=scope=='full'),
            same_class_neighbor_max_absolute_error=same_error,checked_cache_images=len(ids),
            peak_rss_observed_bytes=maximum_rss,checkpoint_sha256=contract['checkpoint_sha256'],
            contract_sha256=sha256(run.path/'EVAL_CONTRACT.json')))
        write_new(run.path/'ROUTE_DECISION.json',decision)
        run.finish(result,status=decision['status'],scientific='exploratory_reanalysis' if scope=='full' else 'engineering_only',
                   next_action=decision['next_action'])
        print('COMPLETED',run.run_id,decision['status'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--scope',choices=('smoke','full'))
    parser.add_argument('--smoke-run',default='')
    args=parser.parse_args()
    if args.preflight:
        expected,environment=inputs()
        print(json.dumps(dict(status='passed',files=len(expected['files']),sources=len(expected['frozen_sources']),environment=environment)))
    elif args.scope:
        execute(args.scope,args.smoke_run)
    else:
        parser.error('scope or preflight required')
