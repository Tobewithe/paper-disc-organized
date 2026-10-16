"""Local full-training closure audit. No model execution and no new optimization."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path

STUDY=Path(__file__).resolve().parent
ROOT=STUDY.parents[1]


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    transport=read(STUDY/'FORMAL_TRANSFER_MANIFEST.json')
    for record in transport['files']:
        item=STUDY/record['path']
        assert item.resolve().is_relative_to(STUDY)
        assert item.stat().st_size==record['size_bytes'] and sha(item)==record['sha256'],record['path']
    freeze=read(STUDY/'EXECUTION_FREEZE.json')
    for relative,digest in freeze['files'].items():assert sha(ROOT/relative)==digest,relative
    for r in read(STUDY/'SCIENTIFIC_FREEZE.json')['files']:assert sha(STUDY/r['path'])==r['sha256']
    for rel,digest in read(STUDY/'FORMAL_CONTROL_FREEZE.json')['files'].items():assert sha(STUDY/rel)==digest
    split=read(STUDY/'SPLIT.json');orders=read(STUDY/'IMAGE_ORDERS.json')['epochs']
    expected=[iid for order in orders for iid in order]
    assert len(expected)==4800 and len(set(expected))==1600 and not set(expected)&set(split['heldout_image_ids'])
    cache=ROOT/'experiments/mask_readout_mechanism_v2_20261012/runs/RUN_S2_CACHE_FULL_61ce0f1ca0e6479ca6704fb2fcceabec'
    trace={r['image_id']:r for r in read(cache/'trace/TRACE_MANIFEST.json')['images']}
    teacher=ROOT/'experiments/mask_attainable_teacher_20261016/runs/RUN_AMT_TEACHER_FULL_24a4d184c8e54f45bf4327b0ab4710ca/TEACHER_ROWS.jsonl'
    counts=Counter(r['image_id'] for r in rows(teacher) if r['accepted'])
    contracts={};report={}
    for rid in transport['run_ids']:
        run=STUDY/'runs'/rid
        meta,c,summary,independent=[read(run/n) for n in ('run.json','EVAL_CONTRACT.json','SUMMARY.json','INDEPENDENT.json')]
        arm=c['arm']
        assert arm in ('REAL','PERM') and arm not in contracts and c['scope']=='full'
        assert meta['execution_status']=='completed' and meta['exit_code']==0 and meta['scientific_status']=='trained_not_evaluated'
        assert read(run/'CONFIG_AUDIT.json')['status']=='passed' and independent['status']=='passed'
        assert independent['kind']=='full_training_serialized_audit' and independent['additional_optimizer_steps']==0
        assert independent['independent_gradient_reexecution'] is False
        assert c['expected_optimizer_steps']==summary['optimizer_steps']==independent['optimizer_steps']==1200
        assert c['expected_image_visits']==summary['images_seen']==4800
        assert c['epochs']==3 and c['fit_images']==1600 and c['image_microbatch']==1 and c['gradient_accumulation']==4
        assert c['operation_authorization']['allow_formal_training'] is True and c['operation_authorization']['allow_full_evaluation'] is False
        assert sha(run/'EVAL_CONTRACT.json')==meta['contract_sha256']==independent['contract_sha256']
        assert sha(run/'PROTOCOL.md')==sha(STUDY/'PROTOCOL.md')
        assert read(run/'SOURCE_MANIFEST.json')['files']==freeze['files']
        for rel,digest in freeze['files'].items():assert sha(run/read(run/'SOURCE_SNAPSHOT_MAP.json')[rel])==digest
        for r in read(run/'ARTIFACT_MANIFEST.json')['artifacts']:assert sha(run/r['relative_path'])==r['sha256']
        assert sha(run/'last.pt')==summary['checkpoint_sha256']
        assert sha(run/'initial.pt')==c['original_initial_sha256']=='ce2de59a316eef3cea6cfe6d1763221931fd10693377c8a91ce321843604bacd'
        assert read(run/'IMAGE_ORDERS.json')==orders
        log=rows(run/'training.jsonl');steps=rows(run/'OPTIMIZER_STEPS.jsonl')
        assert [r['image_id'] for r in log]==expected
        assert len(steps)==1200 and [i for s in steps for i in s['image_ids']]==expected
        for j,r in enumerate(log):
            image=r['image_id']
            assert r['epoch']==j//1600 and r['step']==j//4+1
            assert r['teachers']==counts[image] and r['outputs']==len(trace[image]['prediction_indices'])
            assert r['cache_sha256']==trace[image]['cache_sha256']
            assert all(math.isfinite(r[k]) for k in ('loss','direction_loss','preservation_loss'))
            assert math.isclose(r['loss'],r['direction_loss']+.1*r['preservation_loss'],abs_tol=3e-6,rel_tol=1e-5)
        assert sum(r['teachers'] for r in log)==7530
        assert sum(r['outputs'] for r in log)==528126
        per_epoch=[]
        for epoch in range(3):
            selected=log[epoch*1600:(epoch+1)*1600]
            per_epoch.append(dict(epoch=epoch+1,images=1600,teacher_visits=sum(r['teachers'] for r in selected),
                                 total_loss=math.fsum(r['loss'] for r in selected)/1600,
                                 direction_loss=math.fsum(r['direction_loss'] for r in selected)/1600,
                                 preservation_loss=math.fsum(r['preservation_loss'] for r in selected)/1600))
        contracts[arm]=dict(c);contracts[arm].pop('arm')
        report[arm]=dict(run_id=rid,steps=1200,image_visits=4800,teacher_visits=7530,output_visits=528126,
                         checkpoint_sha256=summary['checkpoint_sha256'],initial_state_sha256=summary['initial_state_sha256'],
                         final_state_sha256=summary['final_state_sha256'],adapter_forward_calls=summary['adapter_forward_calls'],
                         per_epoch_training_only_description=per_epoch,elapsed_seconds=meta['elapsed_seconds'])
    assert set(contracts)=={'REAL','PERM'} and contracts['REAL']==contracts['PERM']
    assert all(x['adapter_forward_calls']==4800 for x in report.values())
    assert report['REAL']['initial_state_sha256']==report['PERM']['initial_state_sha256']
    result=dict(status='passed',at=datetime.now(timezone.utc).isoformat(),verifier_sha256=sha(Path(__file__)),
                scope='local hashes, contracts and full logs; checkpoint/optimizer tensor audit executed independently on laptop',
                transferred_files=len(transport['files']),transferred_bytes=sum(r['size_bytes'] for r in transport['files']),
                configurations_differ_only_in_arm=True,arms=report,total_formal_steps=2400,total_formal_adapter_forward_calls=9600,
                heldout_training_visits=0,heldout_model_evaluation_images=0,new_base_model_forwards=0,new_oracle_solves=0,
                mechanism_and_utility_gates='not_evaluated',native_gain=None,historical_method_increment=None)
    with (STUDY/'FORMAL_VERIFICATION.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
