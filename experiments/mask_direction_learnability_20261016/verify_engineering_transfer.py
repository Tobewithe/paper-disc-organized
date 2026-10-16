"""Local stdlib-only closure audit; never imports the model or starts a remote task."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone

STUDY=Path(__file__).resolve().parent
ROOT=STUDY.parents[1]


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path): return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s]


def main():
    transport=read(STUDY/'ENGINEERING_TRANSFER_MANIFEST.json')
    assert sha(STUDY/'ENGINEERING_TRANSFER_MANIFEST.json')=='b4e222221d9fd267d625388e9366623688e03317286d189d31b22166defac67a'
    for record in transport['files']:
        path=STUDY/record['path']
        assert path.resolve().is_relative_to(STUDY)
        assert path.stat().st_size==record['size_bytes'] and sha(path)==record['sha256'],record['path']
    freeze=read(STUDY/'EXECUTION_FREEZE.json')
    for path,digest in freeze['files'].items(): assert sha(ROOT/path)==digest,path
    for r in read(STUDY/'SCIENTIFIC_FREEZE.json')['files']: assert sha(STUDY/r['path'])==r['sha256']
    reports={}
    for rid in transport['run_ids']:
        run=STUDY/'runs'/rid
        meta,contract,independent,summary=[read(run/f) for f in ('run.json','EVAL_CONTRACT.json','INDEPENDENT.json','SUMMARY.json')]
        assert meta['execution_status']=='completed' and meta['exit_code']==0
        assert meta['scientific_status']=='engineering_only' and independent['status']=='passed'
        assert read(run/'CONFIG_AUDIT.json')['status']=='passed'
        assert sha(run/'EVAL_CONTRACT.json')==meta['contract_sha256']==independent['contract_sha256']
        assert sha(run/'PROTOCOL.md')==meta['protocol_sha256']==sha(STUDY/'PROTOCOL.md')
        assert read(run/'SOURCE_MANIFEST.json')['files']==freeze['files']
        for source,relative in read(run/'SOURCE_SNAPSHOT_MAP.json').items(): assert sha(run/relative)==freeze['files'][source]
        for r in read(run/'ARTIFACT_MANIFEST.json')['artifacts']: assert sha(run/r['relative_path'])==r['sha256']
        assert contract['operation_authorization']['allow_formal_training'] is False
        assert contract['operation_authorization']['allow_full_evaluation'] is False
        assert contract['environment']==read(STUDY/'INHERITED_CONTRACT.json')['source_environment']
        reports[rid]=dict(summary=summary,independent=independent,contract=contract)
    preflight=reports[transport['run_ids'][0]]
    a=read(STUDY/'runs'/transport['run_ids'][0]/'AMPLITUDE.json')
    assert a['teacher_count']==2510 and a['heldout_teachers_used']==0
    assert preflight['summary']['native_rows']==176042
    tests=read(STUDY/'runs'/transport['run_ids'][0]/'BEHAVIOR_TESTS.json')
    assert tests['returncode']==0 and 'Ran 9 tests' in tests['stderr']
    smoke={r['contract']['arm']:(rid,r) for rid,r in reports.items() if r['contract']['stage']=='CLD_training'}
    c_real,c_perm=[dict(smoke[k][1]['contract']) for k in ('REAL','PERM')]
    c_real.pop('arm');c_perm.pop('arm')
    assert c_real==c_perm,'two-arm effective contracts differ beyond the frozen arm variable'
    expected=read(STUDY/'SPLIT.json')['fit_image_ids'][:8]
    steps=forwards=0
    for arm,(rid,r) in smoke.items():
        run=STUDY/'runs'/rid
        actual=rows(run/'training.jsonl')
        assert [x['image_id'] for x in actual]==expected
        assert sum(x['teachers'] for x in actual)==4
        assert r['summary']['optimizer_steps']==2 and r['summary']['images_seen']==8
        assert r['contract']['amplitude']==a['value']
        assert sha(run/'last.pt')==r['summary']['checkpoint_sha256']
        assert sha(run/'initial.pt')==r['contract']['original_initial_sha256']
        assert r['independent']['normal_mixed_masks_verified']==681
        assert r['independent']['additional_optimizer_steps']==0
        steps+=r['summary']['optimizer_steps']
        forwards+=r['summary']['adapter_forward_calls']+r['independent']['adapter_forward_calls']
    assert steps==4 and forwards==80
    assert not list((STUDY/'runs').glob('RUN_CLD_FULL_*')) and not list((STUDY/'runs').glob('RUN_CLD_EVALUATION_*'))
    result=dict(status='passed',at=datetime.now(timezone.utc).isoformat(),verification_scope='local byte/configuration/registration audit; tensor/gradient independent checks executed on laptop',
                verifier_sha256=sha(Path(__file__)),run_ids=transport['run_ids'],transferred_files=len(transport['files']),
                transferred_bytes=sum(r['size_bytes'] for r in transport['files']),source_files=36,scientific_freeze_files=11,
                configurations_differ_only_in_arm=True,common_amplitude=a['value'],smoke_optimizer_steps=steps,formal_optimizer_steps=0,
                adapter_forward_calls_including_independent_checks=forwards,base_model_forward_calls=0,new_oracle_solves=0,
                heldout_training_or_model_evaluation_images=0,scientific_gates='not_evaluated',next_action='Formal matched REAL/PERM training remains unstarted')
    destination=STUDY/'ENGINEERING_VERIFICATION.json'
    with destination.open('x',encoding='utf-8') as stream: stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__': main()
