"""New registered validation Run; retain and hash-copy the failed smoke outputs."""
from lea_common_v2 import *
from lea_verify_v2 import verify

old = STUDY/'runs/RUN_LEA_SMOKE_6b56df7df3d64803bc21c2824fbced70'
require(read_json(old/'run.json')['execution_status'] == 'error', 'Original smoke must remain failed')
require('unexpected keyword argument' in read_json(old/'run.json')['error'], 'Recovery does not match recorded error')
require(read_json(STUDY/'IMPORT_TESTS_V2.json')['status'] == 'passed', 'Import repair not checked')
require(not list((STUDY/'runs').glob('RUN_LEA_SMOKE_RECOVERY_*')), 'Recovery already exists')
files = ['TARGETS.json','ROW_MAPPING.json','INPUT_CACHE_RECORDS.json','ANNOTATIONS_SUBSET.json','PAIRS.jsonl',
         'GATES.jsonl','SUCCESS_SUPPORT.jsonl','SOURCE_GT_LEDGER.jsonl','N_predictions.json','G_predictions.json',
         'G_B_predictions.json','N_MATCHES.jsonl','G_MATCHES.jsonl','G_B_MATCHES.jsonl','GT_CENSUS.jsonl',
         'STATISTICS.json','BOOTSTRAP.npz']
run = Run('smoke', old)
study = read_json(STUDY/'study.json')
try:
    with run:
        cfg, manifest, profile, env = inspect_inputs()
        run.meta.update(retry_of=old.name, recovery_type='validate_existing_smoke_bytes_no_new_outcomes')
        contract = read_json(old/'EVAL_CONTRACT.json')
        contract['engineering_recovery'] = dict(original_failed_run_id=old.name, scientific_configuration_unchanged=True)
        run.freeze(contract)
        copied = {}
        for name in files:
            shutil.copyfile(old/name, run.path/name)
            require(sha256(old/name) == sha256(run.path/name), 'Recovery artifact copy changed')
            copied[name] = sha256(old/name)
        write_new(run.path/'RECOVERED_BYTES.json', dict(original_run=old.name, unchanged_files=copied))
        result = verify(run.path, replay=False)
        write_new(run.path/'INDEPENDENT_IN_RUN.json', result)
        require(result['status'] == 'passed', 'Recovery verification failed')
        audit = read_json(old/'CONFIG_AUDIT.json')
        audit.update(recovery_registered=True, original_failed_run_retained=True, source_files_verified=len(run.sources),
                     imported_verifier=str(Path(verify.__code__.co_filename).resolve()), copied_scientific_files=len(copied))
        verify_files(manifest['files'])
        write_new(run.path/'CONFIG_AUDIT.json', audit)
        summary = read_json(run.path/'STATISTICS.json')
        run.finish(summary, status='ENGINEERING_SMOKE_RECOVERED', scientific='ENGINEERING_SMOKE_RECOVERED',
                   next_action='Independent native gate replay, then frozen full evaluation once')
finally:
    if (run.path/'run.json').exists():
        meta=read_json(run.path/'run.json')
        study.update(execution_status=meta['execution_status'], scientific_status=meta['scientific_status'],
                     artifact_status=meta['artifact_status'], latest_run_id=run.run_id,
                     current_phase='recovery_waiting_independent_native_replay')
        save(STUDY/'study.json',study)
