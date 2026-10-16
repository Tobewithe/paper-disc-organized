"""Read-only identities and availability before the authorized fixed CLD evaluation."""
from cld_common import *


def main():
    data = inspect_inputs()
    resources = resource_check()
    authorization = read_json(STUDY / 'OPERATION_AUTHORIZATION.json')
    require(authorization['allow_full_evaluation'] is True and authorization['allow_formal_training'] is False
            and authorization['allow_smoke'] is False, 'evaluation-only scope required')
    require(not list((STUDY / 'runs').glob('RUN_CLD_EVALUATION_*')), 'existing evaluation must be inspected, not restarted')
    registry = read_json(STUDY / 'TRAINED_CHECKPOINTS.json')
    require(registry['heldout_model_evaluation_started'] is False, 'checkpoint registration was not before evaluation')
    checked = 0
    for arm, item in registry['checkpoints'].items():
        path = STUDY / 'runs' / item['run_id']
        check_completed(path)
        require(sha256(path / 'last.pt') == item['sha256'], 'last checkpoint changed')
        require(sha256(path / 'EVAL_CONTRACT.json') == item['training_contract_sha256'], 'training contract changed')
        require(sha256(path / 'INDEPENDENT.json') == item['training_independent_sha256'], 'independent training receipt changed')
        contract = read_json(path / 'EVAL_CONTRACT.json')
        require(contract['arm'] == arm and contract['scope'] == 'full'
                and contract['expected_optimizer_steps'] == 1200, 'wrong training identity')
        require(contract['amplitude'] == registry['common_amplitude'] == 2.1651711144258443, 'amplitude changed')
        for record in read_json(path / 'ARTIFACT_MANIFEST.json')['artifacts']:
            require(sha256(path / record['relative_path']) == record['sha256'], 'training artifact drift')
            checked += 1
    trace = read_json(CACHE / 'trace/TRACE_MANIFEST.json')
    for row in trace['images']:
        require(sha256(CACHE / 'trace' / row['cache']) == row['cache_sha256'], 'native cache bytes changed')
    images = read_json(CACHE / 'IMAGE_MANIFEST.json')
    for row in images:
        require(sha256(Path(row['source_image'])) == row['image_sha256'], 'source image bytes changed')
    controls = read_json(STUDY / 'EVALUATION_CONTROL_FREEZE.json')
    for rel, digest in controls['files'].items():
        require(sha256(STUDY / rel) == digest, 'evaluation controller drift')
    result = dict(status='passed', at=now(), scope='read_only_no_model_forward_no_training',
                  checkpoints=registry['checkpoints'], dependency_artifacts_checked=checked,
                  cache_files=len(trace['images']), images=len(images), environment=data['environment'], resources=resources,
                  execution_freeze_sha256=sha256(STUDY / 'EXECUTION_FREEZE.json'),
                  evaluation_control_freeze_sha256=sha256(STUDY / 'EVALUATION_CONTROL_FREEZE.json'),
                  operation_authorization_sha256=sha256(STUDY / 'OPERATION_AUTHORIZATION.json'),
                  model_forward_calls=0, optimizer_steps=0)
    write_new(STUDY / 'EVALUATION_PRELAUNCH_CHECK.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
