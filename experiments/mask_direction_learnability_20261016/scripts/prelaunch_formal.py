"""Read-only full identity/availability audit before authorized formal training."""
from datetime import datetime,timezone
from cld_common import *


def main():
    data=inspect_inputs()
    resources=resource_check()
    authorization=read_json(STUDY/'OPERATION_AUTHORIZATION.json')
    require(authorization['allow_formal_training'] is True and authorization['allow_full_evaluation'] is False,'wrong operation scope')
    require(not list((STUDY/'runs').glob('RUN_CLD_FULL_*')), 'existing formal work must be inspected rather than restarted')
    dependencies=['RUN_CLD_PREFLIGHT_a436d52ceed643638c6f32be616f9cfe','RUN_CLD_SMOKE_REAL_396e8c25780f4357b21bc76c7431db2a','RUN_CLD_SMOKE_PERM_0761445d8fd2437795f783d6b6acdba8']
    count=0
    for rid in dependencies:
        path=STUDY/'runs'/rid
        check_completed(path)
        for row in read_json(path/'ARTIFACT_MANIFEST.json')['artifacts']:
            require(sha256(path/row['relative_path'])==row['sha256'],'dependency artifact drift')
            count+=1
    trace=read_json(CACHE/'trace/TRACE_MANIFEST.json')
    for row in trace['images']:
        require(sha256(CACHE/'trace'/row['cache'])==row['cache_sha256'],'native cache bytes changed')
    images=read_json(CACHE/'IMAGE_MANIFEST.json')
    for row in images:
        require(sha256(Path(row['source_image']))==row['image_sha256'],'source image bytes changed')
    controls=read_json(STUDY/'FORMAL_CONTROL_FREEZE.json')
    for rel,digest in controls['files'].items():
        require(sha256(STUDY/rel)==digest,'formal controller drift')
    result=dict(status='passed',at=now(),scope='read_only_no_model_forward_no_training',
                dependencies=dependencies,dependency_artifacts_checked=count,cache_files=len(trace['images']),images=len(images),
                execution_freeze_sha256=sha256(STUDY/'EXECUTION_FREEZE.json'),formal_control_freeze_sha256=sha256(STUDY/'FORMAL_CONTROL_FREEZE.json'),
                operation_authorization_sha256=sha256(STUDY/'OPERATION_AUTHORIZATION.json'),environment=data['environment'],resources=resources,
                model_forward_calls=0,optimizer_steps=0)
    write_new(STUDY/'FORMAL_PRELAUNCH_CHECK.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
