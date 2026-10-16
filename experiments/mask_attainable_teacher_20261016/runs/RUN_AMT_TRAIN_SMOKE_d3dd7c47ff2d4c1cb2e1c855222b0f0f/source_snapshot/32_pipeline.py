"""One shot: teacher/student smoke, full teacher Gate, fresh training, audit, STOP."""
import os
import socket
import subprocess
import traceback
from common import *

STATUS=STUDY/'AUTONOMOUS_STATUS.json'
state=dict(status='starting',started_at=now(),host=socket.gethostname(),pid=os.getpid(),
    automatic_S3=False,automatic_S4=False,automatic_commit_or_push=False,runs={})

def update(**kw):
    state.update(kw,updated_at=now());save(STATUS,state)
    study=read_json(STUDY/'study.json')
    study.update(execution_status='running' if state['status']=='running' else state['status'],
        artifact_status='partial' if state['status']=='running' else 'pipeline_evidence_on_laptop',
        current_phase=state.get('phase'),run_ids=state['runs'],
        scientific_status='trained_not_evaluated' if state['status']=='completed' else 'unadjudicated')
    save(STUDY/'study.json',study)

def command(name,args,timeout):
    update(status='running',phase=name)
    with (STUDY/(name+'.log')).open('x',encoding='utf8') as log:
        result=subprocess.run([sys.executable,'-B',str(STUDY/'scripts'/args[0]),*map(str,args[1:])],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    require(result.returncode==0,name+' failed; retain log and Run, no restart')

def stage(name,script,args,timeout):
    before=set((STUDY/'runs').glob('*/run.json')) if (STUDY/'runs').exists() else set()
    try:
        command(name,[script,*args],timeout)
    finally:
        added=set((STUDY/'runs').glob('*/run.json'))-before
        if len(added)==1:
            state['runs'][name]=next(iter(added)).parent.name
            update()
    require(len(added)==1,'stage must create exactly one Run')
    run=added.pop().parent
    state['runs'][name]=run.name
    update()
    require(read_json(run/'run.json')['execution_status']=='completed','stage incomplete')
    require(read_json(run/'DECISION.json')['status']=='passed',name+' scientific/preflight Gate did not pass')
    return run

def main():
    guard=STUDY/'.AUTONOMOUS_OWNER.guard'
    require(not STATUS.exists(),'pipeline already launched; do not duplicate')
    require(not list(ROOT.glob('experiments/*/.*.lock')),'another experiment is active')
    require(not list(ROOT.glob('experiments/*/.AUTONOMOUS_OWNER.guard')),'another owner is active')
    write_new(guard,dict(pid=os.getpid(),host=socket.gethostname(),started_at=now(),study=STUDY.name))
    try:
        update()
        inspect_inputs()
        tsm=stage('teacher_smoke','teacher.py',['--scope','smoke'],1020)
        command('verify_teacher_smoke',['verify_teacher.py',tsm],3720)
        ssm=stage('training_smoke','training.py',['--scope','smoke','--teacher-run',tsm],1020)
        teacher=stage('teacher_full','teacher.py',['--scope','full','--smoke-run',tsm],14520)
        command('verify_teacher_full',['verify_teacher.py',teacher],3720)
        train=stage('training_full','training.py',['--scope','full','--teacher-run',teacher,'--smoke-run',ssm],7320)
        command('verify_training_full',['verify_training.py',train],1020)
        update(status='completed',phase='training_and_configuration_audit_completed',ended_at=now(),
            checkpoint_sha256=sha256(train/'last.pt'),scientific_status='trained_not_evaluated',
            next_action='Frozen S2 same-dev/native and control comparison; no automatic S3')
    except BaseException as e:
        (STUDY/'PIPELINE_ERROR.txt').write_text(traceback.format_exc(),encoding='utf8')
        update(status='failed',ended_at=now(),error=str(e),next_action='Read retained evidence; no automatic scientific changes or retry')
        raise
    finally:
        require(read_json(guard)['pid']==os.getpid(),'owner guard changed; refuse to remove')
        guard.unlink()

if __name__=='__main__':
    main()
