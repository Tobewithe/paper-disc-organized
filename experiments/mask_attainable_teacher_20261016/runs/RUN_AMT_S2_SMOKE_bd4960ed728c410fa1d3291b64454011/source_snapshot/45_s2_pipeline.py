"""Single independent laptop job; stops after frozen S2 and independent audit."""
import socket
import subprocess
import traceback
from s2_common import *
STATUS=STUDY/'S2_AUTONOMOUS_STATUS.json'
state=dict(status='starting',started_at=now(),host=socket.gethostname(),pid=os.getpid(),
    automatic_training=False,automatic_S3=False,automatic_S4=False,automatic_commit_or_push=False,runs={})


def update(**kw):
    state.update(kw,updated_at=now());save(STATUS,state)
    study=read_json(STUDY/'study.json')
    study.update(execution_status=state['status'],evaluation_status=state['status'],current_phase=state.get('phase'),
        active_run_id=state.get('active_run_id'),scientific_status=state.get('scientific_status','trained_pending_S2'),
        artifact_status='S2_evidence_on_laptop',transfer_status='S2_not_transferred')
    study.setdefault('run_ids',{}).update(state['runs']);save(STUDY/'study.json',study)


def command(name,args,timeout):
    update(status='running',phase=name)
    with (STUDY/(name+'.log')).open('x',encoding='utf8') as log:
        p=subprocess.run([sys.executable,'-B',str(STUDY/'scripts'/args[0]),*map(str,args[1:])],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    require(p.returncode==0,'stage failed: '+name+'; retained evidence; no retry/redraw')


def evaluate(scope,smoke=None):
    before=set((STUDY/'runs').glob('RUN_AMT_S2_*/run.json'))
    args=['evaluate_s2.py','--scope',scope]
    if smoke:args+=['--smoke-run',smoke.name]
    try:command('s2_'+scope,args,720 if scope=='smoke' else 3720)
    finally:
        added=set((STUDY/'runs').glob('RUN_AMT_S2_*/run.json'))-before
        if len(added)==1:
            state['runs']['s2_'+scope]=next(iter(added)).parent.name
            update(active_run_id=state['runs']['s2_'+scope])
    require(len(added)==1,'stage Run registration differs')
    run=added.pop().parent
    require(read_json(run/'run.json')['execution_status']=='completed','evaluation incomplete')
    command('s2_verify_'+scope,['verify_s2.py',run],720 if scope=='smoke' else 1920)
    require(read_json(run/'INDEPENDENT_LAPTOP.json')['status']=='passed','independent audit incomplete')
    if scope=='smoke':require(read_json(run/'DECISION.json')['status']=='smoke_passed','smoke Gate failed')
    return run


def main():
    guard=STUDY/'.AUTONOMOUS_OWNER.guard'
    require(not STATUS.exists(),'S2 pipeline already launched')
    require(not list(ROOT.glob('experiments/*/.*.lock')),'another active lock')
    require(not list(ROOT.glob('experiments/*/.AUTONOMOUS_OWNER.guard')),'another active owner')
    require(read_json(STUDY/'S2_PREPARATION.json')['status']=='passed','preparation missing')
    write_new(guard,dict(pid=os.getpid(),host=socket.gethostname(),started_at=now(),study=STUDY.name,phase='S2_only'))
    try:
        update();check_contract(inputs())
        smoke=evaluate('smoke')
        full=evaluate('full',smoke)
        decision=read_json(full/'DECISION.json')
        update(status='completed',phase='S2_evaluation_and_independent_audit_completed',active_run_id=None,
            ended_at=now(),scientific_status=decision['scientific_status'],decision=decision['status'],next_action=decision['next_action'])
    except BaseException as e:
        (STUDY/'S2_PIPELINE_ERROR.txt').write_text(traceback.format_exc(),encoding='utf8')
        update(status='failed',ended_at=now(),error=str(e),scientific_status='blocked_engineering')
        raise
    finally:
        require(read_json(guard)['pid']==os.getpid(),'owner guard changed')
        guard.unlink()


if __name__=='__main__':main()
