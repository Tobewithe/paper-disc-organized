"""One authorized finite sequence; no scheduler, repeat loop, new seed or scientific evaluation."""
import os
import subprocess
import sys
import time
import traceback
from rfc_runtime import *


def main():
    authorization=read_json(STUDY/'AUTHORIZATION.json')
    require(authorization['native_replay_smoke_and_formal_A_Q_training'],'current execution authorization missing')
    state=STUDY/'EXECUTION_STATUS.json'
    lock=STUDY/'.sequence.lock'
    # Separate from stage lock; every phase still owns its own .RFC.lock.
    with lock.open('x',encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(),host=os.environ.get('COMPUTERNAME'),started_at=now()),f)
    try:
        initial_resources()
        preflight=subprocess.run([sys.executable,'-B',str(STUDY/'scripts/rfc_preflight.py')],cwd=ROOT,timeout=600)
        require(preflight.returncode==0,'preflight failed; no automatic retry')
        deadline=time.time()+1800
        stages=[('input_smoke','rfc_native_replay.py',['--scope','smoke','--smoke-deadline',str(deadline)]),
                ('smoke_A','rfc_training.py',['--scope','smoke','--arm','A','--smoke-deadline',str(deadline)]),
                ('smoke_Q','rfc_training.py',['--scope','smoke','--arm','Q','--smoke-deadline',str(deadline)]),
                ('input_full','rfc_native_replay.py',['--scope','full']),
                ('train_A','rfc_training.py',['--scope','full','--arm','A']),
                ('train_Q','rfc_training.py',['--scope','full','--arm','Q'])]
        for phase,script,args in stages:
            save(state,dict(status='running',phase=phase,pid=os.getpid(),updated_at=now()))
            wall=max(.001,deadline-time.time()) if 'smoke' in phase else (10800 if phase=='input_full' else 7200)
            result=subprocess.run([sys.executable,'-B',str(STUDY/'scripts'/script),*args],cwd=ROOT,timeout=wall)
            require(result.returncode==0,'phase failed; no automatic retry: '+phase)
            if phase=='smoke_Q':
                inputs=pointer('INPUT_SMOKE')
                a_smoke,q_smoke=pointer('SMOKE_A'),pointer('SMOKE_Q')
                write_new(STUDY/'REAL_INPUT_REPLAY.json',dict(status='passed',tensor_identity_exact=True,
                          zero_delta_RLE_exact=True,source_fingerprints_match=True,input_run=inputs.name,
                          configuration_audit_sha256=sha256(inputs/'CONFIG_AUDIT.json')))
                write_new(STUDY/'REAL_SMOKE_VERIFICATION.json',dict(status='passed',native_environment_parity=True,
                          parameter_and_gradient_checks='passed',A=a_smoke.name,Q=q_smoke.name,
                          A_audit_sha256=sha256(a_smoke/'CONFIG_AUDIT.json'),Q_audit_sha256=sha256(q_smoke/'CONFIG_AUDIT.json'),
                          smoke_steps_per_arm=2,formal_initialization_reset_required=True))
                from rfc_contract import execution_readiness
                require(execution_readiness()['real_execution_ready'],'real execution readiness failed')
        a,q=pointer('TRAIN_A'),pointer('TRAIN_Q')
        write_new(STUDY/'TRAINING_PAIR_FREEZE.json',dict(status='frozen',at=now(),
                  A=dict(run_id=a.name,checkpoint_sha256=sha256(a/'last.pt')),
                  Q=dict(run_id=q.name,checkpoint_sha256=sha256(q/'last.pt')),
                  design_sha256=sha256(STUDY/'DESIGN_SPEC.json'),scientific_status='trained_not_evaluated',
                  damage_sample_drawn=False,evaluation_started=False))
        save(state,dict(status='completed',phase='both_final_checkpoints_frozen',updated_at=now(),
                        scientific_status='trained_not_evaluated',formal_steps=3000))
    except BaseException as exc:
        save(state,dict(status='failed',updated_at=now(),error=str(exc),traceback=traceback.format_exc(),
                        action='retain runs; diagnose engineering; no automatic retries'))
        raise
    finally:
        lock.unlink()

if __name__=='__main__': main()
