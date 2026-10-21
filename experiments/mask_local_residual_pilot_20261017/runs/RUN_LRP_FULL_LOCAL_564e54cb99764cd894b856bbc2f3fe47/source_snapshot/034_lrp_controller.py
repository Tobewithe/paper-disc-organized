"""One authorized process, detached by WMI. No task scheduler or automation."""
from pathlib import Path
import datetime,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'experiments/mask_local_residual_pilot_20261017'
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p): return json.loads(p.read_text(encoding='utf8'))
def save(data):
    tmp=STUDY/'PROCESS_STATUS.tmp'; tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8'); os.replace(tmp,STUDY/'PROCESS_STATUS.json')
def main():
    os.chdir(ROOT)
    os.environ.update(PYTHONPATH=str(ROOT/'_site'),YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(ROOT/'_ultralytics_config'),OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
    os.environ.pop('MKL_NUM_THREADS',None)
    if (STUDY/'PROCESS_STATUS.json').exists(): raise RuntimeError('Existing execution status; do not duplicate')
    state=dict(status='running',pid=os.getpid(),started_at=now(),phases=[],new_scheduler=False,new_goal=False)
    save(state)
    phases=[('TESTS',['lrp_test.py'],900),('PREPARE',['lrp_prepare.py'],7200),
            ('SMOKE_LOCAL',['lrp_train.py','--scope','smoke','--arm','LOCAL'],900),
            ('SMOKE_UNIFORM',['lrp_train.py','--scope','smoke','--arm','UNIFORM'],900),
            ('FULL_LOCAL',['lrp_train.py','--scope','full','--arm','LOCAL'],7200),
            ('FULL_UNIFORM',['lrp_train.py','--scope','full','--arm','UNIFORM'],7200),
            ('EVALUATION',['lrp_evaluate.py'],7200),('VERIFY',['lrp_verify.py'],3600)]
    try:
        for label,argv,budget in phases:
            cmd=[sys.executable,'-B',str(STUDY/'scripts'/argv[0]),*argv[1:]]
            if label=='VERIFY': cmd.append(str(STUDY/'runs'/read(STUDY/'EVALUATION_POINTER.json')['run_id']))
            with (STUDY/f'{label}.stdout.log').open('x',encoding='utf8') as out,(STUDY/f'{label}.stderr.log').open('x',encoding='utf8') as err:
                child=subprocess.Popen(cmd,cwd=ROOT,env=os.environ.copy(),stdout=out,stderr=err,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
                state.update(current_phase=label,child_pid=child.pid); save(state)
                try: code=child.wait(timeout=budget+60)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(); raise RuntimeError('Child exceeded frozen wall budget: '+label)
                state['phases'].append(dict(label=label,pid=child.pid,command=cmd,exit_code=code,ended_at=now())); save(state)
                if code: raise RuntimeError(f'{label} failed with exit {code}; preserve all outputs')
        state.update(status='completed_and_verified',current_phase='awaiting_transfer',child_pid=None,ended_at=now())
    except BaseException as e:
        (STUDY/'CONTROLLER_ERROR.txt').write_text(traceback.format_exc(),encoding='utf8')
        state.update(status='failed',error=str(e),ended_at=now(),child_pid=None)
    finally: save(state)

if __name__=='__main__': main()
