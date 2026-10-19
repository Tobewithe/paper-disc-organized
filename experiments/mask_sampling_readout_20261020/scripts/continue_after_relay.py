"""Finite dependency handoff: wait once for the existing relay process, then launch laptop pipeline.

No recurring task, scheduler, goal, retry loop or experiment polling. Windows
waits on the already identified process handle, bounded by the input deadline.
"""
import base64,ctypes,datetime as dt,json,subprocess,sys,time,traceback,uuid
from pathlib import Path
from prepare_inputs import STUDY,read,write,sha,now

REMOTE='D:/coco_wire/baseline_native_20261013/experiments/mask_sampling_readout_20261020'
PY='C:/Users/28358/anaconda3/envs/pytorch/python.exe'

def main():
    run=STUDY/'runs'/('RUN_SSR_HANDOFF_'+uuid.uuid4().hex);run.mkdir(parents=True)
    relay=read(STUDY/'RELAY_STATUS.json')
    meta={'run_id':run.name,'study_id':'STUDY_MASK_SAMPLING_READOUT_20261020','stage':'finite_input_handoff',
          'started_at':now(),'execution_status':'running','relay_run_id':relay['run_id'],'relay_pid':relay['pid'],
          'pid':__import__('os').getpid(),'formal_training_steps':0,'scheduled_task':False,'goal_mode':False,
          'source_sha256':sha(__file__),'phase':'waiting_existing_relay_handle'}
    write(run/'run.json',meta);write(STUDY/'HANDOFF_STATUS.json',meta)
    try:
        if relay['execution_status']=='running':
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.OpenProcess.restype=ctypes.c_void_p
            handle=kernel.OpenProcess(0x00100000,False,int(relay['pid']))
            if not handle:raise RuntimeError('existing relay process is not available')
            started=dt.datetime.fromisoformat(relay['started_at'])
            remaining=max(0,10800-(dt.datetime.now(dt.timezone.utc)-started).total_seconds())
            kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
            result=kernel.WaitForSingleObject(handle,int(remaining*1000))
            kernel.CloseHandle.argtypes=[ctypes.c_void_p];kernel.CloseHandle(handle)
            if result!=0:raise RuntimeError('input relay did not finish within frozen deadline')
        relay=read(STUDY/'RELAY_STATUS.json')
        if relay['execution_status']!='completed':raise RuntimeError('relay failed; preserving error without retry')
        meta.update(phase='validate_laptop_inputs_and_freeze');write(run/'run.json',meta);write(STUDY/'HANDOFF_STATUS.json',meta)
        for script in ('prepare_inputs.py','finalize_freeze.py'):
            proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','28358lan',PY,'-B',REMOTE+'/scripts/'+script],
                                capture_output=True,text=True,errors='replace',timeout=1200)
            (run/(script+'.stdout.log')).write_text(proc.stdout,encoding='utf-8')
            (run/(script+'.stderr.log')).write_text(proc.stderr,encoding='utf-8')
            if proc.returncode:raise RuntimeError('laptop preparation failed: '+script)
        # WMI creates one ordinary independent process; no scheduled task exists.
        command='"'+PY+'" -B "'+REMOTE+'/scripts/ssr_runner.py" pipeline'
        ps="$ErrorActionPreference='Stop'; $r=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"+command+"';CurrentDirectory='D:/coco_wire/baseline_native_20261013'}; $r | Select-Object ReturnValue,ProcessId | ConvertTo-Json -Compress"
        encoded=base64.b64encode(ps.encode('utf-16le')).decode()
        proc=subprocess.run(['ssh','-o','BatchMode=yes','28358lan','C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-EncodedCommand',encoded],
                            capture_output=True,text=True,errors='replace',timeout=90)
        (run/'LAUNCH.stdout.log').write_text(proc.stdout,encoding='utf-8');(run/'LAUNCH.stderr.log').write_text(proc.stderr,encoding='utf-8')
        if proc.returncode:raise RuntimeError('finite laptop launch command failed')
        receipt=json.loads(proc.stdout.strip())
        if receipt['ReturnValue']!=0 or not receipt['ProcessId']:raise RuntimeError('WMI did not create the process')
        write(run/'LAPTOP_LAUNCH_RECEIPT.json',{'pid':receipt['ProcessId'],'command':command,'remote_host':'YANGLEI',
              'scheduled_task':False,'finite_stages':['smoke','train','evaluate','statistics'],'launched_at':now()})
        meta.update(execution_status='completed',phase='independent_laptop_pipeline_launched',laptop_pid=receipt['ProcessId'],finished_at=now(),exit_code=0)
        write(run/'run.json',meta);write(STUDY/'HANDOFF_STATUS.json',meta)
        print('LAPTOP_PIPELINE_PID',receipt['ProcessId'],flush=True)
    except BaseException as e:
        (run/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
        meta.update(execution_status='failed',phase='failed',error=str(e),finished_at=now(),exit_code=1)
        write(run/'run.json',meta);write(STUDY/'HANDOFF_STATUS.json',meta);raise

if __name__=='__main__':main()
