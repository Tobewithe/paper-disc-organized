"""Desktop/laptop transport only. No model imports, passwords or console windows."""
from pathlib import Path
import os
import subprocess
import json

HOST = 'root@connect.bjb2.seetacloud.com'
PORT = '33953'
ROOT = '/root/autodl-tmp/prototype_guided_evidence_selection_20261003'

def hidden():
    kw = dict(stdin=subprocess.DEVNULL)
    if os.name == 'nt':
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        kw.update(creationflags=subprocess.CREATE_NO_WINDOW, startupinfo=startup)
    return kw

def key_path():
    directory = Path.home()/'.ssh'
    laptop = directory/'paper_disc_bjb2_33953'
    return laptop if laptop.exists() else directory/'id_ed25519'


def socket_worker(action, timeout, **payload):
    python = Path(r'C:\Dpan\envsfiles\CondaData\envs\pytorch\python.exe')
    if os.name != 'nt' or not python.exists():
        return None
    options = hidden();options.pop('stdin')
    request=dict(action=action,timeout=timeout,key=str(key_path()),**payload)
    result=subprocess.run([str(python),'-X','utf8',str(Path(__file__).with_name('transport_worker.py'))],
        input=json.dumps(request),capture_output=True,text=True,encoding='utf-8',errors='replace',
        timeout=timeout+30,**options)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    value=json.loads(result.stdout)
    if value['returncode']:
        raise RuntimeError(f"SSH exit {value['returncode']}: {value['stderr'][-1800:]} {value['stdout'][-800:]}")
    return value

def ssh(command, timeout=60):
    patched=socket_worker('ssh',timeout,command=command)
    if patched is not None:return patched['stdout']
    result = subprocess.run(['ssh','-T','-p',PORT,'-i',str(key_path()),
        '-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=5',
        '-o','ServerAliveCountMax=3','-o','StrictHostKeyChecking=yes',
        HOST,command],capture_output=True,text=True,encoding='utf-8',errors='replace',
        timeout=timeout,**hidden())
    if result.returncode:
        raise RuntimeError(f'SSH exit {result.returncode}: {result.stderr[-1800:]} {result.stdout[-800:]}')
    return result.stdout

def put(paths, destination, timeout=600):
    if isinstance(paths,(str,Path)): paths=[paths]
    patched=socket_worker('put',timeout,paths=[str(p) for p in paths],destination=str(destination))
    if patched is not None:return
    result=subprocess.run(['scp','-B','-P',PORT,'-i',str(key_path()),
        '-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=5',
        '-o','ServerAliveCountMax=3','-X','buffer=1024','-X','nrequests=256',
        *[str(p) for p in paths],HOST+':'+destination],
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout,**hidden())
    if result.returncode: raise RuntimeError(result.stderr[-1800:])

def get(source,destination,timeout=600):
    patched=socket_worker('get',timeout,source=str(source),destination=str(destination))
    if patched is not None:return
    result=subprocess.run(['scp','-B','-P',PORT,'-i',str(key_path()),
        '-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=5',
        '-o','ServerAliveCountMax=3','-X','buffer=1024','-X','nrequests=256',
        HOST+':'+source,str(destination)],
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout,**hidden())
    if result.returncode: raise RuntimeError(result.stderr[-1800:])
