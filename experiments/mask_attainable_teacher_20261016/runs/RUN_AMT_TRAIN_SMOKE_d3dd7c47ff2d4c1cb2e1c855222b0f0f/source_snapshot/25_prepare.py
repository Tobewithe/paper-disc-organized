"""Read-only native inputs/resources/tests, then one immutable launch freeze."""
import os
import socket
import subprocess
from common import *

def main():
    require(not (STUDY/'FREEZE_MANIFEST.json').exists(),'Study already frozen; do not overwrite')
    require(not list(ROOT.glob('experiments/*/.*.lock')),'unresolved experiment lock')
    require(not list(ROOT.glob('experiments/*/.AUTONOMOUS_OWNER.guard')),'another execution owner')
    import psutil
    others=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        if p.pid!=os.getpid() and 'python' in (p.info['name'] or '').lower():
            others.append(p.info)
    require(not others,'other Python task running; no concurrent start')
    ram=psutil.virtual_memory().available
    disk=shutil.disk_usage(ROOT).free
    require(ram>=4*1024**3 and disk>=8*1024**3,'insufficient resource headroom')
    data=inspect_inputs(frozen=False)
    from adapter_data import FrozenCache
    cache=FrozenCache(CACHE/'trace',data['split']['fit_ids'])
    cache.verify()
    tested=subprocess.run([sys.executable,'-B',str(STUDY/'scripts/test_targets.py')],capture_output=True,text=True,encoding='utf8',timeout=180)
    require(tested.returncode==0,'behavior tests failed: '+tested.stdout+tested.stderr)
    write_new(STUDY/'INPUT_EXPECTATIONS.json',dict(files=data['inputs_sha256'],
        external_evaluator_sources=data['external_evaluator_sources'],environment=data['environment']))
    write_new(STUDY/'PREPARATION.json',dict(status='passed',at=now(),host=socket.gethostname(),
        other_python_processes=others,free_ram_bytes=ram,free_disk_bytes=disk,
        native_argument_differences=data['native_argument_differences'],cache_files_checked=2000,
        tests_stdout=tested.stdout,tests_stderr=tested.stderr,tests_exit_code=tested.returncode,
        formal_teacher_calls=0,student_training_steps=0,base_model_forward_calls=0))
    frozen=dict(frozen_at=now(),files={p:sha256(ROOT/p) for p in CODE},
        input_manifest_sha256=sha256(STUDY/'INPUT_EXPECTATIONS.json'),preparation_sha256=sha256(STUDY/'PREPARATION.json'))
    write_new(STUDY/'FREEZE_MANIFEST.json',frozen)
    study=read_json(STUDY/'study.json')
    study.update(plan_status='frozen',frozen_at=frozen['frozen_at'],protocol_sha256=sha256(STUDY/'PROTOCOL.md'),preparation_status='passed')
    save(STUDY/'study.json',study)
    print('AMT_FROZEN',frozen['frozen_at'],len(CODE),'sources',len(data['inputs_sha256']),'inputs',flush=True)

if __name__=='__main__':
    main()
