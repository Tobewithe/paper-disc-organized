"""Verify available immutable sources and resources, then seal the new protocol."""
from common import *
import os
import psutil
import socket
import subprocess

def main():
    require(socket.gethostname().upper()=='YANGLEI','authorized laptop only')
    require(not list(ROOT.glob('experiments/*/.*.lock')) and not list(ROOT.glob('experiments/*/.AUTONOMOUS_OWNER.guard')),'unresolved execution owner')
    others=[p.info for p in psutil.process_iter(['pid','name','cmdline']) if p.info['pid']!=os.getpid() and 'python' in (p.info['name'] or '').lower()]
    require(not others,'another Python task exists: '+str(others))
    data=inspect_inputs(False)
    trace=read_json(CACHE/'trace/TRACE_MANIFEST.json')
    records=[r for r in trace['images'] if r['image_id'] in set(data['split']['fit_ids'])]
    require(len(records)==2000,'missing fit cache')
    for row in records:
        require(sha256(CACHE/'trace'/row['cache'])==row['cache_sha256'],'cache bytes drift')
    ram=psutil.virtual_memory().available
    disk=shutil.disk_usage(ROOT).free
    require(ram>=3*1024**3 and disk>=10*1024**3,'insufficient resources')
    result=subprocess.run([sys.executable,'-B',str(STUDY/'scripts/test_statistics.py')],capture_output=True,text=True)
    require(result.returncode==0,'statistic fixture tests failed: '+result.stdout+result.stderr)
    write_new(STUDY/'INPUT_EXPECTATIONS.json',dict(files=data['files'],external_evaluator_sources=data['external_sources'],matched_original_sources=data['matched_original_sources'],verified_source_artifacts=data['verified_source_artifacts']))
    write_new(STUDY/'PREPARATION.json',dict(status='passed',at=now(),host=socket.gethostname(),environment=data['environment'],fit_images=2000,dev_images=512,cache_files_checked=2000,available_ram_bytes=ram,free_disk_bytes=disk,other_python_processes=others,tests_stdout=result.stdout,tests_stderr=result.stderr,tests_exit_code=result.returncode,new_training_steps=0,new_checkpoint_evaluation_performed=False))
    frozen=dict(frozen_at=now(),files={rel:sha256(ROOT/rel) for rel in CODE},input_manifest_sha256=sha256(STUDY/'INPUT_EXPECTATIONS.json'))
    write_new(STUDY/'FREEZE_MANIFEST.json',frozen)
    study=read_json(STUDY/'study.json')
    study.update(plan_status='frozen',frozen_at=frozen['frozen_at'],protocol_sha256=sha256(STUDY/'PROTOCOL.md'),preparation_status='passed')
    save(STUDY/'study.json',study)
    print(json.dumps(dict(status='frozen',files=len(CODE),input_files=len(data['files']),frozen_at=frozen['frozen_at'])),flush=True)

if __name__=='__main__':
    main()
