"""Engineering launch recovery only; execute the unchanged frozen scientific runner."""
from pathlib import Path
import argparse,sys
STUDY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(STUDY/'scripts'))
from common import MarginRun,read_json,save,sha256,require
from run_audit import execute

p=argparse.ArgumentParser();p.add_argument('--scope',choices=['smoke','full'],required=True);p.add_argument('--parent-run',required=True);p.add_argument('--retry-of');a=p.parse_args()
parent=STUDY/'runs'/a.parent_run
if a.scope=='full':
    require(read_json(parent/'run.json')['execution_status']=='completed','smoke incomplete')
    require(read_json(parent/'INDEPENDENT_LAPTOP.json')['status']=='passed','smoke verification')
    previous=list((STUDY/'runs').glob('RUN_RSM_FULL_*'))
    if previous:
        require(a.retry_of and (STUDY/'runs'/a.retry_of) in previous,'explicit failed-run recovery required')
        require(all(read_json(p/'run.json')['execution_status']=='error' for p in previous),'completed/running full must not be repeated')
run=MarginRun(a.scope,parent)
run.meta.update(engineering_launch_wrapper_sha256=sha256(Path(__file__)),launch_mode='attached_SSH',retry_of=a.retry_of or (a.parent_run if a.scope=='smoke' else None))
study=read_json(STUDY/'study.json')
try:
    with run:
        study.update(execution_status='running',active_run_id=run.run_id,current_phase=a.scope,artifact_status='partial')
        save(STUDY/'study.json',study);execute(run)
finally:
    if (run.path/'run.json').exists():
        m=read_json(run.path/'run.json');study.update(execution_status=m['execution_status'],scientific_status=m['scientific_status'],active_run_id=None,
          latest_run_id=run.run_id,artifact_status=m['artifact_status'],current_phase='full_ready' if a.scope=='smoke' and m['execution_status']=='completed' else 'awaiting_transfer' if m['execution_status']=='completed' else 'engineering_error_retained')
        save(STUDY/'study.json',study)
