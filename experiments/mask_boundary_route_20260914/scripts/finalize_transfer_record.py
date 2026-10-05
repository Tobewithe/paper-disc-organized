"""Finite follow-through: summarize an existing evaluation and append its evidence once."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run-id',required=True);a=p.parse_args()
root=Path(a.project);study=root/'experiments/mask_boundary_route_20260914';scripts=study/'scripts';runs=study/'runs'
protocol=json.loads((study/'PROTOCOL_TRANSFER_V5.json').read_text(encoding='utf-8'))
evaluation=runs/protocol['run_ids']['transfer_eval'];out=runs/a.run_id
deadline=time.monotonic()+4*3600
while time.monotonic()<deadline:
    path=evaluation/'run.json'
    if path.exists():
        try:run=json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:time.sleep(5);continue
        if run.get('status')=='completed' and (evaluation/'SUMMARY.json').exists():break
        if run.get('status') in ('failed','cancelled','interrupted'):raise RuntimeError('Evaluation did not complete: '+run['status'])
    time.sleep(30)
else:raise TimeoutError('Evaluation not completed within four hours')
command=[sys.executable,str(root/'shared/tools/research_runner/runner.py'),'--study',protocol['study_id'],'--run-id',a.run_id,
    '--output',str(out),'--cwd',str(root),'--input',str(evaluation/'SUMMARY.json'),
    '--snapshot',str(scripts/'summarize_transfer_results.py'),'--snapshot',str(scripts/'finalize_transfer_record.py'),
    '--expect',str(out/'SUMMARY.json'),'--expect',str(out/'RESULTS.md'),'--metrics',str(out/'SUMMARY.json'),
    '--',sys.executable,'-u',str(scripts/'summarize_transfer_results.py'),'--project',str(root),'--output',str(out)]
subprocess.run(command,cwd=root,check=True)
cli=['C:/Program Files/PowerShell/7/pwsh.exe','-NoProfile','-File','C:/Dpan/codexproject/Aggregation Workbench/researchctl.ps1','--project','paper-disc']
relative='experiments/mask_boundary_route_20260914/REPORT.md';tag='FIND_frozen_m_to_s_transfer_20260915'
for attempt in range(3):
    read=subprocess.run([*cli,'source',relative],cwd=root,capture_output=True,check=True)
    current=json.loads(read.stdout.decode('utf-8-sig'))
    if tag in current['content']:break
    note=dict(id=tag,kind='finding',title='YOLO26m拟合的校准器在YOLO26s上冻结迁移的完整结果',verdict='observed',coverage='declared',
        scope=dict(dataset='COCO val2017 previously explored4500 images',target='pretrainedYOLO26s-seg fixed predictions',metric='COCO segm AP and fixed-slot R75',protocol='PROTOCOL_TRANSFER_V5.json'),
        limitations=['same previously explored validation images','single target checkpoint','no AP confidence interval','no retraining of gates'],
        evidence=[dict(relation='derived_from',path=(evaluation/'SUMMARY.json').relative_to(root).as_posix(),revision=hashlib.sha256((evaluation/'SUMMARY.json').read_bytes()).hexdigest())])
    text='\n\n## 冻结跨权重结果（V5完成）\n\n<!-- research-note: '+json.dumps(note,ensure_ascii=False)+' -->\n\n'
    text+=(out/'RESULTS.md').read_text(encoding='utf-8').replace('# YOLO26m → YOLO26s 冻结迁移结果','### YOLO26m → YOLO26s')
    text+='\n<!-- /research-note -->\n'
    proposed=out/'report_proposed.txt';proposed.write_text(current['content']+text,encoding='utf-8')
    result=subprocess.run([*cli,'record',relative,'--input',str(proposed),'--expected-revision',current['revision']],cwd=root,capture_output=True)
    if result.returncode==0:break
    if result.returncode!=3:raise RuntimeError(result.stderr.decode(errors='replace'))
else:raise RuntimeError('Concurrent report revision kept changing')
for operation in ('sync','validate'):
    result=subprocess.run([*cli,operation],cwd=root,capture_output=True)
    (out/(operation+'.json')).write_bytes(result.stdout)
    (out/(operation+'_stderr.log')).write_bytes(result.stderr)
print('TRANSFER_REPORT_RECORDED',flush=True)
