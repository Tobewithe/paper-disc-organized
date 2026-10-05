"""Archive completed S048/S049 analyses and update human-readable entry points."""
import hashlib,json,shutil,re
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=BASE.parent.parent
D=BASE/'diagnostics'
C=D/'mask_geometry_failure_census_20260912'
B=D/'crowded_failure_branch_20260912'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

for directory in [C,B]:
    receipt=json.loads((directory/'COMPLETE.json').read_text())
    for name,expected in receipt['hashes'].items():
        if sha(directory/name)!=expected:raise RuntimeError(f'Changed original artifact: {name}')
    print(directory.name,'original hashes verified',len(receipt['hashes']))

protocol=json.loads((B/'protocol.json').read_text())
for name,expected in protocol['pools'].items():
    assert sha(C/name)==expected,('Selection pool changed',name)
assert sha(C/'instances.csv')==protocol['census_sha256']
copies={C:['summarize_mask_geometry_census.py'],B:['summarize_crowded_branches.py','plot_geometry_branch_results.py','archive_geometry_branch_results.py']}
for directory,names in copies.items():
    for name in names:shutil.copy2(BASE/name,directory/name)
    originals=json.loads((directory/'COMPLETE.json').read_text())['hashes']
    analysis_files=[p for p in directory.iterdir() if p.is_file() and p.name not in originals and p.name not in {'COMPLETE.json','ANALYSIS_COMPLETE.json'}]
    dump(directory/'ANALYSIS_COMPLETE.json',dict(status='COMPLETE',original_hashes_verified=len(originals),original_receipt_sha256=sha(directory/'COMPLETE.json'),pool_hashes_unchanged=True,
        analysis_hashes={p.name:sha(p) for p in sorted(analysis_files)},
        note='Same-artifact verification, not an independent model run. S048 undefined geometry uses instances_classified.csv. Plotting needed conda Library/bin on DLL path; no package changes.'))

status='''2026-09-12 S048/S049 COMPLETE：全COCO val5000图/36335普通GT完成同一预测框与mask失败清点；box好mask差5575，进一步初筛支持充分且没有其他保留好mask者4260，真实同类边界拥挤低重叠失败池511/397图、Box90为102。严格输入/输出对照64入选63可评，同类失败32例：固定原框，邻居减距离匹配背景控制，只换系数纹理/颜色IoU +0.305/−0.179点，只换原型+1.402/+2.015，各完整IoU区间与两路径差区间均跨零，尚不支持系数或原型为主因。原型路径增加自身覆盖，未稳定减少邻居/背景错误。全部进程完成，未启动训练或更上游模块注入；当前应复用保存c/P及像素区域定位具体亚型，勿重复已完成队列。报告[全量失败清点](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/MASK_GEOMETRY_FAILURE_RESULTS_20260912.md)、[路径对照](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/CROWDED_BRANCH_RESULTS_20260912.md)。事实记录已更新至S049。'''
for rel in ['refine-logs/EXPERIMENT_TRACKER.md','refine-logs/coco-structure/EXPERIMENT_TRACKER.md']:
    p=ROOT/rel;s=p.read_text(encoding='utf-8');header,rest=s.split('\n',1)
    if status not in s:p.write_text(header+'\n\n'+status+'\n'+rest,encoding='utf-8')
p=ROOT/'refine-logs/coco-structure/MASK_GEOMETRY_FAILURE_PLAN.md'
s=p.read_text(encoding='utf-8')
note='执行状态（2026-09-12）：S048全量清点和S049严格失败输入/输出对照均已完成。结果见同目录MASK_GEOMETRY_FAILURE_RESULTS_20260912.md与CROWDED_BRANCH_RESULTS_20260912.md。保留下方启动设计及带时间戳版本；后续模块注入尚未执行，不按旧计划重复运行。'
if note not in s:
    header,rest=s.split('\n',1);p.write_text(header+'\n\n'+note+'\n'+rest,encoding='utf-8')

reports=[ROOT/'refine-logs/coco-structure'/name for name in ['MASK_GEOMETRY_FAILURE_RESULTS_20260912.md','CROWDED_BRANCH_RESULTS_20260912.md']]
for p in reports:
    for target in re.findall(r'\]\((C:/[^)]+)\)',p.read_text(encoding='utf-8')):
        assert Path(target).exists(),('Broken link',target)
print('Report links verified')

manifest=ROOT/'MANIFEST.md';s=manifest.read_text(encoding='utf-8')
entries=[ROOT/'RESEARCH_FACTS.md',*reports,
    ROOT/'refine-logs/coco-structure/MASK_GEOMETRY_FAILURE_PLAN.md',ROOT/'refine-logs/coco-structure/MASK_GEOMETRY_FAILURE_PLAN_20260912_205700.md',
    ROOT/'refine-logs/EXPERIMENT_TRACKER.md',ROOT/'refine-logs/coco-structure/EXPERIMENT_TRACKER.md']
entries += [BASE/n for n in ['mask_geometry_failure_census.py','summarize_mask_geometry_census.py','crowded_failure_branch_probe.py','summarize_crowded_branches.py','plot_geometry_branch_results.py','archive_geometry_branch_results.py']]
for directory in [C,B]:
    entries.extend(p for p in sorted(directory.iterdir()) if p.is_file())
rows=['\n\nS048/S049 completed outputs (per-image raw artifacts and hashes are indexed by each COMPLETE.json):\n', '| Timestamp | Skill | File | Stage | Description |','| --- | --- | --- | --- | --- |']
for p in entries:
    rel=p.relative_to(ROOT).as_posix()
    if f'| {rel} |' not in s:rows.append(f'| 2026-09-12 | experiment-plan | {rel} | implementation | S048/S049 completed; scope and limitations in reports |')
manifest.write_text(s+'\n'.join(rows)+'\n',encoding='utf-8')
print('Facts, trackers, current plan and manifest registered; no training or automation changes')
