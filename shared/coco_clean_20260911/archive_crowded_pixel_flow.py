"""S050 archive, candidate inventory and status update. Does not launch experiments."""
import csv,json,hashlib,shutil,re
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent.parent;OUT=BASE/'diagnostics/crowded_pixel_flow_20260912';SRC=BASE/'diagnostics/crowded_failure_branch_20260912'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
receipt=json.loads((OUT/'COMPLETE.json').read_text())
for name,h in receipt['hashes'].items():assert sha(OUT/name)==h,name
exclude={r['image_id'] for r in json.loads((SRC/'manifest.json').read_text())['pairs']}
pool=list(csv.DictReader((BASE/'diagnostics/mask_geometry_failure_census_20260912/CROWDED_SUPPORTED_MASK_FAILURE_POOL.csv').open(encoding='utf8')))
remaining=[r for r in pool if int(r['image_id']) not in exclude]
with (OUT/'UNUSED_FAILURE_SUBTYPE_CANDIDATES.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=pool[0].keys());w.writeheader();w.writerows(remaining)
counts={}
for r in remaining:
    key=f"{r['dominant_error']}/{r['area_bin']}";counts[key]=counts.get(key,0)+1
dump(OUT/'NEXT_CANDIDATE_INVENTORY.json',dict(status='INVENTORY_ONLY_NOT_NEW_EXPERIMENT',excluded_images=sorted(exclude),targets=len(remaining),images=len({r['image_id'] for r in remaining}),counts=counts,scope='Remaining original S048 val pool; S049 image exclusion only, not globally unseen, unselected for editability or predicted treatment response. No fixed next experiment quota or launch.'))
for n in ['plot_crowded_pixel_flow.py','archive_crowded_pixel_flow.py']:shutil.copy2(BASE/n,OUT/n)
dump(OUT/'ANALYSIS_COMPLETE.json',dict(status='COMPLETE',original_receipt=sha(OUT/'COMPLETE.json'),original_hashes_verified=len(receipt['hashes']),analysis_hashes={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in receipt['hashes'] and p.name not in {'COMPLETE.json','ANALYSIS_COMPLETE.json'}},note='Same S049 output reuse; no training/model forward. Subgroups exploratory.'))
status='2026-09-12 S050 COMPLETE：复用S049全部63图c/P，25.516秒解码2268输出，无新模型前向/训练。32主失败中29小目标、30背景误报/2自身遗漏、0同类邻居主导，故不代表粘连失败池。单换原型邻居相对原完整IoU纹理/颜色−0.054/−0.132点，背景控制−1.455/−2.147点：此前相对优势主要含对照损伤。仍有局部临邻自身边界净补回+0.569[0.198,1.005]/+0.715[0.306,1.179]自身面积点，但完整IoU未改善。两填充均相对控制>1点原型12例、均<−1点9例，保留正反例。下一项须按同类误报/自身遗漏及大小固定组成，候选清单已存但未启动新输入实验；不再重复同一四格。报告[PIXEL_FLOW_RESULTS_20260912.md](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/PIXEL_FLOW_RESULTS_20260912.md)。事实记录至S050，所有本轮进程完成，无自动化修改。'
for rel in ['refine-logs/EXPERIMENT_TRACKER.md','refine-logs/coco-structure/EXPERIMENT_TRACKER.md']:
    p=ROOT/rel;s=p.read_text(encoding='utf8')
    if status not in s:
        head,rest=s.split('\n',1);p.write_text(head+'\n\n'+status+'\n'+rest,encoding='utf8')
p=ROOT/'RESEARCH_FACTS.md';s=p.read_text(encoding='utf8');s=s.replace('旧 COCO 审计及 S000—S049','旧 COCO 审计及 S000—S050')
fact='| F16 | S050复用S049发现：32主失败中29小目标、30背景误报、2自身遗漏，未含同类邻居误报主导者。单换原型邻居−原完整IoU仅−0.054/−0.132点，而背景−原−1.455/−2.147点；原型相对优势包含控制损伤。 | 临邻自身边界净补回+0.569[0.198,1.005]/+0.715[0.306,1.179]自身面积点，支持局部敏感性，尚无完整恢复。不能把队列结论外推90个同类泄漏池，下一步须固定失败/大小组成。 | [S050](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/PIXEL_FLOW_RESULTS_20260912.md) |\n\n'
if '| F16 |' not in s:s=s.replace('数值约定：',fact+'数值约定：')
row='| S050 | 复用S04963图，25.516秒/2268解码，全部完整IoU重放、七区域及代数分解通过。32失败的单原型路径局部临邻边界补回0.569/0.715点，但完整IoU未改善；背景对照损伤解释主要相对优势。原型两填充相对有利12、不利9、其他11。 | 同源探索性像素分解，无新模型前向/训练。29小目标、30背景误报，缺失同类泄漏主导类；不能再将其当全部拥挤失败代表。附正例/控制损伤例/反例和未用图片候选清单。[报告](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/PIXEL_FLOW_RESULTS_20260912.md) |\n\n'
if '| S050 |' not in s:s=s.replace('部分报告合并了多个实验，',row+'部分报告合并了多个实验，')
s=s.replace('复用已存区域/张量，分清自身遗漏与邻居/背景误报的逐图响应；亚型发现须在新冻结样本确认，不按这32例最有利响应改主结果。','S050已完成区域复算：主队列缺失同类泄漏类且偏小目标；下一次按原错误类型/大小固定样本组成，局部边界恢复命题在新冻结实例确认，不按这32例有利响应改主结果。')
s=s.replace('更上游模块特征注入/逆向恢复尚未执行','S050像素流分解已完成；更上游模块特征注入/逆向恢复尚未执行')
update='- 2026-09-12：S050复用张量做像素流，记录队列组成偏移、背景控制损伤及真实局部边界补回。全量/逐实例负例不删，后续候选仅存清单、未启动新输入实验。\n'
if update not in s:s+='\n'+update
p.write_text(s,encoding='utf8')
# Keep S049 interpretation connected to its now-completed spatial followup.
p=ROOT/'refine-logs/coco-structure/CROWDED_BRANCH_RESULTS_20260912.md';s=p.read_text(encoding='utf8')
note='后续更正与细分：S050已完成像素流复算。只换原型的邻居−原图完整IoU接近零，相对背景优势主要含背景对照损伤；队列为29小目标、30背景误报/2自身遗漏，不含同类泄漏主导类。见[像素流报告](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/PIXEL_FLOW_RESULTS_20260912.md)。'
if note not in s:
    head,rest=s.split('\n',1);p.write_text(head+'\n\n'+note+'\n'+rest,encoding='utf8')
entries=[ROOT/'RESEARCH_FACTS.md',ROOT/'refine-logs/coco-structure/PIXEL_FLOW_RESULTS_20260912.md',ROOT/'refine-logs/coco-structure/CROWDED_BRANCH_RESULTS_20260912.md',ROOT/'refine-logs/EXPERIMENT_TRACKER.md',ROOT/'refine-logs/coco-structure/EXPERIMENT_TRACKER.md']
entries +=[BASE/n for n in ['crowded_pixel_flow_probe.py','summarize_crowded_pixel_flow.py','plot_crowded_pixel_flow.py','archive_crowded_pixel_flow.py']]
entries +=[p for p in sorted(OUT.iterdir()) if p.is_file()]
manifest=ROOT/'MANIFEST.md';s=manifest.read_text(encoding='utf8')
s+='\n\nS050 saved-output pixel flow (per-image RLE files indexed in COMPLETE.json):\n\n| Timestamp | Skill | File | Stage | Description |\n| --- | --- | --- | --- | --- |\n'
for p in entries:s+=f'| 2026-09-12 | experiment-plan | {p.relative_to(ROOT).as_posix()} | implementation | S050 same-output spatial analysis; no new training/inference |\n'
manifest.write_text(s,encoding='utf8')
for p in [ROOT/'refine-logs/coco-structure/PIXEL_FLOW_RESULTS_20260912.md',ROOT/'refine-logs/coco-structure/CROWDED_BRANCH_RESULTS_20260912.md']:
    for link in re.findall(r'\]\((C:/[^)]+)\)',p.read_text(encoding='utf8')):assert Path(link).exists(),link
print('S050 archive and report links passed; next candidates inventory only:',len(remaining),counts)
