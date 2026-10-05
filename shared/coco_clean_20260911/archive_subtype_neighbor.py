"""Register S051 completed evidence and update persistent research facts."""
import json,hashlib,re,shutil
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent.parent;SRC=BASE/'diagnostics/subtype_neighbor_20260912';REG=SRC/'regional_followup';REPORT=ROOT/'refine-logs/coco-structure/SUBTYPE_NEIGHBOR_RESULTS_20260912.md'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
for n in ['summarize_subtype_neighbor.py','subtype_region_followup.py','summarize_subtype_regions.py','report_subtype_neighbor.py','archive_subtype_neighbor.py']:
    shutil.copy2(BASE/n,SRC/n)
for directory in [SRC,REG]:
    receipt=json.loads((directory/'COMPLETE.json').read_text())
    for name,h in receipt['hashes'].items():assert sha(directory/name)==h,('Changed original',name)
    extra={str(p.relative_to(directory)):sha(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name not in receipt['hashes'] and p.name not in {'COMPLETE.json','ANALYSIS_COMPLETE.json'}}
    if directory==SRC:extra['regional_followup/ANALYSIS.json']=sha(REG/'ANALYSIS.json')
    dump(directory/'ANALYSIS_COMPLETE.json',dict(status='COMPLETE',original_receipt_sha256=sha(directory/'COMPLETE.json'),original_hashes_verified=len(receipt['hashes']),analysis_hashes=extra,note='Same artifact verification only; not independent GPU replication. Followup regions exploratory after aggregate; no additional inference.'))
for link in re.findall(r'\]\((C:/[^)]+)\)',REPORT.read_text(encoding='utf8')):assert Path(link).exists(),link
protocol=json.loads((SRC/'protocol.json').read_text());assert sha(Path(protocol['pool']))==protocol['pool_sha256']
status='2026-09-12 S051 COMPLETE：错误类型×大小配额的局部邻居干预完成44/44（同类误报15、自身遗漏15、背景14，小中大不替补），107.187秒；原型/系数输出空间复算1584臂、14.594秒。两填充各3seed正常流程稳定Mask75恢复同类3例、自身2例。自身遗漏只换原型相对原IoU纹理+2.541[0.650,4.777]、颜色+1.347[−2.115,4.510]；临邻自身净补回+2.259[0.404,5.372]/+2.783[0.636,6.031]自身面积点，但未编辑邻居错误亦增加。系数未单独解释主要个体恢复，原型主因/容量不足未确认；背景控制损伤、填充敏感性、正反例保留。下一可判别项为原输入下预定层特征注入与逆向恢复，本轮未启动；不重复S051、不训练CCL。报告[SUBTYPE_NEIGHBOR_RESULTS_20260912.md](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SUBTYPE_NEIGHBOR_RESULTS_20260912.md)。事实至S051；全部进程完成，无定时任务修改。'
for rel in ['refine-logs/EXPERIMENT_TRACKER.md','refine-logs/coco-structure/EXPERIMENT_TRACKER.md']:
    p=ROOT/rel;s=p.read_text(encoding='utf8')
    if status not in s:
        header,rest=s.split('\n',1);p.write_text(header+'\n\n'+status+'\n'+rest,encoding='utf8')
p=ROOT/'RESEARCH_FACTS.md';s=p.read_text(encoding='utf8').replace('旧 COCO 审计及 S000—S050','旧 COCO 审计及 S000—S051')
s=s.replace('S049 严格失败队列的系数/原型输出对照。','S049 严格失败队列的系数/原型输出对照、S050空间复算及S051错误类型×大小分层局部邻居输入干预。')
fact='| F17 | S051明确分型/大小后44例完成。同类误报15、自身遗漏15，各6小6中3大；背景14。正常流程两填充各3seed均Mask75者同类3例、自身2例。自身遗漏只换原型实际IoU+2.541[0.650,4.777]/+1.347[−2.115,4.510]点，临邻自身净补回2.259/2.783点。 | 有具体框好mask差实例可经上下文变化恢复；系数路径不能单独解释它们。仍有邻居错误取舍、背景控制损伤和反例，整组尚未跨填充确认完整IoU优势。不证明原型容量不足或通用方法有效。 | [S051](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SUBTYPE_NEIGHBOR_RESULTS_20260912.md) |\n'
if '| F17 |' not in s:
    pos=s.index('数值约定：');s=s[:pos].rstrip()+'\n'+fact+'\n'+s[pos:]
row='| S051 | 同类误报/自身遗漏/背景15/15/14例，错误×大小固定配额，局部邻居16输入像素半径，44/44精确crop/source/input通过。单原型自身遗漏纹理实际IoU+2.541[0.650,4.777]、颜色+1.347区间跨零，目标临邻净恢复两填充排零但邻居误报增加。正常流程6输入全Mask75同类3、自身2。 | 新增有原图绝对恢复的机制个体；不得以相对背景+9.635点全当真实恢复。局部剂量不同于S049整邻居；666余文件哈希只是产物核验。含完整类型×大小、正例和明显受损例。[报告](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SUBTYPE_NEIGHBOR_RESULTS_20260912.md) |\n'
row=row.replace('666余文件','668个主产物')
if '| S051 |' not in s:
    pos=s.index('部分报告合并了多个实验，');s=s[:pos].rstrip()+'\n'+row+'\n'+s[pos:]
s=s.replace('S050已完成区域复算：主队列缺失同类泄漏类且偏小目标；下一次按原错误类型/大小固定样本组成，局部边界恢复命题在新冻结实例确认，不按这32例有利响应改主结果。','S051已完成失败类型×大小分层局部输入：同类3、自身2例在6填充均正常Mask75恢复，完整组内仍未跨填充确认。下一项应检验预定层特征注入/逆向恢复，不再重复选样或把局部变化当普遍主因。')
s=s.replace('当前无需再验证“有没有框好mask差”；若具体亚型显示受控区域修复，再预定模块/逆向恢复实验。不能由张量变化幅度直接判根因。','S051已有固定原框的实际个体恢复。预定上游特征位置/层与背景控制、逆向恢复，检查修复是否可传递或撤回；不能由输出变化幅度直接判模块根因。')
s=s.replace('S050像素流分解已完成；更上游模块特征注入/逆向恢复尚未执行','S050像素流分解及S051分型局部邻居输入干预已完成；更上游模块特征注入/逆向恢复尚未执行')
s=s.replace('不得按旧计划重复S047—S049','不得按旧计划重复S047—S051')
# Keep Markdown tables contiguous after append-only entries.
s=s.replace(' |\n\n| F16',' |\n| F16').replace(' |\n\n| F17',' |\n| F17').replace(' |\n\n| S050',' |\n| S050').replace(' |\n\n| S051',' |\n| S051')
update='- 2026-09-12：S051固定失败类型×大小、局部邻居输入44例全部完成；记录实际个体恢复、原型输出线索、未编辑邻居副作用及反例。精确正常流程/固定框结果分开，层级注入尚未开始。\n'
if update not in s:s+='\n'+update
p.write_text(s,encoding='utf8')
current=ROOT/'refine-logs/coco-structure/SUBTYPE_NEIGHBOR_PLAN.md';launch=ROOT/'refine-logs/coco-structure/SUBTYPE_NEIGHBOR_PLAN_20260912_215811.md'
plan=launch.read_text(encoding='utf8');head,rest=plan.split('\n',1);current.write_text(head+'\n\n当前状态：S051已完成44/44及空间复算，详见SUBTYPE_NEIGHBOR_RESULTS_20260912.md。保留下方启动设计；层级特征注入尚未执行，不重复运行本轮。\n'+rest,encoding='utf8')
entries=[ROOT/'RESEARCH_FACTS.md',REPORT,current,launch,ROOT/'refine-logs/EXPERIMENT_TRACKER.md',ROOT/'refine-logs/coco-structure/EXPERIMENT_TRACKER.md']
entries +=[BASE/n for n in ['subtype_neighbor_probe.py','summarize_subtype_neighbor.py','subtype_region_followup.py','summarize_subtype_regions.py','report_subtype_neighbor.py','archive_subtype_neighbor.py']]
entries +=[p for directory in [SRC,REG] for p in sorted(directory.iterdir()) if p.is_file()]
manifest=ROOT/'MANIFEST.md';s=manifest.read_text(encoding='utf8');s+='\n\nS051 completed local-neighbor subtype experiment (raw per-image outputs indexed in COMPLETE.json):\n\n| Timestamp | Skill | File | Stage | Description |\n| --- | --- | --- | --- | --- |\n'
for p in entries:s+=f'| 2026-09-12 | run-experiment | {p.relative_to(ROOT).as_posix()} | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |\n'
manifest.write_text(s,encoding='utf8')
print('S051 receipts, immutable source/pool, report links, facts, trackers and manifest complete')
