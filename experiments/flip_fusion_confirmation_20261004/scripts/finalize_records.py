"""Write the completed research interpretation from verified returned JSON."""
from pathlib import Path
import json,hashlib,tempfile,subprocess,os
P=Path(r'C:\Dpan\codexproject\paper-disc-organized')
R=Path(__file__).resolve().parents[1]
CLI=Path(r'C:\Dpan\codexproject\Aggregation Workbench\researchctl.ps1')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def record(path,text,new=False):
    revision='new' if new else sha(path)
    if new and path.exists():raise RuntimeError('Preserve existing '+str(path))
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',suffix=path.suffix,delete=False) as f:
        f.write(text);name=f.name
    try:
        done=subprocess.run(['pwsh','-NoProfile','-File',str(CLI),'--project','paper-disc','record',path.relative_to(P).as_posix(),'--input',name,'--expected-revision',revision],capture_output=True,text=True,encoding='utf-8',errors='replace',creationflags=0x08000000)
        if done.returncode:raise RuntimeError(done.stdout+done.stderr)
        print(done.stdout,flush=True)
    finally:os.unlink(name)

def main():
    ids=read(R/'RUN_IDS.json');run=R/'runs'/ids['runs']['evaluation']
    summary=read(run/'SUMMARY.json');verification=read(R/'POST_RUN_AUDIT.json')
    assert verification['passed'] and read(run/'COMPLETE.json')['completed']
    assert not summary['decisions']['primary_and_overall_guard_passed']
    collection='COLLECT_11a9ac4f88d44c018a952d99813ceab2'
    receipt=read(R/'_transfer'/collection/'COLLECTION_MANIFEST.json')
    pop=summary['population'];T=summary['tables']
    c=lambda g:T[g]['comparisons']['MIX_minus_A']
    pp=lambda v:f'{v*100:+.4f}' if v is not None else '未定义'
    interval=lambda m:f"{pp(m['delta'])} [{pp(m['ci95'][0])}, {pp(m['ci95'][1])}]"
    lines=['# 冻结翻转融合确认：主要IoU信号未确认，保留小幅排序与候选收益','',
        '**结论：当前固定MIX没有通过预定确认条件，结束这一配置，不追加图片或调alpha/匹配/投影。** 新清单的框好mask差组macro IoU +0.3330pp，95% CI [−0.0225,+0.7117]；全体仅+0.0077pp [−0.0541,+0.0664]。全体146修复、133损伤，净Mask75区间仍跨零。目标组候选平均IoU +0.2707pp [+0.0758,+0.4796]及AUC有次要正信号，伴随原成功实例损伤及全体coverage下降；不能把它写成已兑现的整体掩码提升，也不能说双视图完全没有有用信息。','',
        '## 已完成的固定比较','',
        f"Study `{ids['study_id']}`。smoke `{ids['runs']['smoke']}` 29.02秒通过；正式 `{ids['runs']['evaluation']}` 690.18秒，均在协议预算内。2026-10-03T20:25:12.768689Z远端pipeline completed，真实PID已退出，GPU无研究进程。两Run共1607文件已SHA核验回传，集合 `{collection}`。",'',
        f"前瞻固定1536张train2017，{pop['effective_images']}张有官方one-to-one候选、{pop['candidates']}个候选；11张无正样本按原清单记录，不补样。只运行原模型A和冻结普通MIX，无训练/系数求解/新阈值。原图与整canvas水平翻转各自产生P和c，匹配后逆翻完整logit，以固定0.5融合，再用原框、原阈值及原逆变换。最终掩码有额外视图计算成本，本轮不是推理延迟基准。",'',
        '## 主要结果','',
        '下表为图片macro变化，单位均为百分点；区间为5000次按图片成簇配对bootstrap。唯一主要指标是预定框好mask差组IoU，其余指标/分层为次要描述。','',
        '| 对象 | 图片／候选 | IoU差 [95% CI] | 候选均值IoU差 [95% CI] | Mask75修复／损伤 |',
        '|---|---:|---:|---:|---:|']
    labels={'all':'全体','box_good_mask_bad':'A框好mask差','original_success':'A原成功','original_failure':'A原失败','retained_exactclass_correct':'原top300/conf且类别正确的TAL子集','P3':'P3','P4':'P4','P5':'P5'}
    for g,label in labels.items():
        q=c(g);cross=q['crossings']
        lines.append(f"| {label} | {T[g]['images']}／{T[g]['candidates']} | {interval(q['iou']['image_macro'])} | {interval(q['iou']['candidate'])} | {cross['repair']}／{cross['damage']} |")
    lines+=['','| 对象 | Coverage差 [95% CI] | 框内AUC差 [95% CI] | 框内FPR差 [95% CI] | 净Mask75 macro差 [95% CI] |',
        '|---|---:|---:|---:|---:|']
    for g in ('all','box_good_mask_bad','original_success','retained_exactclass_correct'):
        q=c(g);lines.append('| '+labels[g]+' | '+' | '.join(interval(q[m]['image_macro']) for m in ('coverage','auc','fpr','mask75'))+' |')
    lines+=['','## 按预定规则判断','',
        '- 目标组候选平均IoU +0.2707pp [+0.0758,+0.4796]是可保留的次指标改善；它与图片macro的权重不同，不能替代预定主要判定。原成功候选平均IoU −0.0434pp [−0.0867,−0.0017]的损伤也须并列报告。',
        '- 目标组点估计低于0.5pp，且区间跨零，因此本轮没有建立原0.548pp初筛信号的稳定复现。该区间也仍包含0.5pp，不能反过来声称已证明目标组效应必然小于0.5pp，或效应不存在。',
        '- 全体IoU区间下界−0.0541pp高于−0.1pp，总体损伤护栏通过；这不是总体收益通过。其上界仅+0.0664pp，整体0.2pp量级收益不受本轮支持。',
        '- 全体AUC +0.0726pp [+.0287,+.1151]、目标组AUC +0.2871pp [+.1322,+.4547]：冻结双视图确实保留可重复的像素排序信息。固定支持上的正斜率标量变换不能改变排序，因此这不完全等于统一阈值效应；它仍不定位原生系数失败的唯一根因。',
        '- 全体coverage −0.1636pp、原成功coverage −0.1966pp，区间均为负。FPR下降与更保守的二值mask相容，但AUC正信号说明不能只用保守化解释全部变化。收益和代价应并列保留。','',
        'A框好mask差组2158候选中，124个越过Mask75；但1069个IoU上升、1009个下降，其中603个上升超过1pp、502个下降超过1pp。该组的零Mask75损伤由分组定义决定，不是无误伤证据。全体6831个原成功候选中133个跌破Mask75；MIX产生2个空mask（A为0），全部保留在统计中。','',
        '## 是否与实际输出位置有关','',
        '10224个候选位于原head top300、分数>.001且输出类别与原预测argmax和GT类一致的子集。该子集IoU +0.0164pp [−0.0329,+0.0625]，132修复／119损伤，未建立可靠IoU/净Mask75提升。AUC改善仍在，coverage下降也仍在。','',
        '这只是官方监督候选与原输出的对应联系；retained在空mask过滤之前定义，未覆盖其他推理FP或无TAL对应GT。不能据此报告完整部署收益、AP或对所有实际输出的性能保证。评价GT只用于TAL冻结、分层、类别核验与指标，未进入MIX的选源或是否修正决策。','',
        '## 审计、独立性与数值范围','',
        '- 375个可解析历史身份来源，排除19294个trainID；从剩98993中固定SHA排序取1536。与当前10000fit+1000dev以及早期800fit+200dev均无重叠。',
        '- 只称有已知范围的“方法开发外确认”。官方预训练和旧本项目完整train2017训练已见这些图片；两个0字节旧清单及未索引历史保持未知，不称全项目全新盲测。取样前发现annotationID/count tuple误当imageID，错误版保留，最终修正改变25个计划ID，发生于任何新图推理之前。',
        '- 旧两图7候选的A/MIX五指标与历史误差为0；新图A逐候选连续logit解码与官方process_mask输入/原图二值mask逐像素差为0。原模型参数/BN、源P/c/box回放、身份唯一和完整队列检查通过。',
        f"- 本地只用标准库复核返回指标，{verification['checks']}项均值最大误差{verification['max_mean_error']:.3g}；没有本地模型计算。原匹配和指标依赖SHA与初筛一致。",
        '- 固定全部raw来源中有2次镜像fallback，4297个源分数<.001，208个源与原框IoU<.5，均按事先规则保留。源框IoU均值0.9146；不能把它当正确实例掩码对应的保证，不事后删低分源。','',
        '## 决策及后续边界','',
        '**停止当前冻结MIX作为有效掩码改进的推进；保留目标组候选平均IoU与AUC互补的有限改善及普通TTA参照。** 不追加图片、融合权重、匹配门槛、投影正则、gate或新的训练来挽救这份结果。PROJ此前相对MIX退化的停止条件保持。','',
        '本轮没有检验全体可能的多视图方法，也没有证明现有表示完全缺乏修正信息。若未来提出新方法，必须有独立机制和依据，不能把本轮改名或增加训练预算。普通融合不是新的系数生成机制，且本轮未取得实际量级的正常掩码确认收益，研究总Goal未达成。','',
        '## 记录入口','',
        f'- [逐候选原始记录](runs/{ids["runs"]["evaluation"]}/PER_CANDIDATE.jsonl)、[逐图记录](runs/{ids["runs"]["evaluation"]}/PER_IMAGE.jsonl)、[全部统计](runs/{ids["runs"]["evaluation"]}/SUMMARY.json)。',
        f'- [运行核验](runs/{ids["runs"]["evaluation"]}/AUDIT.json)、[本地独立复算](POST_RUN_AUDIT.json)、[数据暴露审计](EXPOSURE_AUDIT.json)、[功效规划](POWER_PLAN.json)。',
        f'- [协议](PROTOCOL.md)、[代码冻结](CODE_READY.json)、[回传清单](_transfer/{collection}/COLLECTION_MANIFEST.json)。',
        '- Run快照/输入hash/失败日志规则均保留；runner冻结的是列明依赖，不声称完整传递依赖图已知。','']
    record(R/'REPORT.md','\n'.join(lines),True)
    meta=read(R/'study.json');meta.update(status='completed',execution_status='completed',artifact_status='available',transfer_status='verified',report='REPORT.md',scientific_decision='stop_fixed_mix_mask_utility_not_confirmed_retain_auc_signal',finished_at='2026-10-03T20:25:12.768689Z',collection_id=collection,verified_files=1607)
    record(R/'study.json',json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    goal=P/'project/research_notes/PROTOTYPE_COEFFICIENT_ACTIVE_GOAL_20261003.md'
    text=goal.read_text(encoding='utf-8-sig')
    heading='## 最新收尾：冻结MIX确认未过，保留有限排序与候选收益（2026-10-04）'
    assert heading not in text
    entry='''## 最新收尾：冻结MIX确认未过，保留有限排序与候选收益（2026-10-04）

[flip_fusion_confirmation_20261004](../../experiments/flip_fusion_confirmation_20261004/REPORT.md)已2026-10-03T20:25:12.768689Z completed，两个Run共1607文件SHA核验回传，COLLECT_11a9ac4f88d44c018a952d99813ceab2。前瞻1536计划/1525有效train图、11034固定官方候选，正式690.18秒；方法开发外而非全项目或预训练未见。无训练、仅原A及原封不动MIX。

主要目标组817图2158候选macro IoU+.3330pp CI[−.0225,+.7117]，低于.5pp且CI跨零，未通过预定确认；全体+.0077pp CI[−.0541,+.0664]，146修复133损伤，净Mask75不确定。全体/目标AUC+.0726/+.2871pp区间正，保留可重复排序信息；目标组候选平均IoU+.2707pp CI[+.0758,+.4796]为正，但原成功候选IoU−.0434pp CI[−.0867,−.0017]为负，两者均为次指标且不能替换图片macro主判定。全体coverage−.1636pp、原成功coverage−.1966pp区间负，不能宣称已确认整体mask收益。实际保留且类正确TAL子集10224候选IoU+.0164pp仍不确定，不是完整部署/AP。

当前固定MIX的掩码改进推进结束，不补图/扫alpha/匹配/投影/gate/新训练挽救；PROJ保持停止。并非证明所有多视图无用或完全无信息，普通TTA不作新系数创新。旧2图A/MIX指标0误差、全部A逐像素正常解码一致、本地260均值复算max2.22e−16。当前无研究Run在跑，下一项尚未登记，总Goal未完成。下一轮需独立机制、真实历史/数学/代码/文献依据，不重复已停配置；本线程只3080Ti服务器，笔记本另一线程负责。

'''
    marker='## 最新执行：冻结MIX的方法开发外确认（2026-10-04）'
    assert marker in text
    record(goal,text.replace(marker,entry+marker,1))

if __name__=='__main__':main()
