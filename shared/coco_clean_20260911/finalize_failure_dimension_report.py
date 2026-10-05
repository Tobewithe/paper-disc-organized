"""Publish verified S067-S069 findings and correct the research entry point."""
from pathlib import Path
import json
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
ANALYSIS = ROOT / "diagnostics/failure_dimension_analysis_20260913"
EXTENT = ROOT / "diagnostics/joint_failure_extent_20260913"
STATS = ROOT / "diagnostics/recent_pilot_statistics_corrected_20260913"
REPORT_DIR = PROJECT / "refine-logs/coco-structure"


def main():
    d = pd.read_csv(EXTENT / "fixed_slot_effects.csv")
    d["own_coverage"] = 1-d.own_fn/d.own_pixels
    # Algebraic necessary condition, on the current *binary output*:
    # max IoU by removal alone = TP / |GT|. This is not a prototype bound.
    d["residual_state"] = np.select([
        d.new_iou >= .75,
        (d.new_iou < .75) & (d.own_coverage < .75)],
        ["rescued_by_rectangle", "requires_target_pixel_recovery"],
        default="sufficient_true_pixels_but_residual_false_pixels")
    d["target_recovery_proxy"] = np.select([
        d.residual_state.ne("requires_target_pixel_recovery"),
        d.support_proxy < .75],
        ["not_applicable", "proxy_support_also_below75"],
        default="proxy_support_at_least75_but_target_pixels_missing")
    d.to_csv(ANALYSIS / "joint_failure_residuals.csv", index=False)
    rows = []
    for group in ["all", "low", "middle", "high", "undefined"]:
        z = d if group == "all" else d[d.density.eq(group)]
        for state,g in z.groupby("residual_state"):
            rows.append(dict(group=group, state=state,n=len(g),total=len(z),percent=100*len(g)/len(z)))
    residual = pd.DataFrame(rows)
    residual.to_csv(ANALYSIS / "joint_failure_residual_summary.csv",index=False)
    d.groupby(["density","target_recovery_proxy"]).size().reset_index(name="n").to_csv(ANALYSIS/"target_recovery_support_proxy.csv",index=False)
    task = json.loads((EXTENT / "task_summary.json").read_text())
    support = pd.read_csv(ROOT / "diagnostics/joint_failure_support_20260913/summary.csv").set_index("group")
    state_names = {
        "rescued_by_rectangle":"仅矩形外裁切已达到Mask75",
        "requires_target_pixel_recovery":"自身已检出像素不足75%，必须补回自身",
        "sufficient_true_pixels_but_residual_false_pixels":"自身像素已够75%，但矩形内误报仍过多"}
    section = ["", "## 6. 进一步区分：仅纠正矩形范围能否恢复（S068/S069）", "",
        f"6,584个框差掩码差实例中，高E4组共2,292个：框覆盖GT代理中位数{100*support.loc['high','support_median']:.2f}%，只有298个（13.00%）代理覆盖低于75%。这提示‘低Box IoU’不能直接等同于‘框把大部分目标截掉’。", "",
        f"固定原二值掩码与候选，仅与自身GT掩码的整数紧外接矩形求交，保留所有原本正确的像素。完整COCO Mask AP由43.698至{100*task['mask_ap']:.3f}（+{task['delta_ap_points']:.3f}点），高E4 R75由52.132%至{100*task['hit75_high']:.3f}%，低高差由8.609至{100*task['gap_low_high']:.3f}点。高组619/2292（27.01%）固定槽位达到Mask75。", "",
        "这测量的是最终输出上理想矩形范围修复的收益，不是修改原型分辨率上的process_mask裁切。GT参与定位，无法作为方法成绩；它不补回裁切外遗漏，也不证明只有这些实例能经预测框改进恢复。", "",
        "再用一个精确的输出层恒等式分流：在不增加任何正确像素的条件下，即使删除全部误报，最大IoU也是现有TP/GT。它把剩余目标拆成必须恢复自身像素者与仍有可删除误报者。", "",
        "| 联合失败的后续状态 | 全部数量 | 高E4数量 | 高E4占该失败组 |", "|---|---:|---:|---:|"]
    for state in state_names:
        allr=residual[(residual.group=="all") & (residual.state==state)].iloc[0]
        hi=residual[(residual.group=="high") & (residual.state==state)].iloc[0]
        section.append(f"| {state_names[state]} | {int(allr.n):,} | {int(hi.n):,} | {hi.percent:.2f}% |")
    section += ["", "这个三分法只针对原有6,584个固定失败槽位，不能替代官方漏检率或AP分解。矩形内残余误报仍可能是背景、同类或异类目标；自身缺失也可能受裁切、原型或读出共同影响。下一次有模型前向的实验应按这两个残余状态预选样本，分别测精确裁切支持和固定原型读出能力，避免把性质不同的失败混在一起平均。", "",
        "[S069逐实例结果](../joint_failure_extent_20260913/fixed_slot_effects.csv) · [S069官方AP](../joint_failure_extent_20260913/task_summary.json) · [残余状态表](joint_failure_residual_summary.csv)", ""]
    with (ANALYSIS/"REPORT.md").open("a",encoding="utf-8") as f:
        f.write("\n".join(section))
    # Static exportable research figure, generated only from measured results.
    ap=pd.read_csv(ROOT/"diagnostics/failure_dimension_ap_20260913/task_summary.csv").set_index("arm")
    bars=["supported_goodbox_mask_gt","limited_goodbox_mask_gt","badbox_mask_gt"]
    values=[ap.loc[k,"delta_ap_points"] for k in bars]+[task["delta_ap_points"]]
    labels=["Good box, support >=95%\nideal mask (4,417)","Good box, support <95%\nideal mask (1,158)",
            "Poor box + mask\nideal mask (6,584)","Poor box + mask\nGT rectangle only (6,584)"]
    fig,axs=plt.subplots(1,2,figsize=(12,4.2),gridspec_kw={"width_ratios":[1.35,1]})
    colors=["#427b9b","#a2bdd0","#d58937","#745ba2"]
    ys=np.arange(4)
    axs[0].barh(ys,values,color=colors,height=.58)
    axs[0].set_yticks(ys,labels,fontsize=9);axs[0].invert_yaxis()
    axs[0].set_xlim(0,6.7);axs[0].set_xlabel("Mask AP gain (points), full COCO val2017")
    for i,v in enumerate(values):axs[0].text(v+.10,i,f"+{v:.2f}",va="center",fontsize=10)
    axs[0].set_title("GT-assisted output repair; gains are not additive",fontsize=10)
    gap=pd.read_csv(ANALYSIS/"gap_by_state.csv").set_index("state")
    gvalues=[gap.loc["box_bad_mask_bad","high_minus_low_miss_contribution"],
             gap.loc["box_good_mask_bad","high_minus_low_miss_contribution"],
             gap.loc["no_bbox50_assignment","high_minus_low_miss_contribution"]+
             gap.loc["box_good_mask_good","high_minus_low_miss_contribution"]]
    axs[1].barh(np.arange(3),gvalues,color=["#d58937","#427b9b","#8d969d"],height=.58)
    axs[1].set_yticks(np.arange(3),["Poor box + mask","Good box, poor mask","Other states"],fontsize=9);axs[1].invert_yaxis()
    axs[1].set_xlim(0,8.5);axs[1].set_xlabel("Contribution to low-high E4 R75 gap (points)")
    for i,v in enumerate(gvalues):axs[1].text(v+.10,i,f"{v:.2f}",va="center",fontsize=10)
    axs[1].set_title("Observed miss accounting; not a causal decomposition",fontsize=10)
    for ax in axs:
        ax.spines[["top","right"]].set_visible(False)
        ax.xaxis.grid(True,alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ANALYSIS/"failure_opportunity.png",dpi=180,bbox_inches="tight")
    fig.savefig(ANALYSIS/"failure_opportunity.pdf",bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(__file__, ANALYSIS/Path(__file__).name)
    # Central human-readable entry points.
    report=(ANALYSIS/"REPORT.md").read_text(encoding="utf-8")
    (REPORT_DIR/"FAILURE_DIMENSION_AP_RESULTS_20260913.md").write_text(
        "# S067—S069：COCO失败分类、AP机会与矩形范围诊断\n\n"
        "当前完整报告及逐实例数据：[REPORT.md](../../experiments/coco_clean_20260911/diagnostics/failure_dimension_analysis_20260913/REPORT.md)。\n\n"
        "重点结果：官方E4低高R75差8.609点中，7.449点在固定槽位框差掩码差状态；类别×大小共同支持标准化后该状态仍贡献6.959点。其理想完整掩码替换AP+5.629，但仅剪去真实矩形外像素AP+1.659，说明粗范围只解释一部分，不能把联合失败直接命名为框错误根因。4,417框好支持充分失败组理想掩码修复AP+4.630，但差距仅缩小0.453点。\n\n"
        "![失败机会](../../experiments/coco_clean_20260911/diagnostics/failure_dimension_analysis_20260913/failure_opportunity.png)\n",encoding="utf-8")
    means=pd.read_csv(STATS/"seed_means.csv")
    seeds=pd.read_csv(STATS/"paired_seed_results.csv")
    ci=json.loads((STATS/"paired_image_ci.json").read_text())
    text=["# S063/S065/S066：近期加权读出复算与停止决定", "",
        "同一冻结73D残差头，1,200训练图、300张已反复探索的train2017 transfer图、每臂三个种子各15轮。每个方法均与配对的普通raw COCO读出比较；没有重新训练网络或选择最佳种子。高组为ICI>0.5，非高组为ICI≤0.5；与S067的E4分组不同。", "",
        "| 实验 | s0 ΔAP | s1 ΔAP | s2 ΔAP | 平均ΔAP | 方法AP均值±种子标准差 | 平均高组ΔR75 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for _,r in means.iterrows():
        ss=seeds[seeds.experiment==r.experiment].sort_values("seed")
        text.append(f"| {r.experiment} | {ss.iloc[0].delta_ap:+.3f} | {ss.iloc[1].delta_ap:+.3f} | {ss.iloc[2].delta_ap:+.3f} | {r.mean_delta_ap:+.3f} | {r.mean_method_ap:.3f} ± {r.std_method_ap:.3f} | {r.mean_delta_high:+.3f} |")
    text += ["", "普通raw对照AP均值53.313。S065高组每个种子净增加1个GT；S066为+3/+3/+2个GT，平均AP仅+0.024点。它们不构成机制确认或稳定方法收益。", "",
        "旧S063/S065的isin图片重采样实现会去掉重复抽样计数，相关CI撤回。现以2,000次图片簇重采样保留重复计数与全部300张图，三种子先作为同一个实例的配对响应平均：", "",
        "| 实验 | 高组ΔR75 95%CI | 非高−高差距变化95%CI |", "|---|---|---|"]
    for name,val in ci.items():
        h=val['high'];g=val['gap_nonhigh_high']
        text.append(f"| {name} | {h['delta_points']:+.3f} [{h['ci95_points'][0]:+.3f}, {h['ci95_points'][1]:+.3f}] | {g['delta_points']:+.3f} [{g['ci95_points'][0]:+.3f}, {g['ci95_points'][1]:+.3f}] |")
    text += ["", "这些区间条件于三个已训练种子，不覆盖全部训练随机性，也未校正反复试验选择；没有计算AP显著性。", "",
        "实现限制：S063的1+3E4只加权BCE且未归一化，同时改变BCE/Dice比例；S065使用GT框采样位置上的平均冻结响应不确定性加权BCE；S066又对Dice乘factor，而raw没有，同时改变像素BCE平均尺度。三者均不能隔离密度或不确定性这一单一机制。实际每轮checkpoint保留，2臂×3种子×15=90个，旧完成回执继承135字段错误；实验ID及label-only干预描述也继承S032而不准确，补充更正记录，历史文件不改。", "",
        "决定：不扩这三个具体配方、不继续扫权重。转向S067揭示的联合失败状态，先做能区分空间支持与矩形内部错误的干预。S066只保留为探索性小信号，不能作为论文方法。", "",
        "[逐种子原始数表](../../experiments/coco_clean_20260911/diagnostics/recent_pilot_statistics_corrected_20260913/paired_seed_results.csv) · [统计代码与更正](../../experiments/coco_clean_20260911/diagnostics/recent_pilot_statistics_corrected_20260913/README.md)", ""]
    (REPORT_DIR/"RECENT_WEIGHTED_PILOT_CORRECTIONS_20260913.md").write_text("\n".join(text),encoding="utf-8")
    # Source captured after execution; explicitly say so, never rewrite receipts.
    snap=STATS/"source_captured_after_run"
    snap.mkdir(exist_ok=False)
    for name in ["run_density_weighted_readout.py","_generated_density_weighted_runner.py",
                 "run_uncertainty_weighted_readout.py","_generated_uncertainty_weighted_runner.py",
                 "run_pixel_uncertainty_readout.py","_generated_pixel_uncertainty_runner.py",
                 "train_pixel_uncertainty_readout.py","rich_pixel_readout.py"]:
        shutil.copy2(ROOT/name,snap/name)
    # Repair the malformed facts table and replace superseded interpretations.
    facts=PROJECT/"RESEARCH_FACTS.md"
    original=facts.read_text(encoding="utf-8")
    shutil.copy2(facts,ANALYSIS/"RESEARCH_FACTS_before_correction.md")
    early=re.findall(r"^\| F(?:2[4-9]|3[01]) \|.*$",original,re.M)
    value=re.sub(r"^\| F(?:2[4-9]|3[01]) \|.*\n?","",original,flags=re.M)
    loc=value.index("\n",value.index("| F23 |"))
    value=value[:loc]+"\n"+"\n".join(early)+value[loc:]
    replacement={
      "F24":"S058按同一预测槽位划分COCO val2017的36,335个普通GT：未分配同类Box50槽位3,704；框差掩码差6,584；框差掩码好1,106；框好掩码差5,575；框好掩码好19,366。后4,417个框好掩码差具有≥95%的框支持代理量。 | 支持是几何代理，不能等同精确decoder约束。固定槽位状态与官方任务匹配分开；具体当前表见S067。",
      "F29":"S062三种子AP52.772–52.805，低于同预算raw53.256–53.372。旧称信任域的实现实际是软hinge惩罚；非高ICI−高ICI差11.987–12.104，比原11.585更大。 | 背景权重及Dice缩放也变化，不能隔离支持保持机制；停止该配方，不列为更强基线。旧“低组”实际为非高组。",
      "F30":"更正S064/v5：4,417个支持充分的框好掩码差实例中157个有任意保留同类Mask75，严格排除后为4,260。25只计最佳框候选的Mask75，旧4,392池仍混入132个有好mask候选者。 | 撤回“25证明候选问题很小”和4,392干净池解释。v4未多计2,238成功；v5 evidence_tier旧字符串又把9,980状态落入默认成功。新表已显式修正，历史CSV不覆盖。",
      "F31":"S063复算：相对raw三种子平均AP−0.061，高ICI R75+0.112点，正确图片簇CI[−0.883,+1.365]；非高−高差变化−0.190[−1.369,+0.797]点。 | 旧isin重采样CI撤回。1+3E4只乘BCE且未归一化，包含损失尺度混杂；停止当前配方，不据此立机制主线。"
    }
    evidence="[S067—S069](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/failure_dimension_analysis_20260913/REPORT.md)"
    recent="[统计更正](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/RECENT_WEIGHTED_PILOT_CORRECTIONS_20260913.md)"
    for key,body in replacement.items():
        link=recent if key in ["F29","F31"] else evidence
        value=re.sub(rf"^\| {key} \|.*$",f"| {key} | {body} | {link} |",value,flags=re.M)
    value=value.replace("低−高差扩大至12.843–13.036","非高ICI−高ICI差扩大至12.843–13.036")
    appended=[
      f"| F32 | S065/S066加权pilot相对同容量raw控制三种子平均AP−0.083/+0.024；高ICI R75+0.336/+0.895点，正确图片簇CI均跨零。S065每种子净+1个高组GT，S066净+3/+3/+2。 | 不构成稳定方法。权重尺度、Dice缩放/GT支持采样等未完全隔离；300图已反复探索，停止继续扫权重。实际各90 checkpoint，旧135回执更正。 | {recent} |",
      f"| F33 | S067全val官方低E4−高E4 R75差8.609点，其中7.449点发生在框差掩码差状态，框好掩码差0.911，未分配Box50槽位0.239。类别×大小共同支持标准化后差8.072点，联合差组贡献6.959点。 | 这是共同分母的观测记账，不是框导致的因果比例；支持优先追6,584联合失败，而非只盯系数相似。分组为E4，不是ICI非高/高。 | {evidence} |",
      f"| F34 | S067各自独立替换理想GT掩码：框好支持充分4,417例AP+4.630，框好支持不足1,158例+0.864，框差掩码差6,584例+5.629。严格4,260例+4.462。联合差组修复使E4差距8.609→1.150，而支持充分好框组仅→8.156。 | GT输出修复非方法，完整掩码可能越过原框，不能称框修复贡献。AP各臂独立且不可相加，严格池为子集。无Box50槽位没有估算，不填0。 | {evidence} |",
      f"| F35 | S068/S069：高E4联合差2,292例框支持代理中位94.33%，298例低于75%。仅删除原mask在自身GT紧矩形外的像素，高组619例(27.01%)达到固定Mask75；全val AP+1.659，高E4 R75+5.078，差距8.609→7.369。 | 保持所有原TP的GT空间范围干预，非process_mask重裁切或自动方法。粗范围只能修一部分；残余按现有TP/GT<.75必须补自身，或已有足够TP但矩形内误报多继续分流。 | {evidence} |"
    ]
    pos=value.index("\n",value.index("| F31 |"))
    value=value[:pos]+"\n"+"\n".join(appended)+value[pos:]
    value=value.replace("当前覆盖可追溯的旧 COCO 审计及 S000—S057","当前覆盖可追溯的旧 COCO 审计及 S000—S069")
    intro="\n当前优先入口：[S067—S069失败状态、AP机会与矩形范围诊断](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/failure_dimension_analysis_20260913/REPORT.md)。按完整COCO任务缺口，下一机制追查优先针对框与掩码共同失败，并分清自身缺失与矩形内误报；不再继续盲扫加权损失。\n\n更正：F30、S063/S065旧重采样区间及S061—S066的“低组”、checkpoint回执含义以本次更正为准。历史原始文件保留，索引表修复了表头错位。\n"
    value=value.replace("## 1. 后续探索先记住这些事实",intro+"\n## 1. 后续探索先记住这些事实",1)
    value += "\n2026-09-13 S067—S069更新：完成失败类型独立GT掩码修复AP、官方差距记账与类别大小共同支持复算、联合失败支持代理和理想矩形范围干预；所有新增结论见F33—F35。同步纠正v5候选池、证据标签、近期pilot bootstrap及元数据。\n"
    facts.write_text(value,encoding="utf-8")
    tracker=REPORT_DIR/"EXPERIMENT_TRACKER.md"
    with tracker.open("a",encoding="utf-8") as f:
        f.write("\n\n# 2026-09-13 S064更正、S065/S066收尾、S067—S069完成\n\n"
        "S064旧4,392干净池及v4多计成功的解释撤回：严格池4,260；v5 evidence_tier默认成功错误已在新表显式修复。原始回执不覆盖。S063/S065旧isin bootstrap区间撤回，重复采样计数保留的复算见RECENT_WEIGHTED_PILOT_CORRECTIONS_20260913.md；S066 AP均值只比raw高0.024点，高R75+0.895的CI跨零。S061/S062亦有损失项缩放混杂，停止继续扫权重。\n\n"
        "S067 COMPLETE：完整5,000图36,335 GT，五臂153秒，baseline parity通过。支持充分好框掩码理想修复AP+4.630，支持不足+0.864，框差掩码差+5.629；后者缩小E4差距7.459点。官方失败记账中该状态贡献7.449/8.609点差距，非因果比例。S068为确定性支持/候选再分层；S069 COMPLETE：同6,584槽位只做GT紧矩形输出裁切，全部原TP保留、原IoU逐项对齐，AP+1.659，高E4固定槽位恢复619/2292。事实支持从联合差状态继续区分粗范围、自身缺失与矩形内误报，不能直接认定系数或框为根因。完整报告FAILURE_DIMENSION_AP_RESULTS_20260913.md。\n")
    print(residual[residual.group.isin(["all","high"])].to_string(index=False))
    print("Reports, factual ledger, tracker, source snapshots, PNG and PDF updated.")


if __name__ == "__main__":
    main()
