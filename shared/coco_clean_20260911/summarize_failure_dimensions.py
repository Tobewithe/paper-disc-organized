"""Descriptive official-recall gap accounting and ideal-mask AP report.

Unlike mutually exclusive mechanism claims, occurrence states are observable.
Official miss contributions are normalized by ALL GT in each density group,
so their high-minus-low differences sum exactly to the official recall gap.
"""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STATES = ["no_bbox50_assignment", "box_bad_mask_bad", "box_bad_mask_good", "box_good_mask_bad", "box_good_mask_good"]
NAMES = {"no_bbox50_assignment": "未分配同类 Box50 槽位", "box_bad_mask_bad": "框差、掩码差",
         "box_bad_mask_good": "框差、掩码好", "box_good_mask_bad": "框好、掩码差", "box_good_mask_good": "框好、掩码好"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--taxonomy", type=Path, required=True)
    p.add_argument("--ap", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    d = pd.read_csv(a.taxonomy / "instances_improved.csv")
    d["miss"] = d.task_mask75.eq("miss")
    rows = []
    for group in ["low", "middle", "high", "undefined"]:
        g = d[d.mask_density.eq(group)]
        for state in STATES:
            z = g[g.primary_state.eq(state)]
            rows.append(dict(group=group, state=state, n=len(z), denominator=len(g),
                occurrence_percent=100*len(z)/len(g), official_misses=int(z.miss.sum()),
                miss_contribution_points=100*z.miss.sum()/len(g)))
    accounting = pd.DataFrame(rows)
    accounting.to_csv(a.out / "official_gap_accounting.csv", index=False)
    gap = accounting[accounting.group.isin(["low", "high"])].pivot(index="state", columns="group", values="miss_contribution_points").reindex(STATES)
    gap["high_minus_low_miss_contribution"] = gap.high-gap.low
    gap.to_csv(a.out / "gap_by_state.csv")
    task = pd.read_csv(a.ap / "task_summary.csv").set_index("arm")
    assert np.isclose(gap.high.sum()-gap.low.sum(), 100*task.loc["original", "gap_low_high"])
    # Composition diagnostic with a fixed minimum common support of 5 GT per
    # density group and COCO category x official area bin. This is not causal.
    hl = d[d.mask_density.isin(["low", "high"])].copy()
    keys = ["category_id", "area_bin"]
    counts = hl.groupby(keys+["mask_density"]).size().unstack(fill_value=0)
    eligible = counts[(counts.low >= 5) & (counts.high >= 5)]
    common = hl.merge(eligible.reset_index()[keys], on=keys, how="inner", validate="many_to_one")
    weights = eligible.sum(axis=1)/eligible.to_numpy().sum()
    standardized = {}
    for group in ["low", "high"]:
        q = common[common.mask_density.eq(group)]
        rates = q.groupby(keys).miss.mean()
        standardized[group] = float(100*(rates*weights).sum())
    common_raw_gap = 100*(common[common.mask_density.eq("high")].miss.mean()-common[common.mask_density.eq("low")].miss.mean())
    composition = dict(strata="category_id x COCO annotation area_bin", min_gt_per_group=5,
        strata_count=len(eligible), original_counts=hl.groupby("mask_density").size().to_dict(),
        retained_counts=common.groupby("mask_density").size().to_dict(),
        original_gap_points=float(gap.high.sum()-gap.low.sum()),
        common_population_raw_gap_points=float(common_raw_gap),
        common_population_standardized_gap_points=standardized["high"]-standardized["low"],
        weighting="pooled low+high common-support GT counts; no adjustment for Box IoU or outcome",
        caveat="Observational composition check, no causal effect; excluded strata are explicitly counted")
    standard_rows = []
    for state in STATES:
        vals = {}
        for group in ["low", "high"]:
            q = common[common.mask_density.eq(group)].copy()
            q["state_miss"] = q.primary_state.eq(state) & q.miss
            rates = q.groupby(keys).state_miss.mean()
            vals[group] = float(100*(rates*weights).sum())
        standard_rows.append(dict(state=state, low=vals["low"], high=vals["high"], contribution=vals["high"]-vals["low"]))
    assert np.isclose(sum(x["contribution"] for x in standard_rows), composition["common_population_standardized_gap_points"])
    pd.DataFrame(standard_rows).to_csv(a.out / "composition_adjusted_gap.csv", index=False)
    (a.out / "composition.json").write_text(json.dumps(composition, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.crosstab(d.primary_state, d.task_mask75).to_csv(a.out / "fixed_state_vs_official_task.csv")
    names = {"original": "原模型", "supported_goodbox_mask_gt": "仅理想修复：框好、支持≥95%、掩码差",
             "limited_goodbox_mask_gt": "仅理想修复：框好、支持<95%、掩码差",
             "badbox_mask_gt": "仅理想修复：框差、掩码差",
             "strict_supported_mask_gt": "仅理想修复：严格 4,260 实例池"}
    report = ["# COCO 失败维度与理想修复 AP（S067）", "",
        "2026-09-13。5,000 张 COCO val2017、36,335 个普通 GT、446,097 个缓存最终候选。原模型采用已冻结的 one-to-many + NMS 分支。复算基线 AP/AP50/AP75 和 R75 与 S056 完全一致；本次没有训练，也没有改变模型预测分数。", "",
        "## 1. 五个主状态，其他维度独立保留", "",
        "主状态使用同类别 Box IoU≥0.50 的官方一对一诊断匹配，并在同一预测槽位上计算 Box75 与 Mask75。未分配 Box50 槽位不等于完全没有候选。官方分割任务匹配另存，不能用固定槽位成功数代替任务召回。", "",
        "| 同一槽位状态 | GT数 | 占全部GT | 低 E4 组发生率 | 中 E4 组发生率 | 高 E4 组发生率 |", "|---|---:|---:|---:|---:|---:|"]
    for state in STATES:
        n = int(d.primary_state.eq(state).sum())
        z = accounting[accounting.state.eq(state)].set_index("group")
        report.append(f"| {NAMES[state]} | {n:,} | {100*n/len(d):.2f}% | {z.loc['low','occurrence_percent']:.2f}% | {z.loc['middle','occurrence_percent']:.2f}% | {z.loc['high','occurrence_percent']:.2f}% |")
    report += ["", "E4 表示自身可见掩码边界在 4 个模型输入等效像素内邻近同类 GT 的比例。沿用 S048 分组：低=0，中=(0,0.2)，高≥0.2；各组21,391/5,342/9,453，另149个几何未定义。ICI（框拥挤）及按正暴露中位数分组均是其他口径，不能混称。", "",
        "框好掩码差的5,575个实例进一步保留支持代理量：4,417个≥95%，1,158个<95%。前4,417个中，157个存在另一个最终同类 Mask75 候选；严格排除任何可用同类Mask75及官方任务成功后剩4,260个。最佳框候选达到Mask75仅25个，不能据此把余下4,392个称为无好候选池。支持代理量不是精确解码裁切上界。", "",
        "保留六类独立维度：①固定槽位框/掩码状态；②官方任务成败及最终候选可用性；③框对GT的几何支持；④自身遗漏、同类独占/异类独占/歧义邻居/背景误报；⑤C/I/L/S/O/M/X关系图；⑥GT边界E4、ICI、类别与大小。关系、空间误差或阈值状态均不能直接命名网络根因。", "",
        "## 2. 官方高低拥挤召回差距落在哪些状态", "",
        "以下每项是‘该状态且官方 Mask75 失败的 GT数 / 该拥挤组全部 GT数’。各项高减低之和严格等于低组减高组的官方 R75 差距，属于观测记账，不是因果分解。", "",
        "| 固定槽位状态 | 低组失败贡献 | 高组失败贡献 | 高−低差（百分点） |", "|---|---:|---:|---:|"]
    for state, row in gap.iterrows():
        report.append(f"| {NAMES[state]} | {row.low:.3f} | {row.high:.3f} | {row.high_minus_low_miss_contribution:+.3f} |")
    report += [f"| 合计 | {gap.low.sum():.3f} | {gap.high.sum():.3f} | {gap.high.sum()-gap.low.sum():+.3f} |", "",
        f"类别×大小共同支持检查保留低组{composition['retained_counts']['low']:,}、高组{composition['retained_counts']['high']:,}个GT，共{len(eligible)}个分层。相同保留总体的未调整差距为{common_raw_gap:.3f}点，按共同类别×大小组成标准化后为{composition['common_population_standardized_gap_points']:.3f}点。这个比较仍不能排除遮挡、形状等其他因素。", "",
        "## 3. 每类独立理想修复后的真实 COCOeval AP", "",
        "仅将对应失败槽位掩码替换为它的完整 COCO GT，保持分数、类别、顺序和候选数；每个实验独立从原预测开始，全部GT参与官方匹配后再分组。完整GT可能超出预测框，因此这是理想输出修复，不是‘只修改框’或可部署方法，也不是每类AP损失的加性分解。", "",
        "| 处理 | 修改GT数 | Mask AP | ΔAP | 高E4 R75 | 低−高R75差 |", "|---|---:|---:|---:|---:|---:|"]
    for arm, row in task.iterrows():
        report.append(f"| {names[arm]} | {int(row.selected_gt):,} | {100*row.mask_ap:.3f} | {row.delta_ap_points:+.3f} | {100*row.hit75_high:.3f}% | {100*row.gap_low_high:.3f} |")
    report += ["", "未分配 Box50 槽位组没有可直接替换的槽位，本实验不为它编造 AP 收益，也不能填0。需要中间候选和分数/筛选证据后设计不同干预。严格4,260池是4,417池的子集，不能把两者增益相加。所有AP都来自完整验证集，不能由失败数量折算；AP没有置信区间或显著性声明。", "",
        "## 4. 对后续路线的直接作用", "",
        "框好、支持充分但掩码差确实有较大的整体AP机会；不过其理想修复几乎同步改善高低E4组，差距仅从8.609缩至8.156点。框差掩码差组的理想修复将差距缩至1.150点，因而成为当前密集差距目标的优先追查对象。这不证明框是原因：定位误差与掩码错误也可能共同来自尺度、可见形状或特征混杂。", "",
        "下一次实验应回答一个区分性问题：在该组固定原型和系数时，仅纠正裁切支持能恢复多少；若仍差，再固定支持比较原系数与GT辅助系数的可恢复性。加入同类别、相近大小的低拥挤对照，完整记录各组无效支持与残余错误。如果收益主要来自裁切，追定位/支持一致性；如果支持修正后仍有大幅系数可恢复空间，再追掩码读出。使用新采样的训练开发图做机制实验，原验证集不作为不断挑超参数的开发集。", "",
        "## 5. 更正与可追溯性", "",
        "撤回旧v5对4,392‘干净池’以及25个足以排除候选问题的解释。v4的18,234是条件成功数，并没有比v5多计2,238个成功；此前把层级展开说成修正成功数是错误表述。v5的evidence_tier又因旧标签字符串未更新，把9,980个框差/支持不足状态落入默认成功提示；新表按五个观测状态显式映射。历史CSV与回执不覆盖，新表为后续入口。", "",
        f"- 逐实例分类：`{a.taxonomy.resolve() / 'instances_improved.csv'}`",
        f"- AP与逐GT任务结果：`{a.ap.resolve()}`",
        "- 本目录：`official_gap_accounting.csv`、`gap_by_state.csv`、`composition_adjusted_gap.csv`、`composition.json`、`fixed_state_vs_official_task.csv`。", ""]
    (a.out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    shutil.copy2(__file__, a.out / Path(__file__).name)
    print(gap.round(4).to_string())
    print(json.dumps(composition, ensure_ascii=False, indent=2))
    print(task[["mask_ap", "delta_ap_points", "hit75_high", "gap_low_high"]].round(5).to_string())


if __name__ == "__main__":
    main()
